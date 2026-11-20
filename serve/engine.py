"""Nemotron ft_v3 엔진: NeMo 공식 스트리밍 파이프라인(CacheAwareRNNTPipeline)을 감싼다 (engine_handoff 3.1 ~ 3.3).

서버의 나머지 부분은 torch·NeMo를 모른다. 이 파일만 numpy ↔ torch Frame을 바꾼다.
테스트에서는 같은 모양(chunk_samples, step, warmup)의 가짜 엔진을 넣는다 (tests/fakes.py).

시험한 NeMo: GitHub NVIDIA/NeMo main eaf0098dde. 다른 커밋에서는 Frame, TranscribeStepOutput 모양이 다를 수 있다.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
BASE_CFG = "examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml"


@dataclass
class Req:
    """파이프라인 한 조각. samples는 늘 chunk_samples 길이이고, 마지막 조각은 0으로 채운 뒤 length에 실제 길이를 적는다."""
    stream_id: int
    samples: np.ndarray
    is_first: bool
    is_last: bool
    length: int
    audio_end_ms: int  # 이 조각까지 넣은 오디오 끝 (세션 시작 기준)


@dataclass
class StepOut:
    stream_id: int
    final_text: str  # 직전 발화 끝(EoU) 이후 이번에 확정된 글자. 앞에 구분 공백이 붙을 수 있다
    final_segments: list = field(default_factory=list)  # [(start_s, end_s, conf)]
    partial_text: str = ""  # 마지막 확정 이후 지금까지 (확정 전)


def make_cfg(s):
    from omegaconf import OmegaConf
    base = OmegaConf.load(Path(s.nemo_repo) / BASE_CFG)
    over = {
        "asr": {"model_name": s.nemo_model, "strip_lang_tags": True, "compute_dtype": s.compute_dtype,
                "use_amp": False, "decoding": {"strategy": "greedy_batch",
                                               "greedy": {"preserve_frame_confidence": True}}},
        "lang": s.language_code,
        "streaming": {"att_context_size": list(s.att_context), "chunk_size_in_secs": None,
                      "batch_size": s.batch_max, "num_slots": s.num_slots, "request_type": "frame"},
        "endpointing": {"stop_history_eou": s.eou_ms},
        "enable_itn": False, "enable_nmt": False, "asr_output_granularity": "segment",
        "return_tail_result": True,  # colab/pipeline_eval.py(EXP-015 ~ 017)와 같다
        "log_level": 30,
    }
    return OmegaConf.merge(base, OmegaConf.create(over))


class NemotronEngine:
    def __init__(self, s):
        import torch
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

        if not s.nemo_model or not Path(s.nemo_model).exists():
            raise FileNotFoundError(f"ML_NEMO_MODEL이 없다: {s.nemo_model!r}")
        self._torch, self._Frame = torch, Frame
        # 첫 조각에만 붙인다. 파이프라인 기본값(lang, endpointing)과 같은 값을 명시해 둔다
        self._opts = ASRRequestOptions(language_code=s.language_code, stop_history_eou=s.eou_ms)
        t = time.perf_counter()
        self.pipe = PipelineBuilder.build_pipeline(make_cfg(s))
        self.pipe.open_session()
        self.load_s = time.perf_counter() - t
        self.chunk_samples = int(round(self.pipe.chunk_size_in_secs * SR))
        self.gpu = {"name": torch.cuda.get_device_name(0)} if torch.cuda.is_available() else {}

    def step(self, reqs):
        """GPU 작업자 스레드 하나에서만 부른다. 한 번에 같은 stream_id 조각을 둘 넣지 않는다 (3.3)."""
        frames = [self._Frame(samples=self._torch.from_numpy(r.samples), stream_id=r.stream_id,
                              is_first=r.is_first, is_last=r.is_last, length=r.length,
                              options=self._opts if r.is_first else None) for r in reqs]
        outs = self.pipe.transcribe_step(frames)
        return [StepOut(o.stream_id, o.final_transcript,
                        [(float(g.start), float(g.end), float(g.conf)) for g in (o.final_segments or [])],
                        o.partial_transcript) for o in outs]

    def warmup(self, batch_max):
        """준비 완료(/healthz 200) 전에 1개, batch_max개 묶음으로 몇 조각씩 돌려 둔다."""
        n = self.chunk_samples
        sid = 10**9
        for b in sorted({1, batch_max}):
            ids = list(range(sid, sid + b))
            sid += b
            for k in range(4):
                self.step([Req(i, np.zeros(n, np.float32), k == 0, k == 3, n, 0) for i in ids])
