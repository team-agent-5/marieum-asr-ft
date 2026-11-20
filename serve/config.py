"""서버 설정 (환경변수). docs/engine_handoff.md 3.2, 3.5.

VAD·길이 설정은 marieum-ai server/config.py(commit 2ccebf6)와 이름과 기본값이 같다.
colab/marieum_stream.Segmenter가 이 값들을 그대로 읽는다.
"""

import json
import os
from dataclasses import dataclass, field


def _i(name, default):
    return int(os.environ.get(name, default))


def _f(name, default):
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    # ---- 모델 (3.2) ----
    nemo_model: str = os.environ.get("ML_NEMO_MODEL", "")  # ft_v3.nemo 경로. 레포에 넣지 않는다
    nemo_repo: str = os.environ.get("ML_NEMO_REPO", "NeMo")  # NeMo 클론 (examples/ 아래 기본 설정을 읽는다)
    model_version: str = os.environ.get("ML_MODEL_VERSION", "marieum-nemotron-ft_v3")
    att_context: list = field(default_factory=lambda: json.loads(os.environ.get("ML_NEMO_ATT_CONTEXT", "[56,3]")))
    language_code: str = os.environ.get("ML_NEMO_LANG", "ko-KR")
    compute_dtype: str = os.environ.get("ML_NEMO_DTYPE", "bfloat16")  # EXP-015: bf16 27.86 < fp32+amp 28.51

    # ---- GPU 작업자 (3.3, 3.5) ----
    batch_max: int = _i("ML_NEMO_BATCH_MAX", 8)
    eou_ms: int = _i("ML_NEMO_EOU_MS", 800)  # 파이프라인 글자 확정용. 끄면 5배 넘게 느려진다 (3.4)
    step_wait_ms: int = _i("ML_NEMO_STEP_WAIT_MS", 10)
    max_backlog_ms: int = _i("ML_NEMO_MAX_BACKLOG_MS", 5000)
    max_sessions: int = _i("ML_NEMO_MAX_SESSIONS", 16)
    # POST /asr 파일도 끝날 때까지 상태 칸 하나를 쓴다. 넘는 요청은 기다린다
    asr_max_concurrent: int = _i("ML_ASR_MAX_CONCURRENT", 4)

    # ---- VAD·구간 (v1과 같은 값, marieum-ai CALIBRATION.md §0.2) ----
    vad_threshold: float = _f("ML_VAD_THRESHOLD", 0.5)
    speech_pad_ms: int = _i("ML_SPEECH_PAD_MS", 200)
    min_speech_ms: int = _i("ML_MIN_SPEECH_MS", 150)
    end_silence_ms: int = _i("ML_END_SILENCE_MS", 1750)
    utterance_max_s: float = _f("ML_UTTERANCE_MAX_S", 10.0)
    max_segment_s: float = _f("ML_MAX_SEGMENT_S", 15.0)
    session_max_s: float = _f("ML_SESSION_MAX_S", 600.0)
    asr_max_s: float = _f("ML_ASR_MAX_S", 60.0)
    # Segmenter가 읽지만 v2에서는 쓰지 않는다 (interim을 따로 만들지 않는다)
    interim_step_ms: int = _i("ML_INTERIM_STEP_MS", 500)
    min_interim_speech_ms: int = _i("ML_MIN_INTERIM_SPEECH_MS", 300)

    @property
    def num_slots(self):
        """파이프라인 상태 칸. 칸이 모자라면 NeMo가 step 전체를 실패시킨다 (같이 묶인 세션까지).
        세션 + 파일 한도에, 끊긴 세션이 마지막 조각으로 칸을 비우기 전까지 쓸 여유(batch_max)를 더한다."""
        return max(32, self.max_sessions * 2, self.max_sessions + self.asr_max_concurrent + self.batch_max)


settings = Settings()
