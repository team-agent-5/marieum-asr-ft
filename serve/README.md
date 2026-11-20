# serve: Nemotron ft_v3 ML 서버

[`docs/engine_handoff.md`](../docs/engine_handoff.md) 9절의 `/asr-stream/v2`와 `POST /asr`를 구현한 서버다. marieum-ai에 붙이기 전에 이 레포에서 엔진에 맞춰 만들고 시험한다. 교체가 정해지면 `serve/`를 marieum-ai `server/` 옆으로 옮기고 `ML_ENGINE` 스위치를 붙인다.

| API | 내용 |
|---|---|
| `WS /asr-stream/v2` | 실시간. `start` → `ready` → (`segment_start` → `transcript_delta`* → `segment_final`)* → `session_end`. 메시지 스펙은 handoff 9.3 ~ 9.5 |
| `POST /asr` | 파일 (WAV 또는 PCM16 + `?sample_rate=`, 60초까지). 응답 모양은 v1과 같다. `segments`는 파이프라인 확정(EoU 800ms) 단위 |
| `GET /healthz` | 로딩 중 503. 준비되면 `engine`, `sessions`, `gpu_pending`, `step_ms_p50/p95`, `batch_mean` |
| `GET /livez` | 프로세스 생존 |

v1 `/asr-stream`(Whisper)은 이 서버에 없다. 백엔드가 v1 모양을 그대로 써야 하면 handoff 3.7 어댑터를 따로 만든다.

## 구조

| 파일 | 하는 일 |
|---|---|
| `engine.py` | NeMo `CacheAwareRNNTPipeline` 만들기(3.2 설정), numpy ↔ `Frame`, 워밍업. torch·NeMo는 이 파일만 안다 |
| `scheduler.py` | GPU 스레드 하나. 세션마다 가장 오래된 0.32초 조각 하나씩 최대 `ML_NEMO_BATCH_MAX`개를 묶어 `transcribe_step` |
| `stream_v2.py` | 세션 하나의 상태. VAD 구간(운영 `Segmenter`, 쉼 0.8초)과 파이프라인 글자를 섞는다 (3.4) |
| `app.py` | FastAPI. 연결당 세션 하나, 메시지 순서(`seq`), 오류 코드, 한도 |

문장 끊기 규칙 (`stream_v2.py` 맨 위 설명):
- 파이프라인 확정은 그것을 낸 조각의 오디오 위치로 VAD 구간에 붙인다
- 구간이 닫혀도 파이프라인이 그 위치까지 처리한 뒤에 `segment_final`을 보낸다
- 150ms보다 짧은 소리(숨소리)는 구간을 열지 않는다

모드 (`start.mode`): 파이프라인은 두 모드 모두 세션 오디오 전체를 stream 하나로 이어서 처리한다. 모드는 그 글자를 앱에 보일 문장으로 어떻게 묶느냐만 바꾼다.

| | `conversation` (기본) | `utterance` |
|---|---|---|
| 문장 끝 | 서버 VAD 쉼 0.8초(`ML_END_SILENCE_MS`), 쉼 없이 15초면 자름 | 백엔드가 `end`를 보낼 때 |
| 세션 하나에 문장 | 여러 개 (`segment_id` 0, 1, 2…) | 하나 |
| 세션 길이 상한 | 600초 | 10초 |

## handoff와 다르게 한 곳

| handoff | 구현 | 이유 |
|---|---|---|
| 3.4: 확정 글자를 `final_segments` 시각으로 VAD 구간에 나눠 붙인다 | 구간이 닫히면 파이프라인이 그 위치까지 처리하기를 기다렸다가, 그때까지 나온 글자(확정 + 확정 전)를 모두 그 구간에 넣는다 | 11-20 VAD 쉼을 0.8초로 줄였다. 확정(말 끝 뒤 p50 1.1초, EXP-017)보다 구간이 먼저 닫히므로 시각으로 나눌 확정이 아직 없다 |
| 3.4: VAD 쉼 1.75초 | 0.8초 (11-20). 닫을 때 확정 전 글자를 그 구간에 넣고, 나중에 같은 글자가 확정되어 다시 오면 다음 구간 앞에서 뺀다 (`stream_v2.sent_partial`). 그 구간 confidence는 확정 전 토큰들의 confidence(`partial_confs`, NeMo `state.confidences`)로 낸다 | 문장 확정을 빨리. 실제 ft_v3 + Silero VAD로 TTS 60문장(구 사이 쉼 0.3~1.5초)을 돌려 실시간 문장들을 이은 글자가 파일 전사와 60/60 같았다 (0.8초, 1.75초 모두). 0.8초에서 문장 수는 평균 1.02 → 2.02 (말 중간 1초 쉼에서 나뉜다) |
| 3.3: `start`에서 빈 `stream_id`를 받는다 | 1, 2, 3…으로 계속 늘린다 | NeMo가 `stream_id`를 상태 칸에 딕셔너리로 연결한다 (`context_manager.py`). 번호가 커도 된다 |
| 3.5에 없음 | `ML_ASR_MAX_CONCURRENT`(4): `POST /asr` 동시 처리 한도. 넘는 요청은 기다린다 | 파일도 끝날 때까지 상태 칸 하나를 쓴다. 칸이 모자라면 NeMo가 step 전체를 실패시켜서, 같이 묶인 실시간 세션까지 `INTERNAL`로 끝난다. `num_slots`는 세션 한도 + 파일 한도 + `batch_max` 이상으로 잡는다 |

**confidence 집계 (11-20):** 파이프라인은 확정마다 그 토큰들의 평균 confidence를 준다. 구간(`segment_final`)과 파일(`segments`, 전체)의 confidence는 그 확정들을 **토큰 수로 가중 평균**한다 (`stream_v2.token_conf`). 확정마다 단순 평균하면 1토큰 확정(0.35) 하나가 13토큰 확정(0.76)과 같은 무게가 되어 0.55로 떨어진다. 토큰 수는 확정 글자를 모델 토크나이저로 다시 나눠 센다 (`engine.n_tokens`).

| TTS 360문장 실험 (9목소리 × 2속도 × 20문장, 구 사이 쉼 0.9~1.5초, 잡음 SNR 없음/15/5/0) 중 글자가 나온 102문장 | 확정 단순 평균 | 토큰 가중 평균 |
|---|---|---|
| Spearman(confidence, CER) (음수일수록 좋다) | -0.562 | **-0.594** |
| AUROC: CER > 0.2 문장을 낮은 confidence로 가려내기 | 0.731 | **0.758** (차이 95% CI +0.003 ~ +0.053) |
| AUROC: CER > 0 | 0.766 | **0.794** (CI -0.009 ~ +0.069) |

차이는 작지만 모든 지표에서 토큰 가중이 낫다.

**confidence 계산 방식 (11-20, 같은 360문장, 토큰 가중):** 방식을 바꿔도 전사 글자는 360문장 모두 같았다. 점수의 크기와 가려내는 힘만 달라진다.

| `confidence.method_cfg` | 맞은 문장(CER 0) 중앙값 (범위) | 틀린 문장(CER > 0.2) 중앙값 | AUROC CER > 0.2 |
|---|---|---|---|
| entropy tsallis α=0.5 exp (NeMo 기본, 지금) | 0.561 (0.372 ~ 0.698) | 0.332 | **0.758** |
| entropy tsallis α=0.33 exp | 0.213 (0.034 ~ 0.439) | 0.062 | 0.762 |
| entropy gibbs α=1 lin | 0.963 (0.943 ~ 0.979) | 0.934 | 0.719 |
| max_prob | 0.914 (0.857 ~ 0.952) | 0.856 | 0.703 |

지금 방식을 둔다. 점수가 낮게 나오는 척도이므로 백엔드 기준값은 이 점수로 다시 정한다 (`docs/backend_api_v2.md`). 같은 실험에서 360문장 중 258문장은 글자가 하나도 안 나왔다 (잡음 없는 문장도 85개 중 37개). 사람 목소리가 아닌 TTS라서 생긴 일인지는 따로 봐야 한다 (handoff 6절 5단계 정상 발화 확인).

`transcript_delta`는 열린 구간에만 보낸다. 오디오를 실제 속도보다 빨리 보내서 VAD가 구간을 먼저 닫으면, 그 구간 글자는 delta 없이 `segment_final`로만 온다.

## 실행 (GPU 서버)

11-20 GPU 서버(L4, Ubuntu 24.04, sudo 없음)에 이 순서로 설치해서 띄웠다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # ~/.local/bin/uv
uv python install 3.12 && uv venv --managed-python --python 3.12 .venv-nemo && . .venv-nemo/bin/activate
printf 'torch==2.11.0+cu128\ntorchaudio==2.11.0+cu128\n' > /tmp/torch-constraints.txt
IDX="--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match"
uv pip install $IDX torch==2.11.0+cu128 torchaudio==2.11.0+cu128
git clone https://github.com/NVIDIA/NeMo ~/NeMo && git -C ~/NeMo checkout eaf0098ddeffb804ef1b1fb3a955b3bf022f4171
uv pip install $IDX -c /tmp/torch-constraints.txt -e ~/NeMo[asr]
uv pip install $IDX -c /tmp/torch-constraints.txt -r serve/requirements.txt && uv pip check

ML_NEMO_MODEL=~/models/ft_v3.nemo ML_NEMO_REPO=~/NeMo \
  uvicorn serve.app:app --host 0.0.0.0 --port 8001 --workers 1
```

- **Python 3.12 이상:** NeMo `eaf0098`이 요구한다. 3.11로는 설치되지 않는다
- **Python 헤더가 있는 Python:** 모델을 불러올 때 Triton이 C 파일을 `gcc`로 컴파일하면서 `Python.h`를 찾는다. 시스템 `python3.12`에 `python3.12-dev`가 없으면 `model_load_failed`(`fatal error: Python.h`)로 멈춘다. sudo가 없으면 uv 관리 Python(`--managed-python`)을 쓴다
- **torch 고정:** NeMo 의존성이 torch를 바꾸지 못하게 constraints로 묶는다
- 실측 (11-20, L4): 모델 로딩 30.7초, 워밍업 24.3초, `chunk_ms` 320
- `ft_v3.nemo`는 Drive `cdy-hack-ag5/marieum-asr-ft/ckpt/ft_v3/v1/checkpoints/`에 있다 (2,553,098,240 바이트). 서버에 올린 뒤 `sha256sum`을 남긴다. 모델 파일과 Drive 링크는 레포에 넣지 않는다
- 환경변수는 handoff 3.5 표와 `config.py`. VAD 값(`ML_END_SILENCE_MS` 등)은 v1과 같다
- 반드시 `--workers 1`

## 테스트 (GPU 없이)

```bash
pip install -r serve/requirements.txt
pytest
```

`tests/fakes.py`의 가짜 엔진이 NeMo 파이프라인 규칙(stream별 상태, `is_first`/`is_last`, 한 step에 같은 stream 하나, 상태 칸 수, 글자 안의 띄어쓰기)을 어기면 바로 실패한다. 테스트 29개 (11-20 통과).

발화마다 남는 로그 (handoff 9.6, 글자는 찍지 않는다): `segment_final` 이벤트에 `speech_ms`, `first_delta_ms`(말 시작 오디오 도착 → 첫 delta, delta가 없었으면 null), `final_ms`(말 끝 오디오 도착 → final, `latency_ms`와 같은 값), `chars`, `deltas`, `revised`, `reason`, `confidence`.

## GPU에서 아직 확인하지 않은 것

11-20 GPU 서버에 띄워 1번은 확인했고, 4번은 값이 나오는 것까지 봤다. 맥 TTS 한국어 문장 하나(2.8초)로 `POST /asr`와 `/asr-stream/v2`가 끝까지 돈다.

1. ~~`NemotronEngine`이 뜨는가~~ **확인 (11-20):** `model_loaded`, `chunk_ms` 320, `/healthz` 200
2. 마지막 조각을 0으로 채우고 `length`에 실제 길이를 넣는 방식이 파이프라인에서 문제없는가. 남은 오디오가 없으면 무음 한 조각(0.32초)을 `is_last`로 보낸다
3. `final_segments`의 `start`/`end`: NeMo 코드로는 초 단위다 (`bpe_decoder.py`에서 토큰 위치 × 토큰 길이(초)). 실제 값이 stream 시작 기준인지 본다. `POST /asr`의 `start_ms`/`end_ms`에만 쓴다
4. `confidence`가 0이 아닌 값으로 나오는가. **11-20:** 나온다. 같은 소리면 파일과 실시간이 같은 값이다 (0.7029). 처음 본 실시간 0.55는 시험 스크립트가 WAV 헤더(`FLLR` 4KB)를 소리로 넣어서 생긴 값이었다. 분포는 아직 모른다. NeMo 기본은 `preserve_frame_confidence: false`(이때 모두 0.0)이고 `engine.py`가 `true`로 켠다. 방식은 entropy(tsallis, alpha 0.5)의 평균이다. 지금까지 실험(EXP-015 ~ 017)은 신뢰도를 끄고 돌렸으므로 값의 분포를 처음 본다. 백엔드가 confidence 기준값을 쓰면 test_v2 녹음으로 분포를 보고 정한다. 켠 상태의 step 시간도 다시 잰다
5. 같은 녹음을 `conversation`과 `utterance`로 넣어 이어 붙인 글자가 같은가 (코드상 같아야 한다)
6. test_v2 녹음 몇 개를 v2로 실제 속도로 흘려 CER이 EXP-015(27.86)와 같은 수준인가, 지연 목표(말 끝 → 마지막 delta p95 ≤ 800ms, 4세션 `OVERLOADED` 0, step p95 ≤ 160ms)를 지키는가 (handoff 6절 4단계)

## 알려진 제약

- `sample_rate`가 16000이 아니면 WebSocket 프레임마다 따로 16kHz로 바꾼다. 프레임 경계에서 필터가 이어지지 않아 소리가 조금 어긋날 수 있다. 앱은 16kHz로 보내는 것을 권한다
- `POST /asr`는 한 step에 그 파일의 조각 하나씩만 넣는다 (같은 stream 조각 둘을 한 step에 넣지 않는 규칙). 60초 파일은 조각 188개라서, step 하나가 60ms면 11초쯤 걸린다. GPU에서 재 본다
