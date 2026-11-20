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
| `stream_v2.py` | 세션 하나의 상태. VAD 구간(운영 `Segmenter`, 쉼 1.75초)과 파이프라인 글자를 섞는다 (3.4) |
| `app.py` | FastAPI. 연결당 세션 하나, 메시지 순서(`seq`), 오류 코드, 한도 |

문장 끊기 규칙 (`stream_v2.py` 맨 위 설명):
- 파이프라인 확정은 그것을 낸 조각의 오디오 위치로 VAD 구간에 붙인다
- 구간이 닫혀도 파이프라인이 그 위치까지 처리한 뒤에 `segment_final`을 보낸다
- 150ms보다 짧은 소리(숨소리)는 구간을 열지 않는다

모드 (`start.mode`): 파이프라인은 두 모드 모두 세션 오디오 전체를 stream 하나로 이어서 처리한다. 모드는 그 글자를 앱에 보일 문장으로 어떻게 묶느냐만 바꾼다.

| | `conversation` (기본) | `utterance` |
|---|---|---|
| 문장 끝 | 서버 VAD 쉼 1.75초, 쉼 없이 15초면 자름 | 백엔드가 `end`를 보낼 때 |
| 세션 하나에 문장 | 여러 개 (`segment_id` 0, 1, 2…) | 하나 |
| 세션 길이 상한 | 600초 | 10초 |

## handoff와 다르게 한 곳

| handoff | 구현 | 이유 |
|---|---|---|
| 3.4: 확정 글자를 `final_segments` 시각으로 VAD 구간에 나눠 붙인다 | 확정을 낸 조각의 오디오 끝(`emit_ms`)으로 붙인다 | 확정은 말 끝 + 800ms 뒤에 나오고, 구간은 말 끝 + 1.75초에 닫힌다. 그래서 보통 결과가 같다. `segment_final`은 파이프라인이 구간 끝까지 처리한 직후에 보내므로, 시각 기준으로 바꿔도 그보다 늦게 오는 확정은 어차피 붙일 수 없다 |
| 3.4: 구간이 닫히면 남은 확정 전 글자도 이어 `segment_final`로 보낸다 | 마지막 구간(`end`, `too_long`)만 그렇게 한다. 중간 구간에서 남은 확정 전 글자는 다음 구간으로 넘긴다 | 그 글자는 나중에 파이프라인이 확정하면서 한 번 더 나온다. 앞 구간에 넣으면 두 번 보이게 된다. 쉼 1.75초 뒤에는 800ms 확정이 이미 끝나 있으므로, 보통은 넘길 글자가 없다. 15초 자르기(`max_length`)에서는 넘어갈 수 있다 |
| 3.3: `start`에서 빈 `stream_id`를 받는다 | 1, 2, 3…으로 계속 늘린다 | NeMo가 `stream_id`를 상태 칸에 딕셔너리로 연결한다 (`context_manager.py`). 번호가 커도 된다 |
| 3.5에 없음 | `ML_ASR_MAX_CONCURRENT`(4): `POST /asr` 동시 처리 한도. 넘는 요청은 기다린다 | 파일도 끝날 때까지 상태 칸 하나를 쓴다. 칸이 모자라면 NeMo가 step 전체를 실패시켜서, 같이 묶인 실시간 세션까지 `INTERNAL`로 끝난다. `num_slots`는 세션 한도 + 파일 한도 + `batch_max` 이상으로 잡는다 |

`transcript_delta`는 열린 구간에만 보낸다. 오디오를 실제 속도보다 빨리 보내서 VAD가 구간을 먼저 닫으면, 그 구간 글자는 delta 없이 `segment_final`로만 온다.

## 실행 (GPU 서버)

```bash
python3.11 -m venv .venv-nemo && . .venv-nemo/bin/activate
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/NVIDIA/NeMo && git -C NeMo checkout eaf0098ddeffb804ef1b1fb3a955b3bf022f4171
pip install -e "NeMo[asr]" && pip install -r serve/requirements.txt && pip check

ML_NEMO_MODEL=/models/ft_v3.nemo ML_NEMO_REPO=$PWD/NeMo \
  uvicorn serve.app:app --host 0.0.0.0 --port 8001 --workers 1
```

- `ft_v3.nemo`는 Drive `cdy-hack-ag5/marieum-asr-ft/ckpt/ft_v3/v1/checkpoints/`에 있다 (2,553,098,240 바이트). 서버에 올린 뒤 `sha256sum`을 남긴다. 모델 파일과 Drive 링크는 레포에 넣지 않는다
- 환경변수는 handoff 3.5 표와 `config.py`. VAD 값(`ML_END_SILENCE_MS` 등)은 v1과 같다
- 반드시 `--workers 1`

## 테스트 (GPU 없이)

```bash
pip install -r serve/requirements.txt
pytest
```

`tests/fakes.py`의 가짜 엔진이 NeMo 파이프라인 규칙(stream별 상태, `is_first`/`is_last`, 한 step에 같은 stream 하나, 상태 칸 수)을 어기면 바로 실패한다. 테스트 23개 (11-20 통과).

발화마다 남는 로그 (handoff 9.6, 글자는 찍지 않는다): `segment_final` 이벤트에 `speech_ms`, `first_delta_ms`(말 시작 오디오 도착 → 첫 delta, delta가 없었으면 null), `final_ms`(말 끝 오디오 도착 → final, `latency_ms`와 같은 값), `chars`, `deltas`, `revised`, `reason`, `confidence`.

## GPU에서 아직 확인하지 않은 것

가짜 엔진으로는 서버 규칙만 확인했다. 실제 모델로 처음 띄울 때 이것부터 본다.

1. `NemotronEngine`이 뜨는가: `PipelineBuilder.build_pipeline`, `chunk_ms`가 320으로 나오는가 (`/healthz` 로그의 `model_loaded`)
2. 마지막 조각을 0으로 채우고 `length`에 실제 길이를 넣는 방식이 파이프라인에서 문제없는가. 남은 오디오가 없으면 무음 한 조각(0.32초)을 `is_last`로 보낸다
3. `final_segments`의 `start`/`end`: NeMo 코드로는 초 단위다 (`bpe_decoder.py`에서 토큰 위치 × 토큰 길이(초)). 실제 값이 stream 시작 기준인지 본다. `POST /asr`의 `start_ms`/`end_ms`에만 쓴다
4. `confidence`가 0이 아닌 값으로 나오는가. NeMo 기본은 `preserve_frame_confidence: false`(이때 모두 0.0)이고 `engine.py`가 `true`로 켠다. 방식은 entropy(tsallis, alpha 0.5)의 평균이다. 지금까지 실험(EXP-015 ~ 017)은 신뢰도를 끄고 돌렸으므로 값의 분포를 처음 본다. 백엔드가 confidence 기준값을 쓰면 test_v2 녹음으로 분포를 보고 정한다. 켠 상태의 step 시간도 다시 잰다
5. 같은 녹음을 `conversation`과 `utterance`로 넣어 이어 붙인 글자가 같은가 (코드상 같아야 한다)
6. test_v2 녹음 몇 개를 v2로 실제 속도로 흘려 CER이 EXP-015(27.86)와 같은 수준인가, 지연 목표(말 끝 → 마지막 delta p95 ≤ 800ms, 4세션 `OVERLOADED` 0, step p95 ≤ 160ms)를 지키는가 (handoff 6절 4단계)

## 알려진 제약

- `sample_rate`가 16000이 아니면 WebSocket 프레임마다 따로 16kHz로 바꾼다. 프레임 경계에서 필터가 이어지지 않아 소리가 조금 어긋날 수 있다. 앱은 16kHz로 보내는 것을 권한다
- `POST /asr`는 한 step에 그 파일의 조각 하나씩만 넣는다 (같은 stream 조각 둘을 한 step에 넣지 않는 규칙). 60초 파일은 조각 188개라서, step 하나가 60ms면 11초쯤 걸린다. GPU에서 재 본다
