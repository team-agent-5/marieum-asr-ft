# 서비스 엔진 연결 핸드오프: Nemotron ft_v3 → marieum-ai

2026-11-17 작성, 같은 날 고침: 스트리밍 처리를 직접 만들지 않고 NeMo 공식 스트리밍 파이프라인을 감싸는 구조로 바꿨다 (3절).
이 문서만 읽고 marieum-ai에 새 엔진을 붙일 수 있게 정리했다. 숫자는 모두 [`experiments/results.csv`](../experiments/results.csv)의 실험 번호(EXP-)로 확인할 수 있다.

## 0. 한눈에

- **무엇을:** marieum-ai에 Nemotron 스트리밍 엔진을 붙이고, 환경변수 하나로 Whisper(yoona-J)와 고른다. 기본값은 지금처럼 Whisper다
- **왜:** 정확도는 yoona-J와 비슷하다. 실시간 표시는 확실히 낫다 (화면 글자 바뀜 0%, 말이 끝나고 0.26초 뒤 문장 완성, 반복 루프 0)
- **어떻게 (권장 구조):**
  - 모델 실행은 **NeMo 공식 스트리밍 파이프라인 `CacheAwareRNNTPipeline`**에 맡긴다. 세션별 상태, 특징 만들기, 여러 세션 묶기, 발화 끝 판정, 신뢰도, 단어 시각이 다 들어 있다
  - 우리가 만드는 것은 그 위의 **WebSocket 층**뿐이다: 세션 ↔ `stream_id`, 오디오 → `Frame`, 결과 → 메시지
  - 새 엔드포인트 `/asr-stream/v2`(9절)를 연다. 지금 `/asr-stream`(v1)은 Whisper 그대로 두거나, 같은 코어 위의 어댑터로 붙인다 (3.7)
- **아직 정하지 않은 것:** 문장 끊기를 모델의 발화 끝 판정으로 할지, 지금처럼 VAD 쉼 1.75초로 할지. 구음장애 화자는 말 중간에 길게 쉬어서 실험으로 정한다 (3.4, 6절 2단계)
- **일정:** 11-21 교체 판정, 11-22 ~ 24 연결·배포, 11-25 리허설, 11-26 시연. 판정 전에는 기본값을 Whisper로 둔다
- **끝난 모습:** `ML_ENGINE=nemotron`으로 띄우면 `/asr-stream/v2`가 열리고 delta/final이 나간다. `ML_ENGINE=whisper`로 되돌리면 지금과 똑같다. 6절의 확인을 모두 마친다

## 1. 모델

| | |
|---|---|
| 파일 | `ft_v3.nemo` (2,553,098,240 바이트) |
| 위치 | 팀 Google Drive `marieum-asr-ft/ckpt/ft_v3/v1/checkpoints/ft_v3.nemo`. 소유자가 필요한 사람에게만 지정 공유한다 |
| 만든 방법 | Nemotron 3.5 ASR Streaming 0.6B → ft_v1 (3000 step) → ft_v2 (+2000) → ft_v3 (+2000). [`docs/plan.md`](plan.md) |
| 설정 | `att_context_size=[56,3]` (0.32초 조각), 언어 프롬프트 `ko-KR`, 언어 태그 제거, RNNT greedy |
| 시험한 NeMo | GitHub NVIDIA/NeMo main `eaf0098dde` (2026-11-16, 3.1.0 계열). Colab 기본 torch 위에 `pip install -e NeMo[asr]` |
| 전처리 정규화 | `preprocessor.normalize = NA` (확인함) |

**GPU 서버로 옮기기:** 브라우저로 Drive에서 내려받아 `scp`로 서버에 올린다. 서버에서 `sha256sum ft_v3.nemo`를 남겨 두고, 이 문서와 배포 기록에 적는다.

**올리면 안 되는 곳:** marieum-ai는 공개 레포다. 모델 파일, Drive 링크·파일 ID는 넣지 않는다. 모델이 학습 대본(AI Hub 전사)을 외우고 있어서, 공개하면 재배포 금지 데이터가 새어 나갈 수 있다. 코드에는 `ML_NEMO_MODEL=/path/to/ft_v3.nemo`처럼 경로만 쓴다.

## 2. 측정에 쓴 코드와 운영 코드

- **5절의 숫자는 NeMo의 예전 경로로 쟀다:** `conformer_stream_step`을 직접 부르는 방식
  - CER: [`colab/zeroshot_nemotron.py`](../colab/zeroshot_nemotron.py)가 NeMo 예제 `speech_to_text_cache_aware_streaming_infer.py`를 부른다
  - 실시간 지표: [`colab/streaming_metrics.py`](../colab/streaming_metrics.py)의 `load_nemo()`, `stream_batch()`
  - 설정: `att_context_size [56,3]`, `RNNTDecodingConfig(fused_batch_size=-1)`, `set_inference_prompt("ko-KR")`, `set_strip_lang_tags(True)`, `pad_and_drop_preencoded=True`, 가중치 fp32 + bf16 autocast
- **운영은 NeMo의 새 스트리밍 파이프라인으로 한다 (3절).** 같은 모델을 같은 방식(cache-aware, 조각 단위)으로 돌리지만 코드가 다르다
- 그래서 **파이프라인으로 같은 CER이 나오는지 먼저 다시 잰다** (6절 1단계). 차이가 0.5점 넘게 나면 설정을 맞추기 전에는 넘어가지 않는다

## 3. 서비스 구조 (권장)

marieum-ai 지금 구조: `server/app.py` 세션 관리, `server/stream.py`(Segmenter, Agreement) 구간 나누기와 interim 확정, `server/scheduler.py` GPU 작업자, `server/engine.py` Whisper. 프로토콜은 marieum-ai `CALIBRATION.md` §0.2 ~ 0.3.

### 3.1 왜 NeMo 파이프라인을 쓰나

처음 쓴 안(특징 이어 만들기, 조각 묶기, 캐시 쌓고 나누기를 직접 구현)은 NeMo에 이미 있는 것을 다시 만드는 일이었다. NeMo `nemo.collections.asr.inference`(우리가 시험한 커밋에 들어 있다)에 이렇게 있다:

| 필요한 것 | NeMo에 있는 것 |
|---|---|
| 세션마다 상태 | `stream_id`별 상태 칸. `streaming.num_slots` (기본 256). 첫 요청(`is_first`)에서 만들고 마지막 요청(`is_last`) 뒤 지운다 |
| 조금씩 오는 오디오 → 특징 | 오디오 버퍼와 특징 캐시 (`streaming/buffering/`, `use_feat_cache`) |
| 여러 세션 묶기 | `transcribe_step(requests)`에 여러 세션의 `Frame`을 한 번에 넣는다 (`streaming.batch_size`) |
| 다국어 프롬프트 모델 | `lang` 설정과 요청별 `language_code`. 설정 파일 주석에 nemotron-3.5와 `[56,*]`가 적혀 있다 |
| 발화 끝 판정 | `RNNTGreedyEndpointing`: 모델이 `stop_history_eou`(기본 800ms) 동안 빈 출력만 내면 발화 끝 |
| 신뢰도 | `confidence` 설정 (entropy 방식), `preserve_frame_confidence` |
| 단어·구간 시각 | `final_segments`, `asr_output_granularity: word / segment` |
| 단어 힌트 | 세션별 `biasing_cfg`, `boosting_tree.key_phrases_list`. 마리음 AAC 자주 쓰는 말을 넣어 볼 수 있다 (선택) |
| 속도 | bf16 계산, CUDA graph(선택) |

NVIDIA의 서비스 제품(Speech NIM)도 같은 모델을 WebSocket으로 내보내고, 결과를 `...transcription.delta`(새 글자)와 `...transcription.completed`(확정)로 보낸다. 9절 메시지가 같은 모양이다.

### 3.2 파이프라인 설정

NeMo `examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml`을 바탕으로 이것만 바꾼다.

| 키 | 값 | 이유 |
|---|---|---|
| `asr.model_name` | `ft_v3.nemo` 경로 | |
| `lang` | `ko-KR` | 다국어 프롬프트 |
| `asr.strip_lang_tags` | `true` | 평가와 같다 |
| `streaming.att_context_size` | `[56,3]` | 0.32초 조각. `[56,13]`은 정확도가 같고 늦기만 하다 (EXP-011, 014) |
| `streaming.chunk_size_in_secs` | `null` | 모델 기본 (1 + 미리보기 프레임 = 0.32초) |
| `streaming.batch_size` | 8 | 한 번에 묶는 세션 수 |
| `streaming.num_slots` | 32 | 동시 세션 한도보다 크게 |
| `streaming.request_type` | `frame` | 오디오를 그대로 넣는다 |
| `asr.decoding.strategy` | `greedy_batch` | 평가와 같다 |
| `asr.compute_dtype` / `asr.use_amp` | `bfloat16` / `false` (파이프라인 기본) | 평가는 fp32 가중치 + bf16 autocast였다. 6절 1단계에서 둘 다 재 보고 CER이 같은 쪽 중 빠른 것을 고른다 |
| `asr.decoding.greedy.preserve_frame_confidence` | `true` | `confidence` 값 |
| `endpointing.stop_history_eou` | 3.4에서 정한다 | 발화 끝 판정 |
| `enable_itn` | `false` | 숫자 변환(ITN)은 영어용 설정이다. 한국어 ITN은 확인하지 않았다 |
| `asr_output_granularity` | `segment` | |

### 3.3 서버 루프

```
시작할 때
  pipeline = PipelineBuilder.build_pipeline(cfg)    # 모델 불러오기
  pipeline.open_session()
  워밍업: 가짜 stream 몇 개로 transcribe_step 몇 번 → /healthz 200

WebSocket 세션 하나 (이벤트 루프)
  start  → stream_id 하나 받기 (빈 번호), 오디오 버퍼 만들기
  바이너리 → PCM16 → float32 → 16kHz → 버퍼에 쌓기
           → 버퍼가 pipeline.chunk_size_in_secs(0.32초) 넘을 때마다
             Frame(samples=0.32초, stream_id, is_first=첫 조각인지) 을 GPU 대기열에
  end    → 남은 오디오(0.32초보다 짧으면 length로 실제 길이를 주고 0으로 채움)를
           Frame(..., is_last=True) 로 대기열에

GPU 작업자 (스레드 하나)
  대기열에서 세션마다 가장 오래된 Frame 하나씩, 최대 batch_size개
  (첫 Frame이 생기고 ML_NEMO_STEP_WAIT_MS가 지나면 모인 만큼)
  outputs = pipeline.transcribe_step(frames)
  → 이벤트 루프로 넘긴다 (loop.call_soon_threadsafe)

결과 처리 (이벤트 루프, 세션마다)
  out.partial_transcript   : 직전 발화 끝 이후 지금까지 글자 → transcript_delta (9.4)
  out.final_transcript     : 비어 있지 않으면 발화 하나가 확정됨 → segment_final
  out.final_segments       : 그 발화의 시작·끝 시각
  is_last Frame의 결과 뒤  → session_end. 상태는 파이프라인이 지운다
```

- 세션 안에서는 Frame 순서가 중요하다. 한 번의 `transcribe_step`에 같은 세션 Frame을 둘 넣지 않는다 (NeMo 예제도 `n_frames_per_stream=1`)
- `transcribe_step`은 GPU 스레드 하나에서만 부른다. 파이프라인 상태는 스레드 안전하다고 가정하지 않는다
- 세션 오디오는 끊지 않고 계속 넣는다. 발화 끝에서도 상태를 새로 시작하지 않는다 (구간마다 새로 시작하면 빠뜨림이 16.4%에서 21.8%로 늘었다, EXP-006). 쉬는 동안의 무음도 넣는다
- 한도: 처리 못 한 오디오가 세션당 `ML_NEMO_MAX_BACKLOG_MS`(5초)를 넘으면 `OVERLOADED`로 세션을 닫는다. 동시 세션은 `ML_NEMO_MAX_SESSIONS`(16)

### 3.4 문장 끊기: 모델 판정 vs VAD 쉼

| | A. 모델의 발화 끝 판정 (NeMo 기본) | B. 지금 방식 (Silero VAD, 쉼 1.75초) |
|---|---|---|
| 어떻게 | 모델이 `stop_history_eou`ms 동안 빈 출력만 내면 끝 | marieum-ai `Segmenter` 그대로. 구간이 닫히면 그때까지의 `partial_transcript`를 final로 본다 |
| 장점 | 정석. 추가 모델이 없다. final이 빨리 나온다 (기본 800ms) | 이미 앱과 맞춰 둔 값. 구음장애 화자의 긴 쉼에 덜 끊긴다 |
| 걱정 | 구음장애 화자는 말 중간에 길게 쉰다. 800ms면 문장 중간에서 끊길 수 있다 (11-01에 0.8초를 1.75초로 늘린 이유와 같다) | Whisper 때문에 생긴 구조다. final이 늘 1.75초 늦다. 모델 상태와 VAD 구간이 어긋날 수 있다 |

- **6절 2단계 실험으로 정한다:** A를 `stop_history_eou` 800 / 1200 / 1600 / 2000ms로, B를 1.75초로 같은 녹음에 돌린다. 두 가지를 본다
  - 문장이 끊기는 횟수와 위치: 한 문장이 둘로 쪼개지는가 (B 구간 수와 비교)
  - 말이 끝나고 final이 오기까지의 시간
- 정하기 전에는 B로 시작해도 된다. 앱 입장에서는 delta가 0.26초면 다 떠 있으므로 final이 늦어도 기다리는 느낌이 적다
- `Frame.vad_segments`에 VAD 구간을 넘겨 발화 끝 판정에 쓰는 기능도 있다. A와 B를 섞을 때 검토한다

### 3.5 설정 (환경변수)

| 변수 | 기본값 | 의미 |
|---|---|---|
| `ML_ENGINE` | `whisper` | `whisper` / `nemotron`. 되돌리기 스위치. 한 프로세스에는 고른 엔진 하나만 올린다 (`--workers 1` 그대로) |
| `ML_NEMO_MODEL` | (없음) | `.nemo` 경로 |
| `ML_NEMO_ATT_CONTEXT` | `[56,3]` | 0.32초 |
| `ML_NEMO_BATCH_MAX` | 8 | `streaming.batch_size` |
| `ML_NEMO_ENDPOINTING` | `vad` | `vad`(3.4 B) / `model`(3.4 A). 실험 뒤 바꾼다 |
| `ML_NEMO_EOU_MS` | 실험으로 정함 | `model`일 때 `stop_history_eou` |
| `ML_NEMO_STEP_WAIT_MS` | 10 | 첫 Frame이 생긴 뒤 다른 세션 Frame을 기다리는 최대 시간 |
| `ML_NEMO_MAX_BACKLOG_MS` | 5000 | 세션당 처리 못 한 오디오 한도 |
| `ML_NEMO_MAX_SESSIONS` | 16 | 동시 세션 한도 (`num_slots`보다 작게) |

VAD·길이 설정(`ML_VAD_THRESHOLD`, `ML_END_SILENCE_MS`, `ML_UTTERANCE_MAX_S`, `ML_SESSION_MAX_S` 등)은 v1과 같은 값을 같이 쓴다.

### 3.6 GPU

- 실측 (EXP-013, 예전 경로, L4, 4세션 묶음): 조각 하나 p50 62ms / p95 64ms. 실시간은 320ms. 실시간보다 늦은 조각 0개
- 파이프라인으로 바꾼 뒤 1, 4, 8세션으로 다시 잰다 (6절 4단계)
- GPU 메모리: 0.6B 모델 약 2.4GB(fp32 기준) + 세션 상태. L4 24GB에 충분하다
- `/healthz`에 `engine`, `sessions`, `step_ms_p95`, `batch_mean`을 더한다

### 3.7 v1 `/asr-stream` 어댑터 (선택)

백엔드가 v2로 옮기기 전까지 지금 엔드포인트에도 Nemotron을 쓰고 싶으면, 같은 코어 위에서 메시지만 바꾼다.

| v1 메시지 | 만드는 법 |
|---|---|
| `transcript_interim` | `text` = 이 구간 `partial_transcript`. `stable_text` = 마지막 단어 앞까지, `unstable_text` = 마지막 단어 (마지막 단어는 "화장" → "화장실에"처럼 자랄 수 있다). 최소 300ms 간격 (프로토콜 §7.1) |
| `transcript_final` | 구간이 끝날 때 (3.4 A 또는 B). `text` = 그 구간 글자 |
| `is_last` final | `end` → 마지막 Frame(`is_last=True`)의 결과 |
| `confidence` | 파이프라인 신뢰도 (3.2) |
| `no_speech_prob` | 값이 없다. 글자가 없으면 1.0, 있으면 0.0 |
| `model_version` | 예: `marieum-nemotron-ft_v3` |

`POST /asr`(파일)는 파일 하나를 stream 하나로 보고 같은 파이프라인에 Frame을 차례로 넣는다. `final_segments`로 `segments`를 만든다. 구간마다 따로 돌리지 않는다 (EXP-006).

## 4. 설치

- 서버에는 Whisper용으로 torch 2.11 cu128, transformers 4.53.3이 고정돼 있다 (marieum-ai `requirements.txt`). **NeMo가 요구하는 버전과 부딪칠 수 있다.** 먼저 서버에 새 venv를 만들고 같은 torch 위에 NeMo(`eaf0098dde`, `[asr]`)를 설치한 뒤 `pip check`, `PipelineBuilder.build_pipeline`, Frame 몇 개 돌리기까지 확인한다
- 부딪치면 엔진별로 venv를 나눠도 된다. 엔진을 바꿀 때 프로세스를 다시 띄우므로 큰 문제가 아니다
- 시스템 패키지: `libsndfile1`, `ffmpeg`
- ITN을 켜지 않으므로 `nemo_text_processing`(pynini)은 필요 없다

## 5. 성능 요약 (예전 경로로 잰 값, 2절)

정확도 (test_v2 30명, 화자별 CER 평균, 녹음 전체를 스트리밍으로)

| CER % | yoona-J 운영 방식 (EXP-005) | ft_v3 0.32초 (EXP-010) |
|---|---|---|
| 전체 30명 | 25.0 | 28.4 |
| 낭독 대본을 뺀 19개 | 33.7 | 32.6 |
| ㄴ 경증 / 중등도 / 중증 | 21.2 / 20.5 / 55.8 | 23.8 / 17.2 / 54.3 |
| 처음 보는 글자만 (참고용, 화자 2명 분량) | 79.0 | 57.1 |
| 반복 루프 / 지어내기 | 690자 / 6.8% | 0 / 2.5% |

실시간 (녹음 30개 앞 3분, 실제 속도 흉내, L4, EXP-012 ~ 013)

| | yoona-J (1세션) | ft_v3 (4세션 묶음) |
|---|---|---|
| 말이 끝나고 문장 글자가 다 뜨기까지 p50 / p95 | 1952 / 2110ms | 255 / 638ms |
| 화면 글자가 바뀐 구간 / 최종 100자당 지워진 글자 | 90% / 209자 | 0% / 0자 |
| 말 시작 후 첫 글자 p50 / p95 | 403 / 1270ms | 862 / 1631ms |
| GPU 한 번 계산 p95 | interim 393ms, final 489ms | 조각 64ms |

이 숫자에는 네트워크와 앱이 빠져 있다. 서비스에 붙인 뒤 다시 잰다 (6절).

## 6. 확인 순서

1. **파이프라인이 같은 답을 내는가 (Colab L4, 약 1 CU):** NeMo `examples/asr/asr_streaming_inference/asr_streaming_infer.py`에 3.2 설정과 ft_v3로 test_v2를 돌려 이 레포 `eval/score.py`로 채점한다. EXP-010(28.4)과 0.5점 안이어야 한다. bf16 계산과 fp32+autocast를 둘 다 본다. 이때는 발화 끝 판정이 결과에 영향을 주지 않게 `stop_history_eou`를 크게(예: 100000) 둔다
2. **문장 끊기 실험 (Colab L4):** 3.4의 A(800 / 1200 / 1600 / 2000ms)와 B(VAD 1.75초)를 녹음 30개 앞 3분에 돌린다. 구간 수, 한 문장이 쪼개지는지, final까지의 시간을 비교해 `ML_NEMO_ENDPOINTING`과 `ML_NEMO_EOU_MS`를 정한다. 숫자만 남기고 문장은 저장하지 않는다
3. **단위 테스트:** `ML_ENGINE=whisper`에서 기존 `pytest`가 모두 통과한다. 가짜 파이프라인(조각마다 정해진 글자를 내는)으로 v2 테스트를 만든다: `seq`와 구간 순서, `text`가 이어지는지, `revised` 처리, `end`, `OVERLOADED`
4. **지연:** `scripts/stream_client_v2.py`(`stream_client.py`를 고쳐서)로 wav를 실제 속도로 보내 1, 4, 8세션을 잰다. 목표(제안): 말 끝 → 그 구간 마지막 delta p95 ≤ 800ms, 4세션에서 `OVERLOADED` 0, step p95 ≤ 160ms. Whisper 기록(marieum-ai `TEST_RESULTS.md`, 4세션 interim p95 283 ~ 380ms)과 나란히 적는다
5. **정상 발화:** 구음장애가 없는 팀원 목소리로 문장 몇 개를 두 엔진에 넣어 본다. **아직 한 번도 안 쟀다.** 학습 데이터에 정상 발화가 1시간도 안 돼서 원래 모델보다 나빠졌을 수 있다. 녹음은 팀원 것만 쓰고 AI Hub 음성은 쓰지 않는다
6. **판정 (11-21):** 위 결과와 5절 숫자로 정한다. 바꾸면 서버 환경변수만 `ML_ENGINE=nemotron`. 문제가 생기면 `whisper`로 되돌리고 다시 시작한다

## 7. 알려진 약점과 주의

- **경증 화자는 yoona-J보다 나쁘다** (19개 기준 23.8 vs 21.2). 화자 수로는 19명 중 8명만 ft_v3가 낫다. 평균은 비슷해도 사람마다 갈린다
- **테스트 점수는 실제보다 좋게 나온다.** test_v2 글자의 86%가 학습 라벨에도 있는 대본이고, ft_v2 → ft_v3에서 오른 폭은 대부분 이 대본 쪽이다 (겹침 80% 이상 파일 29.7 → 24.9, 나머지 38.9 → 38.1). 시연은 처음 듣는 말이므로 "대본 밖 19개"와 "처음 보는 글자" 숫자를 기준으로 본다. yoona-J도 낭독 대본을 외우고 있어서(대본 11개 CER 9.9), 시연을 AI Hub 낭독 대본으로 하면 yoona-J가 더 좋아 보인다
- **첫 글자는 yoona-J보다 늦게 뜬다** (p50 0.86초 vs 0.40초). 대신 한번 뜬 글자가 바뀌지 않는다. v2의 `segment_start`로 앱이 "듣는 중"을 먼저 띄운다
- **"글자가 바뀌지 않는다"는 지금 설정에서 잰 성질이다.** greedy, ITN 끔, 단어 힌트 끔에서 지워진 글자 0이었다. NeMo는 partial이 바뀔 수 있다고 적어 둔다. beam 탐색, ITN, 단어 힌트를 켜면 바뀔 수 있으므로 v2는 바뀌는 경우도 처리한다 (9.4 `revised`)
- **출력 모양:** 마침표, 쉼표, 물음표가 붙는다 (2.4만 자에 . 240개, , 22개, ? 13개). 숫자는 거의 한글로 나온다 (2.4만 자에 숫자 4개, 예: "세 시"). 앱이나 백엔드가 숫자 모양에 기대면 확인한다
- **로그:** 전사 문장을 로그에 찍지 않는 지금 원칙을 그대로 지킨다. 심사장에서는 `ML_DEBUG_SAVE=0`

## 8. 참고

| 자료 | 내용 |
|---|---|
| NeMo `nemo/collections/asr/inference/pipelines/cache_aware_rnnt_pipeline.py` | 운영에 쓸 파이프라인 (`CacheAwareRNNTPipeline`) |
| NeMo `nemo/collections/asr/inference/pipelines/base_pipeline.py` | `transcribe_step`, `TranscribeStepOutput` (final / partial / current step) |
| NeMo `nemo/collections/asr/inference/streaming/framing/request.py` | `Frame(samples, stream_id, is_first, is_last, length, vad_segments, options)` |
| NeMo `examples/asr/asr_streaming_inference/` | 파이프라인 예제 (`asr_streaming_infer.py`), 설정 `conf/asr_streaming_inference/cache_aware_rnnt.yaml` |
| [NVIDIA Speech NIM Realtime API](https://docs.nvidia.com/nim/speech/26.10.0/reference/api-references/asr/realtime-asr.html) | 같은 계열 모델을 WebSocket으로 내보내는 NVIDIA 제품의 메시지 (delta / completed) |
| [Stateful Conformer with Cache-based Inference (arXiv 2312.17279)](https://alphaxiv.org/abs/2312.17279) | cache-aware 스트리밍 모델 논문 |
| [Turning Whisper into Real-Time Transcription System (arXiv 2307.14743)](https://arxiv.org/pdf/2307.14743) | v1이 쓰는 LocalAgreement의 출처. Whisper를 스트리밍처럼 쓰기 위한 우회책이다 |
| [`colab/streaming_metrics.py`](../colab/streaming_metrics.py) | 실시간 지표 측정 코드 (예전 경로) |
| [`colab/marieum_stream.py`](../colab/marieum_stream.py) | marieum-ai `server/stream.py` 복사본 (commit 2ccebf6) |
| [`colab/zeroshot_nemotron.py`](../colab/zeroshot_nemotron.py) | CER 평가 (예전 경로) |
| [`experiments/results.csv`](../experiments/results.csv), [`docs/plan.md`](plan.md) | 실험 기록 EXP-000 ~ 014, 날짜별 결정 |
| marieum-ai `CALIBRATION.md` §0, `TEST_RESULTS.md` | 지금 프로토콜, 설정, 지연 목표와 Whisper 기록 |

## 9. 새 WebSocket 엔드포인트 `/asr-stream/v2` (예시 스펙)

Nemotron의 성질(글자가 조각마다 붙는다)에 맞춘 엔드포인트다. 서버 내부는 3.3을 따른다.

- `/asr-stream`(v1)은 그대로 둔다. 지금 앱과 백엔드는 그대로 돈다
- `/asr-stream/v2`는 `ML_ENGINE=nemotron`일 때만 연다 (Whisper일 때 연결하면 `MODEL_UNAVAILABLE`로 닫는다)
- 앱은 interim 교체나 확정분 계산을 하지 않아도 된다. 보통은 새 글자를 뒤에 붙이기만 하고, 드물게 `revised`가 오면 그 구간 글자를 바꾼다

### 9.1 v1과 다른 점

| | `/asr-stream` (v1) | `/asr-stream/v2` |
|---|---|---|
| 중간 결과 | `transcript_interim`: 0.5초마다 구간 전체를 다시 받아써서 보낸다 (`stable_text` + `unstable_text`) | `transcript_delta`: 0.32초 조각마다 **새로 붙은 글자**(`append`)와 구간 글자 전체(`text`) |
| 말 시작 알림 | 없음 | `segment_start`. 첫 글자가 뜨기 전(p50 0.86초)에 앱이 "듣는 중"을 띄울 수 있다 |
| 구간 final | `transcript_final` (VAD 쉼 1.75초) | `segment_final` (3.4에서 정한 방식, 시작·끝 시각, 닫힌 이유, 신뢰도) |
| 세션 끝 | `is_last: true` final | `is_last: true` `segment_final` 뒤에 `session_end` |
| 연결당 세션 | 여러 개 (trace_id로 구분) | **한 번에 하나**. 끝나면 같은 연결에서 다음 세션을 시작할 수 있다 |
| 순서 번호 | 없음 | 세션 안 모든 메시지에 `seq` (1부터 1씩) |

NVIDIA Speech NIM Realtime API와 비교하면 `transcript_delta` ≈ `conversation.item.input_audio_transcription.delta`, `segment_final` ≈ `...transcription.completed`, `is_last` ≈ `is_last_result`다. NIM은 오디오를 base64 JSON(`input_audio_buffer.append`)으로 받지만, v2는 v1처럼 바이너리 PCM16으로 받는다 (더 가볍고 지금 백엔드 코드를 그대로 쓴다). 백엔드가 NIM 형식을 원하면 메시지 이름만 바꿔도 된다.

### 9.2 연결

- URL: `ws://<ML 서버>:8001/asr-stream/v2`. 접근 제한은 v1과 같다 (marieum-ai `CALIBRATION.md` §3 보안그룹)
- 연결 직후 서버가 먼저 보내는 것은 없다. 클라이언트가 `start`를 보낸다
- 모델이 준비 안 됐으면 `error`(`MODEL_UNAVAILABLE`, `fatal: true`) 뒤 1013으로 닫는다 (v1과 같다)

### 9.3 클라이언트 → 서버

**`start`** (텍스트, JSON)

```json
{"type": "start", "trace_id": "abc123", "encoding": "pcm16", "sample_rate": 16000,
 "mode": "conversation", "language": "ko", "patient_id": "p-01"}
```

| 필드 | 필수 | 값 |
|---|---|---|
| `trace_id` | 예 | 영문·숫자·`_-` 64자 이내 (v1 `TRACE_RE`와 같다) |
| `encoding` | 아니오 | `pcm16`만 (기본) |
| `sample_rate` | 예 | 8000 ~ 48000 정수. 16000 권장 (다르면 서버가 16kHz로 바꾼다) |
| `mode` | 아니오 | `conversation`(기본): 발화마다 final, 세션은 계속. `utterance`: 세션 하나가 발화 하나, `end`에서 final 하나 |
| `language` | 아니오 | `ko`만 (파이프라인 `language_code`는 `ko-KR`) |
| `patient_id` | 아니오 | 로그용 64자 |

**오디오** (바이너리): PCM16 little-endian 모노. 프레임 크기는 아무거나 받는다. 20 ~ 100ms 권장. 실제 속도로 보낸다 (5초 넘게 앞서 보내면 `OVERLOADED`. 파일은 `POST /asr`).

**`end`**: `{"type": "end", "trace_id": "abc123"}`. 남은 오디오를 끝까지 처리하고 마지막 발화를 닫는다.

**`ping`**: `{"type": "ping", "trace_id": "abc123"}` → `pong`.

### 9.4 서버 → 클라이언트

모든 메시지에 `type`, `trace_id`, `seq`가 붙는다 (`pong`과 세션 밖 `error`는 `seq` 없음).

**`ready`**: 세션을 만들었다.

```json
{"type": "ready", "trace_id": "abc123", "seq": 1, "engine": "nemotron",
 "model_version": "marieum-nemotron-ft_v3", "chunk_ms": 320, "sample_rate": 16000}
```

**`segment_start`**: 새 발화가 시작됐다. `start_ms`는 세션 시작 기준 위치.
- 3.4 B(VAD)면 VAD가 구간을 연 때 보낸다
- 3.4 A(모델 판정)면 발화 끝 뒤 처음으로 글자가 생긴 때 보낸다. 이 경우 "듣는 중" 표시용으로 VAD를 따로 돌리지 않으면 더 일찍 알릴 방법이 없다. 필요하면 VAD는 알림용으로만 둔다

```json
{"type": "segment_start", "trace_id": "abc123", "seq": 2, "segment_id": 0, "start_ms": 1180}
```

**`transcript_delta`**: 이 발화의 글자가 늘었다 (또는 바뀌었다).

```json
{"type": "transcript_delta", "trace_id": "abc123", "seq": 3, "segment_id": 0,
 "append": "안녕하", "text": "안녕하", "revised": false, "audio_end_ms": 2240}
```

| 필드 | 뜻 |
|---|---|
| `text` | 이 발화에서 지금까지의 글자 전체 (파이프라인 `partial_transcript`) |
| `append` | 앞서 보낸 `text` 뒤에 새로 붙은 글자. 앞에 공백이 오면 새 단어, 없으면 단어 안에서 이어진다 |
| `revised` | `false`면 `text = 이전 text + append`. `true`면 앞서 보낸 글자가 바뀌었으니 **이 발화 글자를 `text`로 통째로 바꾼다** (`append`는 빈 문자열). 지금 설정(greedy, ITN·힌트 끔)에서는 0번이었다 |
| `audio_end_ms` | 이 결과를 낼 때까지 처리한 오디오의 끝 (세션 시작 기준). 지연 측정용 |

**`segment_final`**: 발화가 확정됐다. 이 뒤로 이 발화의 `transcript_delta`는 오지 않는다.

```json
{"type": "segment_final", "trace_id": "abc123", "seq": 9, "segment_id": 0,
 "text": "안녕하세요 저는", "start_ms": 1180, "end_ms": 3420, "reason": "eou",
 "confidence": 0.91, "no_speech_prob": 0.0, "is_last": false}
```

- `text`가 그 발화의 기준 글자다. 앱은 표시를 이것으로 맞춘다 (보통 마지막 delta의 `text`와 같다)
- `start_ms`, `end_ms`: 파이프라인 `final_segments` 시각 (3.4 B면 VAD 구간)
- `reason`: `eou`(모델 판정) / `silence`(VAD 쉼) / `max_length` / `end` / `too_long`
- 말소리가 있었는데 글자가 없으면 `text: ""`, `no_speech_prob: 1.0`
- `confidence`: 파이프라인 신뢰도 (3.2). 백엔드가 쓰는 곳을 확인한다

**`session_end`**: `is_last: true`인 `segment_final` 다음에 한 번.

```json
{"type": "session_end", "trace_id": "abc123", "seq": 10, "segments": 3, "audio_ms": 18320}
```

**`error`**

```json
{"type": "error", "trace_id": "abc123", "seq": 5, "code": "OVERLOADED", "message": "...", "fatal": true}
```

| code | 뜻 | fatal |
|---|---|---|
| `INVALID_FORMAT` | JSON, 필드, 형식 오류 (v1과 같다) | 아니오 (세션을 안 만든다) |
| `NO_SESSION` | `start` 없이 오디오나 `end` (v1과 같다) | 아니오 |
| `SESSION_BUSY` | 세션이 진행 중인데 다른 `trace_id`로 `start` | 아니오 |
| `MODEL_UNAVAILABLE` | 모델 로딩 중이거나 `ML_ENGINE`이 nemotron이 아님 | 예 (연결 닫음) |
| `AUDIO_TOO_LONG` | `utterance` 10초, `conversation` 600초 초과 (v1과 같다). 남은 발화를 `too_long`으로 닫는다 | 예 (세션 끝) |
| `OVERLOADED` | 처리 못 한 오디오가 5초를 넘었다 (세션 끝), 또는 동시 세션이 한도에 찼다 (세션을 안 만든다) | 앞은 예, 뒤는 아니오 |
| `INTERNAL` | 서버 오류 | 예 (세션 끝) |

**순서 보장**
- 한 세션 안에서 `seq`는 1씩 늘고, 메시지는 `seq` 순서로 온다
- 발화마다 `segment_start` → `transcript_delta` 0개 이상 → `segment_final` 순서다. 발화 k의 `segment_final`은 발화 k+1의 `segment_start`보다 먼저 온다

### 9.5 예: 대화 모드 한 세션

예시 글자는 지어낸 것이다.

```
→ {"type":"start","trace_id":"t1","sample_rate":16000,"mode":"conversation"}
← {"type":"ready","seq":1,"chunk_ms":320,...}
→ (오디오 100ms씩)
← {"type":"segment_start","seq":2,"segment_id":0,"start_ms":1180}
← {"type":"transcript_delta","seq":3,"segment_id":0,"append":"안녕","text":"안녕","revised":false,"audio_end_ms":2240}
← {"type":"transcript_delta","seq":4,"segment_id":0,"append":"하세요","text":"안녕하세요","revised":false,"audio_end_ms":2560}
   … (말이 끝나고 발화 끝 판정)
← {"type":"segment_final","seq":7,"segment_id":0,"text":"안녕하세요 …","end_ms":3420,"reason":"eou","is_last":false}
← {"type":"segment_start","seq":8,"segment_id":1,...}
   …
→ {"type":"end","trace_id":"t1"}
← {"type":"segment_final","seq":15,"segment_id":2,...,"reason":"end","is_last":true}
← {"type":"session_end","seq":16,"segments":3,"audio_ms":18320}
```

### 9.6 로그 (글자는 찍지 않는다)

- 발화마다: `speech_ms`, `first_delta_ms`(시작 → 첫 delta), `final_ms`(말 끝 → final), `chars`, `deltas`, `revised` 횟수, `reason`
- 10초마다: `sessions`, `step_ms` p50/p95, `batch_mean`, 가장 긴 대기열
- 형식은 v1과 같다 (JSON 한 줄, `trace_id`, `latency_ms`)

### 9.7 만드는 순서

1. **엔진 코어** (`server/engine_nemotron.py`): 3.2 설정으로 파이프라인 만들기, 워밍업, Frame 만들기, GPU 작업자 (3.3). 6절 1단계를 여기서 통과시킨다
2. **`/asr-stream/v2`**: 세션 ↔ `stream_id`, 결과 → 메시지 (9.4), 한도. 가짜 파이프라인으로 6절 3단계 테스트
3. **문장 끊기** (3.4): 6절 2단계 실험 결과로 기본값을 정한다
4. **시험 클라이언트와 지연** (6절 4단계)
5. (선택) **v1 어댑터** (3.7), `POST /asr`
