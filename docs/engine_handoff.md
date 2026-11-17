# 서비스 엔진 연결 핸드오프: Nemotron ft_v3 → marieum-ai

2026-11-17 작성. 이 문서만 읽고 marieum-ai에 새 엔진을 붙일 수 있게 정리했다.
숫자는 모두 [`experiments/results.csv`](../experiments/results.csv)의 실험 번호(EXP-)로 확인할 수 있다.

## 0. 한눈에

- **무엇을:** marieum-ai의 Whisper 엔진(yoona-J) 옆에 Nemotron 스트리밍 엔진을 붙이고, 환경변수 하나로 고른다. 기본값은 지금처럼 Whisper다
- **왜:** 정확도는 yoona-J와 비슷하다. 실시간 표시는 확실히 낫다 (화면 글자 바뀜 0%, 말이 끝나고 0.26초 뒤 문장 완성, 반복 루프 0)
- **일정:** 11-21 교체 판정, 11-22 ~ 24 연결·배포, 11-25 리허설, 11-26 시연. 판정 전에는 기본값을 Whisper로 둔다
- **두 가지 길:** 지금 `/asr-stream`에 끼워 넣기 (3절), 또는 Nemotron에 맞춘 새 엔드포인트 `/asr-stream/v2` (9절, 새 글자만 보내는 메시지). 엔진 코어는 같다
- **끝난 모습:** `ML_ENGINE=nemotron`으로 띄우면 같은 WebSocket 프로토콜로 interim/final이 나간다. `ML_ENGINE=whisper`로 되돌리면 지금과 똑같다. 4세션 동시 지연 시험과 정상 발화 확인까지 마친다 (6절)

## 1. 모델

| | |
|---|---|
| 파일 | `ft_v3.nemo` (2,553,098,240 바이트) |
| 위치 | 팀 Google Drive `marieum-asr-ft/ckpt/ft_v3/v1/checkpoints/ft_v3.nemo`. 소유자가 필요한 사람에게만 지정 공유한다 |
| 만든 방법 | Nemotron 3.5 ASR Streaming 0.6B → ft_v1 (3000 step) → ft_v2 (+2000) → ft_v3 (+2000). [`docs/plan.md`](plan.md) |
| 설정 | `att_context_size=[56,3]` (0.32초 조각), 언어 프롬프트 `ko-KR`, 언어 태그 제거, RNNT greedy, bf16 autocast |
| 시험한 NeMo | GitHub NVIDIA/NeMo main `eaf0098dde` (2026-11-16, 3.1.0 계열). Colab 기본 torch 위에 `pip install -e NeMo[asr]` |
| 전처리 정규화 | `preprocessor.normalize = NA` (확인함). 조각마다 특징을 따로 뽑아도 녹음 전체로 뽑은 것과 같다 |

**GPU 서버로 옮기기:** 브라우저로 Drive에서 내려받아 `scp`로 서버에 올린다. 서버에서 `sha256sum ft_v3.nemo`를 남겨 두고, 이 문서와 배포 기록에 적는다.

**올리면 안 되는 곳:** marieum-ai는 공개 레포다. 모델 파일, Drive 링크·파일 ID는 넣지 않는다. 모델이 학습 대본(AI Hub 전사)을 외우고 있어서, 공개하면 재배포 금지 데이터가 새어 나갈 수 있다. 코드에는 `ML_NEMO_MODEL=/path/to/ft_v3.nemo`처럼 경로만 쓴다.

## 2. 검증된 사용법 (그대로 따라 한다)

측정한 숫자는 모두 아래 코드로 나왔다. 새 엔진은 이 설정과 같게 만든다.

- 모델 불러오기: [`colab/streaming_metrics.py`](../colab/streaming_metrics.py)의 `load_nemo()`
  1. `ASRModel.restore_from(path, map_location="cuda")`
  2. `encoder.set_default_att_context_size([56, 3])`
  3. `change_decoding_strategy(RNNTDecodingConfig(fused_batch_size=-1))`. 모델에 `cur_decoder`가 있으면 `decoder_type="rnnt"`도 준다
  4. `set_inference_prompt("ko-KR")`, `decoding.set_strip_lang_tags(True)`
  5. `.eval()`, 그다음 `torch.autocast("cuda", dtype=torch.bfloat16)` 안에서 돌린다. 가중치는 fp32로 둔다 (NeMo가 cache-aware 모델의 fp16/bf16 가중치를 지원하지 않는다)
- 스트리밍: 같은 파일의 `stream_batch()`
  - `CacheAwareStreamingAudioBuffer(pad_and_drop_preencoded=True)`
  - 조각마다 `conformer_stream_step(..., drop_extra_pre_encoded=streaming_cfg.drop_extra_pre_encoded, return_transcription=True)`
  - 캐시 3개(`cache_last_channel`, `cache_last_time`, `cache_last_channel_len`), `previous_hypotheses`, `previous_pred_out`를 다음 조각으로 넘긴다
  - `pad_and_drop_preencoded=True`면 첫 조각도 나머지와 크기가 같다 (32프레임 = 0.32초). 세션 여러 개를 한 번에 묶기 쉬워진다
- 평가 스크립트 (CER 측정): [`colab/zeroshot_nemotron.py`](../colab/zeroshot_nemotron.py). NeMo 예제 `speech_to_text_cache_aware_streaming_infer.py`를 같은 설정으로 부른다

## 3. 서비스에 붙이는 설계 (제안)

marieum-ai 구조: `server/app.py`가 세션 관리, `server/stream.py`(Segmenter, Agreement)가 구간 나누기와 interim 확정, `server/scheduler.py`가 GPU 작업자, `server/engine.py`가 Whisper. 프로토콜은 marieum-ai `CALIBRATION.md` §0.2 ~ 0.3.

### 3.1 설정

| 변수 | 기본값 | 의미 |
|---|---|---|
| `ML_ENGINE` | `whisper` | `whisper` / `nemotron`. 되돌리기 스위치 |
| `ML_NEMO_MODEL` | (없음) | `.nemo` 경로 |
| `ML_NEMO_ATT_CONTEXT` | `[56,3]` | 0.32초. `[56,13]`(1.12초)은 정확도가 같고 늦기만 하다 (EXP-011, 014) |
| `ML_NEMO_BATCH_MAX` | 8 | 한 번에 묶는 세션 수 |

한 프로세스에는 고른 엔진 하나만 올린다. 바꾸려면 다시 시작한다 (`--workers 1` 고정은 그대로).

### 3.2 세션 흐름

- **구간 나누기는 지금 Segmenter를 그대로 쓴다.** VAD, 쉼 1.75초, 구간 최대 15초, `segment_id`, `is_last` 규칙이 그대로 유지된다
- **Nemotron은 세션 오디오 전체를 끊지 않고 흘려 넣는다.** 구간마다 상태를 새로 시작하지 않는다. 구간마다 새로 시작하면 빠뜨림이 16.4%에서 21.8%로 늘었다 (EXP-006, 조각 시작마다 앞부분을 버린다). 쉬는 동안의 무음도 넣는다 (4세션 기준 계산이 실시간의 20%라 부담이 없다)
- **오디오 넣기:** 세션마다 16kHz 오디오를 모아 두고, 새 오디오가 0.32초 쌓이면 특징(멜)을 뽑아 한 조각을 돌린다
  - 특징은 새 오디오 앞에 이전 오디오를 조금(STFT 창 25ms 이상) 붙여 뽑고, 새 프레임만 쓴다. `CacheAwareStreamingAudioBuffer`는 파일 전체를 한 번에 뽑으므로 실시간 경로는 따로 만들어야 한다
  - **같은지 꼭 확인한다:** 실시간 경로로 test 클립을 넣은 결과가 파일 방식(`stream_batch`) 결과와 같아야 한다 (6절 1단계)
  - 고정된 NeMo 버전에 실시간 cache-aware 파이프라인(`nemo.collections.asr.inference` 아래)이 있으면 그것을 먼저 검토한다
- **글자 → 구간:** 가설 글자는 세션 전체로 이어진다. 구간이 닫힐 때 직전 final 이후 새로 붙은 글자를 그 구간의 글자로 본다. 말이 끝난 뒤 마지막 글자가 붙기까지 p95 0.64초이고 구간은 1.75초 뒤에 닫히므로 앞 구간 글자가 뒤로 넘어가지 않는다. 15초 상한으로 잘린 구간에서는 단어 하나가 두 구간에 걸칠 수 있다 (드물다)

### 3.3 메시지 대응

| 메시지 | 만드는 법 |
|---|---|
| `transcript_interim` | 글자가 새로 붙었을 때 보낸다 (최소 300ms 간격, 프로토콜 §7.1. 0.32초 조각이라 저절로 맞는다). `text` = 이 구간에 붙은 글자 전체. **`stable_text` = 마지막 단어를 뺀 앞부분, `unstable_text` = 마지막 단어**. 글자는 덧붙기만 하지만 마지막 단어는 "화장" → "화장실에"처럼 자랄 수 있어서, 지금 Agreement와 같은 규칙으로 둔다. `stable_revised`는 항상 false가 된다. `audio_end_ms` = 이 조각까지 처리한 오디오 끝 |
| `transcript_final` | Segmenter가 구간을 닫을 때. `text` = 그 구간 글자 전체. 구간 final 뒤에 그 구간 interim은 보내지 않는다 (지금 규칙) |
| `is_last` final | `end`를 받으면 남은 오디오를 마지막 조각(`keep_all_outputs=True`)까지 돌리고 남은 구간을 닫는다 |
| `confidence` | Whisper는 토큰 확률 평균이다. RNNT에는 같은 값이 없다. NeMo 디코딩의 confidence 설정(`confidence_cfg`, 토큰 confidence 보존)으로 토큰 확률 평균을 내는 것을 제안한다. 백엔드가 이 값을 어디에 쓰는지 먼저 확인한다 (열린 질문) |
| `no_speech_prob` | 해당 값이 없다. 글자가 없으면 1.0, 있으면 0.0을 제안한다. 무음 판정은 VAD + "글자 없음"으로 한다. Nemotron은 지어내기가 적었다 (2.5%, 반복 0). 다만 무음만 넣는 시험은 따로 하지 않았다 |
| `model_version` | 예: `marieum-nemotron-ft_v3` |

- `utterance` 모드: 세션 하나가 발화 하나다. `end`에서 마지막 조각까지 돌리고 final 하나를 보낸다
- `POST /asr` (파일): 파일 전체를 스트리밍으로 한 번에 돌린다 (CER을 잰 방식과 같다). `split_speech()` 구간 경계로 글자를 나눠 `segments`를 만든다. 구간마다 따로 돌리지 않는다 (위 EXP-006)
- **끝 무음 1.75초는 v1에서 그대로 둔다.** Nemotron은 말이 끝나고 0.26초면 글자가 다 떠 있으므로, 앱이 interim의 `stable_text`를 바로 보여 주면 사용자는 기다리지 않는다. final을 더 빨리 내고 싶으면 `ML_END_SILENCE_MS`를 줄인다. 다만 11-01에 "문장이 너무 일찍 끊긴다"는 앱 문제로 0.8초에서 늘린 값이니 앱 담당과 같이 정한다

### 3.4 GPU 작업자

- 지금 `GpuScheduler`는 interim/final 작업을 묶는다. Nemotron은 **조각 작업** 하나뿐이다. 세션이 0.32초 조각을 내면 작업자가 여러 세션의 조각을 묶어 `conformer_stream_step` 한 번으로 돌린다
- 묶기: 세션별 캐시를 배치 축으로 쌓고 결과를 다시 나눈다. `cache_last_channel`과 `cache_last_time`은 [층, 배치, ...] 모양, `cache_last_channel_len`은 [배치]. `previous_hypotheses`와 `previous_pred_out`은 세션별 목록이다. 정확한 모양은 고정한 NeMo 버전에서 `get_initial_cache_state(batch_size)`로 확인한다
- 실측 (EXP-013, L4, 4세션 묶음): 조각 하나 p50 62ms / p95 64ms. 실시간(320ms)보다 늦은 조각 0개. 1세션씩 따로 돌려도 4세션은 들어갈 것으로 보지만 재 보지 않았다. 묶기를 처음부터 넣는 것을 권한다
- 세션 상태 크기는 일정하다 (왼쪽 문맥 56프레임 캐시). 세션 600초 동안 늘어나는 것은 가설 토큰 목록뿐이다
- 워밍업: 시작할 때 1 ~ `ML_NEMO_BATCH_MAX` 세션 크기로 몇 조각 돌린 뒤 `/healthz`를 200으로 바꾼다 (지금과 같은 원칙)

## 4. 설치

- 서버에는 Whisper용으로 torch 2.11 cu128, transformers 4.53.3이 고정돼 있다 (marieum-ai `requirements.txt`). **NeMo가 요구하는 버전과 부딪칠 수 있다.** 먼저 서버에 새 venv를 만들고 같은 torch 위에 NeMo(`eaf0098dde`)를 설치한 뒤 `pip check`, 모델 불러오기, 조각 하나 돌리기까지 확인한다
- 부딪치면 Whisper 되돌리기를 위해 두 엔진이 같은 venv에서 돌아야 하는지 정한다. 다른 venv여도 프로세스를 다시 띄워 바꾸면 되므로 큰 문제는 아니다
- 시스템 패키지: `libsndfile1`, `ffmpeg`
- GPU 메모리: 0.6B 모델 fp32 가중치 약 2.4GB + 세션 캐시. L4 24GB에 충분하다

## 5. 성능 요약

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

이 숫자에는 네트워크와 앱이 빠져 있다. 서비스에 붙인 뒤 `scripts/stream_client.py`로 다시 잰다 (6절).

## 6. 확인 순서

1. **엔진 단독:** 실시간 경로(3.2)로 test 클립 몇 개를 넣은 결과가 `stream_batch()`(파일 방식)와 같은지 본다. 다르면 특징 뽑기나 첫 조각 처리를 고친다. 기준: 글자가 같거나, CER 차이 0.5점 이내
2. **테스트:** `ML_ENGINE=whisper`에서 기존 `pytest`가 모두 통과한다. 가짜 엔진으로 새 단위 테스트를 만든다: 덧붙기만 하는지, 글자가 구간에 맞게 나뉘는지, `end`에서 마지막 조각까지 도는지, `stable_text`/`unstable_text` 나누기
3. **지연:** `scripts/stream_client.py`로 1세션과 4세션을 재고 marieum-ai `TEST_RESULTS.md`의 목표와 비교한다 (interim p50 ≤ 300ms, p95 ≤ 500ms, final p95 ≤ 800ms. final은 `end` 뒤 기준). Whisper 기록(4세션 interim p95 283 ~ 380ms)과 나란히 적는다
4. **정상 발화:** 구음장애가 없는 팀원 목소리로 문장 몇 개를 두 엔진에 넣어 본다. **아직 한 번도 안 쟀다.** 학습 데이터에 정상 발화가 1시간도 안 돼서 원래 모델보다 나빠졌을 수 있다. 시연에 진행자나 정상 발화가 섞이면 꼭 본다. 녹음은 팀원 것만 쓰고 AI Hub 음성은 쓰지 않는다
5. **판정 (11-21):** 위 결과와 5절 숫자로 바꿀지 정한다. 바꾸면 서버 환경변수만 `ML_ENGINE=nemotron`으로 바꾼다. 문제가 생기면 `whisper`로 되돌리고 다시 시작한다

## 7. 알려진 약점과 주의

- **경증 화자는 yoona-J보다 나쁘다** (19개 기준 23.8 vs 21.2). 화자 수로는 19명 중 8명만 ft_v3가 낫다. 평균은 비슷해도 사람마다 갈린다
- **테스트 점수는 실제보다 좋게 나온다.** test_v2 글자의 86%가 학습 라벨에도 있는 대본이고, ft_v2 → ft_v3에서 오른 폭은 대부분 이 대본 쪽이다 (겹침 80% 이상 파일 29.7 → 24.9, 나머지 38.9 → 38.1). 시연은 처음 듣는 말이므로 "대본 밖 19개"와 "처음 보는 글자" 숫자를 기준으로 본다. yoona-J도 낭독 대본을 외우고 있어서(대본 11개 CER 9.9), 시연을 AI Hub 낭독 대본으로 하면 yoona-J가 더 좋아 보인다
- **첫 글자는 yoona-J보다 늦게 뜬다** (p50 0.86초 vs 0.40초). 대신 한번 뜬 글자는 바뀌지 않는다. 앱의 "듣는 중" 표시로 덮는 것을 제안한다
- **출력 모양:** 마침표, 쉼표, 물음표가 붙는다 (2.4만 자에 . 240개, , 22개, ? 13개). 숫자는 거의 한글로 나온다 (2.4만 자에 숫자 4개, 예: "세 시"). 앱이나 백엔드가 숫자 모양에 기대면 확인한다
- **로그:** 전사 문장을 로그에 찍지 않는 지금 원칙을 그대로 지킨다. 심사장에서는 `ML_DEBUG_SAVE=0`

## 8. 참고 파일

| 파일 | 내용 |
|---|---|
| [`colab/streaming_metrics.py`](../colab/streaming_metrics.py) | 모델 불러오기·스트리밍 코드 (2절), 실시간 지표 |
| [`colab/marieum_stream.py`](../colab/marieum_stream.py) | marieum-ai `server/stream.py` 복사본 (commit 2ccebf6) |
| [`colab/zeroshot_nemotron.py`](../colab/zeroshot_nemotron.py) | CER 평가 (NeMo 예제 스크립트 호출) |
| [`experiments/results.csv`](../experiments/results.csv) | 실험 기록 EXP-000 ~ 014 |
| [`docs/plan.md`](plan.md) | 날짜별 기록과 결정 |
| marieum-ai `CALIBRATION.md` §0 | 지금 프로토콜, 설정, 실측 |
| marieum-ai `TEST_RESULTS.md` | 지연 목표와 Whisper 기록 |

## 9. 새 WebSocket 엔드포인트 `/asr-stream/v2` (예시 스펙)

3절은 **지금 엔드포인트 `/asr-stream`(v1)에 Nemotron을 끼워 넣는 방법**이다. 이 절은 Nemotron의 성질(글자가 덧붙기만 한다)에 맞춘 **새 엔드포인트**를 따로 두는 안이다. 두 안은 같은 엔진 코어(9.5)를 쓴다.

- `/asr-stream`(v1)은 그대로 둔다. 메시지와 동작이 바뀌지 않으므로 지금 앱과 백엔드는 그대로 돈다
- `/asr-stream/v2`는 `ML_ENGINE=nemotron`일 때만 연다 (Whisper일 때 연결하면 `MODEL_UNAVAILABLE`로 닫는다)
- 백엔드가 v2로 옮기면 앱이 interim 교체나 확정분 계산을 하지 않아도 된다. 서버가 보낸 새 글자를 뒤에 붙이기만 하면 된다

### 9.1 v1과 다른 점

| | `/asr-stream` (v1) | `/asr-stream/v2` |
|---|---|---|
| 중간 결과 | `transcript_interim`: 구간 전체 글자를 매번 다시 보낸다 (`stable_text` + `unstable_text`) | `transcript_delta`: **새로 붙은 글자만** 보낸다 (`append`). 앞서 보낸 글자는 바뀌지 않는다 |
| 보내는 때 | 0.5초마다 구간 전체를 다시 받아쓴 뒤 | 0.32초 조각을 처리해서 글자가 늘었을 때 |
| 말 시작 알림 | 없음 | `segment_start` (VAD가 말소리를 잡은 때). 첫 글자가 뜨기 전(p50 0.86초)에 앱이 "듣는 중"을 띄울 수 있다 |
| 구간 final | `transcript_final` | `segment_final` (구간 글자 전체, 시작·끝 시각, 닫힌 이유) |
| 세션 끝 | `is_last: true` final | `segment_final`의 `is_last: true` 뒤에 `session_end` (요약 숫자) |
| 연결당 세션 | 여러 개 (trace_id로 구분, 바이너리는 최근 `start` 세션) | **한 번에 하나**. 끝나면 같은 연결에서 다음 세션을 시작할 수 있다 |
| 순서 번호 | 없음 | 세션 안 모든 메시지에 `seq` (1부터 1씩 증가) |

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
| `mode` | 아니오 | `conversation`(기본): 쉼마다 구간 final, 세션은 계속. `utterance`: 세션 하나가 발화 하나, `end`에서 final 하나 |
| `language` | 아니오 | `ko`만 |
| `patient_id` | 아니오 | 로그용 64자 |

**오디오** (바이너리): PCM16 little-endian 모노. 프레임 크기는 아무거나 받는다. 20 ~ 100ms 권장. 실제 속도로 보낸다 (빨리 보내면 9.5.5의 밀림 한도에 걸릴 수 있다).

**`end`**: `{"type": "end", "trace_id": "abc123"}`. 남은 오디오를 끝까지 처리하고 마지막 구간을 닫는다.

**`ping`**: `{"type": "ping", "trace_id": "abc123"}` → `pong`.

### 9.4 서버 → 클라이언트

모든 메시지에 `type`, `trace_id`, `seq`가 붙는다 (`pong`, 세션 밖 `error`는 `seq` 없음).

**`ready`**: `start`를 받아 세션을 만들었다.

```json
{"type": "ready", "trace_id": "abc123", "seq": 1, "engine": "nemotron",
 "model_version": "marieum-nemotron-ft_v3", "chunk_ms": 320, "sample_rate": 16000}
```

**`segment_start`**: VAD가 새 구간을 열었다. `start_ms`는 세션 시작 기준 말소리 시작 위치 (앞 여유 200ms 포함).

```json
{"type": "segment_start", "trace_id": "abc123", "seq": 2, "segment_id": 0, "start_ms": 1180}
```

**`transcript_delta`**: 글자가 새로 붙었다.

```json
{"type": "transcript_delta", "trace_id": "abc123", "seq": 3, "segment_id": 0,
 "append": "안녕하", "text": "안녕하", "audio_end_ms": 2240}
```

| 필드 | 뜻 |
|---|---|
| `append` | 이번에 새로 붙은 글자. **앱은 이 구간 글자 뒤에 그대로 붙인다.** 앞에 공백이 올 수도 있고(새 단어), 없이 단어 안에서 이어질 수도 있다 |
| `text` | 이 구간에서 지금까지 붙은 글자 전체. 메시지를 놓쳤을 때 맞추는 용도. 항상 `이전 text + append`와 같다 |
| `audio_end_ms` | 이 글자를 낼 때까지 처리한 오디오의 끝 (세션 시작 기준). 지연 측정용 |

**`segment_final`**: 구간이 닫혔다 (쉼 1.75초, 최대 15초, 또는 `end`). 이 뒤로 이 구간의 `transcript_delta`는 오지 않는다.

```json
{"type": "segment_final", "trace_id": "abc123", "seq": 9, "segment_id": 0,
 "text": "안녕하세요 저는", "start_ms": 1180, "end_ms": 3420, "reason": "silence",
 "confidence": null, "no_speech_prob": 0.0, "is_last": false}
```

- `text`는 그 구간의 `transcript_delta`를 모두 이은 것과 같다 (다시 받아쓰지 않는다). 앱은 확정 표시만 하면 된다
- `end_ms` = 마지막 말소리 위치 (+ 뒤 여유 200ms). 쉼을 기다린 1.75초는 들어가지 않는다
- `reason`: `silence` / `max_length` / `end` / `too_long`
- 말소리가 있었는데 글자가 없으면 `text: ""`, `no_speech_prob: 1.0`
- `confidence`는 3.3절의 열린 질문이다. 정하기 전에는 `null`로 보낸다

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
| `AUDIO_TOO_LONG` | `utterance` 10초, `conversation` 600초 초과 (v1과 같다). 남은 구간을 `too_long` final로 닫는다 | 예 (세션 끝) |
| `OVERLOADED` | 처리 못 한 오디오가 밀림 한도(9.5.5)를 넘었거나, 동시 세션이 한도에 찼다 | 밀림이면 예 (세션 끝), 세션 한도면 아니오 |
| `INTERNAL` | 서버 오류 | 예 (세션 끝) |

**순서 보장**
- 한 세션 안에서 `seq`는 1씩 늘고, 메시지는 `seq` 순서로 온다
- 구간마다 `segment_start` → `transcript_delta` 0개 이상 → `segment_final` 순서다. 구간 k의 `segment_final`은 구간 k+1의 `segment_start`보다 먼저 온다
- 앞서 보낸 글자를 고치는 메시지는 없다

### 9.5 서버 파이프라인

```
WebSocket 받기 (이벤트 루프)
  │ 바이너리
  ▼
[vad 스레드] PCM16 → float → 16kHz 변환 → 32ms 창마다 Silero VAD           (v1 on_audio와 같다)
  │ (창, 확률)
  ▼
[이벤트 루프] Segmenter.push_windows → 구간 열림: segment_start 보냄
  │                                     구간 닫힘: FinalGate에 등록 (9.5.4)
  │ 16kHz 음성 (VAD 결과와 상관없이 전부)
  ▼
[이벤트 루프] FeatureFrontend.push(음성) → 새 특징 32프레임(0.32초)마다 Chunk 하나 → 세션 큐
  │
  ▼
[GPU 스레드 1개] GpuStepper: 준비된 세션의 Chunk를 최대 ML_NEMO_BATCH_MAX개 묶어
                 conformer_stream_step 한 번 → 세션별 캐시·가설 갱신 → 새 글자
  │ (세션, 새 글자, audio_end_ms)
  ▼
[이벤트 루프] 세션: 열린 구간에 글자를 붙이고 transcript_delta 보냄 → FinalGate 확인 → segment_final
```

#### 9.5.1 세션 상태

| 항목 | 내용 |
|---|---|
| `segmenter` | v1과 같은 `Segmenter` (VAD 확률, 구간 번호, 닫힘) |
| `frontend` | 아직 특징으로 못 바꾼 음성 꼬리, 지금까지 만든 특징 프레임 수, 앞 조각 특징 끝 몇 프레임 (pre-encode 캐시) |
| `cache` | `cache_last_channel`, `cache_last_time`, `cache_last_channel_len` (배치 1짜리 텐서, GPU에 둔다) |
| `hyp`, `pred_out` | `previous_hypotheses`, `previous_pred_out`의 이 세션 몫 |
| `emitted` | 가설 글자 중 이미 보낸 길이 |
| `processed_ms` | GPU가 처리한 오디오 끝 위치 |
| `open_seg` | 지금 열린 구간 번호와 그 구간 글자 (없으면 None) |
| `orphan` | 열린 구간이 없을 때 생긴 글자 (9.5.4) |
| `queue` | 처리를 기다리는 Chunk 목록 |

#### 9.5.2 특징 만들기 (FeatureFrontend)

- NeMo 전처리기(`AudioToMelSpectrogramPreprocessor`)를 그대로 쓴다. 10ms 홉, 25ms 창, pre-emphasis 0.97, STFT center. 평가 모드에서는 dither가 꺼진다 (NeMo `features.py`, training일 때만 켠다). 정규화는 `NA`
- 조금씩 들어오는 음성에 바로 돌리면 조각 경계에서 값이 달라진다 (STFT 가장자리 채우기, pre-emphasis가 앞 샘플을 쓴다). 그래서:
  1. 지금까지의 음성에서 **앞쪽 여유 몇 프레임을 붙여** 잘라 낸다
  2. 그 조각에 전처리기를 돌린다
  3. 가장자리 프레임을 버리고 **새로 확정된 프레임만** 쌓는다
- 프레임 f는 샘플 `f×160 ± 200`이 다 들어와야 확정된다 (약 12.5ms 늦음, 무시해도 된다)
- **확인:** test 클립을 100ms씩 넣어 만든 특징이 파일 전체로 만든 특징(`CacheAwareStreamingAudioBuffer.append_audio_file`)과 같아야 한다. 차이 최댓값을 테스트로 남긴다

#### 9.5.3 조각 만들기와 GPU 묶기 (GpuStepper)

- 새 특징 32프레임마다 Chunk 하나를 만든다: `앞 조각 특징 끝 P프레임 + 새 32프레임`
  - P = `streaming_cfg.pre_encode_cache_size` (리스트면 `[1]`). 세션 첫 조각은 앞을 0으로 채운다
  - `pad_and_drop_preencoded=True`와 같은 방식이고, 평가 숫자도 이 방식으로 나왔다
  - 덕분에 모든 Chunk의 길이가 같아 세션 여러 개를 쌓을 수 있다
- GpuStepper 한 번:
  1. 큐가 빈 세션은 건너뛴다. **세션마다 가장 오래된 Chunk 하나씩**을 오래 기다린 순으로 최대 `ML_NEMO_BATCH_MAX`개 고른다. 세션 안에서는 조각 순서가 중요해서 한 번에 하나만 꺼낸다
  2. 첫 Chunk가 생긴 뒤 `ML_NEMO_STEP_WAIT_MS`(기본 10ms)가 지나면 모인 만큼 돌린다
  3. 특징 `[B, 80, P+32]`, 길이 `[B]`, 캐시를 배치 축으로 이어 붙인다. `cache_last_channel`과 `cache_last_time`은 축 1, `cache_last_channel_len`은 축 0이다. 모양은 `get_initial_cache_state(batch_size)`로 확인한다
  4. `torch.autocast("cuda", bfloat16)` 안에서 `conformer_stream_step(..., keep_all_outputs=False, previous_hypotheses=[세션별], previous_pred_out=[세션별], drop_extra_pre_encoded=streaming_cfg.drop_extra_pre_encoded, return_transcription=True)`
  5. 캐시를 세션별로 다시 나눠 넣는다. 세션별 가설 글자에서 `emitted` 뒤를 새 글자로 꺼내고 `processed_ms`를 늘린다
- 마지막 조각(`end`, `too_long`): 남은 프레임이 32보다 짧으면 그 세션만 따로 `keep_all_outputs=True`로 돌린다 (묶지 않는다)
- 계산은 GPU 스레드 하나에서만 한다. 이벤트 루프는 결과를 `loop.call_soon_threadsafe`로 받는다
- 실측 기준 (EXP-013): 4세션 묶음 조각 하나 62ms, 실시간은 320ms. 여유가 크지만 8세션 이상은 재 봐야 한다

#### 9.5.4 글자를 구간에 붙이기와 final (FinalGate)

- 새 글자가 나왔을 때:
  - 열린 구간이 있으면 그 구간에 붙이고 `transcript_delta`를 보낸다
  - 열린 구간이 없으면 `orphan`에 모은다. 다음 `segment_start` 바로 뒤 첫 `transcript_delta`로 보낸다. 세션이 끝날 때까지 구간이 안 열리면 버리고, 로그에 `orphan_chars` 숫자만 남긴다
- 구간이 닫혀도(Segmenter) 바로 `segment_final`을 보내지 않는다. **GPU가 그 구간이 닫힌 위치까지 처리했을 때**(`processed_ms ≥ 구간 close 위치`) 보낸다. 실시간이면 구간이 닫힐 때(말 끝 + 1.75초) 이미 처리가 끝나 있어서 기다리지 않는다. 밀릴 때만 기다린다
- 쉼 1.75초 동안 생긴 글자도 그 구간에 붙는다 (말이 끝나고 마지막 글자가 붙기까지 p95 0.64초)
- 15초 상한으로 잘린 구간(`max_length`)은 그 위치까지 처리된 글자로 닫고, 뒤 글자는 다음 구간으로 간다. 단어 하나가 둘로 갈릴 수 있다

#### 9.5.5 밀림과 한도

| 한도 | 기본값 | 넘으면 |
|---|---|---|
| 처리 못 한 오디오 (`queue` 길이 × 0.32초) | `ML_NEMO_MAX_BACKLOG_MS=5000` | `OVERLOADED`, 세션 끝 |
| 동시 세션 | `ML_NEMO_MAX_SESSIONS=16` | 새 `start`에 `OVERLOADED` (fatal 아님, 세션을 안 만든다) |
| 세션 길이 | v1과 같다 (utterance 10초, conversation 600초) | `AUDIO_TOO_LONG` |

오디오를 실제 속도보다 빨리 보내면(파일 재생) 큐가 쌓이고, 5초 넘게 앞서면 `OVERLOADED`가 난다. 파일은 `POST /asr`를 쓴다.

### 9.6 설정 추가

| 변수 | 기본값 | 의미 |
|---|---|---|
| `ML_NEMO_STEP_WAIT_MS` | 10 | 첫 Chunk가 생긴 뒤 다른 세션 Chunk를 기다리는 최대 시간 |
| `ML_NEMO_MAX_BACKLOG_MS` | 5000 | 세션당 처리 못 한 오디오 한도 |
| `ML_NEMO_MAX_SESSIONS` | 16 | v2 동시 세션 한도 |

- `ML_ENGINE`, `ML_NEMO_MODEL`, `ML_NEMO_ATT_CONTEXT`, `ML_NEMO_BATCH_MAX`는 3.1절에 있다
- VAD·구간 설정(`ML_VAD_THRESHOLD`, `ML_END_SILENCE_MS`, `ML_MAX_SEGMENT_S` 등)은 v1과 같은 값을 같이 쓴다
- `/healthz`에 `engine`, `v2_sessions`, `step_ms_p95`, `batch_mean`을 더한다

### 9.7 예: 대화 모드 한 세션

예시 글자는 지어낸 것이다.

```
→ {"type":"start","trace_id":"t1","sample_rate":16000,"mode":"conversation"}
← {"type":"ready","seq":1,"chunk_ms":320,...}
→ (오디오 100ms씩)
← {"type":"segment_start","seq":2,"segment_id":0,"start_ms":1180}
← {"type":"transcript_delta","seq":3,"segment_id":0,"append":"안녕","text":"안녕","audio_end_ms":2240}
← {"type":"transcript_delta","seq":4,"segment_id":0,"append":"하세요","text":"안녕하세요","audio_end_ms":2560}
   … (말 끝 3.42초, 쉼 1.75초)
← {"type":"segment_final","seq":7,"segment_id":0,"text":"안녕하세요 …","end_ms":3420,"reason":"silence","is_last":false}
← {"type":"segment_start","seq":8,"segment_id":1,...}
   …
→ {"type":"end","trace_id":"t1"}
← {"type":"segment_final","seq":15,"segment_id":2,...,"reason":"end","is_last":true}
← {"type":"session_end","seq":16,"segments":3,"audio_ms":18320}
```

### 9.8 로그 (글자는 찍지 않는다)

- 구간마다 `segment_final` 로그: `speech_ms`, `first_delta_ms`(말 시작 → 첫 delta), `final_wait_ms`(구간 닫힘 → final, 밀림), `chars`, `deltas`, `orphan_chars`, `reason`
- 10초마다: `v2_sessions`, `step_ms` p50/p95, `batch_mean`, 가장 긴 `queue`
- 형식은 v1과 같다 (JSON 한 줄, `trace_id`, `latency_ms`)

### 9.9 만들기와 확인 순서

1. **엔진 코어** (`server/engine_nemotron.py`): 모델 불러오기(2절), FeatureFrontend, 배치 1짜리 `step`. 여기서 특징 같음 확인(9.5.2)과 결과 같음 확인(6절 1단계)을 통과시킨다
2. **GpuStepper**: 세션 여러 개 묶기. 배치 4로 돌린 결과가 배치 1로 하나씩 돌린 결과와 같은지 본다
3. **`/asr-stream/v2`**: 세션 상태, FinalGate, 메시지, 한도. 가짜 엔진(조각마다 정해진 글자를 내는)으로 단위 테스트를 만든다: `seq`와 구간 순서, `text = 이전 text + append`, final이 delta를 이은 것과 같은지, orphan, `end` 처리, `OVERLOADED`
4. **시험 클라이언트** (`scripts/stream_client_v2.py`, `stream_client.py`를 고쳐서): wav를 실제 속도로 보내고 `segment_start`, `transcript_delta`, `segment_final`의 도착 시각을 잰다. 1, 4, 8세션
5. **목표 (제안)**: 말 끝 → 그 구간 마지막 delta p95 ≤ 800ms, `segment_final`은 쉼 1.75초 + 100ms 안, 4세션에서 `OVERLOADED` 0, step p95 ≤ 160ms (조각 시간의 절반)
6. **v1 어댑터** (3절): 같은 코어 위에서 `transcript_delta`를 `transcript_interim`(구간 `text` 전체, `stable_text`는 마지막 단어 앞까지)으로, `segment_final`을 `transcript_final`로 바꿔 `/asr-stream`에도 붙일 수 있다. 백엔드가 v2로 옮기기 전까지의 다리다
