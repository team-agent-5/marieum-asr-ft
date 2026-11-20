# 마리음 ASR v2 백엔드 연동 가이드

2026-11-20 작성. 서버 코드는 [`serve/`](../serve/README.md), 설계 배경은 [`engine_handoff.md`](engine_handoff.md) 9절.

## 한눈에

Nemotron ft_v3 ML 서버가 GPU 서버 `10.1.2.6:8001`에서 돌고 있다. 백엔드는 실시간은 WebSocket `/asr-stream/v2`, 녹음 파일은 `POST /asr`로 붙는다.

| API | 용도 |
| --- | --- |
| `WS ws://10.1.2.6:8001/asr-stream/v2` | 실시간 전사. 0.32초마다 새로 붙은 글자(`transcript_delta`), 말이 끝나면 문장 확정(`segment_final`) |
| `POST http://10.1.2.6:8001/asr` | 녹음 파일 전사 (WAV 또는 PCM16, 60초까지) |
| `GET /healthz` | 모델 준비 여부, 세션 수, GPU 지연 |
| `GET /livez` | 프로세스 생존 |

지금 배포: `main` (11-20, confidence 토큰 수 가중 평균 포함), 모델 `marieum-nemotron-ft_v3`, GPU NVIDIA L4. 서버는 내부망 주소라 같은 망(예: `inno-server-001`)에서만 열린다.

백엔드가 꼭 지킬 규칙:

- 오디오는 **실제 속도로** 보낸다. 5초 넘게 앞서 보내면 `OVERLOADED`로 세션이 끊긴다. 파일은 WebSocket이 아니라 `POST /asr`로 보낸다
- WebSocket 연결 하나에 **세션은 한 번에 하나**. `session_end`를 받은 뒤 같은 연결에서 다음 `start`를 보낼 수 있다
- 화면 표시는 `segment_final.text`로 맞춘다. **`text`가 빈 문자열인 final은 무시**한다 (말이 없었다는 뜻)
- 세션의 끝은 `session_end`로 판단한다. `is_last: true`인 `segment_final` 바로 뒤에 한 번 온다

## 상태 확인

`GET /healthz`가 200이어야 세션을 열 수 있다. 서버를 띄운 뒤 모델 로딩과 워밍업에 약 1분(실측 30.7초 + 24.3초)이 걸리고, 그동안은 503이다.

```json
{"status":"ok","engine":"nemotron","models":["asr"],"model_version":{"asr":"marieum-nemotron-ft_v3"},
 "gpu":{"name":"NVIDIA L4"},"uptime_sec":67,"sessions":0,"gpu_pending":0,
 "step_ms_p50":39.2,"step_ms_p95":43.2,"batch_mean":1.0}
```

- 로딩 중 또는 로딩 실패: HTTP 503, `{"status":"loading"|"error","engine":"nemotron","uptime_sec":…,"message":"…"}`. 이때 WebSocket에 붙으면 `MODEL_UNAVAILABLE` 뒤 1013으로 닫힌다
- `sessions`: 지금 열린 실시간 세션 수 (한도 16)
- `step_ms_p95`: GPU 한 번 계산 시간. 0.32초(320ms)보다 한참 작아야 실시간을 따라간다
- `GET /livez`: 항상 `{"status":"alive"}`. 프로세스가 살아 있는지만 본다

## 파일 전사: POST /asr

녹음 파일 하나를 본문에 바이트 그대로 보내면 한 번에 전사한다. multipart가 아니다.

| 항목 | 내용 |
| --- | --- |
| 본문 | WAV 파일 바이트 그대로, 또는 PCM16 LE 모노 바이트 + `?sample_rate=` |
| WAV 형식 | PCM 8/16/24/32비트, 채널 수 무관, 8000~48000Hz. 서버가 16kHz 모노로 바꾼다. MP3, M4A, WebM은 안 된다 |
| 최대 길이 | 60초 |
| trace_id | `X-Trace-Id` 헤더 또는 `?trace_id=` (영문·숫자·`_-`, 64자 이내). 없으면 서버가 만든다 |
| 동시 처리 | 파일 4개까지. 넘는 요청은 오류 없이 기다렸다가 처리된다 |

```bash
curl -X POST -H "Content-Type: audio/wav" -H "X-Trace-Id: abc123" \
     --data-binary @rec.wav http://10.1.2.6:8001/asr
curl -X POST --data-binary @rec.pcm "http://10.1.2.6:8001/asr?sample_rate=16000&trace_id=abc123"
```

응답 (200, 실제 응답 그대로):

```json
{"trace_id":"smoke2","text":"안녕하세요. 오늘 배가 조금 아파요","confidence":0.7029,"duration_ms":2783,
 "model_version":"marieum-nemotron-ft_v3",
 "segments":[{"text":"안녕하세요. 오늘 배가 조금 아파요","confidence":0.7029,"start_ms":400,"end_ms":2640}]}
```

- `segments`: 파이프라인이 확정한 단위 (쉼 약 0.8초마다). `start_ms`, `end_ms`는 파일 시작 기준 말의 시작과 끝
- `text`: 구간 글자를 공백으로 잇는다. 마침표·쉼표·물음표가 붙고, 숫자는 거의 한글로 나온다 (예: "세 시")
- 말소리가 없으면 `text: ""`, `segments: []`, `confidence: 0.0`. LLM이나 TTS로 넘기지 말고 다시 말해 달라고 처리한다
- 처리 시간: 2.8초 파일 455ms (실측). 60초 파일은 아직 재지 않았다 (계산상 수 초~11초)

| HTTP | code | 뜻 |
| --- | --- | --- |
| 400 | `INVALID_FORMAT` | 본문이 비었거나 trace_id·sample_rate 형식 오류 |
| 413 | `AUDIO_TOO_LONG` | 60초 초과 |
| 415 | `INVALID_FORMAT` | PCM WAV가 아니거나, WAV가 아닌 바이트인데 `sample_rate`가 없음 |
| 503 | `MODEL_UNAVAILABLE` | 모델 로딩 중 |
| 500 | `INTERNAL` | 전사 중 서버 오류 |

오류 본문: `{"code":"AUDIO_TOO_LONG","message":"…","trace_id":"abc123"}`

## 실시간 전사: WS /asr-stream/v2

`ws://10.1.2.6:8001/asr-stream/v2`에 연결하고 `start`를 보내면 서버가 `ready`로 답한다. 그다음 오디오를 바이너리로 보내고, 끝낼 때 `end`를 보낸다. 연결 직후 서버가 먼저 보내는 것은 없다.

### 백엔드 → 서버

| 메시지 | 형식 | 내용 |
| --- | --- | --- |
| `start` | 텍스트 JSON | `{"type":"start","trace_id":"abc123","sample_rate":16000,"mode":"conversation","encoding":"pcm16","language":"ko","patient_id":"p-01"}` |
| 오디오 | 바이너리 | PCM16 little-endian 모노. 프레임 크기는 아무거나 되지만 20~100ms 권장. 실제 속도로 |
| `end` | 텍스트 JSON | `{"type":"end","trace_id":"abc123"}`. 남은 오디오를 끝까지 처리하고 마지막 문장을 닫는다 |
| `ping` | 텍스트 JSON | `{"type":"ping","trace_id":"abc123"}` → `{"type":"pong","trace_id":"abc123"}` |

`start` 필드:

| 필드 | 필수 | 값 |
| --- | --- | --- |
| `trace_id` | 예 | 영문·숫자·`_-`, 64자 이내. 이 세션의 모든 메시지에 그대로 돌아온다 |
| `sample_rate` | 예 | 8000~48000 정수. 16000 권장 (다르면 서버가 16kHz로 바꾸는데, 프레임마다 따로 바꿔서 소리가 조금 어긋날 수 있다) |
| `mode` | 아니오 | `conversation`(기본) 또는 `utterance`. 아래 "모드" |
| `encoding` | 아니오 | `pcm16`만 |
| `language` | 아니오 | `ko`만 |
| `patient_id` | 아니오 | 로그용, 64자까지 |

### 서버 → 백엔드

세션 안 메시지에는 모두 `type`, `trace_id`, `seq`(1부터 1씩)가 붙는다. `pong`과 세션 밖 `error`에는 `seq`가 없다.

| type | 언제 | 필드 |
| --- | --- | --- |
| `ready` | `start`를 받아 세션을 만들었을 때 | `engine`("nemotron"), `model_version`, `chunk_ms`(320), `sample_rate` |
| `segment_start` | 말소리가 150ms 넘게 이어져 새 문장이 열렸을 때. 첫 글자보다 먼저 온다 | `segment_id`(0부터), `start_ms`(세션 시작 기준) |
| `transcript_delta` | 이 문장의 글자가 늘었을 때 (최대 0.32초마다) | `segment_id`, `append`(새로 붙은 글자), `text`(이 문장의 지금까지 글자 전체), `revised`, `audio_end_ms` |
| `segment_final` | 문장이 확정됐을 때. 이 뒤로 이 문장의 delta는 없다 | `segment_id`, `text`, `start_ms`, `end_ms`, `reason`, `confidence`, `no_speech_prob`, `is_last` |
| `session_end` | `is_last: true`인 `segment_final` 바로 다음 한 번 | `segments`(문장 수), `audio_ms`(받은 오디오 길이) |
| `error` | 오류 | `code`, `message`, `fatal` |

글자 처리 규칙:

- `revised: false` → 화면의 이 문장 뒤에 `append`를 붙인다. `append` 앞에 공백이 있으면 새 단어, 없으면 단어가 이어진 것이다 (예: `"안"` 다음 `"녕하"`)
- `revised: true` → 앞서 보낸 글자가 바뀌었다. 이 문장 글자를 `text`로 통째로 바꾼다 (`append`는 빈 문자열). 지금 설정에서는 드물다
- `segment_final.text`가 그 문장의 기준이다. 보통 마지막 delta의 `text`와 같지만, 확정 직전에 나온 글자는 delta 없이 final에만 들어올 수 있다

`segment_final.reason`: `silence`(0.8초 쉬었다) / `max_length`(쉬지 않고 15초) / `end`(`end`를 받음) / `too_long`(세션 길이 초과).

### 순서 보장

- 한 세션 안에서 `seq`는 1씩 늘고, 메시지는 `seq` 순서로 온다
- 문장마다 `segment_start` → `transcript_delta` 0개 이상 → `segment_final`. 문장 k의 `segment_final`은 문장 k+1의 `segment_start`보다 먼저 온다
- 정상 종료는 항상 `segment_final(is_last: true)` → `session_end`로 끝난다

### 오류 코드

| code | 언제 | fatal | 그 뒤 |
| --- | --- | --- | --- |
| `INVALID_FORMAT` | JSON이 아님, 필드 값 오류, 모르는 type | 아니오 | 세션을 만들지 않는다. 고쳐서 다시 보낸다 |
| `NO_SESSION` | `start` 없이 오디오나 `end`, 또는 다른 `trace_id`의 `end` | 아니오 | 무시된다 |
| `SESSION_BUSY` | 세션이 진행 중인데 또 `start` | 아니오 | 진행 중 세션은 그대로 |
| `OVERLOADED` | 동시 세션 16개가 찼다 | 아니오 | 세션을 만들지 않는다. 잠시 뒤 다시 `start` |
| `OVERLOADED` | 처리 못 한 오디오가 5초를 넘었다 (너무 빨리 보냄) | 예 | 세션이 끝난다. `session_end`는 오지 않는다 |
| `AUDIO_TOO_LONG` | `utterance` 10초, `conversation` 600초 초과 | 예 | 남은 문장을 `reason: too_long`으로 닫고 `session_end`까지 온다 |
| `INTERNAL` | 서버 오류 | 예 | 세션이 끝난다. `session_end`는 오지 않는다 |
| `MODEL_UNAVAILABLE` | 모델 로딩 중에 연결 | 예 | 서버가 1013으로 연결을 닫는다 |

fatal 오류로 세션이 끝나도 연결은 열려 있다 (`MODEL_UNAVAILABLE` 제외). 같은 연결에서 새 `start`를 보낼 수 있다.

## 모드

모드는 글자를 몇 개의 문장으로 나눌지만 바꾼다. 인식 방식은 같아서, 어느 모드를 고를지는 앱 화면 흐름으로 정한다.

| | `conversation` (기본) | `utterance` |
| --- | --- | --- |
| 맞는 화면 | 마이크를 켜 두고 계속 듣는 대화 화면 | 버튼을 누르고 한 마디 하는 화면 |
| 문장 끝을 정하는 곳 | 서버 (0.8초 쉬면). 쉬지 않고 15초면 자른다 | 백엔드 (`end`를 보낼 때) |
| 세션 하나에 문장 | 여러 개 (`segment_id` 0, 1, 2…) | 하나 (`segment_id` 0) |
| 쉬었다 다시 말하면 | 0.8초 넘게 쉬면 다음 문장. 말 중간에 1초쯤 쉬면 한 문장이 둘로 나뉜다 | 같은 문장에 계속 붙는다 |
| `segment_final` | 문장이 닫힐 때마다 (`is_last: false`), `end` 뒤 마지막 하나 (`is_last: true`) | `end` 뒤 한 번 (`is_last: true`) |
| 세션 길이 상한 | 600초 | 10초 |

다음 문장을 `utterance`로 따로 받으려면 `end` → `session_end`를 받은 뒤 같은 연결에서 새 `start`를 보낸다.

`mode`에는 `conversation`, `utterance` 두 값만 받는다. 다른 값(예: `accuracy`, `realtime`)은 `INVALID_FORMAT`으로 거절된다.

## 예시 세션과 클라이언트 코드

실제 서버(`a073d63`, VAD 쉼 0.8초)에 "안녕하세요. 오늘 배가 조금 아파요."(2.8초) + 쉼 3초를 100ms씩 실제 속도로 보낸 실측이다. 왼쪽 시간은 연결부터.

```
→ {"type":"start","trace_id":"t1","sample_rate":16000,"mode":"conversation"}
   2ms ← {"type":"ready","seq":1,"engine":"nemotron","model_version":"marieum-nemotron-ft_v3","chunk_ms":320,"sample_rate":16000}
→ (오디오 100ms씩)
 124ms ← {"type":"segment_start","seq":2,"segment_id":0,"start_ms":0}
 662ms ← {"type":"transcript_delta","seq":3,"segment_id":0,"append":"안녕","text":"안녕","revised":false,"audio_end_ms":640}
 963ms ← {"type":"transcript_delta","seq":4,"append":"하세요","text":"안녕하세요",...}
1259ms ← {"type":"transcript_delta","seq":5,"append":".","text":"안녕하세요.",...}
1561ms ← {"type":"transcript_delta","seq":6,"append":" 오늘","text":"안녕하세요. 오늘",...}
   ...  (seq 7~9: " 배가", " 조금", " 아파")
2870ms ← {"type":"transcript_delta","seq":10,"append":"요","text":"안녕하세요. 오늘 배가 조금 아파요",...}
   ...  (말 끝 2624ms, 0.8초 쉬면)
3571ms ← {"type":"segment_final","seq":11,"segment_id":0,"text":"안녕하세요. 오늘 배가 조금 아파요",
          "start_ms":0,"end_ms":2624,"reason":"silence","confidence":0.704,"no_speech_prob":0.0,"is_last":false}
→ {"type":"end","trace_id":"t1"}
5830ms ← {"type":"segment_start","seq":12,"segment_id":1,"start_ms":5760}
5877ms ← {"type":"segment_final","seq":13,"segment_id":1,"text":"","reason":"end","confidence":0.0,"no_speech_prob":1.0,"is_last":true}
5878ms ← {"type":"session_end","seq":14,"segments":2,"audio_ms":5783}
```

말이 끝나고 `segment_final`까지 약 0.95초다 (VAD 쉼 0.8초 + 처리).

마지막 빈 문장(seq 12, 13)은 말없이 쉬다가 `end`를 보내면 생긴다. `is_last` 문장을 꼭 보내기 때문이고, 백엔드는 `text`가 비었으니 무시하면 된다.

Python 클라이언트 (`pip install websockets`). WAV는 헤더 길이가 파일마다 다르므로(예: macOS `say` 파일은 4096바이트) `wave` 모듈로 읽는다. 앞 44바이트만 건너뛰면 헤더가 소리로 들어가 인식이 틀어진다.

```python
import asyncio, json, wave
import websockets

URL = "ws://10.1.2.6:8001/asr-stream/v2"

async def transcribe(path, trace_id="t1", mode="conversation"):
    with wave.open(path) as w:  # 16kHz 모노 PCM16 WAV
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    shown = {}  # segment_id -> 화면 글자
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"type": "start", "trace_id": trace_id, "sample_rate": 16000, "mode": mode}))
        assert json.loads(await ws.recv())["type"] == "ready"

        async def send_audio():
            for i in range(0, len(pcm), 3200):  # 100ms = 1600샘플 = 3200바이트
                await ws.send(pcm[i:i + 3200])
                await asyncio.sleep(0.1)  # 실제 속도
            await ws.send(json.dumps({"type": "end", "trace_id": trace_id}))

        sender = asyncio.create_task(send_audio())
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "transcript_delta":
                sid = m["segment_id"]
                shown[sid] = m["text"] if m["revised"] else shown.get(sid, "") + m["append"]
            elif m["type"] == "segment_final":
                if m["text"]:  # 빈 문장은 무시
                    shown[m["segment_id"]] = m["text"]
                    print("문장:", m["text"], m["confidence"])
            elif m["type"] == "error":
                print("오류:", m["code"], m["message"])
                if m.get("fatal"):
                    break
            elif m["type"] == "session_end":
                break
        sender.cancel()

asyncio.run(transcribe("rec.wav"))
```

## 주의할 점과 아직 확인 안 된 것

백엔드가 처리해야 할 동작:

- **빈 문장:** `segment_final.text`가 `""`(`no_speech_prob: 1.0`)이면 말이 없었다는 뜻이다. LLM이나 TTS로 넘기지 않는다
- **delta 없는 글자:** delta는 열려 있는 문장에만 온다. 오디오를 실제 속도보다 빨리 보내면 문장 글자가 delta 없이 `segment_final`에만 올 수 있다
- **`start_ms` 기준:** 실시간은 소리 감지(VAD) 기준, `POST /asr`는 모델 정렬 기준이다. 같은 음성도 값이 다를 수 있다 (예: 0 vs 400)
- **출력 모양:** 마침표·쉼표·물음표가 붙고, 숫자는 거의 한글로 나온다. 숫자 모양에 기대는 로직이 있으면 확인한다

confidence:

- 0~1 사이 값이다. 모델이 토큰마다 낸 신뢰도(entropy 방식)를 문장 안 토큰 수로 가중 평균한다. 같은 소리면 실시간과 파일이 같은 값을 준다 (실측 0.7029)
- **값이 낮게 나오는 척도다.** 맞게 인식한 문장도 대체로 0.4~0.7이다 (TTS 실험에서 맞은 문장 중앙값 0.56, 범위 0.37~0.70. 11-20 실제 목소리 테스트 0.23~0.66). 다른 척도에서 정한 기준값(예: 0.45)을 그대로 쓰면 맞은 문장도 많이 걸러진다
- 기준값은 이 서버 점수로 다시 정한다. TTS 기준 임시값: 0.38이면 맞은 문장의 90%가 통과한다 (틀린 문장도 46% 통과). 실제 녹음으로 맞음/틀림을 표시해 다시 정하는 것이 정확하다
- 다른 계산 방식(max_prob, gibbs)은 점수가 0.9대로 높게 나오지만 틀린 문장도 같이 높아서 가려내는 힘이 더 약했다. 그래서 지금 방식을 쓴다

운영 상태:

- 서버 주소 `10.1.2.6`은 내부망이다. `inno-server-001`에서는 열리는 것을 확인했다. 백엔드가 다른 서버에서 돈다면 `curl http://10.1.2.6:8001/healthz`로 먼저 확인한다
- 서버는 지금 터미널에서 띄운 상태다. 다시 띄우면 약 1분간 503이다
- `/docs`(Swagger)에는 HTTP API만 나온다. WebSocket은 OpenAPI에 들어가지 않아서 이 문서가 기준이다. 내부망 밖에서 보려면 `ssh -N -L 8001:10.1.2.6:8001 agent5` 뒤 `http://localhost:8001/docs`

아직 확인 안 된 것:

- [ ] 동시 세션 1·4·8개 지연 (목표: 말 끝 → 마지막 글자 p95 800ms 이하, 4세션에서 `OVERLOADED` 0번)
- [ ] 정상 발화(구음장애가 없는 목소리). TTS 360문장 실험에서 258문장이 글자 없이 나왔다 (잡음 없는 문장도 85개 중 37개). 사람 목소리로 다시 봐야 하고, 그전까지 빈 결과를 "다시 말해 주세요"로 처리하는 것이 안전하다
- [ ] 60초 파일 처리 시간
- [ ] confidence 기준값 (실제 환자 녹음으로 맞음/틀림을 표시해 정한다)
