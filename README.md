# marieum-asr-ft

마리음 실시간 음성 인식 모델을 구음장애 음성으로 파인튜닝하고 평가하는 레포다. 서빙은 [marieum-ai](https://github.com/team-agent-5/marieum-ai)가 맡고, 이 레포는 **모델과 성능표, 서비스 연결 안내**를 넘긴다.

- 지금 서비스 모델: Whisper-small `yoona-J/ASR_Whisper_Disease_General` (말이 끝나고 쉼 1.75초 뒤에 문장 확정)
- 바꿀 모델: NVIDIA Nemotron 3.5 ASR Streaming 0.6B (cache-aware FastConformer-RNNT, 0.32초마다 글자를 덧붙인다)를 AI Hub 구음장애 음성(608)으로 파인튜닝한 **ft_v3**
- 서비스 연결: [`docs/engine_handoff.md`](docs/engine_handoff.md). 그 스펙대로 만든 서버: [`serve/`](serve/README.md)
- 전체 과정 설명 (공유용 페이지): https://claude.ai/artifact/SjWUALX9iqvzHV4aSDx7Tp

```
데이터 전처리 → 문장 조각 정렬 → 학습 (Colab A100) → 평가 (Colab L4) → results.csv → marieum-ai 엔진 교체
```

## 지금 결과 (2026-11-17)

테스트: 학습에 쓰지 않은 30명(380분), 화자별 CER의 평균. 자세한 숫자는 [`experiments/results.csv`](experiments/results.csv), 날짜별 기록은 [`docs/plan.md`](docs/plan.md).

| CER % (낮을수록 좋음) | yoona-J (운영 방식) | Nemotron 학습 전 | ft_v1 | ft_v2 | **ft_v3** |
|---|---|---|---|---|---|
| 전체 30명 | 25.0 | 54.7 | 40.9 | 32.1 | **28.4** |
| 낭독 대본 11개 (yoona-J가 외우고 있음) | 9.9 | | 42.7 | 29.2 | 21.2 |
| 그 밖 19개 | 33.7 | | 39.9 | 33.9 | **32.6** |
| 처음 보는 글자만 (`eval/unseen_text.py`, 참고용) | 79.0 | | 58.2 | 57.5 | 57.1 |
| 반복 루프 (글자) | 690 | | 0 | 0 | **0** |
| 지어내기 (삽입) | 6.8% | | 4.1% | 2.4% | 2.5% |

실시간으로 쓸 때 (녹음 30개 앞 3분, 실제 속도로 흘려 넣는 흉내, L4, `colab/streaming_metrics.py`):

| | yoona-J (운영 방식) | **ft_v3 (0.32초)** |
|---|---|---|
| 말이 끝나고 문장 글자가 다 뜨기까지 p50 / p95 | 1.95초 / 2.11초 | **0.26초 / 0.64초** |
| 화면 글자가 바뀐 문장 | 90% | **0%** |
| 최종 100자당 지워진 글자 | 209자 | **0자** |
| 말 시작 후 첫 글자 p50 | **0.40초** | 0.86초 |
| GPU 한 번 계산 p95 | 0.39초 (1명) | **0.06초 (4명 동시)** |

주의
- test_v2 글자의 86%가 학습 라벨에도 있는 대본이다. test_v2 점수는 처음 듣는 말(시연)보다 좋게 나온다. ft_v2 → ft_v3에서 오른 폭은 대부분 이 대본 쪽이다
- 경증 화자는 아직 yoona-J보다 나쁘다 (대본 밖 19개 중 경증 23.8 vs 21.2). 화자 수로는 19명 중 8명만 ft_v3가 낫다
- **정상 발화(구음장애 없는 사람) 성능은 아직 안 쟀다**

## 모델 파일

| | |
|---|---|
| 파일 | `ft_v3.nemo` (2,553,098,240 바이트) |
| 위치 | 팀 Google Drive `cdy-hack-ag5/marieum-asr-ft/ckpt/ft_v3/v1/checkpoints/ft_v3.nemo` |
| 공유 | 소유자가 필요한 팀원에게만 지정 공유한다. **링크 공개, GitHub, 공개 저장소에 올리지 않는다** (아래 데이터 규칙) |
| 설정 | `att_context_size=[56,3]` (0.32초), 언어 프롬프트 `ko-KR`, 언어 태그 제거, bf16 |
| 시험한 NeMo | GitHub main `eaf0098dde` (2026-11-16, 3.1.0 계열) |

이전 모델 `ft_v1.nemo`, `ft_v2.nemo`도 같은 Drive `ckpt/` 아래에 남아 있다.

## 데이터 규칙

- 원본: AI Hub 구음장애 음성인식 데이터(608), 외장하드 `E:\Dataset_Backup\구음장애_음성인식_데이터`
- **재배포 금지다.** 음성, 라벨 JSON, 전사 문장, 전사가 들어간 manifest는 git에 넣지 않는다 (`.gitignore`). 레포에는 코드, 설정, 화자 ID 목록, 숫자로 된 결과만 둔다
- 데이터는 외장하드, 이 PC, GPU 서버, 팀 Google Drive(Colab 학습용)에만 둔다 (Drive 업로드는 2026-11-13 확인)
- 채점용 정답은 레포 밖 `~/marieum-private/`에만 두고 VM에 올리지 않는다
- 로그와 출력에 전사 문장을 찍지 않는다. 필요하면 글자 수나 CER만 찍는다 (학습 검증 로그도 `model.log_prediction=false`)
- **학습한 모델도 공개하지 않는다.** 모델이 학습 대본을 외우고 있어서 전사가 새어 나갈 수 있다. marieum-ai는 공개 레포라 모델 파일을 넣지 않는다

## 구조

| 폴더 | 내용 |
|---|---|
| `data_prep/` | `inventory.py` 라벨 목록, `make_listening_set.py` 듣고 채점, `make_splits.py` 화자 분할, `extract_test_audio.py` 테스트 음성, `select_train_audio.py` 학습 녹음 고르기, `align_segments.py` 긴 녹음을 문장 조각으로 정렬, `build_train_manifests.py` 학습 목록 (앞뒤 쉼 붙이기, 이어 붙이기, 묶음별 반복) |
| `colab/` | Colab VM 스크립트. `train_nemotron.py` 학습, `zeroshot_nemotron.py` 평가, `run_ft_v*.py` 학습+평가 묶음, `streaming_metrics.py` 실시간 지표, `job.py` 연결이 끊겨도 도는 작업, `check_train_config.py` 학습 설정 사전 검사. 순서는 [`docs/colab_runbook.md`](docs/colab_runbook.md) |
| `eval/` | `score.py` 화자별·묶음별 CER, `unseen_text.py` 처음 보는 글자만 CER, `audit_colab_run.py` 외부 에이전트가 VM을 돌린 뒤 검사 |
| `splits/` | 화자 분할 (`train_v1`, `val_v1`, `test_v2`), 307명 난이도 점수 |
| `experiments/` | `results.csv` 실험 하나에 한 줄 (EXP-000 ~ 014) |
| `serve/` | Nemotron ft_v3 서버. `/asr-stream/v2`, `POST /asr`, `/healthz` (NeMo 스트리밍 파이프라인 위). 실행과 남은 확인은 [`serve/README.md`](serve/README.md) |
| `tests/` | `serve/` 테스트. GPU 없이 가짜 엔진으로 돈다 (`pytest`) |
| `docs/` | `plan.md` 일정과 기록, `colab_runbook.md` VM 실행 순서, `engine_handoff.md` 서비스 연결, 조사 기록 |
| `work/` | 중간 산출물 (git 밖) |

## 분할 원칙

- **화자 단위로 나눈다.** 화자 ID = `질환코드-이니셜-성별나이-지역` (예: `21-HHS-M30-SU`). 학습 268명, 검증 10명, 테스트 30명
- 테스트 `test_v2` 30명: 23명은 학습에서 한 번도 안 들은 사람, 7명은 같은 사람의 다른 녹음이 학습에 있다. 이 중 22명은 직접 들어 점수를 매긴 `test_v1` (쉬움 8, 어려움 14)
- 난이도: 직접 들어 0~5점, 경증(0~1)·중등도(2~3)·중증(4~5) 세 묶음으로만 쓴다
- yoona-J 학습 화자(26번 폴더 등)는 기존 모델과 비교할 때 따로 본다
- 화자당 학습 분량에 상한(20분)을 두고, 중증 3번·중등도 2번 반복한다

## 실행

학습·평가는 Colab VM에서 한다. VM 만들기와 Drive 연결은 사람이 하고, 그 뒤 단계는 [`docs/colab_runbook.md`](docs/colab_runbook.md)대로 한다. 끝나면 이 PC에서 검사하고 채점한다.

```bash
python eval/audit_colab_run.py --expect-head <커밋>
```

```bash
python eval/score.py --hyps work/eval/ft_v3_320ms/hyps.jsonl --out work/eval/ft_v3_320ms
```

Windows 콘솔에서 한글이 깨지면 `PYTHONIOENCODING=utf-8`을 붙인다.
