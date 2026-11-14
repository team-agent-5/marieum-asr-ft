# marieum-asr-ft

마리음 실시간 음성 인식 모델을 구음장애 음성으로 파인튜닝하고 평가하는 레포다. 서빙은 [marieum-ai](../marieum-ai)가 맡고, 이 레포는 **체크포인트와 성능표**만 넘긴다.

```
데이터 전처리 → 학습 → 평가 → results.csv 한 줄 → (좋으면) marieum-ai의 ML_MODEL_DIR로 교체
```

## 데이터 규칙

- 원본: AI Hub 구음장애 음성인식 데이터(608), 외장하드 `E:\Dataset_Backup\구음장애_음성인식_데이터`
- **재배포 금지다.** 음성, 라벨 JSON, 전사 문장, 전사가 들어간 manifest는 git에 넣지 않는다 (`.gitignore`). 레포에는 코드, 설정, 화자 ID 목록, 숫자로 된 결과만 둔다
- 데이터는 외장하드, 이 PC, GPU 서버, 팀 Google Drive(Colab 학습용)에만 둔다 (Drive 업로드는 2026-11-13 확인)
- 로그와 출력에 전사 문장을 찍지 않는다. 필요하면 글자 수나 CER만 찍는다

## 구조

| 폴더 | 내용 |
|---|---|
| `data_prep/` | `inventory.py` 라벨만 읽어 파일·화자 목록 (`work/inventory/`). `make_listening_set.py` 직접 들어 난이도를 매길 화자별 20초 클립과 채점 페이지 (`work/listen/`) |
| `splits/` | 화자 분할. `test_v1.csv`는 직접 들어 점수를 매긴 22명 (고정, 바꾸지 않는다). `listen_scores_v1.csv` 307명 난이도 점수. `train_v1` / `val_v1` / `test_v2`는 `data_prep/make_splits.py`로 만든다 |
| `experiments/` | `results.csv` 실험 하나에 한 줄 |
| `docs/` | 조사 기록. [`nemotron_finetune_cases.md`](docs/nemotron_finetune_cases.md) Nemotron 3.5 파인튜닝 사례와 우리 설정에 반영할 점 |
| `work/` | 중간 산출물 (git 밖) |

학습(`train/`), 평가(`eval/`), 설정(`configs/`), 환경(`envs/`)은 필요해질 때 만든다.

## 분할 원칙

- **화자 단위로 나눈다.** 화자 ID = `질환코드-이니셜-성별나이-지역` (예: `21-HHS-M30-SU`)
- **테스트는 `splits/test_v1.csv` 22명으로 고정한다.** 점수(0~5)는 비전문가 1명이 들어 매긴 주관 점수다. 쉬움(0~2) 8명, 어려움(3~5) 14명
- `yoonaj_train=True`(26.청각+뇌신경 5명)는 기존 모델 yoona-J가 학습한 화자다. 기존 모델과 비교할 때는 따로 본다
- 같은 대본 문장이 여러 화자에게 나온다. 학습에 없던 문장과 자기소개·자유발화의 CER을 따로 낸다
- 학습 후보: 뇌신경장애, 조음장애(21), 감음신경성(28). 후두장애와 전음성은 뺀다
- 화자당 학습 분량에 상한을 둔다 (화자 중간값이 약 3.8시간이라 몇 명이 학습을 독차지하지 않게)

## 실험 기록

모든 실험은 같은 테스트 세트, 같은 평가 스크립트로 잰다. 기준선은 `EXP-000` yoona-J, `EXP-001` 제로샷 후보 모델이다.

## 실행

```bash
python data_prep/inventory.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터"
```

```bash
python data_prep/make_listening_set.py --root "E:/Dataset_Backup/구음장애_음성인식_데이터"
```

`work/listen/index.html`을 크롬으로 열어 듣고 채점한다 (ffmpeg 필요). 채점 결과는 페이지의 **CSV 저장**으로 `work/listen/scores/`에 둔다. 메모 칸에 전사 문장을 적지 않는다.

Windows 콘솔에서 한글이 깨지면 `PYTHONIOENCODING=utf-8`을 붙인다.
