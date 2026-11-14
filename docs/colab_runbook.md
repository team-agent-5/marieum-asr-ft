# Colab 실행 순서 (제로샷 비교)

agy가 이 순서대로 `colab` CLI를 실행한다. 명령은 모두 레포 루트(`D:\codyssey-hack\marieum-asr-ft`)에서 Git Bash로 돌린다.

> **Git Bash에서는 먼저 `export MSYS_NO_PATHCONV=1`을 실행한다.** 안 하면 Git Bash가 `/content/...` 같은 VM 경로를
> `C:/Program Files/Git/content/...`로 바꿔 `upload`·`download`·`ls`가 500 오류로 실패한다 (11-14 확인).
> PowerShell에서 `bash -lc "..."`로 한 줄씩 부르면 매번 `MSYS_NO_PATHCONV=1 colab ...`처럼 앞에 붙인다.

## 지킬 것

- **정답 전사는 레포 밖(`~/marieum-private/`)에 있다. 읽거나 올리지 않는다.** 채점은 사용자가 이 PC에서 한다
- 로그나 출력에 전사 문장을 붙여 넣지 않는다. 숫자(CER 등)만 보고한다
- 작업이 끝나면 반드시 `colab stop`으로 VM을 반납하고 `colab sessions`로 남은 VM이 없는지 확인한다
- 인증은 `--auth=oauth2`를 쓴다 (이 PC에 토큰이 있다)
- `colab drivemount`, `colab auth`는 브라우저 동의가 필요해서 에이전트가 돌릴 수 없다. 제로샷에는 필요 없다

## 0. 상태 확인

```bash
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```

다른 프로젝트 세션(`hc-e05-pamap2-*`)이 켜져 있으면 사용자에게 알린다. 끄지 않는다.

## 1. Windows에서 CLI가 되는지 짧게 시험 (CPU, 1~2분)

```bash
colab --auth=oauth2 new -s smoke
echo 'import sys; print(sys.version)' | colab --auth=oauth2 exec -s smoke
colab --auth=oauth2 upload -s smoke README.md /content/README.md
colab --auth=oauth2 ls -s smoke /content
colab --auth=oauth2 stop -s smoke
```

하나라도 실패하면 멈추고 오류를 보고한다. Windows는 CLI 공식 지원 대상이 아니라서, 실패하면 WSL에서 다시 시도한다.

## 2. 테스트 음성 묶기 (이 PC, 이미 했으면 건너뜀)

```bash
bash colab/pack_test.sh
```

`work/upload/test_v2.tar.part-00`, `-01` (합계 약 200MB)이 생긴다. 안에는 `manifest.jsonl`과 `audio/*.flac`만 있다.

## 3. GPU VM 빌리고 올리기

```bash
colab --auth=oauth2 new -s nemo --gpu L4
echo 'import os; os.makedirs("/content/upload", exist_ok=True); print(sorted(os.listdir("/content")))' | colab --auth=oauth2 exec -s nemo
for f in work/upload/test_v2.tar.part-*; do colab --auth=oauth2 upload -s nemo "$f" "/content/upload/$(basename "$f")" || break; done
colab --auth=oauth2 ls -s nemo /content/upload
```

- **올리기 전에 `/content/upload` 폴더를 먼저 만든다.** 없는 폴더에 올리면 500 오류가 난다 (11-14, 있는 폴더 `/content`에는 같은 방식으로 성공)
- 조각은 50MB, 4개다 (`part-00`~`part-03`). 마지막 `ls`에 4개가 다 보여야 한다
- 폴더를 만든 뒤에도 500이 나면 크기 문제다. `colab/pack_test.sh`의 `50M`을 `20M`으로 바꿔 다시 묶고, 올린 조각을 지운 뒤 다시 올린다
- L4가 안 잡히면(400 오류) `--gpu T4`로 바꾼다. 제로샷 추론은 T4로도 된다

## 4. Nemotron 제로샷 (처음엔 설치 포함 15~30분)

```bash
colab --auth=oauth2 exec -s nemo -f colab/zeroshot_nemotron.py
mkdir -p work/eval/nemotron_zs_320ms
colab --auth=oauth2 download -s nemo /content/out/nemotron_zs_320ms/hyps.jsonl work/eval/nemotron_zs_320ms/hyps.jsonl
colab --auth=oauth2 download -s nemo /content/out/nemotron_zs_320ms/run_info.json work/eval/nemotron_zs_320ms/run_info.json
```

- 기본 지연은 320ms(`[56,3]`)다. 다른 지연은 스크립트 위쪽 `ATT_CONTEXT`를 바꿔 다시 돌린다
- 실패하면 `/content/out/nemotron_zs_320ms/run.log` 끝부분을 확인한다:
  `echo 'print(open("/content/out/nemotron_zs_320ms/run.log").read()[-3000:])' | colab --auth=oauth2 exec -s nemo`

## 5. 기존 모델(yoona-J) 같은 조건 전사 (10~20분)

```bash
colab --auth=oauth2 exec -s nemo -f colab/transcribe_whisper.py
mkdir -p work/eval/whisper_yoonaj
colab --auth=oauth2 download -s nemo /content/out/whisper_yoonaj/hyps.jsonl work/eval/whisper_yoonaj/hyps.jsonl
colab --auth=oauth2 download -s nemo /content/out/whisper_yoonaj/run_info.json work/eval/whisper_yoonaj/run_info.json
```

## 6. 반납

```bash
colab --auth=oauth2 stop -s nemo
colab --auth=oauth2 sessions
```

## 7. 채점 (이 PC)

```bash
python eval/score.py --hyps work/eval/nemotron_zs_320ms/hyps.jsonl --out work/eval/nemotron_zs_320ms
python eval/score.py --hyps work/eval/whisper_yoonaj/hyps.jsonl --out work/eval/whisper_yoonaj
```

결과는 각 폴더의 `summary.csv`, `per_speaker.csv`다. 숫자만 보고한다.

## 채점 시 알아 둘 것

- **CER:** 공백 없는 글자 단위. 라벨 기호(잡음 태그, `*`, `+`, 이중 전사)는 `eval/textnorm.py` 규칙으로 정리한다
- **평균:** 화자별 CER의 평균이다
- **앞 600초만 쓴 녹음 6개:** 정답이 녹음 전체 전사라서, 들은 부분까지만 맞춰 채점한다
- **진행자 목소리:** 정답에는 환자 말만 있을 수 있다. 진행자 말을 받아 적으면 삽입 오류가 된다. 두 모델에 똑같이 적용되니 비교에는 문제가 없지만, 절대값은 실제보다 나쁘게 나올 수 있다
- **`track=seen` 7명:** 지금은 학습 전이라 모든 화자가 처음 보는 사람이다. 파인튜닝 뒤부터 `seen`과 `unseen`을 나눠 본다

---

# 문장 나누기 (학습 데이터 만들기)

학습 녹음 128시간(278명, 583개 파일)을 말소리 구간으로 자르고 yoona-J로 받아쓴다. 정답 붙이기는 이 PC에서 한다.
**모든 colab 명령 앞에 `MSYS_NO_PATHCONV=1`을 붙인다 (Git Bash).**

## A. 묶어서 Drive에 올리기 (사용자, 이 PC)

```bash
bash colab/pack_train.sh
```

`work/upload/train_audio.tar.part-00` ~ `-04` (1GB × 4 + 0.3GB, 정답 없음)가 생긴다.
**브라우저로 Google Drive `marieum-asr-ft/data/` 폴더에 5개를 모두 올린다.**

## B. VM 띄우고 Drive 연결

```bash
colab --auth=oauth2 new -s seg --gpu L4
colab --auth=oauth2 drivemount -s seg
```

`drivemount`는 브라우저 동의 후 Enter가 필요하다. **사용자가 직접 실행한다** (에이전트는 못 한다).

## C. 풀고, 스크립트 올리고, 백그라운드로 실행

```bash
echo 'import subprocess; subprocess.run("cat /content/drive/MyDrive/marieum-asr-ft/data/train_audio.tar.part-* | tar -x -C /content", shell=True, check=True); import os; print(len(os.listdir("/content/train_audio")), "speakers")' | colab --auth=oauth2 exec -s seg
colab --auth=oauth2 upload -s seg colab/segment_transcribe.py /content/segment_transcribe.py
echo 'import subprocess, os; os.makedirs("/content/drive/MyDrive/marieum-asr-ft/runs/segments", exist_ok=True); subprocess.Popen("nohup python /content/segment_transcribe.py > /content/drive/MyDrive/marieum-asr-ft/runs/segments/run.log 2>&1 &", shell=True); print("started")' | colab --auth=oauth2 exec -s seg
```

- 첫 줄 출력이 `278 speakers`여야 한다
- 처음 몇 분은 패키지 설치와 모델 내려받기다

## D. 진행 확인 (몇십 분마다)

```bash
echo 'from pathlib import Path; d=Path("/content/drive/MyDrive/marieum-asr-ft/runs/segments"); print((d/"progress.txt").read_text() if (d/"progress.txt").exists() else "아직 없음"); print("DONE" if (d/"DONE").exists() else ""); print(open(d/"run.log").read()[-1500:])' | colab --auth=oauth2 exec -s seg
```

- `progress.txt`에 `k/278 화자 ...`가 나온다. 예상 1~3시간
- 로그에 오류가 있으면 멈추고 보고한다. 결과는 화자 단위로 Drive에 저장되므로, VM이 끊겨도 C단계부터 다시 하면 남은 화자만 이어서 한다

## E. 결과 받기와 반납

```bash
echo 'import subprocess; subprocess.run("tar -czf /content/segments.tgz -C /content/drive/MyDrive/marieum-asr-ft/runs segments", shell=True, check=True); import os; print(os.path.getsize("/content/segments.tgz"))' | colab --auth=oauth2 exec -s seg
mkdir -p work/segments
colab --auth=oauth2 download -s seg /content/segments.tgz work/segments/segments.tgz
colab --auth=oauth2 stop -s seg
colab --auth=oauth2 sessions
```

## F. 정답 붙이기 (이 PC)

```bash
mkdir -p work/segments/vm && tar -xzf work/segments/segments.tgz -C work/segments/vm --strip-components=1
python data_prep/align_segments.py --segs work/segments/vm
```

출력: `~/marieum-private/segments_train.jsonl`, `segments_val.jsonl` (학습 manifest, 레포 밖), `work/segments/align_stats.csv` (화자별 숫자).
