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

## C. 풀고, 스크립트 올리고, 실행 (커널이 끝까지 기다리게)

> **백그라운드(nohup)로 띄우지 않는다.** 커널이 쉬면 Colab이 VM을 회수한다 (11-15: nohup으로 띄우고 exec가 바로 끝났더니
> 30분 안에 세션이 사라졌다). 아래 셋째 명령은 작업이 끝날 때까지 커널을 붙잡는다. CLI가 중간에 `TimeoutError`로
> 끝나도 VM 안 작업은 계속된다 (제로샷 때 확인). 진행은 D단계처럼 커널을 거치지 않고 확인한다.

```bash
echo 'import subprocess; subprocess.run("cat /content/drive/MyDrive/marieum-asr-ft/data/train_audio.tar.part-* | tar -x -C /content", shell=True, check=True); import os; print(len(os.listdir("/content/train_audio")), "speakers")' | colab --auth=oauth2 exec -s seg
colab --auth=oauth2 upload -s seg colab/segment_transcribe.py /content/segment_transcribe.py
echo 'import subprocess, os; d="/content/drive/MyDrive/marieum-asr-ft/runs/segments"; os.makedirs(d, exist_ok=True); p=subprocess.run("python /content/segment_transcribe.py >> "+d+"/run.log 2>&1", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s seg
```

- 첫 줄 출력이 `278 speakers`여야 한다
- 셋째 명령은 1~3시간 붙잡고 있다. 에이전트는 이 명령을 백그라운드 셸로 돌리고 기다린다. CLI가 `TimeoutError`로 끝나도 다시 실행하지 않는다
- 이미 Drive에 저장된 화자는 건너뛴다 (중간에 VM이 사라졌으면 B단계부터 다시 하면 이어서 한다)

## D. 진행 확인 (30분마다, 커널을 쓰지 않는 명령만)

```bash
colab --auth=oauth2 status -s seg
colab --auth=oauth2 download -s seg /content/drive/MyDrive/marieum-asr-ft/runs/segments/progress.txt work/segments/progress.txt && cat work/segments/progress.txt
colab --auth=oauth2 ls -s seg /content/drive/MyDrive/marieum-asr-ft/runs/segments
```

- `status`가 BUSY여야 정상이다. IDLE인데 `DONE`이 없으면 작업이 죽은 것이다. 그때 `run.log`를 내려받아 끝부분의 오류만 보고한다
- `progress.txt`에 `k/278 화자 ...`가 나온다. `ls`에 `DONE`이 보이면 E단계로 간다
- 이 단계에서 `colab exec`를 쓰지 않는다. 커널이 바빠서 줄을 서고, 앞 작업을 방해할 수 있다
- 사용자는 브라우저로 Drive `marieum-asr-ft/runs/segments/`의 `progress.txt`와 화자별 `.jsonl` 개수를 봐도 된다

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

---

# 파인튜닝 (Nemotron 3.5 Streaming 0.6B)

**모든 colab 명령 앞에 `MSYS_NO_PATHCONV=1`을 붙인다.** 긴 실행은 커널이 끝까지 기다리게 한다 (nohup 금지).

## 0. 학습 manifest 만들기 (이 PC, 문장 나누기 F단계 다음)

```bash
python data_prep/build_train_manifests.py
```

- 출력: `~/marieum-private/manifests/train_v2.jsonl`, `val_v2.jsonl` (전사 포함, 레포 밖)
- 화자당 상한 20분 (기준점 10분), 반복 중증 3·중등도 2·경증 1·기준점 2. 묶음별 시간과 비율이 출력된다
- v2부터 같은 녹음에서 바로 이어지는 구간(쉼 3초 이하)을 절반 확률로 18초까지 이어 붙인다. 쉬었다 이어 말할 때 글자를 놓치지 않게
- **사용자가 두 파일을 브라우저로 Drive `marieum-asr-ft/data/manifests/`에 올린다**

## 0-1. 설정 미리 검사 (이 PC, 학습 스크립트를 고쳤으면 매번)

```bash
python colab/check_train_config.py
```

NeMo 공식 설정에 대해 덮어쓰기 키를 하나씩 검사한다. `실패 0개`가 아니면 VM을 띄우지 않는다 (11-15 시험 학습이 키 오류로 2분 만에 실패).

## 1. VM과 Drive (사용자)

```bash
colab --auth=oauth2 new -s train --gpu A100
colab --auth=oauth2 drivemount -s train
```

- 본 학습은 A100(또는 H100)을 쓴다. L4 24GB는 메모리가 빠듯하다 (`docs/nemotron_finetune_cases.md`). 시험 학습은 L4도 된다
- **Drive 여유 공간:** 체크포인트(.ckpt, 이어하기용)는 개당 약 7GB, .nemo는 약 2.4GB다. 최소 20GB 여유가 필요하다

## 2. 시험 학습 (MAX_STEPS=1000)

```bash
colab --auth=oauth2 upload -s train colab/train_nemotron.py /content/train_nemotron.py
echo 'import subprocess, os; os.environ.update(MAX_STEPS="1000", RUN_NAME="pilot_v1"); p=subprocess.run("python /content/train_nemotron.py", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s train
```

- 처음 10~20분은 압축 풀기·NeMo 설치·모델 내려받기다
- CLI가 `TimeoutError`로 끝나도 다시 보내지 않는다. VM 안 학습은 계속된다

## 3. 진행 확인 (커널을 쓰지 않는 명령만, 30분마다)

```bash
colab --auth=oauth2 status -s train
colab --auth=oauth2 download -s train /content/drive/MyDrive/marieum-asr-ft/runs/train/pilot_v1.log work/train_pilot_v1.log && tail -5 work/train_pilot_v1.log
colab --auth=oauth2 ls -s train /content/drive/MyDrive/marieum-asr-ft/ckpt/pilot_v1/v1/checkpoints
```

- 로그 끝에 step 숫자, `train_loss`, 검증 때 `val_wer`가 보인다. 메모리 부족(`CUDA out of memory`)이면 멈추고 보고한다
  → 다시 할 때 `BATCH_DURATION`을 줄인다 (예: `os.environ.update(BATCH_DURATION="30")`)
- `status`가 IDLE이 되면 끝난 것이다. `.nemo`가 생겼는지 `ls`로 본다

## 4. 평가 (같은 VM)

test_v2 묶음을 올리고(제로샷 3단계와 같다), 파인튜닝 모델로 추론한다.

```bash
echo 'import os; os.makedirs("/content/upload", exist_ok=True); print("ok")' | colab --auth=oauth2 exec -s train
for f in work/upload/test_v2.tar.part-*; do colab --auth=oauth2 upload -s train "$f" "/content/upload/$(basename "$f")" || break; done
colab --auth=oauth2 upload -s train colab/zeroshot_nemotron.py /content/zeroshot_nemotron.py
echo 'import subprocess, os, glob; m=sorted(glob.glob("/content/drive/MyDrive/marieum-asr-ft/ckpt/pilot_v1/v1/checkpoints/*.nemo"))[-1]; os.environ.update(MODEL=m, OUT_NAME="pilot_v1_320ms"); p=subprocess.run("python /content/zeroshot_nemotron.py", shell=True); print("exit", p.returncode, m)' | colab --auth=oauth2 exec -s train
mkdir -p work/eval/pilot_v1_320ms
colab --auth=oauth2 download -s train /content/out/pilot_v1_320ms/hyps.jsonl work/eval/pilot_v1_320ms/hyps.jsonl
colab --auth=oauth2 download -s train /content/out/pilot_v1_320ms/run_info.json work/eval/pilot_v1_320ms/run_info.json
colab --auth=oauth2 stop -s train
colab --auth=oauth2 sessions
```

## 4-1. 평가만 다시 (VM이 회수됐을 때)

학습이 끝난 뒤 평가 중에 VM이 회수돼도 `.nemo`는 Drive에 있다. 새 VM에서 평가만 한다. 추론은 GPU를 다 쓰지 않으므로 **L4로 충분하다** (A100보다 CU가 훨씬 적게 든다). 추론은 약 30분.

- **CU가 0이면 A100·L4를 못 받는다** (`colab --auth=oauth2 usage`로 잔액 확인). 무료 T4로도 평가할 수 있다: `--gpu T4`. T4는 bf16이 없어서 스크립트가 fp16으로 바꿔 돈다 (`run_info.json`의 `amp_dtype`). 추론 1~1.5시간 예상

```bash
colab --auth=oauth2 new -s evalft --gpu L4      # 사용자
colab --auth=oauth2 drivemount -s evalft        # 사용자
```

이후는 4단계와 같다. 세션 이름만 `evalft`로 바꾼다. 결과는 `/content/out/<OUT_NAME>/`과 Drive `runs/eval/<OUT_NAME>/` 두 곳에 저장된다 (11-16부터). 내려받기 전에 VM이 또 회수되면 Drive에서 브라우저로 받는다.

## 4-2. NCP 서버에서 평가 (Colab CU가 없을 때)

NCP GPU 서버(`gp1l4-g3`, L4 24GB)에서 같은 스크립트로 평가한다. L4는 bf16을 지원하므로 A100 평가와 조건이 같다. 평가는 학습이 아니라서 서빙 GPU를 써도 된다 (단, 시연 서버가 돌고 있으면 끝난 뒤에).

**서버에 필요한 것:** Ubuntu, NVIDIA 드라이버(`nvidia-smi`가 된다), Python 3.10 이상, 디스크 여유 15GB

**1. 파일 옮기기 (사용자)**
- Drive `marieum-asr-ft/ckpt/ft_v1/v1/checkpoints/ft_v1.nemo`(약 2.4GB)를 브라우저로 이 PC에 받는다
- 이 PC에서 서버로 보낸다. 정답 파일(refs)은 보내지 않는다

```bash
ssh <서버> "mkdir -p ~/asr-eval/upload"
scp work/upload/test_v2.tar.part-* <서버>:~/asr-eval/upload/
scp colab/zeroshot_nemotron.py <ft_v1.nemo 받은 경로> <서버>:~/asr-eval/
```

**2. 서버에서 실행**

```bash
sudo apt-get install -y libsndfile1 ffmpeg git python3-venv
python3 -m venv ~/asr-eval/venv && source ~/asr-eval/venv/bin/activate
cd ~/asr-eval && WORKDIR=~/asr-eval MODEL=~/asr-eval/ft_v1.nemo OUT_NAME=ft_v1_320ms nohup python zeroshot_nemotron.py > run_ft_v1.out 2>&1 &
tail -f run_ft_v1.out
```

- 처음 10~20분은 NeMo 설치다. 추론은 약 30~40분
- 서버는 Colab과 달리 `nohup`으로 띄워도 회수되지 않는다. ssh가 끊겨도 계속 돈다

**3. 결과 받기와 정리**

```bash
mkdir -p work/eval/ft_v1_320ms
scp <서버>:~/asr-eval/out/ft_v1_320ms/hyps.jsonl <서버>:~/asr-eval/out/ft_v1_320ms/run_info.json work/eval/ft_v1_320ms/
ssh <서버> "rm -rf ~/asr-eval/test_v2 ~/asr-eval/upload"
```

- 평가가 끝나면 서버의 테스트 음성은 지운다 (AI Hub 데이터는 필요한 곳에만 둔다). 모델과 NeMo는 서빙에 다시 쓰므로 남겨도 된다

## 5. 채점 (이 PC)

```bash
python eval/score.py --hyps work/eval/pilot_v1_320ms/hyps.jsonl --out work/eval/pilot_v1_320ms
```

제로샷(EXP-001)과 비교한다. 중증 CER이 내려가고 경증이 크게 나빠지지 않으면 본 학습(MAX_STEPS 3000, RUN_NAME ft_v1)으로 간다.

---

# 중증 구간 다시 건지기 (본 학습 전)

시험 학습 모델(pilot_v1)로 중등도·중증 72명의 말소리 구간(약 12.5h)을 다시 받아쓴다. 이 PC에서 yoona-J 받아쓰기와 합쳐 정렬한다.
**모든 colab 명령 앞에 `MSYS_NO_PATHCONV=1`.** 긴 실행은 커널이 끝까지 기다리게 한다.

## 1. VM과 Drive (사용자)

```bash
colab --auth=oauth2 new -s retx --gpu A100
colab --auth=oauth2 drivemount -s retx
```

추론만 하므로 L4도 된다 (A100보다 1.5~2배 느림).

## 2. 실행

```bash
colab --auth=oauth2 upload -s retx colab/retranscribe_segments.py /content/retranscribe_segments.py
colab --auth=oauth2 upload -s retx work/upload/retranscribe_speakers.txt /content/retranscribe_speakers.txt
echo 'import subprocess; p=subprocess.run("python /content/retranscribe_segments.py", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s retx
```

- 처음 15~20분: 학습 녹음 풀기, NeMo 설치, 구간 자르기 (약 9,200개)
- 받아쓰기: A100 약 1시간 (예상). 11-15 실제로는 전체 9분 (녹음·NeMo가 VM에 남아 있지 않았는데도)
- CLI가 `TimeoutError`로 끝나도 다시 보내지 않는다

## 3. 진행 확인 (커널을 쓰지 않는 명령만)

```bash
colab --auth=oauth2 status -s retx
colab --auth=oauth2 ls -s retx /content/drive/MyDrive/marieum-asr-ft/runs/segments_pilot
```

`DONE`이 보이면 4단계. IDLE인데 `DONE`이 없으면 `run.log`를 내려받아 끝부분 오류를 보고한다.

## 4. 결과 받기와 반납

```bash
echo 'import subprocess; subprocess.run("tar -czf /content/segments_pilot.tgz -C /content/drive/MyDrive/marieum-asr-ft/runs segments_pilot", shell=True, check=True); import os; print(os.path.getsize("/content/segments_pilot.tgz"))' | colab --auth=oauth2 exec -s retx
colab --auth=oauth2 download -s retx /content/segments_pilot.tgz work/segments/segments_pilot.tgz
colab --auth=oauth2 stop -s retx
colab --auth=oauth2 sessions
```

## 5. 합쳐 정렬하고 학습 목록 v3 만들기 (이 PC)

```bash
mkdir -p work/segments/vm2 && tar -xzf work/segments/segments_pilot.tgz -C work/segments/vm2 --strip-components=1
python data_prep/align_segments.py --segs work/segments/vm --segs2 work/segments/vm2
python data_prep/build_train_manifests.py --name v3 --stats work/manifests/train_v3_stats.csv
```

- 출력 끝에 "남긴 구간의 출처"가 나온다: `p1` yoona-J 길잡이, `p2` 시험 학습 모델 길잡이, `-sandwich` 앞뒤 순서
- `train_v3.jsonl`, `val_v3.jsonl`을 브라우저로 Drive `data/manifests/`에 올린다
- 본 학습: `TRAIN_MANIFEST=train_v3.jsonl VAL_MANIFEST=val_v3.jsonl MAX_STEPS=3000 RUN_NAME=ft_v1`
- 11-15 결과: 9,233구간 중 빈 출력 1,418개. 정렬 구간 train 37.2h → 38.7h (중등도 5.76 → 6.68h, 중증 3.42 → 4.03h). 두 길잡이가 모두 남긴 구간은 자른 위치가 중등도 89%, 중증 79% 같다(차이 20% 미만). 시험 모델 길잡이로 새로 남긴 중증 구간의 받아쓰기-정답 CER 중앙값 0.14. 학습 목록 반복 후 49.0h → 51.9h (중증 10.0 → 11.5h, 중등도 11.5 → 12.9h). 화자당 20분 상한 때문에 늘어난 만큼 다 들어가지는 않는다

주의: 시험 학습 모델은 1차에서 남은 구간을 학습했다. 그 구간에서는 받아쓰기가 정답을 외워 맞힐 수 있지만, 이미 남은 구간이라 결과는 같다. 새로 건지는 구간(1차에서 버려진 것)은 학습에 쓰이지 않았으므로 외운 것이 아니다.

---

# 0단계 실험: 쉼마다 끊기, yoona-J 공정 비교, 지연 1.12초 (11-16)

ft_v1(EXP-004)은 녹음 전체를 한 번에 흘려 넣어 평가했다. 운영 서버는 쉼(1.75초)마다 구간을 끊고 구간마다 모델을 새로 돌린다. 세 가지를 한 VM에서 차례로 잰다.

| 실험 | 스크립트 | 결과 이름 |
|---|---|---|
| B. ft_v1을 쉼마다 끊어서 | `colab/eval_segmented.py` MODE=nemo | `ft_v1_vad_320ms` |
| A. yoona-J를 운영 서버처럼 (공정한 기준선) | `colab/eval_segmented.py` MODE=whisper | `yoonaj_vad` |
| C. ft_v1을 녹음 전체로, 지연 1.12초 | `colab/zeroshot_nemotron.py` ATT_CONTEXT=[56,13] | `ft_v1_1120ms` |

## 1. VM과 Drive (사용자)

```bash
colab --auth=oauth2 new -s stage0 --gpu L4
colab --auth=oauth2 drivemount -s stage0
```

## 2. 실행 (한 번만)

```bash
echo 'import os; os.makedirs("/content/upload", exist_ok=True); print("ok")' | colab --auth=oauth2 exec -s stage0
for f in work/upload/test_v2.tar.part-*; do colab --auth=oauth2 upload -s stage0 "$f" "/content/upload/$(basename "$f")" || break; done
colab --auth=oauth2 upload -s stage0 colab/zeroshot_nemotron.py /content/zeroshot_nemotron.py
colab --auth=oauth2 upload -s stage0 colab/eval_segmented.py /content/eval_segmented.py
colab --auth=oauth2 upload -s stage0 colab/run_stage0.py /content/run_stage0.py
echo 'import subprocess; p=subprocess.run("python /content/run_stage0.py", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s stage0
```

- 전체 약 1.5시간 (B 40분, A 15분, C 35분). L4 약 3 CU
- 셋 중 하나가 실패해도 다음 것은 돈다

## 3. 진행 확인 (커널을 쓰지 않는 명령만, 20분마다)

```bash
colab --auth=oauth2 status -s stage0
colab --auth=oauth2 ls -s stage0 /content/drive/MyDrive/marieum-asr-ft/runs/eval
```

`ft_v1_vad_320ms`, `yoonaj_vad`, `ft_v1_1120ms` 폴더가 끝나는 순서대로 생긴다. IDLE이 되면 4단계.

## 4. 결과 받기와 반납

```bash
for d in ft_v1_vad_320ms yoonaj_vad ft_v1_1120ms; do mkdir -p work/eval/$d; for f in hyps.jsonl run_info.json; do colab --auth=oauth2 download -s stage0 /content/out/$d/$f work/eval/$d/$f; done; done
colab --auth=oauth2 stop -s stage0
colab --auth=oauth2 sessions
```

## 5. 채점 (이 PC)

```bash
for d in ft_v1_vad_320ms yoonaj_vad ft_v1_1120ms; do python eval/score.py --hyps work/eval/$d/hyps.jsonl --out work/eval/$d; done
```

---

# ft_v2: 쉼이 들어간 조각으로 이어 학습 (11-16)

0단계 결과 (EXP-005~007): 자유발화 19개에서 yoona-J 33.7 대 ft_v1 38.2. 남은 오류는 대부분 빠뜨림이고 쉬었다 다시 말하는 문장 첫머리에서 많다.

- **학습 목록 v4:** 구간 앞뒤에 실제 녹음 0.2~0.8초를 더 붙인다 (이웃 말소리까지의 쉼 안에서만). 이어 붙이기를 쉼 6초까지, 확률 80%로 늘린다. 반복 후 60.4h (늘어난 시간은 대부분 앞뒤 쉼)
- **학습:** ft_v1에서 이어서 2000 step, 학습률 2e-5, warmup 100, 최대 길이 25초
- **검증:** ft_v1과 같은 `val_v3.jsonl`. val_wer를 ft_v1 마지막 값 0.399와 바로 비교할 수 있다
- **평가:** 같은 VM에서 이어서 1.12초, 0.32초

## 0. 학습 목록 (이 PC, 완료)

```bash
python data_prep/build_train_manifests.py --name v4 --stats work/manifests/train_v4_stats.csv --context-pad 0.8 --merge-gap 6 --merge-prob 0.8 --merge-max 18
```

- **사용자가 `~/marieum-private/manifests/train_v4.jsonl`을 Drive `marieum-asr-ft/data/manifests/`에 올린다** (`val_v3.jsonl`은 이미 있다)

## 1. VM과 Drive (사용자)

```bash
colab --auth=oauth2 new -s ftv2 --gpu A100
colab --auth=oauth2 drivemount -s ftv2
```

## 2. 실행 (한 번만)

```bash
colab --auth=oauth2 usage
echo 'import os; os.makedirs("/content/upload", exist_ok=True); print("ok")' | colab --auth=oauth2 exec -s ftv2
for f in work/upload/test_v2.tar.part-*; do colab --auth=oauth2 upload -s ftv2 "$f" "/content/upload/$(basename "$f")" || break; done
colab --auth=oauth2 upload -s ftv2 colab/train_nemotron.py /content/train_nemotron.py
colab --auth=oauth2 upload -s ftv2 colab/zeroshot_nemotron.py /content/zeroshot_nemotron.py
colab --auth=oauth2 upload -s ftv2 colab/run_ft_v2.py /content/run_ft_v2.py
echo 'import subprocess; p=subprocess.run("python /content/run_ft_v2.py", shell=True); print("exit", p.returncode)' | colab --auth=oauth2 exec -s ftv2
```

- 약 3시간 (학습 2시간 15분 + 준비 15분 + 평가 40분)

## 3. 진행 확인 (커널을 쓰지 않는 명령만, 30분마다)

```bash
colab --auth=oauth2 status -s ftv2
colab --auth=oauth2 download -s ftv2 /content/drive/MyDrive/marieum-asr-ft/runs/train/ft_v2.log work/train_ft_v2.log && tail -5 work/train_ft_v2.log
colab --auth=oauth2 ls -s ftv2 /content/drive/MyDrive/marieum-asr-ft/runs/eval
```

- 학습 중: 로그 끝의 step, val_wer. 학습이 끝나면 `runs/eval`에 `ft_v2_1120ms`, `ft_v2_320ms`가 차례로 생긴다

## 4. 결과 받기와 반납

```bash
for d in ft_v2_1120ms ft_v2_320ms; do mkdir -p work/eval/$d; for f in hyps.jsonl run_info.json; do colab --auth=oauth2 download -s ftv2 /content/out/$d/$f work/eval/$d/$f; done; done
colab --auth=oauth2 stop -s ftv2
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```
