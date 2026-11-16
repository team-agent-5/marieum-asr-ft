"""Colab VM: 긴 작업을 커널과 떼어 돌리고, 커널은 기다리기만 하게 한다.

11-16 ft_v3에서 colab exec 연결이 53분 만에 끊기며(RuntimeError: Connection was lost.) 학습도 같이 멈췄다.
작업을 커널의 자식이 아닌 새 세션으로 띄우면 exec 연결이 끊겨도 작업은 계속된다.
커널이 놀면 VM이 30분 안에 회수되므로, wait로 커널을 계속 바쁘게 한다. wait는 몇 번을 다시 보내도 안전하다.

    python /content/job.py start run_ft_v3.py   # 한 번만 시작된다 (/content/job.pid가 있으면 아무것도 안 한다)
    python /content/job.py wait                 # 끝날 때까지 기다리고 exit 코드와 출력 끝을 보여 준다
    python /content/job.py peek                 # 기다리지 않고 지금 상태만 본다
"""

import os
import subprocess
import sys
import time

PID, OUT, EXIT = "/content/job.pid", "/content/job.out", "/content/job.exit"


def tail():
    return open(OUT, encoding="utf-8", errors="replace").read()[-3000:] if os.path.exists(OUT) else ""


def main():
    cmd = sys.argv[1]
    if cmd == "start":
        if os.path.exists(PID):
            print("이미 시작됨, pid", open(PID).read())
            return
        p = subprocess.Popen(f"python /content/{sys.argv[2]} > {OUT} 2>&1; echo $? > {EXIT}", shell=True,
                             start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        open(PID, "w").write(str(p.pid))
        print("시작, pid", p.pid)
    elif cmd == "wait":
        while not os.path.exists(EXIT):
            time.sleep(30)
        print("exit", open(EXIT).read().strip())
        print(tail())
    elif cmd == "peek":
        print("끝남, exit " + open(EXIT).read().strip() if os.path.exists(EXIT) else "도는 중" if os.path.exists(PID) else "시작 안 함")
        print(tail()[-1500:])


if __name__ == "__main__":
    main()
