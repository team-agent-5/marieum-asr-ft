"""train_nemotron.py의 NeMo 설정 덮어쓰기를 GPU에 올리기 전에 이 PC에서 검사한다.

    python colab/check_train_config.py

NeMo main 브랜치의 공식 설정(yaml)을 받아, Hydra 덮어쓰기 규칙대로 하나씩 적용해 본다.
  키=값    키가 있어야 하고, 묶음(dict)을 글자로 바꾸면 안 된다
  +키=값   키가 없어야 한다
  ++키=값  있든 없든 된다
  ~키      키가 있어야 한다
11-15 시험 학습이 'use_bucketing' 키 없음으로 2분 만에 실패한 뒤 만들었다.
"""

import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_nemotron as T  # noqa: E402

URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/{T.CONF_DIR}/{T.CONF_NAME}.yaml"


def lookup(cfg, dotted):
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return None, None, False
        node = node[p]
    if not isinstance(node, dict):
        return None, None, False
    return node, parts[-1], parts[-1] in node


def main():
    cfg = yaml.safe_load(urllib.request.urlopen(URL, timeout=30).read().decode("utf-8"))
    ovs = T.overrides("TRAIN.jsonl", "VAL.jsonl", 90.0, 5)
    bad = 0
    for ov in ovs:
        if ov.startswith("~"):
            key, val, mode = ov[1:], None, "del"
        elif ov.startswith("++"):
            (key, val), mode = ov[2:].split("=", 1), "force"
        elif ov.startswith("+"):
            (key, val), mode = ov[1:].split("=", 1), "add"
        else:
            (key, val), mode = ov.split("=", 1), "set"
        if key == "init_from_pretrained_model":  # 최상위 키 (speech_to_text_finetune.py가 읽는다)
            ok = key not in cfg
            msg = "" if ok else "이미 있다 (+ 대신 ++ 필요)"
            cfg[key] = val
        else:
            parent, leaf, exists = lookup(cfg, key)
            ok, msg = True, ""
            if parent is None:
                ok, msg = False, "상위 묶음이 없다"
            elif mode == "del":
                ok = exists
                msg = "" if ok else "지울 키가 없다"
                if ok:
                    del parent[leaf]
            elif mode == "add":
                ok = not exists
                msg = "" if ok else "이미 있다 (+ 대신 ++ 또는 그냥 키=값)"
                parent[leaf] = val
            elif mode == "set":
                if not exists:
                    ok, msg = False, "키가 없다 (++ 필요)"
                elif isinstance(parent[leaf], dict):
                    ok, msg = False, "묶음(dict)을 값 하나로 바꿀 수 없다 (~ 후 + 필요)"
                else:
                    parent[leaf] = val
            else:
                parent[leaf] = val
        bad += not ok
        print(("  OK   " if ok else "  FAIL ") + ov + (f"   <- {msg}" if msg else ""))
    print(f"\n{len(ovs)}개 중 실패 {bad}개 ({URL})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
