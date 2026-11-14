"""CER 계산용 텍스트 정규화. 정답(AI Hub 라벨)과 모델 출력에 똑같이 쓴다.

AI Hub 608 라벨은 KsponSpeech와 비슷한 표기를 쓴다 (2026-11-14 전체 라벨에서 기호 문맥을 세어 확인):
  b/ n/ u/ o/ l/   숨·잡음·불명확·다른 화자·웃음 태그     -> 태그만 지운다
  단어*            불확실한 발음                          -> 표시만 지운다
  단어+            반복·말더듬                            -> 표시만 지운다
  (철자)/(발음)    이중 전사 (예: 숫자)                    -> 기본은 발음(spoken)
  한글(SNS)        한글 읽기 + 원문                        -> 한글만
  company(컴퍼니)  원문 + 한글 읽기                        -> 한글만
  <U+XXXX>         깨진 문자                              -> 지운다
  _x000D_          엑셀 줄바꿈 흔적                        -> 지운다
그 뒤 문장부호와 공백을 모두 지우고 영어는 소문자로 바꾼다 (공백 없는 글자 단위 CER).
"""

import re

_DUAL = re.compile(r"\(([^()]*)\)\s*/\s*\(([^()]*)\)")
_LATIN_THEN_HANGUL = re.compile(r"[A-Za-z0-9]+\s*\(([가-힣\s]+)\)")
_HANGUL_THEN_LATIN = re.compile(r"([가-힣]+)\s*\([A-Za-z0-9\s.&-]+\)")
_UNICODE_ESC = re.compile(r"<U\+[0-9A-Fa-f]{2,6}>")
_EXCEL_ESC = re.compile(r"_x[0-9A-Fa-f]{4}_?")
_NOISE_TAG = re.compile(r"(?<![A-Za-z])[A-Za-z]/")
_KEEP = re.compile(r"[^0-9A-Za-z가-힣]")


def normalize(text: str, dual: str = "spoken") -> str:
    t = text or ""
    t = _EXCEL_ESC.sub(" ", t)
    t = _UNICODE_ESC.sub(" ", t)
    t = _DUAL.sub(lambda m: m.group(2) if dual == "spoken" else m.group(1), t)
    t = _LATIN_THEN_HANGUL.sub(r"\1", t)
    t = _HANGUL_THEN_LATIN.sub(r"\1", t)
    t = _NOISE_TAG.sub(" ", t)
    t = _KEEP.sub("", t)  # 괄호·*·+·/·문장부호·공백
    return t.lower()
