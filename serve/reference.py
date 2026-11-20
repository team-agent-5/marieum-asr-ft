"""읽기 문서(reference)로 인식을 돕는다. 웹소켓·GPU를 모른다.

- hint_words(문장들): 문서의 어절을 단어 힌트(phrase boosting) 목록으로 만든다
- Reference.snap(글자): 인식 글자를 문서에서 가장 가까운 부분으로 바꾼다 (충분히 가까울 때만)

snap은 문서 전체를 한 줄로 이어 놓고, 인식 글자가 그 안의 어느 부분과 가장 가까운지 찾는다
(앞뒤가 자유로운 편집 거리, semi-global alignment). VAD가 문장을 둘로 나눠도("물 한 잔 가져다" | "주세요")
각 조각이 문서의 해당 부분에 맞춰진다. 바뀐 부분은 어절 경계까지 넓혀서 단어가 잘리지 않게 한다.
"""

import numpy as np


def _norm(text):
    """비교용: 글자와 숫자만. 공백·문장부호는 뺀다."""
    return "".join(ch for ch in text if ch.isalnum())


def _edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def hint_words(sentences, limit=1000):
    """문서 문장들 → 어절 목록 (문장부호 빼고, 2글자 이상, 처음 나온 순서, 중복 없이)."""
    out, seen = [], set()
    for s in sentences:
        for w in s.split():
            w = _norm(w)
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                out.append(w)
                if len(out) >= limit:
                    return out
    return out


# 비용: 들은 글자를 버리기 2, 문서 글자를 건너뛰기 1, 다른 글자로 바꾸기 1.
# 인식기는 없는 글자를 지어내는 일이 드물어서(덧붙인 글자 2.5%, EXP-015) 들은 글자는 되도록 문서 글자에 대응시킨다.
# 같은 비용이면 "말이 있고 밖에" → "입고 밖에"처럼 앞 단어를 버리는 쪽이 이긴다
DROP_HEARD = 2


class Reference:
    # 기준값 (11-21, ft_v3 실측 조각 159문장으로 고름): 7글자 이상, 차이 40% 이하만 바꾼다.
    # 4글자까지 허용하면 문서 밖 "배가"→"아빠가요" 같은 짧은 조각이 문서로 잘못 바뀌었다 (47문장 중 3)
    def __init__(self, sentences, max_ratio=0.4, min_chars=7):
        self.text = " ".join(s.strip() for s in sentences if s and s.strip())
        self.idx = [i for i, ch in enumerate(self.text) if ch.isalnum()]  # 비교용 글자 → 원문 위치
        norm = "".join(self.text[i] for i in self.idx)
        self.codes = np.array([ord(ch) for ch in norm], dtype=np.int64)
        self.max_ratio, self.min_chars = max_ratio, min_chars

    def __bool__(self):
        return len(self.codes) > 0

    def _align(self, t):
        """t(비교용 글자)가 문서 안 어디와 가장 가까운지: (편집 거리, 시작, 끝). 행 하나씩 numpy로 계산한다."""
        n = len(self.codes)
        j = np.arange(n + 1)
        prev = np.zeros(n + 1, np.int64)  # 문서 어디서 시작해도 비용 0
        start = j.copy()
        for ch in t:
            cost = prev + DROP_HEARD  # 들은 글자 하나가 문서에 없음
            src = start.copy()
            diag = prev[:-1] + (self.codes != ord(ch))  # 맞추기 / 바꾸기
            better = diag < cost[1:]
            cost[1:] = np.where(better, diag, cost[1:])
            src[1:] = np.where(better, start[:-1], src[1:])
            vals = cost - j  # 문서 글자 하나를 건너뜀: new[j] = min_k<=j (cost[k] + j - k)
            run = np.minimum.accumulate(vals)
            arg = np.maximum.accumulate(np.where(vals == run, j, 0))
            prev, start = run + j, src[arg]
        end = int(np.argmin(prev))
        return int(prev[end]), int(start[end]), end

    def snap(self, text):
        """(바꾼 글자, 바꿨는지). 문서에 충분히 가까운 부분이 없으면 그대로 돌려준다."""
        t = _norm(text)
        if not self or len(t) < self.min_chars:
            return text, False
        dist, s, e = self._align(t)
        if e <= s or dist > self.max_ratio * len(t):
            return text, False
        a, b = self.idx[s], self.idx[e - 1] + 1
        while a > 0 and not self.text[a - 1].isspace():  # 어절 경계까지 넓힌다
            a -= 1
        while b < len(self.text) and not self.text[b].isspace():
            b += 1
        out = self.text[a:b].strip()
        o = _norm(out)
        if _edit(t, o) > self.max_ratio * max(len(t), len(o)):  # 넓힌 뒤에도 가까운지 다시 본다
            return text, False
        return out, out != text.strip()
