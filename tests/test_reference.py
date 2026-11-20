"""읽기 문서 맞추기(serve/reference.py) 단위 테스트. 11-20 실제 오인식 예를 쓴다."""

from serve.reference import Reference, hint_words

DOC = [
    "코끼리는 귀가 아주 큽니다.",
    "아빠가 토끼 풍선을 사왔습니다.",
    "무엇을 먹고 싶니?",
    "책상이 너무 높아요.",
    "바지를 입고 밖에 나갑니다.",
    "엄마가 잘 다녀와 하면서 손을 흔듭니다.",
    "원숭이가 꼬리를 길게 늘어뜨리고 있습니다.",
]


def test_snaps_real_misrecognitions():
    ref = Reference(DOC)
    cases = {
        "코끼리는 귀가 아직 큽니다": "코끼리는 귀가 아주 큽니다.",
        "식상이 너무 높아요": "책상이 너무 높아요.",
        "말이 있고 밖에 나갑니다": "바지를 입고 밖에 나갑니다.",
        "원숭이가 꼬리를 길게 밀어뜨리고 있습니다": "원숭이가 꼬리를 길게 늘어뜨리고 있습니다.",
    }
    for heard, want in cases.items():
        got, changed = ref.snap(heard)
        assert got == want and changed, (heard, got)


def test_fragments_snap_to_their_part_only():
    """VAD가 문장을 나눠도 조각은 문서의 그 부분으로만 바뀐다 (문장 전체로 늘리지 않는다)."""
    ref = Reference(DOC)
    assert ref.snap("엄마가 잘 다녀와") == ("엄마가 잘 다녀와", False)
    assert ref.snap("하면서 손을 흔듭니다")[0] == "하면서 손을 흔듭니다."
    assert ref.snap("코끼리는 귀가")[0] == "코끼리는 귀가"
    assert ref.snap("귀가 아직 큽니다")[0] == "귀가 아주 큽니다."


def test_short_fragments_are_not_snapped():
    """7글자 미만은 바꾸지 않는다: 문서 밖 말의 짧은 조각이 문서로 잘못 바뀌는 것을 막는다 (실측 47문장 중 3 → 0)."""
    ref = Reference(DOC)
    assert ref.snap("아직 큽니다") == ("아직 큽니다", False)
    assert ref.snap("아빠가요") == ("아빠가요", False)
    assert Reference(DOC, min_chars=4).snap("아직 큽니다")[0] == "아주 큽니다."


def test_partial_word_expands_to_whole_word():
    ref = Reference(DOC)
    assert ref.snap("녕하세요 무엇을 먹고")[0] in ("녕하세요 무엇을 먹고",)  # 문서에 없는 앞부분이 길면 그대로
    assert ref.snap("끼리는 귀가 아주")[0] == "코끼리는 귀가 아주"  # 어절 앞이 빠져도 어절 전체로


def test_out_of_document_text_is_kept():
    ref = Reference(DOC)
    for heard in ("물 한 잔 가져다 주세요", "화장실에 가고 싶어요", "오늘은 배가 조금 아파요", "의사 선생님을 불러 주세요"):
        assert ref.snap(heard) == (heard, False), heard


def test_short_or_empty_text_is_kept():
    ref = Reference(DOC)
    assert ref.snap("네") == ("네", False)
    assert ref.snap("") == ("", False)
    assert Reference([]).snap("코끼리는 귀가 아주 큽니다") == ("코끼리는 귀가 아주 큽니다", False)


def test_hint_words():
    w = hint_words(DOC)
    assert w[:4] == ["코끼리는", "귀가", "아주", "큽니다"]
    assert "책상이" in w and "늘어뜨리고" in w and len(w) == len(set(w))
    assert all(len(x) >= 2 for x in w)
    assert len(hint_words(DOC, limit=3)) == 3
