from ttsvs.text import split_text


def test_empty_and_blank():
    assert split_text("") == []
    assert split_text("  \n\n \n ") == []


def test_short_text_single_chunk():
    assert split_text("Hello world.") == ["Hello world."]


def test_sentences_pack_under_limit():
    chunks = split_text("One. Two! Three? Four.", max_chars=10)
    assert all(len(c) <= 10 for c in chunks)
    assert " ".join(chunks) == "One. Two! Three? Four."


def test_paragraphs_never_merge():
    assert split_text("First para.\n\nSecond para.") == ["First para.", "Second para."]


def test_single_newlines_are_whitespace():
    assert split_text("hello\nworld.") == ["hello world."]


def test_long_sentence_splits_on_commas():
    text = "alpha beta gamma, delta epsilon zeta, eta theta iota"
    chunks = split_text(text, max_chars=25)
    assert all(len(c) <= 25 for c in chunks)
    assert chunks[0].endswith(",")
    assert " ".join(chunks) == text


def test_very_long_sentence_splits_on_words():
    text = " ".join(["word"] * 100)
    chunks = split_text(text, max_chars=30)
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks) == text


def test_giant_token_is_hard_split():
    chunks = split_text("x" * 95, max_chars=40)
    assert [len(c) for c in chunks] == [40, 40, 15]
