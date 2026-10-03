import json

import pytest

from ttsvs import config
from ttsvs.lexicon import Lexicon


def make(tmp_path, **entries):
    lex = Lexicon(tmp_path / "lex.json")
    for word, say in entries.items():
        lex.set(word, say)
    return lex


def test_default_path_and_missing_file_is_empty(ttsvs_home):
    lex = Lexicon()
    assert lex.path == config.lexicon_path()
    assert lex.entries() == {}
    assert lex.apply("Nothing changes.") == "Nothing changes."


def test_set_persists_sorted_and_remove(tmp_path):
    path = tmp_path / "sub" / "lex.json"
    lex = Lexicon(path)
    lex.set("zebra", "ZEE bra")
    lex.set("Apple", "AP ul")
    assert list(Lexicon(path).entries()) == ["Apple", "zebra"]  # another instance, sorted by word
    assert json.loads(path.read_text(encoding="utf-8")) == {"Apple": "AP ul", "zebra": "ZEE bra"}
    lex.remove("zebra")
    assert Lexicon(path).entries() == {"Apple": "AP ul"}


def test_remove_missing_raises_keyerror(tmp_path):
    lex = make(tmp_path, a="b")
    with pytest.raises(KeyError):
        lex.remove("zzz")
    with pytest.raises(KeyError):
        Lexicon(tmp_path / "none.json").remove("a")


def test_set_replaces_and_is_case_insensitive_on_key(tmp_path):
    lex = make(tmp_path, TTSVS="T T S V S")
    lex.set("ttsvs", "tee tee ess")
    assert lex.entries() == {"ttsvs": "tee tee ess"}
    lex.remove("TtSvS")
    assert lex.entries() == {}


def test_set_rejects_empty_word(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path).set("   ", "x")


def test_apply_whole_words_only_case_insensitive(tmp_path):
    lex = make(tmp_path, cat="kitty", Nguyen="Win")
    assert lex.apply("The Cat sat. CAT! concatenate cats scatter") == "The kitty sat. kitty! concatenate cats scatter"
    assert lex.apply("Dr. nguyen's") == "Dr. Win's"
    assert lex.apply("cat-like") == "kitty-like"  # a hyphen is not a word character


def test_apply_longest_key_first(tmp_path):
    lex = make(tmp_path, **{"New York": "Noo York", "New": "Nu", "New York City": "the Big Apple"})
    assert lex.apply("New York City, New York and New Jersey") == "the Big Apple, Noo York and Nu Jersey"


def test_apply_is_single_pass(tmp_path):
    lex = make(tmp_path, a="b", b="c")
    assert lex.apply("a b") == "b c"  # not "c c"


def test_apply_keys_with_symbols_at_edges(tmp_path):
    lex = make(tmp_path, **{"C++": "see plus plus", "(tm)": "trademark", ".NET": "dot net", "a.b": "a dot b"})
    assert lex.apply("I like C++ and C++.") == "I like see plus plus and see plus plus."
    assert lex.apply("C++x stays, abc++ stays") == "C++x stays, abc++ stays"
    assert lex.apply("Widget (tm) here") == "Widget trademark here"
    assert lex.apply("Widget(tm)") == "Widget(tm)"  # touching a word character: not a whole word
    assert lex.apply("Use .NET today") == "Use dot net today"
    assert lex.apply("a.b and aXb") == "a dot b and aXb"  # regex metacharacters are escaped


def test_apply_replacement_is_literal(tmp_path):
    lex = make(tmp_path, x=r"\1 \g<0> $1")
    assert lex.apply("x") == r"\1 \g<0> $1"


def test_apply_whitespace_in_key_is_flexible(tmp_path):
    lex = make(tmp_path, **{"New  York": "NY"})
    assert lex.entries() == {"New York": "NY"}
    assert lex.apply("in new\nyork city") == "in NY city"


def test_apply_unicode_words(tmp_path):
    lex = make(tmp_path, **{"café": "kaf-ay", "Zoë": "Zo-ee"})
    assert lex.apply("Un CAFÉ, Zoë; cafés") == "Un kaf-ay, Zo-ee; cafés"


def test_empty_replacement_deletes_the_word(tmp_path):
    lex = make(tmp_path, um="")
    assert lex.apply("so um yes") == "so  yes"


def test_corrupt_or_odd_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "lex.json"
    path.write_text("{not json", encoding="utf-8")
    assert Lexicon(path).entries() == {}
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert Lexicon(path).entries() == {}
    path.write_text(json.dumps({"ok": "fine", "bad": 3, "": "x"}), encoding="utf-8")
    assert Lexicon(path).entries() == {"ok": "fine"}
    lex = Lexicon(path)
    lex.set("new", "n")  # recovers by rewriting a clean file
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": "n", "ok": "fine"}


def test_edits_from_another_instance_are_seen(tmp_path):
    a, b = Lexicon(tmp_path / "lex.json"), Lexicon(tmp_path / "lex.json")
    a.set("foo", "bar")
    assert b.apply("foo") == "bar"
    b.remove("foo")
    assert a.apply("foo") == "foo"
