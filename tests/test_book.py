import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ttsvs import audio
from ttsvs import book as book_mod
from ttsvs.book import BookError, Chapter, load_book, render_book, safe_filename
from ttsvs.synth import Synthesizer
from ttsvs.voices import VoiceLibrary

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

try:
    import pypdf  # noqa: F401
except ImportError:
    HAVE_PYPDF = False
else:
    HAVE_PYPDF = True


def write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


# -- plain text and Markdown ----------------------------------------------------------


def test_txt_chapter_headings(tmp_path):
    path = write(tmp_path, "story.txt", (
        "Chapter 1\n\nIt was a dark night.\nThe wind howled.\n\n"
        "CHAPTER II: The Storm\n\nRain fell.\n\n"
        "part 3\n\nThe end."
    ))
    title, chapters = load_book(path)
    assert title == "story"
    assert [(c.index, c.title) for c in chapters] == [(1, "Chapter 1"), (2, "CHAPTER II: The Storm"), (3, "part 3")]
    assert chapters[0].text == "Chapter 1\n\nIt was a dark night.\nThe wind howled."
    assert chapters[1].text.endswith("Rain fell.") and chapters[2].text.endswith("The end.")


def test_txt_text_before_first_heading_becomes_a_chapter(tmp_path):
    path = write(tmp_path, "novel.txt", "A foreword about this book.\n\nBook 1\n\nStory.")
    _, chapters = load_book(path)
    assert [c.title for c in chapters] == ["novel", "Book 1"]
    assert chapters[0].text == "A foreword about this book."


def test_heading_lookalikes_are_not_headings(tmp_path):
    text = "Partly cloudy today.\nPart mild and nice.\nChapter and verse.\nBook did well.\nChapters 3 and 4 differ.\n"
    _, chapters = load_book(write(tmp_path, "weather.txt", text))
    assert len(chapters) == 1 and chapters[0].title == "weather"
    assert chapters[0].text.count("\n") == 4


def test_empty_chapters_are_dropped_and_reindexed(tmp_path):
    text = ("Contents\nChapter 1\nChapter 2\nChapter 3\n\n"  # table of contents: headings with no body
            "Chapter 4\n\nReal text.\n\nChapter 5\n\n   \n\nChapter 6\n\nMore text.")
    _, chapters = load_book(write(tmp_path, "toc.txt", text))
    assert [(c.index, c.title) for c in chapters] == [(1, "toc"), (2, "Chapter 4"), (3, "Chapter 6")]


def test_markdown_headings(tmp_path):
    path = write(tmp_path, "notes.md", (
        "# My Book\n\n## Opening\n\nSome **bold** words and a [link](http://x.y).\n\n"
        "### Minor heading\n\n- item one\n\n```\ncode = 1\n```\n\n"
        "## Second [Part](u)\n\nAnd more.\n\n---\n"
    ))
    title, chapters = load_book(path)
    assert title == "notes"
    assert [c.title for c in chapters] == ["Opening", "Second Part"]  # lone "# My Book" has no body
    first = chapters[0].text
    assert "bold" in first and "**" not in first and "link" in first and "http" not in first
    assert "Minor heading" in first and "###" not in first and "item one" in first
    assert "code = 1" not in first
    assert "---" not in chapters[1].text


def test_hash_without_space_is_not_a_heading(tmp_path):
    _, chapters = load_book(write(tmp_path, "tags.txt", "#hashtag is trending\n\nand more"))
    assert len(chapters) == 1


def test_no_headings_single_chapter_named_after_file(tmp_path):
    _, chapters = load_book(write(tmp_path, "My Essay.txt", "Just one\r\nparagraph.\r\n\r\n\r\n\r\nAnother."))
    assert len(chapters) == 1 and chapters[0].title == "My Essay" and chapters[0].index == 1
    assert chapters[0].text == "Just one\nparagraph.\n\nAnother."


def test_long_text_without_headings_is_split_into_parts(tmp_path):
    paragraph = "This is a sentence of moderate length, repeated. " * 8  # ~390 chars
    paragraphs = [f"{paragraph.strip()} #{i}" for i in range(80)]  # ~31k chars
    _, chapters = load_book(write(tmp_path, "long.txt", "\n\n".join(paragraphs)))
    assert len(chapters) >= 2
    assert [c.title for c in chapters] == [f"Part {i}" for i in range(1, len(chapters) + 1)]
    assert [c.index for c in chapters] == list(range(1, len(chapters) + 1))
    assert all(len(c.text) <= 16_500 for c in chapters)
    assert all(len(c.text) >= 5_000 for c in chapters[:-1])
    joined = "\n\n".join(c.text for c in chapters)
    assert joined == "\n\n".join(paragraphs)  # cut only at paragraph boundaries, nothing lost


def test_long_text_just_under_the_limit_stays_whole(tmp_path):
    _, chapters = load_book(write(tmp_path, "edge.txt", ("word " * 3900).strip()))  # 19.5k chars
    assert len(chapters) == 1 and chapters[0].title == "edge"


def test_long_wall_of_text_is_cut_at_sentences(tmp_path):
    wall = " ".join(f"Sentence number {i} goes here." for i in range(1500))  # no blank lines at all
    assert len(wall) > 40_000
    _, chapters = load_book(write(tmp_path, "wall.txt", wall))
    assert len(chapters) >= 3 and all(len(c.text) <= 16_500 for c in chapters)
    assert all(c.text.rstrip().endswith(".") for c in chapters)
    assert " ".join(c.text for c in chapters) == wall


def test_single_long_headed_chapter_is_split_with_its_title(tmp_path):
    body = "\n\n".join(f"Paragraph {i}. " + "lorem ipsum dolor " * 40 for i in range(60))
    _, chapters = load_book(write(tmp_path, "one.md", f"# The Whole Thing\n\n{body}"))
    assert len(chapters) >= 2 and chapters[0].title == "The Whole Thing - Part 1"


def test_cp1252_text_file(tmp_path):
    _, chapters = load_book(write(tmp_path, "old.txt", "Café “quoted”", encoding="cp1252"))
    assert chapters[0].text == "Café “quoted”"


def test_utf8_bom_is_removed(tmp_path):
    _, chapters = load_book(write(tmp_path, "bom.txt", "﻿Chapter 1\n\nHello."))
    assert chapters[0].title == "Chapter 1"


def test_load_book_errors(tmp_path):
    with pytest.raises(BookError, match="not found"):
        load_book(tmp_path / "missing.txt")
    with pytest.raises(BookError, match="Unsupported"):
        load_book(write(tmp_path, "book.docx", "x"))
    with pytest.raises(BookError, match="No readable text"):
        load_book(write(tmp_path, "blank.txt", "  \n\n \n"))


# -- EPUB -----------------------------------------------------------------------------

XHTML = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{title}</title>
<style>p {{ color: red; }}</style><script>var hidden = 1;</script></head>
<body>{body}</body></html>"""

CH1 = ("<h1>The Beginning</h1><p>It was the best of times, it was the worst of times, it was the age "
       "of wisdom, it was the age of foolishness, it was the epoch of belief.</p>"
       "<p>Second <em>paragraph</em> with an entity &amp; a caf&eacute;.<br/>Line break here, followed by "
       "enough extra words to keep this chapter comfortably above the minimum length.</p>"
       "<script>alert('no')</script>")
CH2 = ("<h2>Middle Chapter</h2><p>" + "Another long stretch of narrative text. " * 8 + "</p>"
       "<div>A div paragraph.</div>")
COVER = "<p>Cover</p>"


def make_epub(path, spine_order=("ch2", "cover", "ch1"), extra=None, title="Two Chapter Tale"):
    files = {
        "cover.xhtml": XHTML.format(title="Cover", body=COVER),
        "text/ch1.xhtml": XHTML.format(title="Book Title", body=CH1),
        "text/ch2.xhtml": XHTML.format(title="Book Title", body=CH2),
    }
    files.update(extra or {})
    manifest = "".join(
        f'<item id="{i}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, href in {"cover": "cover.xhtml", "ch1": "text/ch1.xhtml", "ch2": "text/ch2.xhtml"}.items()
    )
    manifest += '<item id="css" href="style.css" media-type="text/css"/>'
    spine = "".join(f'<itemref idref="{i}"/>' for i in spine_order)
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title></metadata>"
        f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )
    container = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/style.css", "p {}")
        for name, content in files.items():
            zf.writestr(f"OEBPS/{name}", content)
    return path


def test_epub_follows_spine_and_skips_short_items(tmp_path):
    title, chapters = load_book(make_epub(tmp_path / "tale.epub"))
    assert title == "Two Chapter Tale"
    # spine order is ch2, cover, ch1; the cover page (under 200 chars) is skipped
    assert [(c.index, c.title) for c in chapters] == [(1, "Middle Chapter"), (2, "The Beginning")]
    assert chapters[0].text.startswith("Middle Chapter\n\nAnother long stretch")
    assert chapters[0].text.endswith("A div paragraph.")


def test_epub_text_extraction_details(tmp_path):
    _, chapters = load_book(make_epub(tmp_path / "tale.epub"))
    text = chapters[1].text
    assert "hidden" not in text and "alert" not in text and "color: red" not in text
    assert "Book Title" not in text  # <title> lives in <head>
    assert "Second paragraph with an entity & a café.\nLine break here" in text
    assert text.split("\n\n")[0] == "The Beginning" and len(text.split("\n\n")) == 3  # paragraph breaks kept
    assert "  " not in text


def test_epub_title_falls_back_to_html_title_then_numbered(tmp_path):
    long_text = "<p>" + "Plain words without any heading. " * 10 + "</p>"
    extra = {
        "text/ch1.xhtml": XHTML.format(title="From Title Tag", body=long_text),
        "text/ch2.xhtml": XHTML.format(title="", body=long_text),
    }
    _, chapters = load_book(make_epub(tmp_path / "t.epub", spine_order=("ch1", "ch2"), extra=extra))
    assert [c.title for c in chapters] == ["From Title Tag", "Chapter 2"]


def test_epub_keeps_short_items_when_everything_is_short(tmp_path):
    short = {
        "text/ch1.xhtml": XHTML.format(title="x", body="<h1>One</h1><p>Tiny.</p>"),
        "text/ch2.xhtml": XHTML.format(title="x", body="<h1>Two</h1><p>Tiny too.</p>"),
    }
    _, chapters = load_book(make_epub(tmp_path / "s.epub", spine_order=("ch1", "ch2", "cover"), extra=short))
    assert [c.title for c in chapters] == ["One", "Two", "Cover"]


def test_epub_without_dc_title_uses_file_name(tmp_path):
    path = make_epub(tmp_path / "plain-name.epub")
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    members["OEBPS/content.opf"] = members["OEBPS/content.opf"].replace(b"dc:title", b"dc:subject")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    title, _ = load_book(path)
    assert title == "plain-name"


def test_epub_percent_encoded_hrefs_and_missing_files(tmp_path):
    path = make_epub(tmp_path / "t.epub", extra={"text/ch 1.xhtml": XHTML.format(title="t", body=CH1)})
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    opf = members["OEBPS/content.opf"].decode()
    opf = opf.replace("text/ch1.xhtml", "text/ch%201.xhtml")
    opf = opf.replace("</spine>", '<itemref idref="ghost"/></spine>')  # unknown idref is ignored
    members["OEBPS/content.opf"] = opf.encode()
    del members["OEBPS/text/ch1.xhtml"]
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    _, chapters = load_book(path)
    assert [c.title for c in chapters] == ["Middle Chapter", "The Beginning"]


def test_epub_errors(tmp_path):
    bad = tmp_path / "bad.epub"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(BookError, match="not a valid EPUB"):
        load_book(bad)
    empty = tmp_path / "empty.epub"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
    with pytest.raises(BookError, match="package"):
        load_book(empty)
    entity = tmp_path / "entity.epub"
    with zipfile.ZipFile(entity, "w") as zf:
        zf.writestr("META-INF/container.xml", '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><container/>')
    with pytest.raises(BookError, match="entities"):
        load_book(entity)


# -- PDF ------------------------------------------------------------------------------


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """A minimal valid PDF with one text line per entry (no extra dependencies)."""
    objects: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
        stream = "BT /F1 12 Tf 72 720 Td 16 TL " + " T* ".join(f"({line}) Tj" for line in escaped) + " ET"
        content_no = len(objects) + 2
        page_no = len(objects) + 1
        kids.append(f"{page_no} 0 R")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_no} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode())
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.mark.skipif(not HAVE_PYPDF, reason="pypdf not installed")
def test_pdf_pages_become_text_and_chapters(tmp_path):
    path = make_pdf(tmp_path / "paper.pdf", [
        ["Chapter 1", "The first page says hello", "and keeps going (with parens)."],
        ["on the second page.", "Chapter 2", "Final words."],
    ])
    title, chapters = load_book(path)
    assert title == "paper"
    assert [c.title for c in chapters] == ["Chapter 1", "Chapter 2"]
    assert "and keeps going (with parens).\non the second page." in chapters[0].text
    assert chapters[1].text.endswith("Final words.")


@pytest.mark.skipif(not HAVE_PYPDF, reason="pypdf not installed")
def test_pdf_without_text_or_broken(tmp_path):
    with pytest.raises(BookError, match="No text found"):
        load_book(make_pdf(tmp_path / "blank.pdf", [[""]]))
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 definitely not a pdf")
    with pytest.raises(BookError):
        load_book(broken)


def test_pdf_without_pypdf_explains_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # makes `import pypdf` raise ImportError
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(BookError, match="pip install pypdf"):
        load_book(pdf)


# -- file names -----------------------------------------------------------------------


def test_safe_filename():
    assert safe_filename('Chapter 1: "The/Start" <of> a|b?*\\') == "Chapter 1 TheStart of ab"
    assert safe_filename("  spaced    out  ") == "spaced out"
    assert safe_filename("trailing dots...") == "trailing dots"
    assert safe_filename("???", "fallback") == "fallback"
    long = safe_filename("word " * 40)
    assert len(long) <= 60 and not long.endswith(" ")
    assert safe_filename("Ünïcode – ok") == "Ünïcode – ok"


# -- rendering ------------------------------------------------------------------------


class FakeSynth:
    """Stands in for a Synthesizer; records every speak() call."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def speak(self, text, voice_id=None, progress=None, **params):
        self.calls.append((text, voice_id, params))
        if self.fail_on is not None and text == self.fail_on:
            raise RuntimeError("engine crashed")
        t = np.arange(2400) / 24000
        return 24000, (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


CHAPTERS = [Chapter(1, "Intro", "Intro text."), Chapter(2, "Why: a/b?", "Second text."), Chapter(3, "End", "Last.")]


def test_render_book_writes_one_file_per_chapter(tmp_path):
    synth = FakeSynth()
    out = render_book(synth, CHAPTERS, tmp_path / "out")
    assert [p.name for p in out] == ["01 - Intro.wav", "02 - Why ab.wav", "03 - End.wav"]
    assert all(p.is_file() and p.parent == tmp_path / "out" for p in out)
    assert sf.info(str(out[0])).samplerate == 24000
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [p.name for p in out]  # no .part left
    assert [c[0] for c in synth.calls] == ["Intro text.", "Second text.", "Last."]


def test_render_book_passes_voice_and_params_through(tmp_path):
    synth = FakeSynth()
    render_book(synth, CHAPTERS[:1], tmp_path, voice_id="anna", exaggeration=0.7, speed=1.3)
    assert synth.calls == [("Intro text.", "anna", {"exaggeration": 0.7, "speed": 1.3})]


def test_render_book_resume_skips_finished_files(tmp_path):
    first = FakeSynth(fail_on="Last.")
    with pytest.raises(RuntimeError, match="crashed"):
        render_book(first, CHAPTERS, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["01 - Intro.wav", "02 - Why ab.wav"]  # no .part leftovers

    second = FakeSynth()
    out = render_book(second, CHAPTERS, tmp_path)
    assert [c[0] for c in second.calls] == ["Last."]  # only the missing chapter was rendered
    assert [p.name for p in out] == ["01 - Intro.wav", "02 - Why ab.wav", "03 - End.wav"]

    third = FakeSynth()
    render_book(third, CHAPTERS, tmp_path, resume=False)
    assert len(third.calls) == 3


def test_stale_partial_file_is_not_mistaken_for_a_finished_one(tmp_path):
    (tmp_path / "01 - Intro.part.wav").write_bytes(b"half written")
    synth = FakeSynth()
    out = render_book(synth, CHAPTERS[:1], tmp_path)
    assert len(synth.calls) == 1
    assert sf.info(str(out[0])).frames == 2400
    assert not (tmp_path / "01 - Intro.part.wav").exists()


def test_render_book_keeps_chapter_numbers_for_a_subset(tmp_path):
    out = render_book(FakeSynth(), CHAPTERS[1:], tmp_path)
    assert [p.name for p in out] == ["02 - Why ab.wav", "03 - End.wav"]


def test_render_book_progress(tmp_path):
    seen = []
    render_book(FakeSynth(), CHAPTERS, tmp_path, progress=lambda done, total, title: seen.append((done, total, title)))
    assert seen == [(0, 3, "Intro"), (1, 3, "Why: a/b?"), (2, 3, "End"), (3, 3, "End")]
    seen.clear()
    render_book(FakeSynth(), CHAPTERS, tmp_path, progress=lambda *a: seen.append(a))  # all skipped by resume
    assert [s[0] for s in seen] == [0, 1, 2, 3]
    assert render_book(FakeSynth(), [], tmp_path / "nothing") == []


def test_render_book_creates_out_dir_and_rejects_bad_format(tmp_path):
    synth = FakeSynth()
    with pytest.raises(BookError, match="format"):
        render_book(synth, CHAPTERS, tmp_path, fmt="aac")
    assert synth.calls == []
    nested = tmp_path / "a" / "b"
    render_book(synth, CHAPTERS[:1], nested, fmt=".WAV")
    assert (nested / "01 - Intro.wav").is_file()


def test_render_book_wide_numbers_for_big_books(tmp_path):
    chapters = [Chapter(100, "Big", "Text.")]
    assert render_book(FakeSynth(), chapters, tmp_path)[0].name == "100 - Big.wav"


@needs_ffmpeg
def test_render_book_mp3(tmp_path):
    out = render_book(FakeSynth(), CHAPTERS[:2], tmp_path, fmt="mp3")
    assert [p.name for p in out] == ["01 - Intro.mp3", "02 - Why ab.mp3"]
    assert all(p.stat().st_size > 200 for p in out)
    assert not list(tmp_path.glob("*.part.*"))


def test_render_book_mp3_without_ffmpeg_fails_before_synthesizing(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)
    synth = FakeSynth()
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        render_book(synth, CHAPTERS, tmp_path, fmt="mp3")
    assert synth.calls == [] and list(tmp_path.iterdir()) == []


def test_load_and_render_end_to_end_with_dummy_engine(tmp_path, sample_wav):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    path = write(tmp_path, "tale.txt", "Chapter 1\n\nOnce upon a time.\n\nChapter 2\n\nThe end. Really.")
    title, chapters = load_book(path)
    synth = Synthesizer("dummy", library=lib)
    out = render_book(synth, chapters, tmp_path / "audio", voice_id="anna", speed=1.0)
    assert [p.name for p in out] == ["01 - Chapter 1.wav", "02 - Chapter 2.wav"]
    assert all(sf.info(str(p)).duration > 0.5 for p in out)
    assert book_mod.chapter_filename(chapters[0]) == "01 - Chapter 1.wav"
