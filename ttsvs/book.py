"""Audiobooks: load a .txt/.md/.epub/.pdf, split it into chapters, render one file per chapter.

Only the standard library is needed for text, Markdown and EPUB. PDF support
uses pypdf (``pip install "ttsvs[book]"``).
"""

from __future__ import annotations

import os
import posixpath
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote
from xml.etree import ElementTree

from . import audio

if TYPE_CHECKING:
    from .synth import Synthesizer

# Chapter splitting knobs.
LONG_TEXT_CHARS = 20_000  # a single "chapter" longer than this is cut into parts
PART_TARGET_CHARS = 15_000  # ... of about this size, at paragraph boundaries
MIN_EPUB_CHARS = 200  # shorter spine items (covers, tables of contents) are skipped
MAX_TITLE_FILE_CHARS = 60

_FORMATS = {"wav", "mp3", "flac", "ogg"}


class BookError(Exception):
    """Raised when a book cannot be read or rendered."""


@dataclass
class Chapter:
    index: int  # 1-based
    title: str
    text: str


# -- text helpers ---------------------------------------------------------------


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")  # pragma: no cover (cp1252 gaps only)


def _normalize(text: str) -> str:
    """Unix newlines, no trailing spaces, at most one blank line between paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _one_line(text: str) -> str:
    return " ".join(text.split())


_MD_HEADING = re.compile(r"^(#{1,2})\s+(.*?)\s*#*\s*$")  # '#' or '##', not '###'
_CHAPTER_LINE = re.compile(r"^(chapter|part|book)\s+([0-9]+|[ivxlcdm]+)\b", re.IGNORECASE)
_ROMAN = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)
_MD_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_STRIP = re.compile(r"(\*{1,3}|`)")
_MD_RULE = re.compile(r"^\s*([-_*=]\s*){3,}$")
_MD_PREFIX = re.compile(r"^\s*(?:#{3,6}\s+|>\s?|[-*+]\s+)")


def _clean_markdown(line: str) -> str:
    """Light cleanup so Markdown syntax is not read out loud."""
    if _MD_RULE.match(line):
        return ""
    line = _MD_IMAGE.sub(r"\1", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _MD_PREFIX.sub("", line)
    return _MD_STRIP.sub("", line)


def _heading_title(line: str) -> str | None:
    """The chapter title if this line starts a chapter, else None."""
    stripped = line.strip()
    md = _MD_HEADING.match(stripped)
    if md:
        return _one_line(md.group(2)) or None
    m = _CHAPTER_LINE.match(stripped)
    if m and len(stripped) <= 100:
        numeral = m.group(2)
        if numeral.isdigit() or _ROMAN.match(numeral):
            return _one_line(stripped)
    return None


def _pack_paragraphs(paragraphs: list[str], target: int) -> list[str]:
    """Greedily group paragraphs into parts of at most `target` chars."""
    parts: list[str] = []
    cur = ""
    for para in paragraphs:
        if cur and len(cur) + 2 + len(para) > target:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    # Don't leave a sliver of a final part: fold it into the one before.
    if len(parts) > 1 and len(parts[-1]) < target // 8:
        parts[-2] = f"{parts[-2]}\n\n{parts.pop()}"
    return parts


def _break_long_paragraph(paragraph: str, target: int) -> list[str]:
    """A wall of text with no blank lines: cut at sentence ends, then at spaces."""
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
        while len(sentence) > target:
            cut = sentence.rfind(" ", 0, target)
            cut = cut if cut > 0 else target
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    # Re-join the pieces with spaces into parts no longer than target.
    parts: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > target:
            parts.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}" if cur else piece
    if cur:
        parts.append(cur)
    return parts


def _split_into_parts(text: str) -> list[str]:
    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        paragraphs.extend(_break_long_paragraph(para, PART_TARGET_CHARS) if len(para) > PART_TARGET_CHARS else [para])
    return _pack_paragraphs(paragraphs, PART_TARGET_CHARS)


def _chapters_from_text(text: str, stem: str, markdown: bool = False) -> list[Chapter]:
    """Split plain text / Markdown on headings; fall back to one chapter (or Parts)."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    in_fence = False
    for line in _normalize(text).split("\n"):
        if markdown and _MD_FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        title = _heading_title(line)
        if title is not None:
            sections.append((_clean_markdown(title) if markdown else title, []))
        else:
            sections[-1][1].append(_clean_markdown(line) if markdown else line)

    chapters: list[Chapter] = []
    for title, lines in sections:
        body = _normalize("\n".join(lines))
        if not body:
            continue  # empty section: a table-of-contents entry or a lone book title
        if title is None:
            chapters.append(Chapter(len(chapters) + 1, stem, body))
        else:
            chapters.append(Chapter(len(chapters) + 1, title, f"{title}\n\n{body}"))

    if len(chapters) == 1 and len(chapters[0].text) > LONG_TEXT_CHARS:
        only = chapters[0]
        has_heading = only.title != stem
        chapters = [
            Chapter(n, f"{only.title} - Part {n}" if has_heading else f"Part {n}", part)
            for n, part in enumerate(_split_into_parts(only.text), start=1)
        ]
    return chapters


# -- EPUB -----------------------------------------------------------------------

_BLOCK_TAGS = {
    "p", "div", "section", "article", "aside", "blockquote", "li", "ul", "ol", "tr", "table",
    "pre", "header", "footer", "figure", "figcaption", "dl", "dt", "dd", "hr", "nav", "main",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}


class _TextExtractor(HTMLParser):
    """HTML to text with paragraph breaks kept as blank lines; also notes titles/headings."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.doc_title_parts: list[str] = []
        self.headings: dict[int, str] = {}  # level -> first heading text
        self._skip = 0
        self._in_head = False
        self._in_title = False
        self._heading_level: int | None = None
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "head":
            self._in_head = True
        elif tag == "body":
            self._in_head = False
        elif tag == "title":
            self._in_title = True
        if self._in_head:
            return
        if tag == "br":
            self.parts.append("\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        if tag in ("h1", "h2", "h3"):
            self._heading_level = int(tag[1])
            self._heading_parts = []

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(self._skip - 1, 0)
            return
        if tag == "title":
            self._in_title = False
        if self._skip:
            return
        if tag == "head":
            self._in_head = False
            return
        if self._in_head:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        if self._heading_level is not None and tag == f"h{self._heading_level}":
            text = _one_line("".join(self._heading_parts))
            if text:
                self.headings.setdefault(self._heading_level, text)
            self._heading_level = None

    def handle_data(self, data):
        if self._in_title:
            self.doc_title_parts.append(data)
            return
        if self._skip or self._in_head:
            return
        data = re.sub(r"\s+", " ", data)
        self.parts.append(data)
        if self._heading_level is not None:
            self._heading_parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        return _normalize("\n".join(line.strip() for line in raw.split("\n")))

    def title(self) -> str | None:
        for level in (1, 2, 3):
            if level in self.headings:
                return self.headings[level]
        return _one_line("".join(self.doc_title_parts)) or None


def _parse_xml(data: bytes, what: str) -> ElementTree.Element:
    if b"<!ENTITY" in data:  # no entity tricks (billion laughs) in a book
        raise BookError(f"The EPUB's {what} uses XML entities, which are not supported.")
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise BookError(f"The EPUB's {what} is not valid XML: {exc}") from exc


def _epub_spine(zf: zipfile.ZipFile) -> tuple[str | None, list[str]]:
    """(book title, member paths of the reading-order HTML files)."""
    names = zf.namelist()
    opf_path = None
    if "META-INF/container.xml" in names:
        container = _parse_xml(zf.read("META-INF/container.xml"), "container.xml")
        rootfile = container.find(".//{*}rootfile")
        if rootfile is not None:
            opf_path = unquote(rootfile.get("full-path", "")) or None
    if opf_path is None or opf_path not in names:
        opf_path = next((n for n in names if n.lower().endswith(".opf")), None)
    if opf_path is None:
        raise BookError("This EPUB has no package (.opf) file.")

    opf = _parse_xml(zf.read(opf_path), "package file")
    base = posixpath.dirname(opf_path)
    title_el = opf.find(".//{*}metadata/{*}title")
    book_title = _one_line(title_el.text) if title_el is not None and title_el.text else None

    manifest: dict[str, tuple[str, str]] = {}
    for item in opf.iterfind(".//{*}manifest/{*}item"):
        if item.get("id") and item.get("href"):
            manifest[item.get("id")] = (item.get("href"), item.get("media-type", ""))

    def is_html(href: str, media: str) -> bool:
        return media in ("application/xhtml+xml", "text/html") or href.lower().endswith((".xhtml", ".html", ".htm"))

    paths: list[str] = []
    for ref in opf.iterfind(".//{*}spine/{*}itemref"):
        entry = manifest.get(ref.get("idref", ""))
        if entry and is_html(*entry):
            href = unquote(entry[0].split("#")[0])
            paths.append(posixpath.normpath(posixpath.join(base, href)))
    if not paths:  # no usable spine: fall back to the manifest order
        paths = [
            posixpath.normpath(posixpath.join(base, unquote(h.split("#")[0])))
            for h, m in manifest.values() if is_html(h, m)
        ]
    return book_title, [p for p in paths if p in names]


def _load_epub(path: Path) -> tuple[str, list[Chapter]]:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BookError(f"{path.name} is not a valid EPUB file: {exc}") from exc
    with zf:
        book_title, members = _epub_spine(zf)
        found: list[tuple[str | None, str]] = []
        for member in members:
            parser = _TextExtractor()
            try:
                parser.feed(_decode(zf.read(member)))
                parser.close()
            except Exception as exc:  # a malformed page should not sink the whole book
                raise BookError(f"Could not read {member} in {path.name}: {exc}") from exc
            text = parser.text()
            if text:
                found.append((parser.title(), text))

    kept = [f for f in found if len(f[1]) >= MIN_EPUB_CHARS] or found  # never drop everything
    chapters = [
        Chapter(n, title or f"Chapter {n}", text) for n, (title, text) in enumerate(kept, start=1)
    ]
    return book_title or path.stem, chapters


# -- PDF ------------------------------------------------------------------------


def _load_pdf(path: Path) -> tuple[str, list[Chapter]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise BookError(
            'Reading PDF files needs the pypdf package. Install it with: pip install pypdf '
            '(or pip install "ttsvs[book]").'
        ) from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise BookError(f"{path.name} is password protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
        meta_title = getattr(getattr(reader, "metadata", None), "title", None)
    except BookError:
        raise
    except Exception as exc:
        raise BookError(f"Could not read {path.name}: {exc}") from exc
    text = "\n".join(pages)
    if not text.strip():
        raise BookError(f"No text found in {path.name}. Scanned PDFs need OCR first.")
    title = _one_line(str(meta_title)) if meta_title else path.stem
    return title, _chapters_from_text(text, path.stem)


# -- public API -----------------------------------------------------------------

_TEXT_EXTS = {".txt", ".text"}
_MD_EXTS = {".md", ".markdown"}


def load_book(path: Path) -> tuple[str, list[Chapter]]:
    """Read a book file and return (book title, chapters). Empty chapters are dropped."""
    path = Path(path)
    if not path.is_file():
        raise BookError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext in _TEXT_EXTS | _MD_EXTS:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise BookError(f"Could not read {path}: {exc}") from exc
        title = path.stem
        chapters = _chapters_from_text(_decode(raw), path.stem, markdown=ext in _MD_EXTS)
    elif ext == ".epub":
        title, chapters = _load_epub(path)
    elif ext == ".pdf":
        title, chapters = _load_pdf(path)
    else:
        raise BookError(f"Unsupported file type {ext or '(none)'!r}; use .txt, .md, .epub or .pdf.")
    if not chapters:
        raise BookError(f"No readable text found in {path.name}.")
    return title, chapters


_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_filename(title: str, fallback: str = "untitled", max_len: int = MAX_TITLE_FILE_CHARS) -> str:
    """A title usable in a file name on any OS: no / \\ : * ? " < > |, trimmed to max_len."""
    name = _one_line(_UNSAFE_CHARS.sub("", title or "")).strip(" .")
    name = name[:max_len].rstrip(" .")
    return name or fallback


def chapter_filename(chapter: Chapter, fmt: str = "wav", width: int = 2) -> str:
    """"<NN> - <safe title>.<fmt>", the name render_book gives a chapter's file."""
    title = safe_filename(chapter.title, f"Chapter {chapter.index}")
    return f"{chapter.index:0{width}d} - {title}.{fmt.lower().lstrip('.')}"


def render_book(synth: Synthesizer, chapters: list[Chapter], out_dir: Path, voice_id: str | None = None,
                fmt: str = "wav", resume: bool = True,
                progress: Callable[[int, int, str], None] | None = None, **params) -> list[Path]:
    """Render one audio file per chapter into out_dir and return their paths in order.

    Each file is written under a ".part" name and renamed when complete, so an
    interrupted run never leaves a half-written file that resume would skip.
    With resume=True, chapters whose file already exists are not re-rendered.
    progress(done_chapters, total_chapters, current_title) is called before each
    chapter and once more at the end. Extra keyword arguments (exaggeration,
    speed, ...) go to synth.speak. The synthesizer is used as given: build it
    with save_history=False to keep chapters out of the history.
    """
    fmt = fmt.lower().lstrip(".")
    if fmt not in _FORMATS:
        raise BookError(f"Unsupported format {fmt!r}; use one of: {', '.join(sorted(_FORMATS))}.")
    if fmt == "mp3" and audio.ffmpeg_path() is None:
        raise audio.AudioError("MP3 output needs ffmpeg. Install it or use the wav format.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(chapters)
    width = max([2] + [len(str(c.index)) for c in chapters])
    outputs: list[Path] = []
    for done, chapter in enumerate(chapters):
        if progress:
            progress(done, total, chapter.title)
        final = out_dir / chapter_filename(chapter, fmt, width)
        if resume and final.is_file():
            outputs.append(final)
            continue
        sr, data = synth.speak(chapter.text, voice_id, **params)
        part = final.with_name(f"{final.stem}.part{final.suffix}")
        try:
            audio.save_audio(part, data, sr)
            os.replace(part, final)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        outputs.append(final)
    if progress and chapters:
        progress(total, total, chapters[-1].title)
    return outputs
