"""History of generated clips: audio plus the text and settings that made it."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audio as audio_io
from . import config

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")  # also keeps ids from escaping the root


class HistoryNotFoundError(KeyError):
    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""


@dataclass
class HistoryItem:
    id: str  # sortable, e.g. 20261003-072500-ab12
    created_at: str  # ISO 8601
    text: str
    voice_id: str | None
    voice_name: str | None
    engine: str
    params: dict = field(default_factory=dict)
    audio_path: Path = Path()  # <history dir>/<id>/audio.wav
    duration_s: float = 0.0


def _json_default(value):
    """Let numpy scalars/arrays and paths into item.json."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class History:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else config.history_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- reading ---------------------------------------------------------

    def _read(self, folder: Path) -> HistoryItem | None:
        meta = folder / "item.json"
        wav = folder / "audio.wav"
        if not _ID_RE.match(folder.name) or not meta.is_file() or not wav.is_file():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            params = data.get("params")
            return HistoryItem(
                id=folder.name,
                created_at=str(data.get("created_at", "")),
                text=str(data["text"]),
                voice_id=data.get("voice_id"),
                voice_name=data.get("voice_name"),
                engine=str(data.get("engine", "")),
                params=params if isinstance(params, dict) else {},
                audio_path=wav,
                duration_s=float(data.get("duration_s", 0.0)),
            )
        except (ValueError, KeyError, TypeError, OSError, AttributeError):
            return None

    def list(self, limit: int | None = None) -> list[HistoryItem]:
        """Newest first. Folders that are not complete items are skipped."""
        folders = sorted((f for f in self.root.iterdir() if f.is_dir()), key=lambda f: f.name, reverse=True)
        items: list[HistoryItem] = []
        for folder in folders:
            if limit is not None and len(items) >= limit:
                break
            item = self._read(folder)
            if item is not None:
                items.append(item)
        return items

    def get(self, item_id: str) -> HistoryItem:
        item_id = str(item_id).strip()
        item = self._read(self.root / item_id) if _ID_RE.match(item_id) else None
        if item is None:
            raise HistoryNotFoundError(f"No history item {item_id!r}.")
        return item

    # -- writing ---------------------------------------------------------

    def _new_folder(self) -> tuple[str, Path]:
        """Create a fresh item folder whose name sorts after earlier ones.

        The name is UTC date-time plus four hex digits taken from the
        microseconds, so ids rise in creation order. A clash (same instant or a
        frozen clock) bumps the suffix.
        """
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%d-%H%M%S")
        suffix = now.microsecond // 16  # 0..62499, fits four hex digits
        for _ in range(0x10000):
            item_id = f"{stamp}-{suffix:04x}"
            folder = self.root / item_id
            try:
                folder.mkdir()
            except FileExistsError:
                suffix = (suffix + 1) & 0xFFFF
                continue
            return item_id, folder
        raise OSError("Could not allocate a history id.")  # pragma: no cover

    def add(self, sr: int, audio: np.ndarray, text: str, voice_id: str | None,
            voice_name: str | None, engine: str, params: dict) -> HistoryItem:
        """Save a finished clip (mono float audio at sample rate sr)."""
        data = np.asarray(audio, dtype=np.float32)
        item_id, folder = self._new_folder()
        try:
            wav = audio_io.save_audio(folder / "audio.wav", data, int(sr))
            item = HistoryItem(
                id=item_id,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                text=text,
                voice_id=voice_id,
                voice_name=voice_name,
                engine=engine,
                params=json.loads(json.dumps(dict(params or {}), default=_json_default)),
                audio_path=wav,
                duration_s=round(len(data) / float(sr), 3),
            )
            meta = {k: v for k, v in item.__dict__.items() if k not in ("id", "audio_path")}
            # item.json goes last: a folder without it is an unfinished item and is ignored.
            (folder / "item.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return item

    def delete(self, item_id: str) -> None:
        item = self.get(item_id)
        shutil.rmtree(item.audio_path.parent)

    def clear(self) -> int:
        """Remove every item (and any half-written leftovers); returns how many were removed."""
        removed = 0
        for folder in self.root.iterdir():
            if folder.is_dir() and ((folder / "item.json").exists() or (folder / "audio.wav").exists()):
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        return removed

    def export(self, item_id: str, out_path: Path) -> Path:
        """Write the clip to out_path: .wav (a straight copy), or .mp3 (needs ffmpeg), .flac, .ogg."""
        item = self.get(item_id)
        out_path = Path(out_path)
        if out_path.suffix.lower() in ("", ".wav"):
            if not out_path.suffix:
                out_path = out_path.with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.audio_path, out_path)
            return out_path
        data, sr = audio_io.load_audio(item.audio_path)
        return audio_io.save_audio(out_path, data, sr)
