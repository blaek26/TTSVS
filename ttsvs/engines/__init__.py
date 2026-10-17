"""Engine registry. Engines are constructed lazily; heavy imports stay inside them."""

from __future__ import annotations

from collections.abc import Callable

from .base import EngineError, TTSEngine


def _chatterbox(device: str) -> TTSEngine:
    from .chatterbox import ChatterboxEngine

    return ChatterboxEngine(device)


def _dummy(device: str) -> TTSEngine:
    from .dummy import DummyEngine

    return DummyEngine(device)


_REGISTRY: dict[str, Callable[[str], TTSEngine]] = {
    "chatterbox": _chatterbox,
    "dummy": _dummy,
}


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def get_engine(name: str, device: str = "auto") -> TTSEngine:
    try:
        factory = _REGISTRY[name.lower()]
    except KeyError:
        raise EngineError(f"Unknown engine {name!r}; choose from: {', '.join(available_engines())}.") from None
    return factory(device)


__all__ = ["EngineError", "TTSEngine", "available_engines", "get_engine"]
