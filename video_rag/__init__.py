"""Video RAG: semantic search + clip retrieval over screen-recording videos."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.2.0"

# Resolved on first attribute access (PEP 562) so that importing a light
# submodule — `video_rag.retrieval`, where the ranking logic lives — does not
# drag in torch, faiss and OpenCV via this file. That is what lets the test
# suite and CI run in seconds without a single model download.
_LAZY_EXPORTS = {
    "VideoRAGPipeline": ".pipeline",
    "QueryResult": ".pipeline",
    "Located": ".pipeline",
    "Config": ".config",
    "Segment": ".segment",
    "SearchHit": ".index",
    "SemanticIndex": ".index",
    "Database": ".db",
    "Match": ".db",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]

if TYPE_CHECKING:  # pragma: no cover - hints only; never executed at runtime
    from .config import Config as Config
    from .db import Database as Database
    from .db import Match as Match
    from .index import SearchHit as SearchHit
    from .index import SemanticIndex as SemanticIndex
    from .pipeline import Located as Located
    from .pipeline import QueryResult as QueryResult
    from .pipeline import VideoRAGPipeline as VideoRAGPipeline
    from .segment import Segment as Segment


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
