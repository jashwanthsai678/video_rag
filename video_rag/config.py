"""Configuration for the Video RAG pipeline.

All values can be overridden via environment variables (a .env file is
loaded automatically if python-dotenv is installed and a .env exists).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(key: str, default):
    """Read `key` from the environment, coerced to the type of `default`."""
    val = os.environ.get(key)
    if val is None:
        return default
    # bool must precede int: bool is a subclass of int in Python.
    if isinstance(default, bool):
        return val.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


@dataclass
class Config:
    # --- Storage ---
    data_dir: str = field(default_factory=lambda: _env("VIDEORAG_DATA_DIR", "data"))
    videos_dir: str = field(default_factory=lambda: _env("VIDEORAG_VIDEOS_DIR", "videos"))
    frames_dir: str = field(default_factory=lambda: _env("VIDEORAG_FRAMES_DIR", "data/frames"))
    clips_dir: str = field(default_factory=lambda: _env("VIDEORAG_CLIPS_DIR", "data/clips"))
    index_path: str = field(default_factory=lambda: _env("VIDEORAG_INDEX_PATH", "data/index"))

    # --- Frame extraction ---
    fps_extract: float = field(default_factory=lambda: _env("VIDEORAG_FPS_EXTRACT", 1.0))
    # Scene-change gating works on the *fraction of pixels that moved*, not the
    # mean pixel delta. On flat UI screencasts a mean-difference threshold has
    # to be set so low it admits noise, or so high it discards real events like
    # a dialog opening (10% of pixels changing by 150 is only a mean of ~15).
    scene_pixel_delta: int = field(default_factory=lambda: _env("VIDEORAG_SCENE_PIXEL_DELTA", 25))
    scene_change_min_fraction: float = field(
        default_factory=lambda: _env("VIDEORAG_SCENE_MIN_FRACTION", 0.02)
    )
    # Frames are OCR'd in memory and discarded; enable to keep them on disk for
    # debugging. A 1 fps pass over a long recording is a lot of image files.
    keep_frames: bool = field(default_factory=lambda: _env("VIDEORAG_KEEP_FRAMES", False))

    # --- OCR ---
    ocr_enabled: bool = field(default_factory=lambda: _env("VIDEORAG_OCR_ENABLED", True))
    ocr_confidence: float = field(default_factory=lambda: _env("VIDEORAG_OCR_CONFIDENCE", 0.5))
    ocr_similarity_threshold: float = field(
        default_factory=lambda: _env("VIDEORAG_OCR_SIMILARITY", 0.85)
    )
    # A second OCR pass over a binarized copy of the frame. Roughly doubles
    # indexing time (OCR dominates), so it is opt-in.
    ocr_dual_pass: bool = field(default_factory=lambda: _env("VIDEORAG_OCR_DUAL_PASS", False))
    ocr_languages: str = field(default_factory=lambda: _env("VIDEORAG_OCR_LANGUAGES", "en"))
    min_text_length: int = field(default_factory=lambda: _env("VIDEORAG_MIN_TEXT_LEN", 8))

    # --- Speech transcription ---
    transcription_enabled: bool = field(
        default_factory=lambda: _env("VIDEORAG_TRANSCRIPTION_ENABLED", True)
    )
    whisper_model: str = field(default_factory=lambda: _env("VIDEORAG_WHISPER_MODEL", "base"))
    # Empty string => let Whisper auto-detect the language.
    whisper_language: str = field(default_factory=lambda: _env("VIDEORAG_WHISPER_LANGUAGE", "en"))

    # --- Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: _env("VIDEORAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    # BGE models are trained asymmetrically: retrieval queries must carry this
    # instruction prefix, indexed passages must not. Omitting it measurably
    # hurts recall. Set to "" for symmetric models (e.g. all-MiniLM).
    query_instruction: str = field(
        default_factory=lambda: _env(
            "VIDEORAG_QUERY_INSTRUCTION",
            "Represent this sentence for searching relevant passages: ",
        )
    )
    embedding_batch_size: int = field(default_factory=lambda: _env("VIDEORAG_EMBED_BATCH", 32))

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _env("VIDEORAG_TOP_K", 8))
    # Candidates pulled from each retriever before fusion, as a multiple of
    # top_k. Fusion can only promote what at least one retriever surfaced.
    candidate_multiplier: int = field(default_factory=lambda: _env("VIDEORAG_CANDIDATE_MULT", 4))
    vector_weight: float = field(default_factory=lambda: _env("VIDEORAG_VECTOR_WEIGHT", 1.0))
    keyword_weight: float = field(default_factory=lambda: _env("VIDEORAG_KEYWORD_WEIGHT", 0.6))
    rrf_k: int = field(default_factory=lambda: _env("VIDEORAG_RRF_K", 60))
    aggregate_top_n: int = field(default_factory=lambda: _env("VIDEORAG_AGGREGATE_TOP_N", 3))
    # Minimum cosine similarity of the single best hit. BGE similarities sit
    # around 0.5 even for unrelated text, so a low threshold never rejects
    # anything. Calibrate against real negatives — `videorag eval` reports the
    # score distribution for exactly this purpose.
    relevance_threshold: float = field(
        default_factory=lambda: _env("VIDEORAG_RELEVANCE_THRESHOLD", 0.55)
    )

    # --- Clip selection ---
    # Matched moments further apart than this belong to different clips.
    cluster_max_gap_seconds: float = field(
        default_factory=lambda: _env("VIDEORAG_CLUSTER_MAX_GAP", 30.0)
    )
    clip_buffer_seconds: float = field(
        default_factory=lambda: _env("VIDEORAG_CLIP_BUFFER_SECONDS", 4.0)
    )
    # Hard ceiling, so a pathological cluster can never produce a huge export.
    max_clip_seconds: float = field(
        default_factory=lambda: _env("VIDEORAG_MAX_CLIP_SECONDS", 180.0)
    )
    ffmpeg_binary: str = field(default_factory=lambda: _env("VIDEORAG_FFMPEG", "ffmpeg"))

    # --- LLM (OpenAI-compatible; works with OpenAI or OpenRouter) ---
    llm_enabled: bool = field(default_factory=lambda: _env("VIDEORAG_LLM_ENABLED", True))
    llm_api_key: str = field(default_factory=lambda: _env("VIDEORAG_LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: _env("VIDEORAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    llm_model: str = field(
        default_factory=lambda: _env("VIDEORAG_LLM_MODEL", "openai/gpt-4o-mini")
    )
    llm_max_tokens: int = field(default_factory=lambda: _env("VIDEORAG_LLM_MAX_TOKENS", 800))
    llm_timeout_seconds: float = field(default_factory=lambda: _env("VIDEORAG_LLM_TIMEOUT", 60.0))
    llm_max_context_segments: int = field(default_factory=lambda: _env("VIDEORAG_LLM_MAX_SEGS", 40))

    def __post_init__(self):
        # Fall back to common env var names if the generic one wasn't set.
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                "OPENAI_API_KEY", ""
            )

    # Directory creation is *not* done here: constructing a Config should be a
    # pure act. Writers create their own output directory on demand, and this
    # stays available for callers that want them all up front.
    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.frames_dir, self.clips_dir):
            os.makedirs(path, exist_ok=True)

    @property
    def ocr_language_list(self) -> list:
        return [lang.strip() for lang in self.ocr_languages.split(",") if lang.strip()]
