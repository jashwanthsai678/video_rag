"""Turn a cluster of retrieved segments into a written how-to guide.

Uses any OpenAI-compatible chat completion endpoint - defaults to
OpenRouter so you can point it at Claude, GPT, Gemini, or a local
OpenAI-compatible server just by changing VIDEORAG_LLM_MODEL / BASE_URL.
"""

from __future__ import annotations

import logging
from typing import List

from .config import Config
from .segment import Segment
from .util import format_timestamp

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """User question: "{query}"

Here is what was seen/heard in the matched video, in chronological order:
{context}

Write a clear, numbered, step-by-step guide (Step 1:, Step 2:, ...) that
answers the user's question based only on the content above.

Rules:
1. One step per line - do not write a paragraph.
2. Keep it GENERIC: do not repeat specific example values shown on screen
   (names, IDs, sample text). Use placeholders like "the required field"
   or "the patient's name" instead.
3. If the content above does not actually answer the question, reply
   exactly with: "I couldn't find an answer to that in this video."
"""


class ExplanationGenerator:
    def __init__(self, config: Config):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.config.llm_api_key:
                raise RuntimeError(
                    "No LLM API key configured. Set VIDEORAG_LLM_API_KEY "
                    "(or OPENROUTER_API_KEY / OPENAI_API_KEY) in your environment."
                )
            self._client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                timeout=self.config.llm_timeout_seconds,
            )
        return self._client

    def build_context(self, segments: List[Segment]) -> str:
        ordered = sorted(segments, key=lambda s: s.timestamp)
        lines = []
        for seg in ordered[: self.config.llm_max_context_segments]:
            tag = "seen" if seg.source == "ocr" else "said"
            lines.append(f"[{format_timestamp(seg.timestamp)}] ({tag}) {seg.text}")
        return "\n".join(lines)

    def generate(self, query: str, segments: List[Segment]) -> str:
        if not self.config.llm_enabled or not segments:
            return self._fallback(segments)

        prompt = PROMPT_TEMPLATE.format(query=query, context=self.build_context(segments))
        try:
            resp = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.llm_max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)
            return f"{self._fallback(segments)}\n\n(Note: LLM explanation failed: {e})"

    @staticmethod
    def _fallback(segments: List[Segment]) -> str:
        """No-LLM fallback: just list the matched moments in order."""
        if not segments:
            return "No relevant moments found."
        ordered = sorted(segments, key=lambda s: s.timestamp)
        lines = ["Relevant moments found in the video:"]
        for i, seg in enumerate(ordered[:10], 1):
            lines.append(f"{i}. [{format_timestamp(seg.timestamp)}] {seg.text}")
        return "\n".join(lines)
