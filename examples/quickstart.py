"""Minimal example: index a folder of videos, then ask a question.

Usage:
    python examples/quickstart.py
"""

import logging

from video_rag import VideoRAGPipeline

# Indexing is slow and silent without this — the pipeline logs its progress.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

pipeline = VideoRAGPipeline()

added = pipeline.index_videos()  # reads Config().videos_dir, e.g. "videos/"
print(f"Indexed {added} new segments across {len(pipeline.videos)} video(s).")

result = pipeline.query("How do I add a new patient?")

if result.status == "success":
    print(f"\nBest match: {result.source_video} (similarity {result.score:.2f})")
    print(f"Moment: {result.start_seconds:.0f}s - {result.end_seconds:.0f}s")
    print(f"Clip: {result.clip_path}")
    if result.clusters_found > 1:
        print(f"(chosen from {result.clusters_found} separate matching moments)")
    print(f"\n{result.explanation}")
else:
    print(f"No confident match found (best similarity {result.score:.2f}).")
    print("Lower VIDEORAG_RELEVANCE_THRESHOLD if that looks wrong.")

# Retrieval only — no LLM call, no clip written. Useful for tuning.
found = pipeline.locate("How do I export a report?")
for hit in found.hits[:5]:
    print(f"  {hit.segment.timestamp:7.1f}s  sim={hit.score:.3f}  {hit.segment.text[:60]}")
