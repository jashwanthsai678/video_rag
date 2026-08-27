"""Command-line interface: `videorag index`, `query`, and `eval`."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .util import format_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videorag", description="Video RAG CLI")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show debug-level logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only show warnings and errors"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index videos from a folder")
    p_index.add_argument("--videos-dir", default=None, help="Folder of videos (default: config)")
    p_index.add_argument("--force", action="store_true", help="Re-index already-indexed videos")

    p_query = sub.add_parser("query", help="Ask a question about indexed videos")
    p_query.add_argument("question", help="Your question, in quotes")
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.add_argument(
        "--no-clip", action="store_true", help="Skip cutting a clip (retrieval only)"
    )
    p_query.add_argument(
        "--show-hits", action="store_true", help="List every retrieved moment and its score"
    )

    p_eval = sub.add_parser("eval", help="Measure retrieval quality against labelled data")
    p_eval.add_argument("labels", help="Path to a JSON label file")
    p_eval.add_argument("--top-k", type=int, default=None)
    p_eval.add_argument(
        "--init", action="store_true", help="Write a starter label file at LABELS and exit"
    )

    p_inspect = sub.add_parser(
        "inspect",
        help="Show what's indexed (videos, segment counts) without loading FAISS or torch",
    )
    p_inspect.add_argument("--index-path", default=None, help="Index folder (default: config)")

    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _print_hits(hits) -> None:
    print("\nRetrieved moments:")
    for i, hit in enumerate(hits, 1):
        via = "+".join(hit.found_by) or "-"
        print(
            f"  {i:2d}. [{format_timestamp(hit.segment.timestamp)}] "
            f"{hit.segment.video_name}  sim={hit.score:.3f} via={via}\n"
            f"      {hit.segment.text[:110]}"
        )


def _run_index(args, pipeline) -> int:
    added = pipeline.index_videos(videos_dir=args.videos_dir, force=args.force)
    if not added:
        print("No new segments indexed.")
        return 0
    print(f"Done. Added {added} new segments. Total: {len(pipeline.index.segments)}")
    return 0


def _run_query(args, pipeline) -> int:
    if args.no_clip:
        found = pipeline.locate(args.question, top_k=args.top_k)
    else:
        found = pipeline.query(args.question, top_k=args.top_k)

    if found.status == "no_match":
        print(f"No confident match found (best similarity: {found.score:.3f}).")
        print(
            "If that looks wrong, lower VIDEORAG_RELEVANCE_THRESHOLD "
            f"(currently {pipeline.config.relevance_threshold:.2f})."
        )
        if args.show_hits and found.hits:
            _print_hits(found.hits)
        return 1
    if getattr(found, "error", None):
        print(f"Error: {found.error}")
        return 1

    print(f"Best match video : {found.source_video}")
    print(f"Confidence       : {found.score:.3f}")
    print(
        f"Time range       : {format_timestamp(found.start_seconds)}"
        f" - {format_timestamp(found.end_seconds)}"
        f"  ({found.end_seconds - found.start_seconds:.1f}s)"
    )
    if found.clusters_found > 1:
        print(f"Note             : best of {found.clusters_found} separate matching moments")
    if getattr(found, "clip_path", None):
        print(f"Clip saved to    : {found.clip_path}")

    if args.show_hits:
        _print_hits(found.hits)

    if getattr(found, "explanation", ""):
        print("\nExplanation:\n")
        print(found.explanation)
    return 0


def _run_inspect(args) -> int:
    """Report what's indexed without constructing VideoRAGPipeline at all —
    this is the payoff of SQLite-backed metadata: no FAISS, no torch, no cv2,
    just the standard library's sqlite3 module."""
    from .store import inspect as inspect_store

    index_path = args.index_path or Config().index_path
    report = inspect_store(index_path)
    if not report["videos"]:
        print(f"No index found at {index_path}.")
        return 1

    print(f"Index           : {index_path}")
    print(f"Embedding model : {report['embedding_model']}")
    print(f"Total segments  : {report['total_segments']}")
    print("\nVideos:")
    for name, count in sorted(report["videos"].items()):
        print(f"  {name:30s} {count} segment(s)")
    return 0


def _run_eval(args, pipeline) -> int:
    from .evaluate import evaluate, load_labels, write_template

    if args.init:
        path = write_template(args.labels, pipeline)
        print(f"Wrote a starter label file to {path}. Fill in the questions and timestamps,")
        print(f"then run: videorag eval {path}")
        return 0

    labels = load_labels(args.labels)
    if not labels:
        print(f"No labels found in {args.labels}.")
        return 1

    report = evaluate(pipeline, labels, top_k=args.top_k)
    print(report.format(args.top_k or pipeline.config.top_k))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose, args.quiet)

    if args.command == "inspect":
        # No pipeline construction at all: this command's entire point is
        # answering "what's indexed" without the cost of importing cv2,
        # faiss, or torch.
        return _run_inspect(args)

    # Imported here so `--help` (and `inspect`, above) don't pay for torch/faiss.
    from .index import IndexCompatibilityError
    from .pipeline import VideoRAGPipeline

    try:
        pipeline = VideoRAGPipeline(Config())
    except IndexCompatibilityError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    handlers = {"index": _run_index, "query": _run_query, "eval": _run_eval}
    handler = handlers.get(args.command)
    if handler is None:
        return 1

    try:
        return handler(args, pipeline)
    except (IndexCompatibilityError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
