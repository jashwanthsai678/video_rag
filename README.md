# Video RAG

Ask a plain-English question about a library of screen-recording / walkthrough
videos and get back **the exact clip** where it happens, plus a generated
**step-by-step guide** — instead of scrubbing through hours of footage.

This is Retrieval-Augmented Generation (RAG) applied to video instead of text
documents: the same "embed → index → retrieve → generate" pattern used for
document Q&A, but the retrieved unit is a moment in a video rather than a
paragraph.

## How it works

```
 ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
 │   Video(s)  │ ──▶ │  Frame + OCR │ ──▶ │                │
 │  (mp4, etc) │     │  extraction  │     │  Segments      │
 └─────────────┘     └──────────────┘     │  (text +       │
        │                                  │  timestamp)   │
        ▼             ┌──────────────┐     │                │
   ┌──────────┐  ──▶  │ Whisper      │ ──▶ │                │
   │  Audio   │       │ transcription│     └───────┬────────┘
   └──────────┘       └──────────────┘             │
                                                     ▼
                                    ┌────────────────────────────┐
                                    │ Embeddings + FAISS  (dense) │
                                    │ BM25 keyword index (sparse) │
                                    └──────────┬─────────────────┘
                                                │
   "How do I add a new patient?"  ──▶  both ───┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ reciprocal rank fusion     │
                                  │ → best video               │
                                  │ → best *contiguous* moment │
                                  └─────────────┬─────────────┘
                                                │
                          ┌─────────────────────┴────────────────────┐
                          ▼                                          ▼
                ffmpeg cuts a clip                       LLM writes a generic,
                around the matched                       numbered how-to guide
                cluster of moments                       from the matched text
```

**Indexing** (done once per video):
1. Sample ~1 frame/sec, keeping only frames where enough pixels actually
   changed, so a static screen doesn't produce hundreds of redundant frames.
   Frames are OCR'd in memory and discarded — nothing is written to disk.
2. Run OCR on kept frames to capture on-screen text (buttons, labels, form
   fields).
3. Transcribe the audio track with Whisper to capture narration.
4. Embed every OCR/transcript snippet into a FAISS vector index and add it to
   a BM25 keyword index, tagged with its source video and timestamp.

**Querying** (repeatable, no re-indexing):
1. Search both indexes — dense embeddings for paraphrase, BM25 for the exact
   UI tokens embeddings blur together — and fuse the two rankings.
2. Pick the video with the strongest *concentration* of evidence.
3. Split its matches into contiguous runs and keep the strongest run, so the
   result is one moment rather than a span from the first hit to the last.
4. Cut that window with `ffmpeg`.
5. Ask an LLM to turn the matched text into a clean, numbered how-to guide —
   instructed to generalize away example data (names, IDs) that happened to be
   on screen during recording.

## Why hybrid retrieval?

Two different failure modes, two different retrievers:

- The question is usually **worded differently** from what's on screen ("add a
  new patient" vs. "click New → Patient Record"). Dense embeddings handle this.
- Screen text is full of **exact tokens** — button captions, menu names, field
  labels — that embeddings smear into their neighbours. BM25 handles this.

Their scores live on incompatible scales, so they're combined by [Reciprocal
Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), which
merges *rankings* instead of scores and needs no normalization tuning.

## Install

```bash
git clone https://github.com/<you>/video-rag.git
cd video-rag
pip install -r requirements.txt
# also needs ffmpeg on PATH:
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt install ffmpeg
#   Windows: winget install Gyan.FFmpeg

cp .env.example .env   # then fill in your LLM API key
```

The LLM step uses any OpenAI-compatible endpoint — defaults to
[OpenRouter](https://openrouter.ai) so you can point `VIDEORAG_LLM_MODEL` at
GPT, Claude, Gemini, or a local server without code changes. Set
`VIDEORAG_LLM_ENABLED=false` to skip it entirely and get a plain list of
matched moments instead of a generated guide.

## Usage

**CLI**

```bash
# put your videos in ./videos, then:
videorag index
videorag query "How do I add a new patient?"

videorag query "..." --show-hits   # every retrieved moment, with scores
videorag query "..." --no-clip     # retrieval only, skip ffmpeg
videorag index --force             # re-index (replaces, does not duplicate)
```

**Python**

```python
from video_rag import VideoRAGPipeline

pipeline = VideoRAGPipeline()
pipeline.index_videos()  # indexes everything in ./videos not already indexed

result = pipeline.query("How do I add a new patient?")
print(result.source_video, result.score)
print(result.clip_path)
print(result.explanation)

# Retrieval only: no LLM call, no files written. Use this when tuning.
found = pipeline.locate("How do I add a new patient?")
for hit in found.hits:
    print(hit.segment.timestamp, hit.score, hit.found_by, hit.segment.text)
```

See `examples/quickstart.py` for a runnable end-to-end example.

## Measuring retrieval quality

Every knob below is a guess until it moves a number, so retrieval is
measurable directly. Write ~20 questions with the timestamp that answers each:

```bash
videorag eval labels.json --init   # writes a starter file to fill in
videorag eval labels.json
```

```
Labels: 18 positive, 4 negative

Recall@8         88.9%   (a correct moment appears anywhere in the top 8)
MRR              0.712   (1.0 = correct moment always ranked first)
Video accuracy   94.4%   (picked the right video)
Clip accuracy    83.3%   (the returned clip actually contains the moment)
Specificity     100.0%   (correctly refused to answer)

Score distribution — use this to set VIDEORAG_RELEVANCE_THRESHOLD:
  answerable    min 0.612  median 0.744  max 0.881
  unanswerable  min 0.317  median 0.402  max 0.489
  suggested threshold: 0.55
```

Label a few questions with `"video": null` to mark things the library should
*not* answer — that's the only way to calibrate `relevance_threshold` rather
than guessing at it.

## Configuration

All settings live in `video_rag/config.py` and can be overridden via
environment variables (or a `.env` file) — see `.env.example`. The knobs that
actually matter:

| Variable | Default | What it does |
|---|---|---|
| `VIDEORAG_RELEVANCE_THRESHOLD` | `0.55` | Minimum similarity to answer at all. **Calibrate this with `videorag eval`** — BGE similarities sit near 0.5 even for unrelated text, so a low value never rejects anything |
| `VIDEORAG_KEYWORD_WEIGHT` | `0.6` | BM25's weight in the fusion; `0` gives vector-only search |
| `VIDEORAG_CLUSTER_MAX_GAP` | `30` | Matches further apart than this become separate candidate clips |
| `VIDEORAG_TOP_K` | `8` | Moments retrieved per query |
| `VIDEORAG_SCENE_MIN_FRACTION` | `0.02` | Fraction of pixels that must change to keep a frame. Raise for faster indexing, lower if subtle UI changes are missed |
| `VIDEORAG_FPS_EXTRACT` | `1` | Frames sampled per second before dedup |
| `VIDEORAG_WHISPER_MODEL` | `base` | Whisper model size (`tiny`…`large-v3`) |
| `VIDEORAG_OCR_DUAL_PASS` | `false` | Second OCR pass over a binarized frame; ~2× slower indexing |
| `VIDEORAG_LLM_ENABLED` | `true` | Toggle LLM-written explanations |

Note that changing `VIDEORAG_EMBEDDING_MODEL` invalidates a saved index —
vectors from different models aren't comparable, so loading is refused with an
explanation rather than silently returning nonsense.

## Project layout

```
video_rag/
  config.py       Settings (env-var driven)
  segment.py      Shared data model (a retrievable moment)
  retrieval.py    BM25, rank fusion, temporal clustering  ← no heavy deps
  util.py         Filename/time helpers                   ← no heavy deps
  extract.py      Frame sampling + scene-change gating
  ocr.py          On-screen text extraction (EasyOCR)
  transcribe.py   Speech-to-text (faster-whisper)
  index.py        Hybrid FAISS + BM25 index (with save/load)
  explain.py      LLM-generated step-by-step guide
  clipper.py      ffmpeg clip cutting
  pipeline.py     Wires it all together (VideoRAGPipeline)
  evaluate.py     Recall@k / MRR / threshold calibration
  cli.py          `videorag index` / `query` / `eval`
tests/            140 tests, no model weights, runs in ~1s
examples/
  quickstart.py
```

Every ranking decision lives in `retrieval.py`, which imports nothing heavier
than the standard library. That's deliberate: it's the code most likely to be
wrong, so it's testable — and CI-runnable — without downloading a single model.

## Development

```bash
pip install -e ".[dev]"
pytest          # ~1s, no model downloads
ruff check .
```

## Limitations / roadmap

- Retrieval quality still depends on OCR/transcription quality — noisy audio
  or small UI text will hurt recall.
- No cross-encoder re-ranking yet. Fusion improves ordering, but a reranker
  over the top ~20 candidates would help more.
- OCR is the indexing bottleneck and runs single-threaded.
- No web UI included; a Streamlit/Gradio front-end over `VideoRAGPipeline`
  would be a natural next step.
- Everything runs locally except the final explanation-generation LLM call.

## License

MIT — see [LICENSE](LICENSE).
