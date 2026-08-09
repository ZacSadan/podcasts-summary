# Podcast & YouTube Summarizer

An automated pipeline that monitors Hebrew and English podcast RSS feeds and YouTube channels, fetches new episodes, extracts transcripts, and generates a detailed Hebrew summary for each. Summarization runs on a local GGUF model (Gemma3-4B-Instruct) via `llama-cpp-python`, entirely on the GitHub Actions runner — no external paid APIs, no local setup required.
Results are delivered automatically to a configured Telegram channel.
---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Output Format](#output-format)
- [Transcript Extraction Methods](#transcript-extraction-methods)
- [Summarization Pipeline](#summarization-pipeline)
- [Configuration](#configuration)
- [GitHub Actions Setup](#github-actions-setup)
- [Running Manually](#running-manually)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [State Management](#state-management)

---

## How It Works

```
Every hour (GitHub Actions cron)
        │
        ▼
  Fetch all RSS / YouTube feeds (sequentially)
        │
        ▼
  Filter: skip already-seen episodes
        │
        ▼
  For each new episode (stops after 3h wall-clock; rest deferred):
    ├─ Try to get transcript (8 methods, cheapest first)
    ├─ Summarize locally with Gemma3-4B-Instruct (GGUF, CPU)
    │    ├─ Hebrew source  → summarize directly in Hebrew
    │    └─ non-Hebrew source → summarize in English, then translate to Hebrew
    │    (long transcripts are map-reduce chunked by actual token count)
    ├─ Falls back to BART + Helsinki-NLP if the local LLM fails
    ├─ Format output (Hebrew summary + original description + links)
    ├─ Validate and resolve all links
    └─ Send to Telegram
        │
        ▼
  git commit & push state back to repo
```

The pipeline runs on a free GitHub-hosted Ubuntu runner (2 CPU cores). All state is stored in the repository itself — no database, no external services.

---

## Features

- **Fully automated** — GitHub Actions cron fires every hour, processes new episodes, and commits results back
- **Local LLM summarization** — Gemma3-4B-Instruct (GGUF, `llama-cpp-python`) runs entirely on the Actions runner's CPU, no API key and no external service required
- **Language-aware pipeline** — Hebrew-source transcripts are summarized directly in Hebrew; non-Hebrew transcripts are summarized in English first (the model's strongest language), then translated to Hebrew as a separate step — this avoids repetition loops and script-drift that a small model can hit when asked to reason and generate long-form Hebrew directly from non-Hebrew source text
- **Map-reduce for long transcripts** — Transcripts too long for a single call (measured by actual tokenizer count, not word count, since Hebrew tokenizes far less densely than English) are split into chunks, summarized independently, then combined into one final summary
- **Graceful degradation** — If the local LLM fails to load or produces bad output (refusal, repetition, wrong script) after retries, falls back to BART (`facebook/bart-large-cnn`) + Helsinki-NLP translation models, then to a simple extractive summary as a last resort
- **Long Hebrew summary** — 800–1200 words (1200–1500 for long show notes), bold section headers, bullet points
- **Time-budgeted runs** — Each run stops starting new episodes after 3 hours wall-clock (checked between episodes, never mid-episode); remaining episodes defer to the next cron run rather than risking GitHub Actions' 6-hour job limit
- **8 transcript methods** — Tries every available source (including PDF show notes) before falling back to Whisper audio transcription
- **Transcript caching** — Whisper results are saved to `data/transcripts/` and re-used on subsequent runs, avoiding costly re-transcription. Files older than 30 days are deleted automatically on each run.
- **Whisper budget** — Only 1 audio transcription per run to stay within GitHub Actions runner time limits; remaining episodes are deferred to the next cron run
- **Smart link handling** — Dead links are dropped, `example.com` removed, and URL shorteners (`bit.ly`, `t.co`, etc.) resolved to their final destination
- **Feed filtering** — Run on a specific feed by name via `workflow_dispatch` input
- **Test mode** — Process one small episode per feed type to verify the pipeline without long Whisper jobs
- **Telegram delivery** — Each new summary is sent automatically to a Telegram channel, including which model produced it; supports chunked messages for long summaries and respects rate limits
- **Resend history** — Re-send all existing `results.txt.md` entries to Telegram via a single `workflow_dispatch` toggle (requires `--write-results` to have been used previously)
- **No external paid APIs** — The entire summarization pipeline (primary model and fallback) runs locally on the Actions runner
- **SSRF protection** — All outbound HTTP requests validate the target URL against a blocklist of private/loopback/link-local IPs and cloud metadata endpoints before fetching
- **Download size cap** — RSS transcript and episode-page fetches are capped at 500 MB; link-liveness checks are capped at 1 MB

---

## Output Format

By default, summaries are sent to Telegram only. To also write them to `results.txt.md`, pass `--write-results` when running the pipeline. Each episode produces one block:

```markdown
----
## Chapter Name : <episode title>

**Podcast:** <feed name>
**Author:** <author>
**Date:** <episode publish date> UTC
**Generated:** <summary generation time> UTC
**Link:** <episode URL>

---

**Hebrew Summary:**
<detailed Hebrew summary — 800–1200 words, bold section headers, bullet points>

**Original description:**
<first 600 characters of the RSS description>

**Links mentioned:**
• [Page Title](https://resolved-url.com)
• ...

---
*Pipeline:*
  • Transcript: <method> (<N> words, lang=<lang>) — <audio analysis note>
  • Summary: Gemma3-4B-Instruct (he)
```

The **Pipeline** section shows:
- Which transcript method was used and word count
- Whether the **full audio file was transcribed** (Whisper) or show notes / captions were used instead
- Which summarization model actually produced the summary — e.g. `Gemma3-4B-Instruct (he)` for a Hebrew-source transcript summarized directly, `Gemma3-4B-Instruct (en→he)` for a non-Hebrew source that went through the English-then-translate path, or the BART+Helsinki fallback's step names if the local LLM failed. This line is shown in both `results.txt.md` and the Telegram message.

An **English Summary** block appears in `results.txt.md` whenever an English summary was produced along the way — either the English-first intermediate summary (non-Hebrew source, before translation) or the BART fallback's English summary. It is written between the Hebrew summary and the original description, but it is **never** included in the Telegram message, which always contains only the Hebrew summary and the pipeline footer.

---

## Transcript Extraction Methods

Methods are tried in order from cheapest (no download) to most expensive (full audio). If Whisper fails or is skipped, the pipeline retries the text-based methods once more as a last resort (including a lower-bar description check) before giving up:

| # | Method | Description |
|---|--------|-------------|
| 0 | **Cache** | Loads a previously saved transcript from `data/transcripts/` — skips all other methods |
| 1 | **PDF show notes** | Scans the RSS description and episode page for a linked `.pdf` (e.g. Security Now) and extracts its text with `pypdf`; skippable with `--no-pdf` / the `no_pdf` workflow input |
| 2 | **RSS `<podcast:transcript>` tag** | Parses a transcript URL embedded in the RSS feed (VTT, SRT, or HTML formats) |
| 3 | **YouTube captions** | Uses `youtube-transcript-api` to fetch manual or auto-generated captions; falls back to `yt-dlp --write-auto-sub` |
| 4 | **Episode web page** | Fetches the episode's URL and extracts body text — useful for shows like Reversim where the web page has 3× more content than the RSS description |
| 5 | **RSS description** | Uses the RSS `<description>` field if it is ≥1500 words (likely full show notes) |
| 6 | **Whisper** | Downloads audio via `yt-dlp` (falling back to `pytubefix` for YouTube), transcribes with `faster-whisper` (small model, CPU, int8). Limited to `max_whisper_per_run` per run (default: 1) |
| 7 | **Short description fallback** | If Whisper failed or was skipped, retries methods 2–5 and finally accepts any description ≥50 words as a last resort — helps when YouTube bot-detection blocks yt-dlp on CI runners |

Language priority: Hebrew episodes prefer `he/iw` captions first, then `en`. English episodes prefer `en` first.

---

## Summarization Pipeline

### Primary: Gemma3-4B-Instruct (local GGUF model)

Downloaded automatically from Hugging Face (`bartowski/google_gemma-3-4b-it-GGUF`, Q4_K_M quantization) and cached across runs via `actions/cache`. Loaded once per run via `llama-cpp-python` (`n_ctx=8192`, `n_threads=2`, tuned for the runner's 2 CPU cores) — no API key, no network calls at inference time.

**Why this model and this shape of pipeline:** several smaller/faster models (Qwen2.5 1.5B and 3B, DictaLM 2.0) were tried first and each failed in production on real Hebrew content — repetition loops, code-switching into Chinese script under long-context load, or outright failing to complete. Gemma3-4B was the first to reliably produce accurate, non-repetitive, non-hallucinated summaries across both short and long transcripts, in both Hebrew and English source content.

**Language-dependent routing** (`_is_mostly_hebrew()` check on the transcript text):

- **Hebrew-source transcript** → summarized directly in Hebrew (one model call, or several + a combine call if map-reduce chunking is needed)
- **Non-Hebrew-source transcript** → summarized in English first (the model's most reliable language), then translated to Hebrew in a separate, narrower call. Asking a 4B-class model to reason over and generate long-form Hebrew directly from non-Hebrew source text was a reliable way to trigger repetition/script-drift failures; splitting "understand and summarize" from "translate" avoids this.

**Map-reduce for long transcripts:** the decision to chunk, and the resulting chunk count, is based on the transcript's *actual tokenized length* (`llm.tokenize()`), not a word-count estimate — Hebrew tokenizes far less densely than English in this model, so a word-count-based limit that's safe for English content can silently overflow the 8192-token context window for Hebrew. Each chunk is summarized into working notes, then all notes are combined into one final summary. Every individual LLM call additionally re-truncates its own input against the real token budget as a second line of defense.

**Retry-with-quality-guards:** each call is retried (progressively shorter input) if the output is a refusal, an empty/placeholder response, degenerate sentence repetition, or — for Hebrew-generation steps — contains non-Hebrew script characters (Chinese/Japanese/Korean/Cyrillic), which is a reliable signal the model drifted under load.

**Output length budgets** are tiered per call type instead of one fixed ceiling, since generation time on a 2-core runner scales with the requested token budget: chunk notes get a smaller cap, the final summary/combine call gets a larger one, and the translation call's cap scales with the input length.

### Fallback: BART + Helsinki (local models)

Used only when the primary Gemma3-4B path raises an exception (model failed to load, or every retry attempt was rejected by the quality guards). Runs entirely on CPU inside the GitHub Actions runner using `transformers` + `torch`.

**Hebrew episode pipeline:**
1. Extractive pre-summary (if >1,500 words) to reduce translation cost
2. Translate Hebrew → English (`Helsinki-NLP/opus-mt-tc-big-he-en`)
3. Summarize with BART (`facebook/bart-large-cnn`) in 800-word chunks
4. Translate English summary → Hebrew (`Helsinki-NLP/opus-mt-en-he`)

**English episode pipeline:**
1. Extractive pre-summary (if >4,000 words)
2. Summarize with BART in 800-word chunks
3. Translate English summary → Hebrew

This path is known to be lower quality than the primary model — Helsinki's translation can mistranslate Hebrew idioms/slang, and errors compound across the two translation hops. It exists as a safety net so a failure never produces a crash, only a degraded (but clearly labeled, see [Output Format](#output-format)) summary.

### Extractive fallback

If both the local LLM and the BART+Helsinki fallback fail, a simple sentence-extraction summary is used as a last resort.

---

## Configuration

All feeds and settings live in `config/feeds.yaml`.

### Settings

```yaml
settings:
  hours_lookback: 168              # Look back N hours for new episodes (default: 7 days)
  description_min_length: 1500     # Min word count to treat RSS description as transcript
  max_audio_duration_minutes: 120  # Present in config for documentation purposes; not currently read by the code
  whisper_model: small             # faster-whisper model size (tiny/base/small/medium/large)
  max_whisper_per_run: 1           # Max Whisper jobs per cron run (defers the rest)
  bart_chunk_words: 800            # BART input chunk size in words
  summary_sentences: 8             # Present in config for documentation purposes; not currently read by the code
```

Note: `extractive_max_sentences` (default 15) controls the extractive-summary fallback in code but is not currently set in `feeds.yaml` — it always uses its hardcoded default.

### Hardcoded constants (`main.py`)

A few limits are not exposed in `feeds.yaml` and require a code change to adjust:

| Constant | Default | Purpose |
|---|---|---|
| `MAX_RUN_HOURS` | 3 | Stop starting new episodes once this many hours have elapsed in the current run; remaining episodes defer to the next cron run (guards against GitHub Actions' 6-hour hard job limit) |
| `TRANSCRIPT_RETENTION_DAYS` | 30 | Cached transcripts older than this (by first git-commit date) are deleted at the start of each run |
| `MAX_SEEN_ENTRIES` | 1000 | Cap on `data/seen.json` entries; oldest are pruned first |

### Adding a Feed

```yaml
feeds:
  - name: My Podcast
    url: https://example.com/feed.xml
    # optional — used only as metadata, not for fetching:
    spotify_url: https://open.spotify.com/show/...
```

To force Whisper transcription for a specific feed (skipping captions/description methods):
```yaml
  - name: My Channel
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UC...
    enforce_whisper: true   # always use Whisper, skip captions/description
```

To temporarily disable a feed without removing it from the list:
```yaml
  - name: My Channel
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UC...
    disabled: true   # feed is skipped entirely until removed
```

For YouTube channels, use the channel RSS URL:
```yaml
  - name: My Channel
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UC...
```

Language is auto-detected from feed metadata and Hebrew character ratio. Override is not needed in most cases.

---

## GitHub Actions Setup

### Required Secrets

No API token is required for summarization — Gemma3-4B and its BART+Helsinki fallback both run locally on the Actions runner.

| Secret | Description |
|--------|-------------|
| `TELEGRAM_BOT_TOKEN` | Token for your Telegram bot (from [@BotFather](https://t.me/BotFather)). Optional — if not set, Telegram delivery is silently skipped. |
| `TELEGRAM_CHAT_ID` | Target channel or chat ID (e.g. `@MyChannel` or a numeric ID). The bot must be added as an **admin** of the channel. |

### Workflow Triggers

**Automatic (cron):** Fires every hour at `:00 UTC`. Processes all unseen episodes from the last 7 days.

**Shabbat guard:** `main.py` has an `IsShbbatKodeah()` check that normally skips scheduled runs from Friday 16:00 to Saturday 21:00 Israel time. **This check is currently disabled** (commented out in code, not deleted) — scheduled runs currently proceed through Shabbat. Re-enable by uncommenting the call site in `main()`.

**Manual (`workflow_dispatch`):** Go to the repo → Actions → Podcast Summary → Run workflow.

| Input | Description |
|-------|-------------|
| `feed` | Optional substring to filter by feed name (e.g. `רברס` or `Creative Channel`) |
| `test` | If checked, processes only 1 small episode per feed type (YouTube / Spotify-RSS / other RSS) — fast verification without triggering Whisper |
| `resend_history` | If checked, re-sends every entry already in `results.txt.md` to Telegram (requires `--write-results` to have been used previously) |
| `no_pdf` | If checked, skips the PDF show-notes transcript method (method 1) — useful for before/after comparison testing |

### First Run

GitHub may delay scheduled workflow runs for newly created repositories by several hours (known behavior). To verify the workflow works, use manual `workflow_dispatch` first.

---

## Running Manually

The pipeline is designed to run on GitHub Actions, not locally. To trigger a run without waiting for the cron:

1. Go to the repository on GitHub
2. Click **Actions** → **Podcast Summary** → **Run workflow**
3. Optionally fill in `feed` (e.g. `בזמן שעבדתם`), check `test` for a quick run, and check `no_pdf` to skip PDF show-notes extraction
4. Click **Run workflow**

Results are sent to Telegram after the run. To also write them to `results.txt.md`, add `--write-results` to the workflow inputs.

---

## Project Structure

```
podcasts-summary/
├── main.py                     # Pipeline orchestrator — entry point
├── requirements.txt            # Python dependencies
├── config/
│   └── feeds.yaml              # Feed list + pipeline settings
├── src/
│   ├── fetcher.py              # RSS/YouTube feed parsing, Episode dataclass
│   ├── transcript.py           # All transcript extraction methods
│   └── summarize.py            # Summarization, formatting, link handling
├── data/
│   ├── seen.json               # Tracks processed episode IDs (max 1000 entries)
│   └── transcripts/            # Cached transcript files (one .txt per episode)
└── .github/
    └── workflows/
        └── summarize.yml       # GitHub Actions workflow definition
```

### Key Files

**`data/seen.json`** — JSON object mapping episode IDs to the ISO timestamp when they were processed. Prevents re-processing. Capped at 1,000 entries (oldest are pruned first).

**`data/transcripts/<name>.txt`** — Cached transcript files. Format:
```
Feed: <feed name>
Episode: <episode title>
Method: <whisper|youtube_captions|...>
Language: <he|en|auto>
Words: <count>
URL: <episode URL>

--- TRANSCRIPT ---

<full transcript text>
```

**`results.txt.md`** — Optional append-only output file. Only written when `--write-results` flag is passed. In test mode with `--feed`, new entries are appended. In bare `--test` mode (no feed filter), the file is cleared first.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `llama-cpp-python` | 0.3.16 | Runs the local Gemma3-4B-Instruct GGUF model on CPU |
| `feedparser` | 6.0.11 | RSS/Atom feed parsing |
| `beautifulsoup4` | 4.12.3 | HTML parsing for page content extraction |
| `lxml` | 5.3.0 | Fast HTML/XML parser backend |
| `requests` | 2.32.3 | HTTP client for RSS, transcripts, link checking |
| `PyYAML` | 6.0.2 | Config file parsing |
| `youtube-transcript-api` | 1.2.4 | YouTube caption fetching (no download) |
| `yt-dlp` | 2026.3.17 | Audio download and YouTube subtitle fallback |
| `pytubefix` | ≥8.0.0 | Fallback YouTube audio downloader when `yt-dlp` is blocked |
| `faster-whisper` | 1.0.3 | CPU-optimized Whisper transcription |
| `transformers` | 4.57.6 | BART summarization + Helsinki translation models |
| `torch` | 2.12.0 | PyTorch backend for local models |
| `sentencepiece` | 0.2.0 | Tokenizer for Helsinki translation models |
| `sacremoses` | 0.1.1 | Text normalization for translation |
| `pypdf` | ≥4.0.0 | Text extraction from linked PDF show notes |

---

## Security

### SSRF protection

All outbound HTTP requests made from RSS feed content (transcript URLs, episode page URLs, and extracted links) are validated before fetching. The `_is_safe_url()` check blocks:

- Non-`http`/`https` schemes
- `localhost`, `127.x.x.x`, and all loopback addresses
- `169.254.169.254` and `metadata.google.internal` (cloud instance metadata endpoints)
- All RFC-1918 private ranges (`10.x`, `172.16–31.x`, `192.168.x`) and link-local addresses

### Download size limits

Fetches from untrusted RSS sources are capped to prevent memory exhaustion:

| Fetch type | Cap |
|---|---|
| RSS transcript tag, episode web page | 500 MB |
| PDF show-notes download | 20 MB |
| Link liveness / title check | 1 MB |

### Path traversal protection

Transcript filenames are derived from RSS feed and episode titles. After sanitizing the name, the resolved path is asserted to remain inside `data/transcripts/` before any read or write.

---

## State Management

The pipeline uses two files for state — no database required:

- **`data/seen.json`** — Which episodes have been processed (persisted in git after every run)
- **`data/transcripts/`** — Full transcript text for each processed episode (persisted in git, used as a cache to avoid re-running Whisper). Files are automatically deleted after 30 days based on their first git commit date.

Both files are committed back to `master` by the workflow after every run, so state survives across cron invocations.

To reprocess an episode, delete its entry from `seen.json` and its file from `data/transcripts/`.

### A note on long runs and commit conflicts

If a run takes long enough that another run's commit lands on `master` first, the workflow's `git pull --rebase` can hit a merge conflict on `data/seen.json` and fail to push — in which case **that entire run's processed episodes are lost from state** (already sent to Telegram, but not recorded as seen), and they'll be reprocessed — and re-sent to Telegram — on a future run. This actually happened once in production: a run backlogged to 34 episodes over 4.5 hours, failed to commit, and triggered several subsequent oversized runs (one of which hit GitHub Actions' 6-hour hard job limit) before being manually resolved.

`MAX_RUN_HOURS` (3 hours, see [Configuration](#hardcoded-constants-mainpy)) exists specifically to keep any single run short enough that this is unlikely, but if you ever see a run take unusually long, check whether the eventual `chore: update summaries [skip ci]` commit actually landed before assuming affected episodes were recorded as seen.
