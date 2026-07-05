"""
Production entry point for the podcast/YouTube summarization pipeline.

Usage:
    python main.py                  # normal run: episodes from last 7 days, skip seen
    python main.py --test           # test run: 3 smallest episodes (1 YouTube, 1 RSS-Spotify, 1 other RSS)
    python main.py --write-results  # also append summaries to results.txt.md
"""
import sys
import io
import re
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SEEN_PATH = DATA_DIR / "seen.json"
RESULTS_PATH = ROOT / "results.txt.md"
CONFIG_PATH = ROOT / "config" / "feeds.yaml"
DEBUG_DIR = DATA_DIR / "transcripts"

TRANSCRIPT_RETENTION_DAYS = 30

MAX_SEEN_ENTRIES = 1000


# ── Shabbat guard ─────────────────────────────────────────────────────────────

def IsShbbatKodeah() -> bool:
    """True from Friday 16:00 IL time through Saturday 21:00 IL time.
    The pipeline must not run in GitHub Actions during this window."""
    from zoneinfo import ZoneInfo
    il_now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    minutes = il_now.hour * 60 + il_now.minute
    if il_now.weekday() == 4 and minutes >= 16 * 60:   # Friday from 16:00
        return True
    if il_now.weekday() == 5 and minutes < 21 * 60:    # Saturday until 21:00
        return True
    return False


# ── Config & State ────────────────────────────────────────────────────────────

def load_config() -> tuple[list, dict]:
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["feeds"], config["settings"]


def load_seen() -> dict:
    if SEEN_PATH.exists():
        with open(SEEN_PATH, encoding="utf-8-sig") as f:
            return json.load(f)
    return {"version": 1, "entries": {}}


def save_seen(seen: dict):
    entries = seen["entries"]
    if len(entries) > MAX_SEEN_ENTRIES:
        sorted_items = sorted(entries.items(), key=lambda x: x[1])
        seen["entries"] = dict(sorted_items[-MAX_SEEN_ENTRIES:])
    DATA_DIR.mkdir(exist_ok=True)
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, ensure_ascii=False, indent=2)


def mark_seen(seen: dict, episode_id: str):
    seen["entries"][episode_id] = datetime.now(timezone.utc).isoformat()


def is_seen(seen: dict, episode_id: str) -> bool:
    return episode_id in seen["entries"]


# ── Transcript cleanup ────────────────────────────────────────────────────────

def _git_first_commit_date(path: Path) -> datetime | None:
    """Return the UTC datetime when the file was first committed to git, or None."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%cI", "--", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        date_str = result.stdout.strip()
        if date_str:
            return datetime.fromisoformat(date_str)
    except Exception:
        pass
    return None


def cleanup_old_transcripts():
    """Delete transcript .txt files first committed to git more than TRANSCRIPT_RETENTION_DAYS ago."""
    if not DEBUG_DIR.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=TRANSCRIPT_RETENTION_DAYS)
    deleted = 0
    for txt_file in sorted(DEBUG_DIR.glob("*.txt")):
        first_commit = _git_first_commit_date(txt_file)
        if first_commit is None:
            continue  # untracked / new file — leave it alone
        if first_commit.tzinfo is None:
            first_commit = first_commit.replace(tzinfo=timezone.utc)
        if first_commit < cutoff:
            txt_file.unlink()
            deleted += 1
            logger.info(f"Cleanup: deleted {txt_file.name} (committed {first_commit.date()})")
    if deleted:
        logger.info(f"Cleanup: removed {deleted} transcript(s) older than {TRANSCRIPT_RETENTION_DAYS} days")


# ── Test mode episode selection ───────────────────────────────────────────────

def _enclosure_size(episode) -> int:
    """Return RSS enclosure byte length if available, else sys.maxsize."""
    import sys as _sys
    url = episode.audio_url or ""
    if not url:
        return _sys.maxsize
    # feedparser stores enclosure length in the feed entry; we don't have direct
    # access here, so fall back to sys.maxsize for YouTube / unknown sources.
    return _sys.maxsize


def select_test_episodes(feed_configs: list) -> list:
    """
    Fetch one recent episode per feed across all feeds, then pick:
      - 1 from youtube_rss feeds
      - 1 from rss feeds that have a spotify_url field  (counts as "spotify")
      - 1 from rss feeds without spotify_url            (other RSS)
    Within each bucket, prefer the episode whose audio enclosure is smallest.
    Returns up to 3 episodes total.
    """
    from src.fetcher import fetch_feed

    youtube_bucket = []
    spotify_rss_bucket = []
    other_rss_bucket = []

    for cfg in feed_configs:
        if cfg.get("disabled"):
            continue
        feed_type = cfg.get("type", "rss")
        if feed_type == "spotify":
            continue
        try:
            episodes = fetch_feed(cfg)
        except Exception as e:
            logger.warning(f"Test fetch failed for {cfg['name']}: {e}")
            continue
        if not episodes:
            continue
        ep = episodes[0]

        if feed_type == "youtube_rss":
            youtube_bucket.append(ep)
        elif cfg.get("spotify_url"):
            spotify_rss_bucket.append(ep)
        else:
            other_rss_bucket.append(ep)

    def pick_smallest(bucket):
        return min(bucket, key=_enclosure_size, default=None)

    selected = []
    for bucket in (youtube_bucket, spotify_rss_bucket, other_rss_bucket):
        ep = pick_smallest(bucket)
        if ep:
            selected.append(ep)
    return selected[:3]


# ── Output helpers ────────────────────────────────────────────────────────────

def append_result(text: str):
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("----\n")
        f.write(text)
        f.write("\n")


# ── Telegram ──────────────────────────────────────────────────────────────────

_TG_MAX = 4096


def _md_to_tg_html(text: str) -> str:
    """Convert the markdown used in results.txt.md to Telegram HTML."""
    import re as _re
    # Escape HTML special chars first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # ## Heading → <b>Heading</b>
    text = _re.sub(r'^#{1,3} (.+)$', r'<b>\1</b>', text, flags=_re.MULTILINE)
    # **bold** → <b>bold</b>  (must come before single-star rule)
    text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = text.replace("**", "")  # remove any unmatched ** leftover
    # *italic/bold* (single star, e.g. *Pipeline:*) → <b>text</b>
    text = _re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<b>\1</b>', text)
    # [title](url) → <a href="url">title</a>
    text = _re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    # bare URLs not already inside an href
    text = _re.sub(r'(?<!href=")https?://\S+', lambda m: f'<a href="{m.group()}">{m.group()}</a>', text)
    # strip --- and ---- divider lines
    text = _re.sub(r'^-{2,}$', '', text, flags=_re.MULTILINE)
    # collapse 3+ blank lines to 2
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _tg_split(text: str, limit: int = _TG_MAX) -> list[str]:
    """Split text into chunks that each fit within Telegram's limit,
    breaking on blank lines where possible."""
    chunks = []
    while len(text) > limit:
        split_at = text.rfind('\n\n', 0, limit)
        if split_at == -1:
            split_at = text.rfind('\n', 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def send_telegram(formatted_summary: str):
    import os
    import time as _time
    import requests as _req

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        logger.info("  Telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping")
        return

    md_chunks = _tg_split(formatted_summary)
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    sent = 0
    try:
        for i, md_chunk in enumerate(md_chunks):
            if i > 0:
                _time.sleep(3)  # stay under Telegram's 20 msg/min channel limit
            html = _md_to_tg_html(md_chunk)
            resp = _req.post(api_url, json={
                "chat_id": chat_id,
                "text": html,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=15)
            if resp.ok:
                sent += 1
            elif resp.status_code == 429:
                retry_after = resp.json().get("parameters", {}).get("retry_after", 30)
                logger.info(f"  Telegram: rate limited, waiting {retry_after}s")
                _time.sleep(retry_after + 1)
                resp = _req.post(api_url, json={
                    "chat_id": chat_id, "text": html,
                    "parse_mode": "HTML", "disable_web_page_preview": True,
                }, timeout=15)
                if resp.ok:
                    sent += 1
                else:
                    logger.warning(f"  Telegram: retry failed {resp.status_code} — {resp.text[:200]}")
                    break
            else:
                logger.warning(f"  Telegram: send failed {resp.status_code} — {resp.text[:200]}")
                break
        logger.info(f"  Telegram: {sent}/{len(md_chunks)} message(s) sent")
    except Exception as e:
        logger.warning(f"  Telegram: send error — {e}")


def resend_history():
    """Send every entry already in results.txt.md to Telegram."""
    import time as _time
    if not RESULTS_PATH.exists():
        logger.info("No results.txt.md found — nothing to resend")
        return
    content = RESULTS_PATH.read_text(encoding="utf-8")
    blocks = [b.strip() for b in content.split("----") if b.strip()]
    logger.info(f"Resending {len(blocks)} existing entries to Telegram")
    for i, block in enumerate(blocks, 1):
        logger.info(f"  Sending entry {i}/{len(blocks)}")
        send_telegram(block)
        if i < len(blocks):
            _time.sleep(4)  # pause between entries to avoid rate limiting


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Podcast/YouTube summarization pipeline")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process 3 smallest episodes (1 per feed type), ignore 7d window")
    parser.add_argument("--feed", type=str, default=None,
                        help="Filter feeds by name substring (case-insensitive)")
    parser.add_argument("--resend-history", action="store_true",
                        help="Resend all existing entries in results.txt.md to Telegram")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF show-notes extraction (for before/after comparison tests)")
    parser.add_argument("--write-results", action="store_true",
                        help="Append summaries to results.txt.md (disabled by default)")
    args = parser.parse_args()

    if IsShbbatKodeah():
        logger.info("Shabbat Kodesh (Fri 16:00 - Sat 21:00 IL time) — skipping run entirely.")
        return

    if args.resend_history:
        resend_history()
        return

    cleanup_old_transcripts()

    feed_configs, settings = load_config()
    if args.feed:
        feed_configs = [f for f in feed_configs if args.feed.lower() in f["name"].lower()]
        if not feed_configs:
            logger.error(f"No feed matched: {args.feed!r}")
            return
        logger.info(f"Feed filter: {args.feed!r} → {len(feed_configs)} feed(s)")
    seen = load_seen()

    # ── Collect episodes ──
    if args.test:
        logger.info("Test mode: selecting 3 smallest episodes across feed types")
        # Only wipe results in pure test mode (no feed filter); with --feed, always append
        if args.write_results and not args.feed and RESULTS_PATH.exists():
            RESULTS_PATH.unlink()
        episodes = select_test_episodes(feed_configs)
        logger.info(f"Test episodes selected: {len(episodes)}")
    else:
        from src.fetcher import get_recent_episodes
        hours = settings.get("hours_lookback", 168)
        logger.info(f"Normal mode: fetching episodes from last {hours}h")
        all_recent = get_recent_episodes(feed_configs, hours=hours)
        episodes = [e for e in all_recent if not is_seen(seen, e.id)]
        logger.info(f"New episodes after seen filter: {len(episodes)}")

    if not episodes:
        logger.info("No new episodes to process.")
        return

    from src.transcript import get_transcript
    from src.summarize import summarize_episode

    max_whisper = settings.get("max_whisper_per_run", 1)
    whisper_count = 0
    feed_config_by_name = {f["name"]: f for f in feed_configs}

    for episode in episodes:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Processing: [{episode.feed_type}] {episode.feed_name} — {episode.title}")
        logger.info(f"  Published: {episode.published.strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"  URL: {episode.url}")

        feed_cfg = feed_config_by_name.get(episode.feed_name, {})
        enforce_whisper = feed_cfg.get("enforce_whisper", False)

        # In test mode, don't apply the whisper budget to get_transcript so all 3 run
        transcript = get_transcript(episode, settings,
                                    whisper_count=0 if args.test else whisper_count,
                                    enforce_whisper=enforce_whisper,
                                    no_pdf=args.no_pdf,
                                    transcripts_dir=DEBUG_DIR)

        if transcript is None:
            logger.warning("  No transcript found — skipping episode")
            mark_seen(seen, episode.id)
            save_seen(seen)
            continue

        if transcript.method == "whisper":
            whisper_count += 1
            logger.info(f"  Whisper used ({whisper_count}/{max_whisper})")

        logger.info(f"  Transcript: {transcript.method} ({transcript.word_count} words, lang={transcript.language})")

        # Save transcript to debug file
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^\w\- ]', '_', f"{episode.feed_name} — {episode.title}")[:80]
        debug_path = DEBUG_DIR / f"{safe_name}.txt"
        if not str(debug_path.resolve()).startswith(str(DEBUG_DIR.resolve())):
            logger.warning(f"  Path traversal blocked for transcript save: {safe_name!r}")
        else:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"Feed: {episode.feed_name}\n")
                f.write(f"Episode: {episode.title}\n")
                f.write(f"Method: {transcript.method}\n")
                f.write(f"Language: {transcript.language}\n")
                f.write(f"Words: {transcript.word_count}\n")
                f.write(f"URL: {episode.url}\n")
                f.write("\n--- TRANSCRIPT ---\n\n")
                f.write(transcript.text)
            logger.info(f"  Transcript saved to {debug_path.name}")

        try:
            summary, tg_summary = summarize_episode(episode, transcript, settings)
        except Exception as e:
            logger.error(f"  Summarization failed: {e}")
            mark_seen(seen, episode.id)
            save_seen(seen)
            continue

        if args.write_results:
            append_result(summary)
        mark_seen(seen, episode.id)
        save_seen(seen)
        send_telegram(tg_summary)
        logger.info("  Done.")

        # Stop if whisper budget is exhausted (production only; test mode processes all 3)
        if not args.test and whisper_count >= max_whisper:
            remaining = episodes[episodes.index(episode) + 1:]
            needs_whisper = [
                e for e in remaining
                if not e.transcript_url and not e.youtube_video_id
            ]
            if needs_whisper:
                logger.info(
                    f"Whisper limit reached ({whisper_count}/{max_whisper}). "
                    f"{len(needs_whisper)} episodes deferred to next cron run."
                )
                break

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
