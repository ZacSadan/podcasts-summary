import re
import os
import logging
import tempfile
import ipaddress
from dataclasses import dataclass, field
from typing import Optional, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def _is_safe_url(url: str) -> bool:
    """Block SSRF targets: non-http(s) schemes, private/loopback/link-local IPs, cloud metadata."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = (p.hostname or "").lower()
        if not host:
            return False
        if host in {"localhost", "metadata.google.internal", "169.254.169.254"}:
            return False
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return False
        except ValueError:
            pass  # hostname, not a bare IP — pass through
        return True
    except Exception:
        return False


def _safe_get_text(url: str, timeout: int = 30, headers: dict = None) -> str:
    """GET with SSRF check and 500 MB download cap. Returns decoded text."""
    if not _is_safe_url(url):
        raise ValueError(f"Blocked unsafe URL: {url}")
    r = requests.get(url, timeout=timeout, stream=True,
                     headers=headers or {})
    r.raise_for_status()
    chunks = []
    total = 0
    for chunk in r.iter_content(65536):
        total += len(chunk)
        if total > _MAX_DOWNLOAD_BYTES:
            r.close()
            raise ValueError(f"Response exceeded 500 MB cap: {url}")
        chunks.append(chunk)
    raw = b"".join(chunks)
    encoding = r.encoding or r.apparent_encoding or "utf-8"
    return raw.decode(encoding, errors="replace")


@dataclass
class TranscriptResult:
    text: str
    method: str
    language: str
    word_count: int
    attempted: List[str] = field(default_factory=list)  # methods tried before this succeeded


# Substrings yt-dlp/youtube-transcript-api raise when a video is a premiere or
# scheduled live stream that hasn't started broadcasting yet — a transient
# state, unlike a deleted/private/region-blocked video.
_NOT_YET_LIVE_MARKERS = (
    "this live event will begin in a few moments",
    "premieres in",
    "this live event will begin in",
)


def _is_not_yet_live_error(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _NOT_YET_LIVE_MARKERS)


class NotYetLiveError(Exception):
    """Raised when a YouTube video is a premiere/scheduled stream that hasn't
    started broadcasting yet. Callers should retry later rather than treating
    this as a permanent transcript failure."""


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def strip_vtt(vtt: str) -> str:
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    return " ".join(lines)


# ── Method 0: PDF show notes ─────────────────────────────────────────────────

_PDF_URL_RE = re.compile(r'https?://[^\s\'"<>)]+\.pdf(?:\?[^\s\'"<>)]*)?', re.IGNORECASE)
_PDF_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


def try_pdf_show_notes(episode, min_length: int = 500, page_text: Optional[str] = None) -> Optional[TranscriptResult]:
    """Find PDF show-note links in description or episode page and extract text.

    Runs before all other methods so that rich PDF show notes (e.g. Security Now)
    are preferred over captions or Whisper.

    `page_text`, if provided, is reused instead of re-fetching the episode page —
    lets the orchestrator share one page fetch with try_page_content.
    """
    pdf_urls: list[str] = []
    pdf_urls.extend(_PDF_URL_RE.findall(episode.description or ""))

    if page_text is not None:
        pdf_urls.extend(_PDF_URL_RE.findall(page_text))
    else:
        url = episode.url or ""
        if url and not re.search(r'\.(mp3|m4a|ogg|opus|aac|wav|flac)(\?.*)?$', url, re.IGNORECASE) \
                and "youtube.com/watch" not in url and "youtu.be/" not in url:
            try:
                page = _safe_get_text(url, timeout=20,
                                      headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"})
                pdf_urls.extend(_PDF_URL_RE.findall(page))
            except Exception as e:
                logger.debug(f"PDF page scan failed for {episode.title}: {e}")

    seen: set[str] = set()
    for pdf_url in pdf_urls:
        if pdf_url in seen or not _is_safe_url(pdf_url):
            continue
        seen.add(pdf_url)
        try:
            import io
            from pypdf import PdfReader
            r = requests.get(pdf_url, timeout=30, stream=True,
                             headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"})
            if r.status_code >= 400:
                r.close()
                continue
            chunks, total = [], 0
            for chunk in r.iter_content(65536):
                total += len(chunk)
                if total > _PDF_MAX_BYTES:
                    r.close()
                    break
                chunks.append(chunk)
            text = "\n".join(
                t for page in PdfReader(io.BytesIO(b"".join(chunks))).pages
                if (t := page.extract_text())
            ).strip()
            text = re.sub(r"\s+", " ", text).strip()
            wc = len(text.split())
            if wc >= min_length:
                logger.info(f"  PDF show notes: {pdf_url} ({wc} words)")
                return TranscriptResult(f"[SHOW NOTES — PDF]\n{text}",
                                        "pdf_show_notes", episode.language, wc)
        except Exception as e:
            logger.debug(f"PDF extraction failed for {pdf_url}: {e}")
    return None


# ── Method 1: RSS <podcast:transcript> tag ────────────────────────────────────

def try_rss_transcript(episode) -> Optional[TranscriptResult]:
    if not episode.transcript_url:
        return None
    try:
        text = _safe_get_text(episode.transcript_url, timeout=30)
        mime = (episode.transcript_type or "").lower()
        if "vtt" in mime or "srt" in mime:
            text = strip_vtt(text)
        elif "html" in mime:
            text = strip_html(text)
        text = text.strip()
        if len(text) < 100:
            return None
        return TranscriptResult(text, "rss_tag", "auto", len(text.split()))
    except Exception as e:
        logger.debug(f"RSS transcript tag failed for {episode.title}: {e}")
        return None


# ── Method 2: YouTube captions ────────────────────────────────────────────────

def _lang_priority(language: str) -> list:
    if language == "he":
        return ["he", "iw", "en"]
    return ["en", "he", "iw"]


def try_youtube_captions(video_id: str, language: str) -> Optional[TranscriptResult]:
    # Strategy A: youtube-transcript-api (no download needed)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        tlist = YouTubeTranscriptApi().list(video_id)
        langs = _lang_priority(language)

        for manual in (True, False):
            for lang in langs:
                try:
                    t = (tlist.find_manually_created_transcript([lang]) if manual
                         else tlist.find_generated_transcript([lang]))
                    snippets = t.fetch()
                    text = " ".join(s.text for s in snippets).replace("\n", " ")
                    text = re.sub(r"\s+", " ", text).strip()
                    return TranscriptResult(text, "youtube_captions", t.language_code, len(text.split()))
                except Exception:
                    pass
    except Exception as e:
        if _is_not_yet_live_error(str(e)):
            raise NotYetLiveError(str(e)) from e
        logger.warning(f"youtube-transcript-api failed for {video_id}: {e}")

    # Strategy B: yt-dlp --write-auto-sub fallback
    try:
        import yt_dlp
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tmpl = os.path.join(tmpdir, "%(id)s")
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["he", "iw", "en"],
                "subtitlesformat": "vtt",
                "outtmpl": out_tmpl,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            for f in os.listdir(tmpdir):
                if f.endswith(".vtt"):
                    with open(os.path.join(tmpdir, f), encoding="utf-8") as vf:
                        text = strip_vtt(vf.read()).strip()
                    if len(text) > 100:
                        lang = f.split(".")[-2] if "." in f else "auto"
                        return TranscriptResult(text, "youtube_captions_yt_dlp", lang, len(text.split()))
    except Exception as e:
        if _is_not_yet_live_error(str(e)):
            raise NotYetLiveError(str(e)) from e
        logger.debug(f"yt-dlp captions failed for {video_id}: {e}")

    return None


# ── Method 3: Fetch episode web page for richer show notes ───────────────────

_AUDIO_EXT_RE = re.compile(r'\.(mp3|m4a|ogg|opus|aac|wav|flac)(\?.*)?$', re.IGNORECASE)
_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form"}


def try_page_content(episode, min_length: int, page_text: Optional[str] = None) -> Optional[TranscriptResult]:
    """Fetch the episode's web page and extract body text.

    Useful when the RSS description is short but the episode page has full
    show notes (e.g. Reversim, where the page has 3× more content than RSS).
    Skips audio URLs and YouTube watch pages (handled by other methods).

    `page_text`, if provided, is reused instead of re-fetching the episode page —
    lets the orchestrator share one page fetch with try_pdf_show_notes.
    """
    url = episode.url or ""
    if not url:
        return None
    if _AUDIO_EXT_RE.search(url):
        return None
    if "youtube.com/watch" in url or "youtu.be/" in url:
        return None

    try:
        if page_text is None:
            page_text = _safe_get_text(
                url, timeout=20,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"},
            )
        soup = BeautifulSoup(page_text, "lxml")
        for tag in soup.find_all(_SKIP_TAGS):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logger.debug(f"Page fetch failed for {episode.title}: {e}")
        return None

    # Only use if meaningfully longer than the RSS description
    rss_len = len((episode.description or "").split())
    page_len = len(text.split())
    if page_len < min_length or page_len <= rss_len:
        return None

    return TranscriptResult(
        f"[SHOW NOTES — PAGE]\n{text}",
        "page_content",
        episode.language,
        page_len,
    )


# ── Method 4: Long description as show notes ──────────────────────────────────

def try_description(episode, min_length: int) -> Optional[TranscriptResult]:
    raw = episode.description or ""
    text = strip_html(raw).strip()
    if len(text.split()) < min_length:
        return None
    return TranscriptResult(
        f"[SHOW NOTES]\n{text}",
        "description",
        episode.language,
        len(text.split()),
    )


# ── Method 4: Whisper (speech-to-text fallback) ───────────────────────────────

def _download_audio(url: str, out_tmpl: str, tmpdir: str) -> None:
    """Download audio with three fallbacks. Raises if all methods fail.

    Raises NotYetLiveError instead if every attempt failed because the video
    is a premiere/scheduled stream that hasn't started broadcasting yet."""
    import yt_dlp

    base_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
    }
    not_yet_live = False

    # Attempt 1: yt-dlp default
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            ydl.download([url])
        return
    except Exception as e:
        not_yet_live = not_yet_live or _is_not_yet_live_error(str(e))
        logger.warning(f"yt-dlp default failed: {e}")

    # Attempt 2: yt-dlp with iOS player client (bypasses YouTube bot check)
    try:
        ios_opts = {**base_opts, "extractor_args": {"youtube": {"player_client": ["ios"]}}}
        with yt_dlp.YoutubeDL(ios_opts) as ydl:
            ydl.download([url])
        return
    except Exception as e:
        not_yet_live = not_yet_live or _is_not_yet_live_error(str(e))
        logger.warning(f"yt-dlp ios client failed: {e}")

    # Attempt 3: pytubefix
    try:
        from pytubefix import YouTube
        yt = YouTube(url, 'WEB')
        stream = yt.streams.filter(only_audio=True).order_by("abr").last()
        if not stream:
            raise RuntimeError("no audio stream found")
        stream.download(output_path=tmpdir, filename="audio")
        return
    except Exception as e:
        not_yet_live = not_yet_live or _is_not_yet_live_error(str(e))
        logger.warning(f"pytubefix failed: {e}")

    if not_yet_live:
        raise NotYetLiveError(f"video not yet live: {url}")
    raise RuntimeError(f"all audio download methods failed for {url}")


def try_whisper(episode, model_size: str = "small") -> Optional[TranscriptResult]:
    try:
        import yt_dlp  # noqa: F401
        from faster_whisper import WhisperModel
    except ImportError as e:
        logger.warning(f"Whisper deps not installed ({e}), skipping")
        return None

    if not episode.audio_url and not episode.youtube_video_id:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tmpl = os.path.join(tmpdir, "audio.%(ext)s")
        url = (f"https://www.youtube.com/watch?v={episode.youtube_video_id}"
               if episode.youtube_video_id else episode.audio_url)

        try:
            _download_audio(url, out_tmpl, tmpdir)
        except NotYetLiveError:
            raise
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            return None

        audio_files = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
        if not audio_files:
            return None

        audio_path = os.path.join(tmpdir, audio_files[0])
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            lang = episode.language if episode.language in ("he", "en") else None
            segments, info = model.transcribe(
                audio_path, language=lang, vad_filter=True, beam_size=5,
                condition_on_previous_text=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return TranscriptResult(text, "whisper", info.language, len(text.split()))
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return None


# ── Method 0: Cached transcript from previous run ─────────────────────────────

def try_cached_transcript(episode, transcripts_dir) -> Optional[TranscriptResult]:
    """Load a previously saved transcript file to avoid re-running Whisper."""
    from pathlib import Path
    transcripts_dir = Path(transcripts_dir)
    safe_name = re.sub(r'[^\w\- ]', '_', f"{episode.feed_name} — {episode.title}")[:80]
    cache_path = transcripts_dir / f"{safe_name}.txt"
    if not str(cache_path.resolve()).startswith(str(transcripts_dir.resolve())):
        logger.warning(f"Path traversal blocked for cache read: {safe_name!r}")
        return None
    if not cache_path.exists():
        return None
    try:
        content = cache_path.read_text(encoding="utf-8")
        if "--- TRANSCRIPT ---" not in content:
            return None
        header, _, text = content.partition("--- TRANSCRIPT ---")
        text = text.strip()
        if not text:
            return None
        lang = "auto"
        method = "cached"
        for line in header.splitlines():
            if line.startswith("Language:"):
                lang = line.split(":", 1)[1].strip()
            elif line.startswith("Method:"):
                raw = line.split(":", 1)[1].strip()
                # Avoid double-prefixing if already cached
                method = raw if raw.startswith("cached") else "cached_" + raw
        return TranscriptResult(text, method, lang, len(text.split()))
    except Exception as e:
        logger.debug(f"Cache read failed for {episode.title}: {e}")
        return None


# ── Orchestrator ──────────────────────────────────────────────────────────────

def _fetch_episode_page_once(episode) -> Optional[str]:
    """Fetch the episode's web page a single time, for reuse across
    try_pdf_show_notes and try_page_content. Returns None for audio URLs,
    YouTube watch pages, or missing URLs (neither method would fetch those
    either), or if the request fails."""
    url = episode.url or ""
    if not url or _AUDIO_EXT_RE.search(url) or "youtube.com/watch" in url or "youtu.be/" in url:
        return None
    try:
        return _safe_get_text(url, timeout=20,
                              headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"})
    except Exception as e:
        logger.debug(f"Page fetch failed for {episode.title}: {e}")
        return None


def get_transcript(episode, settings: dict, whisper_count: int = 0,
                   skip_whisper: bool = False,
                   enforce_whisper: bool = False,
                   no_pdf: bool = False,
                   transcripts_dir=None) -> Optional[TranscriptResult]:
    """Returns None if no transcript could be found through any method.

    Raises NotYetLiveError instead if the only failures seen were YouTube
    reporting the video as a premiere/scheduled stream that hasn't started
    broadcasting yet — a transient state the caller should retry later rather
    than treating as a permanent transcript failure."""
    attempted: List[str] = []
    page_text: Optional[str] = None
    page_fetched = False
    not_yet_live = False

    def _get_page_text() -> Optional[str]:
        nonlocal page_text, page_fetched
        if not page_fetched:
            page_text = _fetch_episode_page_once(episode)
            page_fetched = True
        return page_text

    def _return(result: TranscriptResult) -> TranscriptResult:
        result.attempted = attempted[:]
        return result

    def _try_youtube_captions_safe(video_id, language):
        nonlocal not_yet_live
        try:
            return try_youtube_captions(video_id, language)
        except NotYetLiveError:
            not_yet_live = True
            return None

    if transcripts_dir:
        result = try_cached_transcript(episode, transcripts_dir)
        if result:
            logger.info(f"  Transcript via cache ({result.word_count} words)")
            return _return(result)

    if enforce_whisper and episode.youtube_video_id:
        # Even with enforce_whisper, prefer YouTube captions when they are full (≥500 words).
        attempted.append("youtube_captions")
        result = _try_youtube_captions_safe(episode.youtube_video_id, episode.language)
        if result and result.word_count >= 500:
            logger.info(f"  Transcript via youtube_captions (enforce_whisper overridden, {result.word_count} words)")
            attempted.pop()  # succeeded — not a failed attempt
            return _return(result)
        if result:
            logger.info(f"  enforce_whisper: captions too short ({result.word_count} words), falling back to Whisper")
        else:
            logger.info("  enforce_whisper: no captions found, falling back to Whisper")

    if not enforce_whisper:
        if not no_pdf:
            result = try_pdf_show_notes(episode, page_text=_get_page_text())
            if result:
                logger.info(f"  Transcript via pdf_show_notes ({result.word_count} words)")
                return _return(result)
            attempted.append("pdf_show_notes")

        result = try_rss_transcript(episode)
        if result:
            logger.info(f"  Transcript via rss_tag ({result.word_count} words)")
            return _return(result)
        attempted.append("rss_tag")

        if episode.youtube_video_id:
            result = _try_youtube_captions_safe(episode.youtube_video_id, episode.language)
            if result:
                logger.info(f"  Transcript via youtube_captions ({result.word_count} words)")
                return _return(result)
            attempted.append("youtube_captions")

        result = try_page_content(episode, settings.get("description_min_length", 1500), page_text=_get_page_text())
        if result:
            logger.info(f"  Transcript via page_content ({result.word_count} words)")
            return _return(result)
        attempted.append("page_content")

        result = try_description(episode, settings.get("description_min_length", 1500))
        if result:
            logger.info(f"  Transcript via description ({result.word_count} words)")
            return _return(result)
        attempted.append("description")

    if skip_whisper:
        logger.info("  Whisper skipped (test mode)")
        attempted.append("whisper_skipped")
    else:
        max_w = settings.get("max_whisper_per_run", 2)
        if whisper_count >= max_w:
            logger.info(f"  Whisper limit reached ({whisper_count}/{max_w})")
            attempted.append("whisper_limit_reached")
        else:
            attempted.append("whisper")
            try:
                result = try_whisper(episode, settings.get("whisper_model", "small"))
            except NotYetLiveError:
                not_yet_live = True
                result = None
            if result:
                logger.info(f"  Transcript via whisper ({result.word_count} words)")
                attempted.pop()  # succeeded
                return _return(result)

    # Whisper failed or was skipped — fall back to whatever text is available
    logger.info("  Whisper unavailable — falling back to text methods")

    result = try_rss_transcript(episode)
    if result:
        logger.info(f"  Transcript via rss_tag fallback ({result.word_count} words)")
        return _return(result)
    attempted.append("rss_tag")

    if episode.youtube_video_id:
        result = _try_youtube_captions_safe(episode.youtube_video_id, episode.language)
        if result:
            logger.info(f"  Transcript via youtube_captions fallback ({result.word_count} words)")
            return _return(result)
        attempted.append("youtube_captions")

    result = try_page_content(episode, settings.get("description_min_length", 1500), page_text=_get_page_text())
    if result:
        logger.info(f"  Transcript via page_content fallback ({result.word_count} words)")
        return _return(result)
    attempted.append("page_content")

    result = try_description(episode, 50)
    if result:
        logger.info(f"  Transcript via description fallback ({result.word_count} words)")
        return _return(result)
    attempted.append("description")

    if not_yet_live:
        raise NotYetLiveError(f"{episode.title!r} not yet live — no transcript available yet")

    return None
