import re
import html
import logging
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")

_LINK_CHECK_MAX_BYTES = 1 * 1024 * 1024  # 1 MB — enough to find a <title> tag


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
            pass
        return True
    except Exception:
        return False

_HEBREW_RE = re.compile(r"[֐-׿]")
_AUDIO_EXT_RE = re.compile(r'\.(mp3|m4a|ogg|opus|aac|wav|flac)(\?.*)?$', re.IGNORECASE)
_TITLE_RE = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
_SHORTENER_RE = re.compile(
    r'^https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|buff\.ly|'
    r'rebrand\.ly|short\.io|tiny\.cc|is\.gd|cutt\.ly|rb\.gy)/',
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(r'^https?://(?:[^/]*\.)?example\.com', re.IGNORECASE)


def _extract_urls(text: str) -> list:
    return list(dict.fromkeys(_URL_RE.findall(text)))


def _resolve_and_check(url: str) -> tuple[str, str] | None:
    """Fetch url, follow redirects, check liveness. Returns (final_url, title) or None if dead."""
    if _AUDIO_EXT_RE.search(url):
        return None
    if _EXAMPLE_RE.match(url):
        return None
    if not _is_safe_url(url):
        return None
    try:
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"},
                         allow_redirects=True, stream=True)
        if r.status_code >= 400:
            r.close()
            return None
        final_url = r.url
        # Read only enough bytes to find the <title> tag
        chunks = []
        total = 0
        for chunk in r.iter_content(4096):
            chunks.append(chunk)
            total += len(chunk)
            if total >= _LINK_CHECK_MAX_BYTES:
                break
        r.close()
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        title = ""
        m = _TITLE_RE.search(raw[:4096])
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            title = re.sub(r"<[^>]+>", "", title).strip()
            title = html.unescape(title)
            title = title[:120]
        return (final_url, title)
    except Exception:
        return None


def _enrich_urls(urls: list) -> list[tuple[str, str]]:
    """Return list of (final_url, title) for live URLs only, fetched in parallel.
    Dead links, example.com URLs, and audio files are dropped.
    Shortener URLs are resolved to their final destination."""
    results: dict[str, tuple[str, str] | None] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_to_url = {ex.submit(_resolve_and_check, u): u for u in urls}
        try:
            for future in as_completed(future_to_url, timeout=20):
                orig_url = future_to_url[future]
                try:
                    results[orig_url] = future.result()
                except Exception:
                    results[orig_url] = None
        except Exception:
            pass
    # Preserve original order, drop dead links
    return [(final_url, title) for u in urls
            if (r := results.get(u)) is not None
            for final_url, title in [r]]


_TIMESTAMP_RE = re.compile(
    r'\[?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]?'   # [00:51], [1:04:30], 00:51
    r'|\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)'     # (00:51)
)


def _clean_text(text: str, strip_urls: bool = False) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[SHOW NOTES[^\]]*\]", "", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    if strip_urls:
        text = _URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_mostly_hebrew(text: str) -> bool:
    if not text:
        return False
    letters = re.findall(r"[a-zA-Z֐-׿]", text)
    if not letters:
        return False
    return sum(1 for c in letters if _HEBREW_RE.match(c)) / len(letters) > 0.5


def _extractive_summary(text: str, max_sentences: int = 15, max_chars: int = 5000) -> str:
    clean = _clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return clean[:max_chars]
    if len(sentences) <= max_sentences:
        result = " ".join(sentences)
    else:
        head = sentences[: max_sentences - 2]
        tail = sentences[-2:]
        result = " ".join(head) + " [...] " + " ".join(tail)
    return result[:max_chars]


# ── BART + Helsinki models (requires torch + transformers) ─────────────────────

def _bart_summarize(text: str, settings: dict) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    chunk_size = settings.get("bart_chunk_words", 800)
    words = text.split()
    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

    chunk_summaries = []
    for chunk in chunks[:8]:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = model.generate(**inputs, max_length=500, min_length=80, num_beams=4)
        chunk_summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    combined = " ".join(chunk_summaries)
    if len(chunk_summaries) > 1:
        inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = model.generate(**inputs, max_length=800, min_length=150, num_beams=4)
        combined = tokenizer.decode(ids[0], skip_special_tokens=True)
    return combined


def _translate_he_to_en(text: str) -> str:
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()

    words = text.split()
    chunks = [" ".join(words[i: i + 400]) for i in range(0, min(len(words), 3200), 400)]
    parts = []
    for c in chunks:
        inputs = tokenizer(c, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            ids = model.generate(**inputs, max_length=512)
        parts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts)


def _translate_en_to_he(text: str) -> str:
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    model_name = "Helsinki-NLP/opus-mt-en-he"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()

    # Translate sentence-by-sentence to avoid per-sequence length truncation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        sentences = [text]
    parts = []
    for s in sentences:
        inputs = tokenizer([s], return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        parts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts)


_PRE_EXTRACT_HE_WORDS = 1500   # max words to translate (he→en)
_PRE_EXTRACT_EN_WORDS = 4000   # max words to feed into BART

_SUMMARY_PROMPT = """\
You are summarizing a podcast episode. Write a detailed Hebrew summary.

IMPORTANT RULES:
- Keep ALL English tech terms as-is (product names, company names, tools, frameworks, acronyms like AI, AGI, SaaS, API, etc.)
- Summary must be LONG and DETAILED (800-1200 words) — cover every topic discussed
- Use bold section headers (**כותרת**) and bullet points
- Include all numbers, statistics, names, and specific claims made
- Do NOT skip any technological, business, or product topics
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.) — focus only on what was discussed in THIS episode
- Do NOT include the podcast host/owner's own biography, credentials, or company description (his standard intro about himself) — only summarize content actually discussed in the episode, and any biographical info about guests
- Do NOT use hashtags (words starting with #) anywhere. If a keyword is worth mentioning, write it as a normal word with no "#"
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned in the episode" — this applies no matter how that heading is phrased or reworded (Hebrew or English). If the last thing you write is a heading followed by a list of links/topics with no new analysis, delete that heading entirely and instead weave each link into the sentence of the paragraph where that topic was actually discussed
- Write ONLY Hebrew text (except for English tech terms that must stay in English)

Cover EVERY subject: technology topics, business models, products, companies, people mentioned, arguments made, predictions, and all links/resources. 800-1200 words.

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
[your Hebrew summary here]

Episode: {title}
Podcast: {feed_name}

Transcript:
{transcript}"""


_SUMMARY_PROMPT_LONG = """\
You are summarizing a podcast episode that has full, detailed show notes. Write a comprehensive Hebrew summary.

IMPORTANT RULES:
- Keep ALL English tech terms as-is (product names, company names, tools, frameworks, acronyms like AI, AGI, SaaS, API, etc.)
- Summary must be COMPREHENSIVE (1200-1500 words) — cover every topic, detail, and nuance
- Use bold section headers (**כותרת**) and bullet points
- Include all numbers, statistics, names, CVEs, vulnerabilities, tools, and specific claims made
- Do NOT skip any technological, business, security, or product topics
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.) — focus only on what was discussed in THIS episode
- Do NOT include the podcast host/owner's own biography, credentials, or company description (his standard intro about himself) — only summarize content actually discussed in the episode, and any biographical info about guests
- Since this is based on full show notes, be especially thorough and complete
- Do NOT use hashtags (words starting with #) anywhere. If a keyword is worth mentioning, write it as a normal word with no "#"
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned in the episode" — this applies no matter how that heading is phrased or reworded (Hebrew or English). If the last thing you write is a heading followed by a list of links/topics with no new analysis, delete that heading entirely and instead weave each link into the sentence of the paragraph where that topic was actually discussed
- Write ONLY Hebrew text (except for English tech terms that must stay in English)

Cover EVERY subject in depth. 1200-1500 words.

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
[your Hebrew summary here]

Episode: {title}
Podcast: {feed_name}

Show Notes:
{transcript}"""


_LOCAL_LLM_WORD_LIMIT = 3000  # ~4k tokens input, leaves room for a 1200-word output in a 8k context

_REFUSAL_PHRASES = (
    "i'm sorry",
    "i am sorry",
    "too long",
    "falls outside",
    "cannot process",
    "could you provide",
    "please provide",
    "exceeds",
)


def _is_refusal(text: str) -> bool:
    """Return True if the model returned an apology/refusal instead of a summary."""
    lower = text.lower()
    return (
        "HEBREW_SUMMARY:" not in text
        and any(phrase in lower for phrase in _REFUSAL_PHRASES)
    )


def _is_degenerate_repetition(text: str, min_sentences: int = 6) -> bool:
    """Return True if the text is dominated by a small model looping the same
    sentence(s) — a common failure mode under long-context summarization."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
    if len(sentences) < min_sentences:
        return False
    return len(set(sentences)) / len(sentences) < 0.6


_LOCAL_LLM_MODEL = "DictaLM2.0-Instruct"
_LOCAL_LLM_REPO = "dicta-il/dictalm2.0-instruct-GGUF"
_LOCAL_LLM_FILE = "dictalm2.0-instruct.Q4_K_M.gguf"

_llm_instance = None


def _get_local_llm():
    """Load (and cache in-process) the local GGUF model via llama-cpp-python."""
    global _llm_instance
    if _llm_instance is None:
        from llama_cpp import Llama
        _llm_instance = Llama.from_pretrained(
            repo_id=_LOCAL_LLM_REPO,
            filename=_LOCAL_LLM_FILE,
            n_ctx=8192,
            n_threads=2,
            verbose=False,
        )
    return _llm_instance


def _summarize_with_local_llm(episode, text: str, long_summary: bool = False) -> tuple:
    """Returns (hebrew_summary, english_summary, steps) using a local GGUF model
    (DictaLM2.0-Instruct via llama-cpp-python) — no network calls, no API key."""
    llm = _get_local_llm()

    prompt_tpl = _SUMMARY_PROMPT_LONG if long_summary else _SUMMARY_PROMPT
    words = text.split()
    result = ""

    for attempt, limit in enumerate([_LOCAL_LLM_WORD_LIMIT, _LOCAL_LLM_WORD_LIMIT // 2, _LOCAL_LLM_WORD_LIMIT // 4]):
        truncated = " ".join(words[:limit]) if len(words) > limit else text
        prompt = prompt_tpl.format(
            title=episode.title,
            feed_name=episode.feed_name,
            transcript=truncated,
        )
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
            repeat_penalty=1.15,
        )
        candidate = response["choices"][0]["message"]["content"] or ""
        if _is_refusal(candidate):
            logger.warning(
                f"  Local LLM refused (attempt {attempt + 1}, "
                f"{len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        parsed_he = candidate.split("HEBREW_SUMMARY:", 1)[1].strip() if "HEBREW_SUMMARY:" in candidate else ""
        if re.match(r'^\s*\[[^\]]{0,200}\]\s*$', parsed_he) or re.match(r'^\s*<[^>]{0,200}>\s*$', parsed_he):
            logger.warning(f"  Local LLM returned a placeholder — retrying with fewer words")
            continue
        if len(parsed_he) < 50:
            logger.warning(
                f"  Local LLM returned an empty/too-short summary "
                f"(attempt {attempt + 1}, {len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        if _is_degenerate_repetition(parsed_he):
            logger.warning(
                f"  Local LLM output degenerated into repetition "
                f"(attempt {attempt + 1}, {len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        logger.info(f"  Local LLM: used {_LOCAL_LLM_MODEL} ({len(truncated.split())} words)")
        result = candidate
        break

    if not result:
        raise RuntimeError("All local LLM attempts failed")

    if "HEBREW_SUMMARY:" in result:
        hebrew_summary = result.split("HEBREW_SUMMARY:", 1)[1].strip()
    else:
        hebrew_summary = result.strip()

    return hebrew_summary, "", [(f"Summary: {_LOCAL_LLM_MODEL} (he)", "summary")]


def _summarize_with_models(episode, transcript_text: str, lang: str, settings: dict,
                           long_summary: bool = False) -> tuple:
    """Returns (hebrew_summary, english_summary, pipeline_steps).
    pipeline_steps is a list of (text, category) tuples; category is one of
    "transcript", "summary", "translate", "debug". The telegram output drops
    "summary" and "debug" steps.
    Tries the local GGUF LLM first; falls back to BART+Helsinki if that fails."""
    try:
        text = _clean_text(transcript_text, strip_urls=False)
        return _summarize_with_local_llm(episode, text, long_summary)
    except Exception as e:
        logger.warning(f"  Local LLM unavailable ({type(e).__name__}: {e}), falling back to BART+Helsinki")

    # ── Fallback: BART + Helsinki (local LLM failed to load/run) ───────────────
    steps = []
    text = _clean_text(transcript_text, strip_urls=True)

    if lang in ("he", "iw"):
        pre_words = len(text.split())
        en_input = text
        if pre_words > _PRE_EXTRACT_HE_WORDS:
            en_input = _extractive_summary(text, max_sentences=40,
                                           max_chars=_PRE_EXTRACT_HE_WORDS * 7)
            steps.append((f"Pre-extract for translation: {pre_words}→{len(en_input.split())} words", "translate"))
        en_text = _translate_he_to_en(en_input)
        steps.append(("Translate: he→en (Helsinki opus-mt-tc-big-he-en)", "translate"))
        n_chunks = max(1, len(en_text.split()) // settings.get("bart_chunk_words", 800))
        english_summary = _bart_summarize(en_text, settings)
        steps.append((f"English summary: BART facebook/bart-large-cnn ({n_chunks} chunks)", "summary"))
        hebrew_summary = _translate_en_to_he(english_summary)
        steps.append(("Hebrew summary: BART → translate en→he (Helsinki opus-mt-en-he)", "summary"))
        return hebrew_summary, english_summary, steps

    else:
        pre_words = len(text.split())
        if pre_words > _PRE_EXTRACT_EN_WORDS:
            text = _extractive_summary(text, max_sentences=80,
                                       max_chars=_PRE_EXTRACT_EN_WORDS * 6)
            steps.append((f"Pre-extract: {pre_words}→{len(text.split())} words", "translate"))
        n_chunks = max(1, len(text.split()) // settings.get("bart_chunk_words", 800))
        english_summary = _bart_summarize(text, settings)
        steps.append((f"English summary: BART facebook/bart-large-cnn ({n_chunks} chunks)", "summary"))
        hebrew_summary = _translate_en_to_he(english_summary)
        steps.append(("Translate: en→he (Helsinki opus-mt-en-he)", "translate"))
        return hebrew_summary, english_summary, steps


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_output(episode, hebrew_summary: str, english_summary: str,
                   urls: list, pipeline_steps: list) -> tuple[str, str]:
    """Returns (full_text, telegram_text). full_text goes to results.txt.md;
    telegram_text omits English summary and original description."""
    from datetime import datetime, timezone

    # Clear English if it came out as Hebrew (extractive/model error)
    if english_summary and _is_mostly_hebrew(english_summary):
        english_summary = ""

    desc_clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", episode.description or "")).strip()

    url_block = ""
    if urls:
        enriched = _enrich_urls(urls[:20])
        lines = []
        _SKIP_TITLES = {"privacy faq", "privacy policy", "just a moment..."}
        for u, title in enriched:
            if title.lower() in _SKIP_TITLES:
                continue
            lines.append(f"• [{title}]({u})" if title else f"• {u}")
        url_block = "\n\n**Links mentioned:**\n" + "\n".join(lines)

    steps_block = "\n".join(f"  • {text}" for text, _category in pipeline_steps)
    telegram_steps_block = "\n".join(
        f"  • {text}" for text, category in pipeline_steps
        if category not in ("summary", "debug")
    )
    date_str = episode.published.strftime("%d/%m/%Y %H:%M") + " UTC"
    generated_str = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M") + " UTC"
    desc_block = f"\n**Original description:**  \n{desc_clean[:600]}" if desc_clean else ""

    he_block = f"**Hebrew Summary:**  \n{hebrew_summary}\n\n" if hebrew_summary else ""
    en_block = f"**English Summary:**  \n{english_summary}\n\n" if english_summary else ""

    source_label = "Youtube Channel" if episode.feed_type == "youtube_rss" else "Podcast"
    header = (
        f"**{episode.feed_name}** [ {episode.author} ]\n\n"
        f"**{episode.title}**  \n\n"
        f"[{source_label}]\n{date_str}\n{generated_str} [Generated]  \n"
        f"\n---\n\n"
    )
    footer = (
        f"{url_block}\n\n"
        f"---\n\n"
        f"**Link:**\n{episode.url}\n\n"
        f"*Pipeline:*\n{steps_block}\n"
    )
    telegram_footer = (
        f"---\n\n"
        f"**Link:**\n{episode.url}\n\n"
        f"*Pipeline:*\n{telegram_steps_block}\n"
    )

    full_text = header + he_block + en_block + desc_block + "\n" + footer
    telegram_text = header + he_block + telegram_footer
    return full_text, telegram_text


# ── Public API ────────────────────────────────────────────────────────────────

def summarize_episode(episode, transcript, settings: dict) -> tuple[str, str]:
    lang = transcript.language or episode.language
    raw_text = transcript.text
    urls = _extract_urls(raw_text) + _extract_urls(episode.description or "")
    urls = list(dict.fromkeys(urls))

    method = transcript.method
    if "whisper" in method:
        audio_note = "Full audio file transcribed (Whisper)"
    elif method.startswith("youtube_captions"):
        audio_note = "No audio download — YouTube captions used"
    elif method == "rss_tag":
        audio_note = "No audio download — transcript from RSS feed"
    elif method == "pdf_show_notes":
        audio_note = "No audio download — summary based on PDF show notes"
    else:
        audio_note = "No audio download — summary based on show notes / description only"

    transcript_step = f"Transcript: {method} ({transcript.word_count} words, lang={lang}) — {audio_note}"
    pipeline_steps = [(transcript_step, "transcript")]
    if getattr(transcript, "attempted", []):
        pipeline_steps.append((f"Tried and failed: {', '.join(transcript.attempted)}", "debug"))

    is_pdf = method == "pdf_show_notes"
    try:
        hebrew_summary, english_summary, model_steps = _summarize_with_models(
            episode, raw_text, lang, settings, long_summary=is_pdf)
        pipeline_steps.extend(model_steps)
    except Exception as e:
        logger.warning(f"Model pipeline unavailable ({type(e).__name__}: {e}), using extractive fallback")
        max_sent = settings.get("extractive_max_sentences", 15)
        extracted = _extractive_summary(_clean_text(raw_text, strip_urls=True), max_sentences=max_sent, max_chars=5000)
        if lang in ("he", "iw"):
            hebrew_summary = f"[Extractive summary]\n\n{extracted}"
            english_summary = ""
        else:
            hebrew_summary = ""
            english_summary = f"[Extractive summary]\n\n{extracted}"
        pipeline_steps.append((f"Summary: extractive ({max_sent} sentences, BART unavailable: {type(e).__name__})", "summary"))
        pipeline_steps.append(("תרגום: — (לא בוצע)", "translate"))

    return _format_output(episode, hebrew_summary, english_summary, urls, pipeline_steps)
