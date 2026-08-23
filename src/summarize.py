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
You are summarizing a podcast episode. Write a detailed summary IN ENGLISH, regardless of what language the transcript is in.

IMPORTANT RULES:
- Write the summary in English only, even if the transcript below is in Hebrew or another language
- Keep all product names, company names, tools, frameworks, and acronyms (AI, AGI, SaaS, API, etc.) as they appear
- Summary must be {length_instr} — cover every topic discussed
- Use bold section headers (**Heading**) and bullet points
- Include all numbers, statistics, names, and specific claims made
- Do NOT skip any technological, business, or product topics
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.) — focus only on what was discussed in THIS episode
- Do NOT include the podcast host/owner's own biography, credentials, or company description (his standard intro about himself) — only summarize content actually discussed in the episode, and any biographical info about guests
- Do NOT use hashtags (words starting with #) anywhere. If a keyword is worth mentioning, write it as a normal word with no "#"
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned in the episode". If the last thing you write is a heading followed by a list of links/topics with no new analysis, delete that heading entirely and instead weave each link into the sentence of the paragraph where that topic was actually discussed
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information.

ACCURACY RULE (highest priority, never break this):
- Only state facts, names, numbers, quotes, and claims that are literally present in the transcript below. Never invent, guess, extrapolate, or "fill in" a name, company, statistic, or detail that is not actually there, even if it would sound plausible.
- If you are not confident about a specific name, number, or detail because the transcript is unclear or garbled at that point, simply omit that detail instead of guessing at it or inventing a plausible-sounding substitute.
- Do NOT attribute a claim, quote, or fact to a person or company unless the transcript clearly says they made it.

Cover EVERY subject: technology topics, business models, products, companies, people mentioned, arguments made, predictions, and all links/resources. {length_instr}.

Respond EXACTLY in this format (no extra text before or after):
ENGLISH_SUMMARY:
[your English summary here]

Episode: {title}
Podcast: {feed_name}

Transcript:
{transcript}"""


_SUMMARY_PROMPT_LONG = """\
You are summarizing a podcast episode that has full, detailed show notes. Write a comprehensive summary IN ENGLISH, regardless of what language the show notes are in.

IMPORTANT RULES:
- Write the summary in English only, even if the show notes below are in Hebrew or another language
- Keep all product names, company names, tools, frameworks, and acronyms (AI, AGI, SaaS, API, etc.) as they appear
- Summary must be COMPREHENSIVE (1200-1500 words) — cover every topic, detail, and nuance
- Use bold section headers (**Heading**) and bullet points
- Include all numbers, statistics, names, CVEs, vulnerabilities, tools, and specific claims made
- Do NOT skip any technological, business, security, or product topics
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.) — focus only on what was discussed in THIS episode
- Do NOT include the podcast host/owner's own biography, credentials, or company description (his standard intro about himself) — only summarize content actually discussed in the episode, and any biographical info about guests
- Since this is based on full show notes, be especially thorough and complete
- Do NOT use hashtags (words starting with #) anywhere. If a keyword is worth mentioning, write it as a normal word with no "#"
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned in the episode". If the last thing you write is a heading followed by a list of links/topics with no new analysis, delete that heading entirely and instead weave each link into the sentence of the paragraph where that topic was actually discussed
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information.

ACCURACY RULE (highest priority, never break this):
- Only state facts, names, numbers, quotes, and claims that are literally present in the show notes below. Never invent, guess, extrapolate, or "fill in" a name, company, statistic, or detail that is not actually there, even if it would sound plausible.
- If you are not confident about a specific name, number, or detail because the source is unclear at that point, simply omit that detail instead of guessing at it or inventing a plausible-sounding substitute.
- Do NOT attribute a claim, quote, or fact to a person or company unless the source clearly says they made it.

Cover EVERY subject in depth. 1200-1500 words.

Respond EXACTLY in this format (no extra text before or after):
ENGLISH_SUMMARY:
[your English summary here]

Episode: {title}
Podcast: {feed_name}

Show Notes:
{transcript}"""


_CHUNK_SUMMARY_PROMPT = """\
You are summarizing PART {part} OF {total} of a longer podcast transcript. Write detailed notes IN ENGLISH covering everything discussed in this part only, regardless of what language the transcript is in.

IMPORTANT RULES:
- Write the notes in English only, even if the transcript below is in Hebrew or another language
- Keep all product names, company names, tools, frameworks, and acronyms as they appear
- Include all numbers, statistics, names, and specific claims made in this part
- Do NOT summarize or refer to other parts — only what appears in THIS transcript segment
- Do NOT include generic podcast/channel descriptions, host biography, or subscription/social-media info
- This is a working note, not a final summary — plain prose is fine, no need for headers
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information.
- ACCURACY: only state facts, names, numbers, and claims literally present in this transcript segment. Never invent or guess a name, company, or statistic. If unsure about a detail, omit it rather than guessing.

Respond EXACTLY in this format (no extra text before or after):
NOTES:
[your English notes here]

Episode: {title}
Podcast: {feed_name}

Transcript (part {part} of {total}):
{transcript}"""


_COMBINE_SUMMARY_PROMPT = """\
You are given English notes covering different parts of the same podcast episode, in order. Combine them into one detailed, coherent English summary of the whole episode.

IMPORTANT RULES:
- Write in English only
- Keep all product names, company names, tools, frameworks, and acronyms (AI, AGI, SaaS, API, etc.) as they appear
- Summary must be LONG and DETAILED (800-1200 words) — cover every topic discussed across all parts, in order
- Use bold section headers (**Heading**) and bullet points
- Include all numbers, statistics, names, and specific claims made
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.)
- Do NOT include the podcast host/owner's own biography or company description — only content actually discussed
- Do NOT use hashtags (words starting with #) anywhere
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned" — weave each link into the sentence of the paragraph where that topic was discussed
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information.
- ACCURACY: only combine facts, names, numbers, and claims that literally appear in the notes below. Never invent, guess, or add a name, company, or statistic that isn't in the notes, even if it would sound plausible. If a note is unclear or ambiguous, omit that detail rather than guessing at it.

Respond EXACTLY in this format (no extra text before or after):
ENGLISH_SUMMARY:
[your English summary here]

Episode: {title}
Podcast: {feed_name}

Notes from all parts, in order:
{transcript}"""


_TRANSLATE_TO_HEBREW_PROMPT = """\
Translate the following English podcast summary into Hebrew. This is a translation task only — do not summarize further, do not add or remove information, translate the full text faithfully.

RULES:
- Translate into natural, fluent Hebrew
- Keep product names, company names, tools, frameworks, and acronyms (AI, AGI, SaaS, API, etc.) in English, exactly as they appear
- Keep the same structure: headings, bullet points, paragraph breaks
- Write ONLY in Hebrew script and English tech terms. NEVER use Chinese, Russian, Arabic, or any other script — not even one character
- Do NOT write full English sentences or paragraphs anywhere in the output. The ONLY English allowed is the product/company/tool names and acronyms that were already in English in the source text. Every sentence of prose must be in Hebrew.
- This is a literal translation, not a re-summary: do NOT add, invent, or guess any name, number, company, or claim that is not already present in the English summary below
- Do NOT repeat any sentence more than once

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
[the Hebrew translation here]

English summary to translate:
{transcript}"""


# ── Direct-Hebrew prompts (used when the transcript is already mostly Hebrew,
# skipping the English-summarize-then-translate round trip) ────────────────────

_HEBREW_SUMMARY_PROMPT = """\
You are summarizing a podcast episode. Write a detailed Hebrew summary.

LANGUAGE RULE (highest priority, never break this):
- Write ONLY in Hebrew script and English tech terms. NEVER use Chinese, Russian, Arabic, or any other script — not even one character. If you notice yourself writing a non-Hebrew, non-English character, stop and rewrite that word in Hebrew instead.
- Do NOT write full English sentences, clauses, or phrases anywhere in the output. The ONLY English allowed is individual product names, company names, tools, frameworks, and acronyms (AI, AGI, SaaS, API, etc.) embedded inside an otherwise-Hebrew sentence — never a run of ordinary English words like "for example" or "the company said".
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information. If you find yourself about to write something you already said, stop and move to the next topic instead.
- Do NOT add an English translation, gloss, or restatement of any Hebrew sentence — not in parentheses, not on a new line, not anywhere. Write each idea in Hebrew exactly once and move on. The ONLY English allowed is product names, company names, tools, and acronyms embedded naturally inside a Hebrew sentence.

ACCURACY RULE (highest priority, never break this):
- Only state facts, names, numbers, quotes, and claims that are literally present in the transcript below. Never invent, guess, extrapolate, or "fill in" a name, company, statistic, or technical term that is not actually there, even if it would sound plausible or fluent in Hebrew.
- If a name, number, or term in the transcript is unclear, garbled, or ambiguous, do NOT invent a plausible-sounding Hebrew or English substitute for it — omit that specific detail and move on to the next topic.
- Do NOT attribute a claim, quote, or fact to a person or company unless the transcript clearly says they made it.

IMPORTANT RULES:
- Keep ALL English tech terms as-is (product names, company names, tools, frameworks, acronyms like AI, AGI, SaaS, API, etc.)
- Summary must be {length_instr} — cover every topic discussed
- Use bold section headers (**כותרת**) and bullet points
- Include all numbers, statistics, names, and specific claims made
- Do NOT skip any technological, business, or product topics
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.) — focus only on what was discussed in THIS episode
- Do NOT include the podcast host/owner's own biography, credentials, or company description (his standard intro about himself) — only summarize content actually discussed in the episode, and any biographical info about guests
- Do NOT use hashtags (words starting with #) anywhere. If a keyword is worth mentioning, write it as a normal word with no "#"
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned in the episode" — this applies no matter how that heading is phrased or reworded (Hebrew or English). If the last thing you write is a heading followed by a list of links/topics with no new analysis, delete that heading entirely and instead weave each link into the sentence of the paragraph where that topic was actually discussed

Cover EVERY subject: technology topics, business models, products, companies, people mentioned, arguments made, predictions, and all links/resources. {length_instr}.

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
[your Hebrew summary here, in Hebrew script only]

Episode: {title}
Podcast: {feed_name}

Transcript:
{transcript}"""


_HEBREW_SUMMARY_PROMPT_LONG = """\
You are summarizing a podcast episode that has full, detailed show notes. Write a comprehensive Hebrew summary.

LANGUAGE RULE (highest priority, never break this):
- Write ONLY in Hebrew script and English tech terms. NEVER use Chinese, Russian, Arabic, or any other script — not even one character. If you notice yourself writing a non-Hebrew, non-English character, stop and rewrite that word in Hebrew instead.
- Do NOT write full English sentences, clauses, or phrases anywhere in the output. The ONLY English allowed is individual product names, company names, tools, frameworks, and acronyms (AI, AGI, SaaS, API, etc.) embedded inside an otherwise-Hebrew sentence — never a run of ordinary English words like "for example" or "the company said".
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information. If you find yourself about to write something you already said, stop and move to the next topic instead.
- Do NOT add an English translation, gloss, or restatement of any Hebrew sentence — not in parentheses, not on a new line, not anywhere. Write each idea in Hebrew exactly once and move on. The ONLY English allowed is product names, company names, tools, and acronyms embedded naturally inside a Hebrew sentence.

ACCURACY RULE (highest priority, never break this):
- Only state facts, names, numbers, quotes, and claims that are literally present in the show notes below. Never invent, guess, extrapolate, or "fill in" a name, company, statistic, or technical term that is not actually there, even if it would sound plausible or fluent in Hebrew.
- If a name, number, or term in the source is unclear, garbled, or ambiguous, do NOT invent a plausible-sounding Hebrew or English substitute for it — omit that specific detail and move on to the next topic.
- Do NOT attribute a claim, quote, or fact to a person or company unless the source clearly says they made it.

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

Cover EVERY subject in depth. 1200-1500 words.

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
[your Hebrew summary here, in Hebrew script only]

Episode: {title}
Podcast: {feed_name}

Show Notes:
{transcript}"""


_HEBREW_CHUNK_SUMMARY_PROMPT = """\
You are summarizing PART {part} OF {total} of a longer podcast transcript. Write detailed notes in Hebrew covering everything discussed in this part only.

LANGUAGE RULE (highest priority, never break this):
- Write ONLY in Hebrew script and English tech terms. NEVER use Chinese, Russian, Arabic, or any other script — not even one character.
- Do NOT write full English sentences, clauses, or phrases anywhere in the output. The ONLY English allowed is individual product names, company names, tools, frameworks, and acronyms embedded inside an otherwise-Hebrew sentence.
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information.
- Do NOT add an English translation, gloss, or restatement of any Hebrew sentence — not in parentheses, not on a new line, not anywhere. Write each idea in Hebrew exactly once.

ACCURACY RULE (highest priority, never break this):
- Only state facts, names, numbers, and claims literally present in this transcript segment. Never invent or guess a name, company, statistic, or term that is not actually there.
- If a name, number, or term is unclear or garbled, omit that specific detail rather than inventing a plausible-sounding substitute.

IMPORTANT RULES:
- Keep ALL English tech terms as-is (product names, company names, tools, frameworks, acronyms)
- Include all numbers, statistics, names, and specific claims made in this part
- Do NOT summarize or refer to other parts — only what appears in THIS transcript segment
- Do NOT include generic podcast/channel descriptions, host biography, or subscription/social-media info
- This is a working note, not a final summary — plain prose is fine, no need for headers

Respond EXACTLY in this format (no extra text before or after):
NOTES:
[your Hebrew notes here, in Hebrew script only]

Episode: {title}
Podcast: {feed_name}

Transcript (part {part} of {total}):
{transcript}"""


_HEBREW_COMBINE_SUMMARY_PROMPT = """\
You are given Hebrew notes covering different parts of the same podcast episode, in order. Combine them into one detailed, coherent Hebrew summary of the whole episode.

LANGUAGE RULE (highest priority, never break this):
- Write ONLY in Hebrew script and English tech terms. NEVER use Chinese, Russian, Arabic, or any other script — not even one character.
- Do NOT write full English sentences, clauses, or phrases anywhere in the output. The ONLY English allowed is individual product names, company names, tools, frameworks, and acronyms embedded inside an otherwise-Hebrew sentence.
- Do NOT repeat the same sentence, phrase, or idea more than once. Every sentence must add new information.
- Do NOT add an English translation, gloss, or restatement of any Hebrew sentence — not in parentheses, not on a new line, not anywhere. Write each idea in Hebrew exactly once and move on. The ONLY English allowed is product names, company names, tools, and acronyms embedded naturally inside a Hebrew sentence.

ACCURACY RULE (highest priority, never break this):
- Only combine facts, names, numbers, and claims that literally appear in the notes below. Never invent, guess, or add a name, company, statistic, or term that isn't in the notes, even if it would sound plausible. If a note is unclear or ambiguous, omit that detail rather than guessing at it.
- Do NOT attribute a claim, quote, or fact to a person or company unless the notes clearly say they made it.

IMPORTANT RULES:
- Keep ALL English tech terms as-is (product names, company names, tools, frameworks, acronyms like AI, AGI, SaaS, API, etc.)
- Summary must be LONG and DETAILED (800-1200 words) — cover every topic discussed across all parts, in order
- Use bold section headers (**כותרת**) and bullet points
- Include all numbers, statistics, names, and specific claims made
- Do NOT include generic descriptions of the podcast/channel itself (its mission, social links, subscription info, follow us on X/Facebook/TikTok etc.)
- Do NOT include the podcast host/owner's own biography or company description — only content actually discussed
- Do NOT use hashtags (words starting with #) anywhere
- Never close the summary with a standalone heading whose sole purpose is to list links, sources, or "additional things mentioned" — weave each link into the sentence of the paragraph where that topic was discussed

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
[your Hebrew summary here, in Hebrew script only]

Episode: {title}
Podcast: {feed_name}

Notes from all parts, in order:
{transcript}"""


_LOCAL_LLM_N_CTX = 8192
_LOCAL_LLM_WORD_LIMIT = 3000  # fallback word-based slice size, only used if tokenizing fails

# Per-call-type output caps — right-sized to the actual target length of each
# step instead of one generous ceiling, since generation time on this 2-core
# runner scales with max_tokens even when the model would stop earlier on its own.
_MAX_TOKENS_CHUNK_NOTES = 900     # working notes, ~400-650 words observed in practice
_MAX_TOKENS_SUMMARY = 2048        # final summary/combine, up to 1500-word target (~2000-2200 tokens)
_MAX_TOKENS_TRANSLATE_MARGIN = 1.4  # Hebrew translation output budget = input tokens * this margin

# Safety margin subtracted from n_ctx before computing how many prompt tokens
# fit, covering the chat template's own tokens (role markers, BOS/EOS, etc.)
# added on top of the raw text by the tokenizer/template.
_PROMPT_TOKEN_MARGIN = 200


def _truncate_to_token_budget(llm, text: str, max_prompt_tokens: int) -> str:
    """Truncate `text` (word-boundary-safe) so it tokenizes to at most
    max_prompt_tokens under the model's actual tokenizer. Word count is a
    poor proxy for token count — Hebrew text tokenizes far less efficiently
    than English in Gemma3, so a fixed word limit that's safe for English
    can silently overflow the context window for Hebrew. Falls back to
    _LOCAL_LLM_WORD_LIMIT-based truncation if tokenization itself fails."""
    try:
        token_count = len(llm.tokenize(text.encode("utf-8")))
        if token_count <= max_prompt_tokens:
            return text
        words = text.split()
        # Binary search for the longest word-prefix that fits the budget.
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = " ".join(words[:mid])
            if len(llm.tokenize(candidate.encode("utf-8"))) <= max_prompt_tokens:
                lo = mid
            else:
                hi = mid - 1
        return " ".join(words[:lo])
    except Exception as e:
        logger.debug(f"Tokenization failed ({type(e).__name__}: {e}), falling back to word-count truncation")
        words = text.split()
        return " ".join(words[:_LOCAL_LLM_WORD_LIMIT]) if len(words) > _LOCAL_LLM_WORD_LIMIT else text

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


_NON_HEBREW_SCRIPT_RE = re.compile(r"[一-鿿぀-ヿ가-힣Ѐ-ӿ؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

# A tech-term/acronym list ("Google Cloud, Kubernetes, and OpenAI API") can
# contain several consecutive Latin words without being English prose. Real
# English sentences are instead dense in short function words (the, is, that,
# with, ...) — so flag a window of Latin words only if enough of them are
# function words, which a list of product/company names never has.
_ENGLISH_FUNCTION_WORDS = {
    "the", "is", "are", "was", "were", "and", "that", "this", "with", "for",
    "have", "has", "had", "not", "but", "you", "your", "they", "their",
    "what", "when", "which", "from", "about", "into", "than", "then",
    "also", "because", "while", "these", "those", "there", "been", "being",
    "will", "would", "could", "should", "can", "its", "so", "if", "of",
    "to", "a", "an", "as", "on", "in", "at",
}
_LATIN_WORD_RE = re.compile(r"[A-Za-z']+")
_ENGLISH_PROSE_WINDOW = 15
_ENGLISH_PROSE_MIN_HITS = 5


def _has_wrong_script(text: str, max_chars: int = 0) -> bool:
    """Return True if the text contains Chinese/Japanese/Korean/Cyrillic/Arabic
    characters — a sign the model code-switched out of Hebrew under long-context load.
    Even a single stray character (e.g. an Arabic letter dropped into a name like
    "ד"ר רالف") is a real failure, not noise — so this has zero tolerance by default."""
    return len(_NON_HEBREW_SCRIPT_RE.findall(text)) > max_chars


def _has_english_prose_run(text: str) -> bool:
    """Return True if the text contains a run of Latin-script words dense enough
    in English function words to be real English prose — a sign the model wrote
    full English sentences instead of Hebrew prose with embedded tech terms.
    A run of product/company names and acronyms is allowed (no function words);
    an actual English sentence or clause is not."""
    words = _LATIN_WORD_RE.findall(text)
    if len(words) < _ENGLISH_PROSE_WINDOW:
        return False
    for i in range(len(words) - _ENGLISH_PROSE_WINDOW + 1):
        window = words[i:i + _ENGLISH_PROSE_WINDOW]
        hits = sum(1 for w in window if w.lower() in _ENGLISH_FUNCTION_WORDS)
        if hits >= _ENGLISH_PROSE_MIN_HITS:
            return True
    return False


def _is_degenerate_repetition(text: str, min_sentences: int = 6) -> bool:
    """Return True if the text is dominated by a small model looping the same
    sentence(s) — a common failure mode under long-context summarization."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
    if len(sentences) < min_sentences:
        return False
    return len(set(sentences)) / len(sentences) < 0.6


_LOCAL_LLM_MODEL = "Gemma4-E4B-Instruct"
_LOCAL_LLM_REPO = "bartowski/google_gemma-4-E4B-it-GGUF"
_LOCAL_LLM_FILE = "google_gemma-4-E4B-it-Q4_K_M.gguf"

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


def _run_local_llm(llm, prompt_tpl: str, marker: str, text: str, fmt_kwargs: dict,
                   check_hebrew_script: bool = False, max_tokens: int = 2048) -> str:
    """Run one prompt against the local LLM, retrying with progressively shorter
    transcript slices if the output is a refusal, placeholder, too short,
    repetitive, or (when check_hebrew_script is set) in the wrong script.
    Returns the parsed text after `marker`, or raises RuntimeError if every
    attempt failed.

    The transcript slice is truncated by actual tokenized length (not word
    count) so the prompt always fits the context window regardless of how
    densely the source language tokenizes — a fixed word count that's safe
    for English can silently overflow the context for Hebrew."""
    budget = _LOCAL_LLM_N_CTX - max_tokens - _PROMPT_TOKEN_MARGIN
    # Reserve room for the prompt template's own text (headers, instructions)
    # by measuring it once with an empty transcript slot.
    template_tokens = len(llm.tokenize(prompt_tpl.format(transcript="", **fmt_kwargs).encode("utf-8")))
    text_budget = max(200, budget - template_tokens)
    base_text = _truncate_to_token_budget(llm, text, text_budget)

    result = ""
    for attempt, shrink in enumerate([1, 2, 4]):
        truncated = base_text if shrink == 1 else _truncate_to_token_budget(llm, base_text, text_budget // shrink)
        prompt = prompt_tpl.format(transcript=truncated, **fmt_kwargs)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            repeat_penalty=1.15,
        )
        candidate = response["choices"][0]["message"]["content"] or ""
        finish_reason = response["choices"][0].get("finish_reason")
        if _is_refusal(candidate):
            logger.warning(
                f"  Local LLM refused (attempt {attempt + 1}, "
                f"{len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        parsed = candidate.split(marker, 1)[1].strip() if marker in candidate else ""
        if finish_reason == "length":
            # Generation hit max_tokens mid-sentence — drop the dangling
            # incomplete sentence rather than shipping text that cuts off
            # abruptly (e.g. "...אבחון מחלה קש").
            sentences = re.split(r"(?<=[.!?])\s+", parsed)
            if len(sentences) > 1 and not re.search(r'[.!?]\s*$', parsed):
                parsed = " ".join(sentences[:-1]).strip()
                logger.warning(
                    f"  Local LLM output hit max_tokens (attempt {attempt + 1}) — "
                    f"dropped truncated trailing sentence"
                )
        if re.match(r'^\s*\[[^\]]{0,200}\]\s*$', parsed) or re.match(r'^\s*<[^>]{0,200}>\s*$', parsed):
            logger.warning("  Local LLM returned a placeholder — retrying with fewer words")
            continue
        if len(parsed) < 50:
            logger.warning(
                f"  Local LLM returned an empty/too-short result "
                f"(attempt {attempt + 1}, {len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        if _is_degenerate_repetition(parsed):
            logger.warning(
                f"  Local LLM output degenerated into repetition "
                f"(attempt {attempt + 1}, {len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        if check_hebrew_script and _has_wrong_script(parsed):
            logger.warning(
                f"  Local LLM code-switched into a non-Hebrew script "
                f"(attempt {attempt + 1}, {len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        if check_hebrew_script and _has_english_prose_run(parsed):
            logger.warning(
                f"  Local LLM wrote English prose instead of Hebrew "
                f"(attempt {attempt + 1}, {len(truncated.split())} words) — retrying with fewer words"
            )
            continue
        result = parsed
        break

    if not result:
        raise RuntimeError("All local LLM attempts failed")
    return result


def _summarize_with_local_llm(episode, text: str, long_summary: bool = False) -> tuple:
    """Returns (hebrew_summary, english_summary, steps) using a local GGUF model
    (Gemma3-4B-Instruct via llama-cpp-python) — no network calls, no API key.

    If the transcript is already mostly Hebrew, summarize directly in Hebrew
    (one fewer model call, no translation round trip). Otherwise summarize in
    English first — the model's strongest language — then translate to Hebrew
    as a separate, narrower step; this avoids the repetition/script-drift
    failures seen when asking small models to reason and generate long-form
    Hebrew directly from non-Hebrew source text.

    Transcripts too long to fit a single call (measured in actual tokens, not
    words) are split into chunks, summarized independently, then combined
    (map-reduce) so long episodes aren't silently truncated."""
    llm = _get_local_llm()
    words = text.split()
    source_is_hebrew = _is_mostly_hebrew(text)

    # Scale the requested summary length to the source length: asking a 4B
    # model for an 800-1200 word summary of a ~150-word RSS description (no
    # real transcript) reliably produces a short "not enough content" style
    # reply that the quality gate then rejects as too-short, burning all
    # retries for nothing since shrinking an already-tiny input doesn't help.
    if len(words) < 300:
        length_instr = "SHORT and CONCISE (100-200 words) — the source material is brief, so do not pad or repeat"
    elif len(words) < 600:
        length_instr = "MODERATE in length (300-500 words) — cover every topic discussed"
    else:
        length_instr = "LONG and DETAILED (800-1200 words)"
    fmt_kwargs = {"title": episode.title, "feed_name": episode.feed_name, "length_instr": length_instr}

    if source_is_hebrew:
        summary_tpl = _HEBREW_SUMMARY_PROMPT_LONG if long_summary else _HEBREW_SUMMARY_PROMPT
        chunk_tpl, combine_tpl, marker = _HEBREW_CHUNK_SUMMARY_PROMPT, _HEBREW_COMBINE_SUMMARY_PROMPT, "HEBREW_SUMMARY:"
    else:
        summary_tpl = _SUMMARY_PROMPT_LONG if long_summary else _SUMMARY_PROMPT
        chunk_tpl, combine_tpl, marker = _CHUNK_SUMMARY_PROMPT, _COMBINE_SUMMARY_PROMPT, "ENGLISH_SUMMARY:"

    # Decide single-call vs. map-reduce by actual tokenized length, not word
    # count — word count is a poor proxy across languages (Hebrew tokenizes
    # far less densely than English in this model).
    single_call_budget = _LOCAL_LLM_N_CTX - _MAX_TOKENS_SUMMARY - _PROMPT_TOKEN_MARGIN
    total_tokens = len(llm.tokenize(text.encode("utf-8")))
    fits_single_call = total_tokens <= single_call_budget

    if fits_single_call:
        summary = _run_local_llm(llm, summary_tpl, marker, text, fmt_kwargs,
                                 check_hebrew_script=source_is_hebrew, max_tokens=_MAX_TOKENS_SUMMARY)
        logger.info(f"  Local LLM: summary via {_LOCAL_LLM_MODEL} ({len(words)} words, source_he={source_is_hebrew})")
    else:
        chunk_token_budget = _LOCAL_LLM_N_CTX - _MAX_TOKENS_CHUNK_NOTES - _PROMPT_TOKEN_MARGIN - 300
        n_chunks = max(1, -(-total_tokens // chunk_token_budget))  # ceil division
        chunk_word_size = max(1, -(-len(words) // n_chunks))
        chunks = [" ".join(words[i:i + chunk_word_size])
                  for i in range(0, len(words), chunk_word_size)]
        logger.info(f"  Local LLM: transcript split into {len(chunks)} chunks for map-reduce "
                    f"({total_tokens} total tokens, source_he={source_is_hebrew})")
        notes = []
        for i, chunk in enumerate(chunks, 1):
            chunk_kwargs = {**fmt_kwargs, "part": i, "total": len(chunks)}
            note = _run_local_llm(llm, chunk_tpl, "NOTES:", chunk, chunk_kwargs,
                                  check_hebrew_script=source_is_hebrew, max_tokens=_MAX_TOKENS_CHUNK_NOTES)
            logger.info(f"  Local LLM: chunk {i}/{len(chunks)} notes ({len(note.split())} words)")
            notes.append(f"[Part {i}/{len(chunks)}]\n{note}")
        combined_notes = "\n\n".join(notes)
        summary = _run_local_llm(llm, combine_tpl, marker, combined_notes, fmt_kwargs,
                                 check_hebrew_script=source_is_hebrew, max_tokens=_MAX_TOKENS_SUMMARY)
        logger.info(f"  Local LLM: combined {len(chunks)} chunk(s) into final summary")

    if source_is_hebrew:
        return summary, "", [(f"Summary: {_LOCAL_LLM_MODEL} (he)", "summary")]

    translate_tokens = min(_MAX_TOKENS_SUMMARY, max(256, int(len(summary.split()) * 1.5 * _MAX_TOKENS_TRANSLATE_MARGIN)))
    hebrew_summary = _run_local_llm(
        llm, _TRANSLATE_TO_HEBREW_PROMPT, "HEBREW_SUMMARY:", summary, {},
        check_hebrew_script=True, max_tokens=translate_tokens,
    )
    logger.info(f"  Local LLM: translated to Hebrew ({len(hebrew_summary.split())} words)")
    return hebrew_summary, summary, [(f"Summary: {_LOCAL_LLM_MODEL} (en→he)", "summary")]


def _bart_helsinki_fallback(transcript_text: str, lang: str, settings: dict) -> tuple:
    """Returns (hebrew_summary, english_summary, steps) using BART + Helsinki-NLP
    translation models — used when the local LLM is unavailable or fails."""
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


def _summarize_with_models(episode, transcript_text: str, lang: str, settings: dict,
                           long_summary: bool = False) -> tuple:
    """Returns (hebrew_summary, english_summary, pipeline_steps).
    pipeline_steps is a list of (text, category) tuples; category is one of
    "transcript", "summary", "translate", "debug". The telegram output drops
    only "debug" steps.
    Tries the local GGUF LLM first; falls back to BART+Helsinki if that fails."""
    try:
        text = _clean_text(transcript_text, strip_urls=False)
        return _summarize_with_local_llm(episode, text, long_summary)
    except Exception as e:
        logger.warning(f"  Local LLM unavailable ({type(e).__name__}: {e}), falling back to BART+Helsinki")
        return _bart_helsinki_fallback(transcript_text, lang, settings)


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
        if category != "debug"
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
    elif method == "page_content":
        audio_note = "No audio download — summary based on episode page content"
    elif method == "description":
        audio_note = "No audio download — summary based on RSS description only"
    elif method == "cached" or method.startswith("cached_"):
        audio_note = "No audio download — reused a previously fetched transcript"
    else:
        audio_note = f"No audio download — summary based on {method}"

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
