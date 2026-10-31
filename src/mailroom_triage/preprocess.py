"""Input cleaning shared by training, evaluation and serving.

Real mail arrives as HTML, with quoted threads, signatures and broken
whitespace. The benchmark text is already clean; applying the same cleaning
everywhere keeps the serving contract identical to the evaluated one.

Deliberate choices:
- Quoted reply chains are removed: triage should read the new message, not the
  history that is usually duplicated or stale.
- Forwarded content is kept: the evidence is often inside the forwarded part.
- Signature removal is conservative (the "-- " delimiter and "Sent from my ...")
  so we do not delete real evidence.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser

MAX_BODY_CHARS = 20_000
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

_HTML_HINT = re.compile(r"<(?:html|body|div|p|br|table|span|a|td|tr)\b", re.IGNORECASE)
_QUOTE_START = re.compile(
    r"(?im)^(?:"
    r"On\s.{0,300}\bwrote:\s*$"
    r"|Le\s.{0,300}\ba écrit\s*:\s*$"
    r"|-{2,}\s*Original Message\s*-{2,}\s*$"
    r"|_{10,}\s*$"
    r"|From:\s.*\n(?:Sent|Date):\s.*\n(?:To|Cc):\s.*\nSubject:\s.*$"
    r")"
)
_SIGNATURE_DELIMITER = re.compile(r"(?m)^--\s*$")
_SENT_FROM = re.compile(r"(?im)^\s*Sent from my .{0,60}$")
_FORWARD_START = re.compile(
    r"(?i)^-{2,}\s*(?:forwarded message|original message)\s*-{2,}\s*$|^begin forwarded message:\s*$"
)
_HEADER_LINE = re.compile(r"(?i)^(from|date|subject|to|cc|sent|reply-to):")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "head"}:
            self.skip += 1
        elif tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "head"}:
            self.skip = max(0, self.skip - 1)
        elif tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip == 0:
            self.parts.append(data)


def looks_like_html(text: str) -> bool:
    return bool(_HTML_HINT.search(text or ""))


def html_to_text(text: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(text)
        parser.close()
        out = "".join(parser.parts)
        if out.strip():
            return out
    except Exception:  # malformed markup: fall through to a regex strip
        pass
    return re.sub(r"<[^>]+>", " ", text)


def strip_quoted(text: str) -> str:
    match = _QUOTE_START.search(text)
    if match:
        text = text[: match.start()]
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    return "\n".join(lines)


def strip_signature(text: str) -> str:
    match = _SIGNATURE_DELIMITER.search(text)
    if match:
        text = text[: match.start()]
    return _SENT_FROM.sub("", text)


def strip_forward_headers(text: str) -> str:
    """Drop the forward block's From/Date/Subject lines; keep the forwarded body."""
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if _FORWARD_START.match(lines[index].strip()):
            index += 1
            while index < len(lines) and lines[index].strip() and _HEADER_LINE.match(lines[index].strip()):
                index += 1
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out)


def clean_body(body: str) -> str:
    text = body or ""
    if looks_like_html(text):
        text = html_to_text(text)
    text = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH).replace("\xa0", " ")
    text = strip_quoted(text)
    text = strip_signature(text)
    text = strip_forward_headers(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_BODY_CHARS]


def clean_header(value: str, limit: int = 300) -> str:
    if not isinstance(value, str):
        value = format_sender(value)
    text = unicodedata.normalize("NFKC", value or "").translate(ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def format_sender(value) -> str:
    """The corpus stores From as a dict; real mail gives a display string."""
    if isinstance(value, dict):
        name = str(value.get("display_name") or "").strip()
        address = str(value.get("email") or "").strip()
        if name and address:
            return f"{name} <{address}>"
        return address or name
    return str(value or "")


def clean_state(state: dict) -> dict:
    return {
        "subject": clean_header(state.get("subject", "")),
        "from": clean_header(state.get("from", "")),
        "date": clean_header(state.get("date", ""), limit=80),
        "body": clean_body(state.get("body", "")),
    }
