"""Clean-ups for feature metadata fetched from Neuronpedia."""

import difflib
import re
from functools import cache

SIMILAR = 0.85


@cache
def byte_decoder(model_name: str) -> dict[str, int] | None:
    """`{character: byte}` for a byte-level BPE tokenizer, or `None` for one that is not."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    slow = getattr(tok, "byte_decoder", None)
    if slow:
        return dict(slow)
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    if "gpt2" not in (tok.__class__.__name__ or "").lower():
        return None
    return {ch: b for b, ch in bytes_to_unicode().items()}


def decode_token(s: str, dec: dict[str, int] | None) -> str:
    """One token string, out of the byte alphabet and into text."""
    if dec is None or not s:
        return s
    out = bytearray()
    for ch in s:
        b = dec.get(ch)
        if b is None:
            out += ch.encode("utf-8")
        else:
            out.append(b)
    return out.decode("utf-8", errors="replace")


def decode_tokens(tokens: list[str], dec: dict[str, int] | None) -> list[str]:
    """A contiguous WINDOW of tokens, decoded as ONE byte stream."""
    if dec is None:
        return list(tokens)
    import codecs

    step = codecs.getincrementaldecoder("utf-8")("replace")
    out = []
    for t in tokens:
        buf = bytearray()
        for ch in t:
            b = dec.get(ch)
            if b is None:
                buf += ch.encode("utf-8")
            else:
                buf.append(b)
        out.append(step.decode(bytes(buf)))
    tail = step.decode(b"", final=True)
    if tail and out:
        out[-1] += tail
    return out


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.;!?])\s+", text) if p.strip()]


def _key(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


def drop_repeated_sentences(text: str) -> str:
    """One description with any sentence it states twice stated once."""
    seen: set[str] = set()
    kept = []
    for s in _sentences(text):
        k = _key(s)
        if k and k not in seen:
            seen.add(k)
            kept.append(s)
    return " ".join(kept) if kept else text.strip()


def dedupe_activations(acts: list[dict]) -> tuple[list[dict], int]:
    """Neuronpedia's `activations[]`, minus windows already listed. Returns `(kept, n_dropped)`."""
    kept, seen = [], set()
    for a in acts:
        key = (tuple(a.get("tokens") or []), tuple(a.get("values") or []))
        if key in seen:
            continue
        seen.add(key)
        kept.append(a)
    return kept, len(acts) - len(kept)


def tidy_explanations(items: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for raw in items:
        text = drop_repeated_sentences(raw.strip())
        if not text:
            continue
        k = _key(text)
        if any(
            k == _key(x) or difflib.SequenceMatcher(None, k, _key(x)).ratio() >= SIMILAR
            for x in kept
        ):
            dropped.append(text)
            continue
        kept.append(text)
    return kept, dropped
