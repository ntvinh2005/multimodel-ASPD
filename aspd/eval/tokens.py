"""Token-id to display-string decoding that preserves whitespace."""

from collections.abc import Callable


def decode_with_spaces(tokenizer: object) -> Callable[[list[int]], list[str]]:
    """A `list[int] -> list[str]` decoder where each token keeps its leading space / newline."""
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if batch_decode is not None:
        return lambda ids: batch_decode([[i] for i in ids])
    decode = tokenizer.decode  # type: ignore[attr-defined]
    return lambda ids: [decode([i]) for i in ids]
