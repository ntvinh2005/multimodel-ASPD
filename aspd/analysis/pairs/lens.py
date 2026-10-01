"""Logit lens on a component's write direction u_c."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FinalNorm:
    """Where a target keeps its final normalisation and its unembedding."""

    weight: str
    bias: str | None  # LayerNorm has one; RMSNorm does not
    unembed: str
    centres: bool  # LayerNorm centres its input; RMSNorm does not
    unit_offset: bool  # Gemma's RMSNorm scales by `(1 + w)`
    eps: float


_GPT2 = FinalNorm("ln_f.weight", "ln_f.bias", "wte.weight", True, False, 1e-5)
_GEMMA = FinalNorm("model.norm.weight", None, "model.embed_tokens.weight", False, True, 1e-6)


def final_norm_for_model(model_name: str) -> FinalNorm:
    return _GEMMA if "gemma" in model_name else _GPT2


class LogitLens:
    """`u -> [V]`, lazily loading one norm and one unembedding matrix for a target."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.spec = final_norm_for_model(model_name)
        self._loaded = False
        self._gain: Tensor | None = None
        self._beta: Tensor | None = None
        self._unembed: Tensor | None = None
        self._tokens: list[str] | None = None
        self.reason: str | None = None

    def _load(self) -> None:
        """Materialise the norm and the unembedding, recording WHY if either is missing."""
        if self._loaded:
            return
        self._loaded = True
        from aspd.analysis.pairs.features import TargetNorms

        store = TargetNorms(self.model_name)
        try:
            from safetensors import safe_open
            from transformers import AutoTokenizer

            keys = [self.spec.weight, self.spec.unembed] + (
                [self.spec.bias] if self.spec.bias else []
            )
            got: dict[str, Tensor] = {}
            for key in keys:
                with safe_open(store.shard_of(key), "pt") as f:
                    present = set(f.keys())
                    assert key in present, f"{key!r} is not in {self.model_name}'s safetensors"
                    got[key] = f.get_tensor(key).to(torch.float32)
            tok = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001 -- the card must say why, not 500
            self.reason = f"no logit lens for {self.model_name}: {type(exc).__name__}: {exc}"
            return
        w = got[self.spec.weight]
        self._gain = w + 1.0 if self.spec.unit_offset else w
        self._beta = got[self.spec.bias] if self.spec.bias else None
        self._unembed = got[self.spec.unembed]
        from aspd.analysis.pairs.clean import byte_decoder, decode_token

        dec = byte_decoder(self.model_name)
        self._tokens = [
            decode_token(t, dec)
            for t in tok.convert_ids_to_tokens(range(self._unembed.shape[0]))
        ]

    @property
    def available(self) -> bool:
        self._load()
        return self._unembed is not None

    @property
    def d_model(self) -> int | None:
        self._load()
        return None if self._gain is None else int(self._gain.numel())

    def normed(self, u: Tensor) -> Tensor:
        """`norm_f(u)` -- the target's own final normalisation, applied to one direction."""
        self._load()
        assert self._gain is not None
        x = u.to(torch.float32)
        if self.spec.centres:
            x = x - x.mean()
            x = x / (x.var(unbiased=False) + self.spec.eps).sqrt()
        else:
            x = x / (x.pow(2).mean() + self.spec.eps).sqrt()
        x = x * self._gain
        return x if self._beta is None else x + self._beta

    def scores(self, u: Tensor) -> Tensor:
        """`[V]` -- one score per vocabulary token."""
        self._load()
        assert self._unembed is not None, self.reason
        assert u.numel() == self._unembed.shape[1], (
            f"a logit lens needs a {self._unembed.shape[1]}-wide direction, got {u.numel()}"
        )
        return self._unembed @ self.normed(u.flatten())

    def top(self, u: Tensor, k: int = 10) -> dict[str, object]:
        """The `k` tokens this direction promotes and the `k` it suppresses."""
        s = self.scores(u)
        assert self._tokens is not None
        import numpy as np

        k = min(k, s.numel())
        v = s.numpy()
        hi = np.argpartition(v, -k)[-k:]
        lo = np.argpartition(v, k)[:k]
        hi = hi[np.argsort(-v[hi])]
        lo = lo[np.argsort(v[lo])]
        pick = lambda idx: [[self._tokens[int(i)], float(v[int(i)])] for i in idx]
        return {"promoted": pick(hi), "suppressed": pick(lo)}
