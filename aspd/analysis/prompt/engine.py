"""Holds one loaded model and a cache of recent prompt traces."""

from collections import OrderedDict
from pathlib import Path

from torch import Tensor

from aspd.analysis.prompt.trace import PromptTrace, build_trace


class PromptEngine:
    """Lazily loads one run's `ComponentModel` and traces prompts against it."""

    def __init__(self, run_dir: Path, step: int | None = None, device: str = "cpu",
                 cache_size: int = 4):
        self.run_dir = Path(run_dir)
        self.step = step
        self.device = device
        self.cache_size = cache_size
        self._model = None
        self._cfg = None
        self._tok = None
        self._traces: OrderedDict[str, PromptTrace] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoTokenizer

        from aspd.checkpoints import load_run_model
        from aspd.analysis.prompt.load import latest_step

        step = self.step or latest_step(self.run_dir)
        model, cfg = load_run_model(self.run_dir, step, device=self.device)
        self._model, self._cfg = model, cfg
        self._tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)

    def tokenize(self, text: str) -> list[int]:
        self._load()
        return self._tok(text)["input_ids"]  # pyright: ignore[reportOptionalCall, reportIndexIssue]

    def decode(self, token_id: int) -> str:
        self._load()
        return self._tok.decode([token_id])  # pyright: ignore[reportOptionalMemberAccess]

    def single_token(self, text: str) -> int:
        """The id of `text`, asserting it is ONE token."""
        ids = self.tokenize(text)
        assert len(ids) == 1, f"{text!r} is {len(ids)} tokens ({ids}); the target needs one"
        return int(ids[0])

    def trace(self, prompt: str) -> PromptTrace:
        hit = self._traces.get(prompt)
        if hit is not None:
            self._traces.move_to_end(prompt)
            return hit
        self._load()
        assert self._model is not None and self._cfg is not None and self._tok is not None
        tokens: Tensor = self._tok(prompt, return_tensors="pt")["input_ids"].to(self.device)  # pyright: ignore[reportIndexIssue, reportCallIssue]
        pieces = [self._tok.decode([t]) for t in tokens[0].tolist()]
        out = build_trace(self._model, self._cfg, tokens, pieces)
        self._traces[prompt] = out
        while len(self._traces) > self.cache_size:
            self._traces.popitem(last=False)
        return out
