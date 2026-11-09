from __future__ import annotations

from aspd.multimodel.cli.smoke_aspd_v3 import _classify_outcome, _run_alignment


class _CharacterTokenizer:
    is_fast = True

    def __init__(self, *, mismatch: bool = False):
        self.mismatch = mismatch
        self._pieces: dict[int, str] = {}

    def __call__(self, text: str, **_kwargs):
        self._pieces = {index + 1: value for index, value in enumerate(text)}
        count = len(text)
        return {
            "input_ids": [99, *range(1, count + 1), 100],
            "attention_mask": [1] * (count + 2),
            "offset_mapping": [(0, 0), *((index, index + 1) for index in range(count)), (0, 0)],
        }

    def decode(self, token_ids, **_kwargs):
        value = "".join(self._pieces.get(int(token_id), "") for token_id in token_ids)
        return value + ("!" if self.mismatch and value else "")


def _models(status: str) -> dict[str, dict[str, str]]:
    return {key: {"status": status} for key in ("qwen", "gemma", "phi")}


def test_smoke_success_path_requires_models_and_nonzero_alignment() -> None:
    tokenizers = {
        key: _CharacterTokenizer() for key in ("qwen", "gemma", "phi")
    }
    alignment = _run_alignment(tokenizers)
    assert alignment["status"] == "REAL_TOKENIZER_ALIGNMENT_PASS"
    assert alignment["accepted_aligned_anchors"] > 0
    assert _classify_outcome(
        _models("REAL_MODEL_SMOKE_PASS"), alignment["status"]
    ) == ("REAL_MODEL_SMOKE_PASS", 0)


def test_smoke_blocked_path_is_distinct_from_failure() -> None:
    alignment = _run_alignment({"qwen": _CharacterTokenizer()})
    assert alignment["status"] == "BLOCKED_NOT_RUN"
    assert _classify_outcome(
        _models("BLOCKED_NOT_RUN"), alignment["status"]
    ) == ("BLOCKED_NOT_RUN", 2)


def test_smoke_zero_anchor_alignment_is_failure_not_pass() -> None:
    tokenizers = {
        key: _CharacterTokenizer(mismatch=True)
        for key in ("qwen", "gemma", "phi")
    }
    alignment = _run_alignment(tokenizers)
    assert alignment["accepted_aligned_anchors"] == 0
    assert alignment["status"] == "REAL_TOKENIZER_ALIGNMENT_FAIL"
    assert _classify_outcome(
        _models("REAL_MODEL_SMOKE_PASS"), alignment["status"]
    ) == ("REAL_MODEL_SMOKE_FAIL", 1)


def test_smoke_model_correctness_failure_exits_nonzero() -> None:
    rows = _models("REAL_MODEL_SMOKE_PASS")
    rows["phi"] = {"status": "REAL_MODEL_SMOKE_FAIL"}
    assert _classify_outcome(rows, "REAL_TOKENIZER_ALIGNMENT_PASS") == (
        "REAL_MODEL_SMOKE_FAIL",
        1,
    )
