"""Human-only, blinded semantic evaluation helpers for ASPD v3.1.

The public task representation deliberately cannot contain the experimental arm,
site, feature identity, activation magnitude, source-document identity, or answer.
Those fields belong only in the separately stored answer key.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "activation",
        "anchor_id",
        "anchor_index",
        "arm",
        "correct_intruder",
        "document_id",
        "donor_feature",
        "feature",
        "feature_fires",
        "frequency_stratum",
        "intruder_position",
        "site",
    }
)

PATTERN_CATEGORIES = (
    "semantic",
    "syntactic",
    "stylistic",
    "formatting",
    "mixed",
    "unclear",
)


class HumanLlamaJudgeContract(BaseModel):
    """Single-presentation exploratory Llama judge requested alongside human scoring."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    status: Literal["RESEARCHER_AUTHORIZED_EXPLORATORY_RAW_LLM_JUDGE"]
    experiment_family: Literal["aspd_v3_human_items_llama70b_raw_v1"]
    judge_model_id: Literal["meta-llama/Llama-3.3-70B-Instruct"]
    judge_revision: Literal["6f6073b423013f6a7d4d9f39144961bfbfbc386b"]
    local_files_only: Literal[True]
    paid_api: Literal[False]
    forbidden_model_substrings: tuple[Literal["qwen"]]
    dtype: Literal["bfloat16"]
    decoding: Literal["greedy"]
    batch_size: Literal[4]
    max_new_tokens: Literal[96]
    forced_choice: Literal[True]
    cyclic_rotations: Literal[1]
    position_bias_gate: Literal[False]
    prompt_version: Literal["original_aspd_single_forced_choice_v1"]
    bootstrap_seed: Literal[113]
    bootstrap_replicates: Literal[5_000]

    @classmethod
    def load(cls, path: str | Path) -> HumanLlamaJudgeContract:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return the on-disk canonical JSON representation used by frozen artifacts."""

    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_excluded_documents(source_paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    """Union prior document manifests and freeze a deterministic exclusion manifest."""

    rows_by_id: dict[str, dict[str, str]] = {}
    source_hashes: list[dict[str, str]] = []
    for path in source_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"document manifest must be a list: {path}")
        source_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        for row in rows:
            document_id = str(row["document_id"])
            normalized = {
                "document_id": document_id,
                "source_group": str(row.get("source_group", "unknown")),
            }
            previous = rows_by_id.setdefault(document_id, normalized)
            if previous != normalized:
                raise ValueError(f"conflicting provenance for document {document_id}")
    frozen = sorted(rows_by_id.values(), key=lambda row: row["document_id"])
    payload = canonical_json_bytes(frozen)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(payload)
    return {
        "document_count": len(frozen),
        "manifest_sha256": sha256_bytes(payload),
        "source_manifests": source_hashes,
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def validate_public_blinding(items: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed if a public task contains experimental metadata or answers."""

    item_ids: set[str] = set()
    for item in items:
        item_id = str(item.get("item_id", ""))
        if not item_id or item_id in item_ids:
            raise ValueError("public item IDs must be nonempty and unique")
        item_ids.add(item_id)
        leaked = PUBLIC_FORBIDDEN_KEYS.intersection(_walk_keys(item))
        if leaked:
            raise ValueError(f"public item {item_id} leaks protected keys: {sorted(leaked)}")
        examples = item.get("examples")
        if not isinstance(examples, list) or len(examples) != 5:
            raise ValueError(f"public item {item_id} must contain five examples")
        candidate_ids = [str(row.get("candidate_id", "")) for row in examples]
        if candidate_ids != ["1", "2", "3", "4", "5"]:
            raise ValueError(f"public item {item_id} candidate IDs are not canonical")
        for example in examples:
            for view in ("compact", "expanded"):
                parts = example.get(view)
                if not isinstance(parts, Mapping):
                    raise ValueError(f"public item {item_id} lacks {view} context")
                if set(parts) != {"prefix", "anchor", "suffix", "truncated_left", "truncated_right"}:
                    raise ValueError(f"public item {item_id} has malformed {view} context")
                if not str(parts["anchor"]):
                    raise ValueError(f"public item {item_id} has an empty anchor span")


def context_views(row: Mapping[str, Any], compact_characters_per_side: int) -> dict[str, Any]:
    """Convert a validated canonical context into compact and expanded marked views."""

    text = str(row["canonical_context"])
    context_start = int(row["canonical_context_start"])
    context_end = int(row["canonical_context_end"])
    anchor_start = int(row["canonical_anchor_start"])
    anchor_end = int(row["canonical_anchor_end"])
    boundary = int(row["canonical_character_boundary"])
    anchor_text = str(row["canonical_anchor_text"])
    if context_end - context_start != len(text):
        raise ValueError("canonical context offsets do not match retained text")
    if not context_start <= anchor_start < anchor_end == boundary <= context_end:
        raise ValueError("canonical anchor span is outside the retained context")
    local_start = anchor_start - context_start
    local_end = anchor_end - context_start
    if text[local_start:local_end] != anchor_text or not anchor_text:
        raise ValueError("canonical anchor text does not match its source span")

    compact_start = max(0, local_start - compact_characters_per_side)
    compact_end = min(len(text), local_end + compact_characters_per_side)

    def view(lower: int, upper: int) -> dict[str, Any]:
        return {
            "prefix": text[lower:local_start],
            "anchor": anchor_text,
            "suffix": text[local_end:upper],
            "truncated_left": lower > 0 or context_start > 0,
            "truncated_right": upper < len(text),
        }

    return {
        "compact": view(compact_start, compact_end),
        "expanded": view(0, len(text)),
    }


def balanced_positions(
    strata: Sequence[str], *, rng: np.random.Generator, positions: int = 5
) -> list[int]:
    """Assign exactly balanced positions while limiting within-stratum imbalance."""

    count = len(strata)
    base, extra = divmod(count, positions)
    multiset = np.concatenate(
        [np.full(base + int(index < extra), index + 1) for index in range(positions)]
    )
    if len(multiset) != count:
        raise AssertionError("position quota construction failed")
    for _ in range(100_000):
        candidate = rng.permutation(multiset).astype(int).tolist()
        acceptable = True
        for stratum in sorted(set(strata)):
            counts = Counter(
                position for position, observed in zip(candidate, strata, strict=True)
                if observed == stratum
            )
            values = [counts.get(position, 0) for position in range(1, positions + 1)]
            if max(values) - min(values) > 2:
                acceptable = False
                break
        if acceptable:
            return candidate
    raise RuntimeError("could not construct a balanced position assignment")


def balanced_round_robin(
    grouped: Mapping[str, Sequence[dict[str, Any]]], *, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Interleave arm/site groups so every complete four-item block contains each group."""

    queues = {key: list(value) for key, value in grouped.items()}
    for queue in queues.values():
        rng.shuffle(queue)
    output: list[dict[str, Any]] = []
    while any(queues.values()):
        available = [key for key, values in queues.items() if values]
        rng.shuffle(available)
        output.extend(queues[key].pop() for key in available)
    return output


def response_template(dataset_sha256: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "evaluator_id": "",
        "started_at": None,
        "exported_at": None,
        "responses": {
            str(item["item_id"]): {
                "intruder_choice": None,
                "cannot_determine": False,
                "coherence": None,
                "pattern_category": None,
                "confidence": None,
                "explanation": "",
            }
            for item in items
        },
    }


def validate_responses(
    payload: Mapping[str, Any], *, dataset_sha256: str, public_item_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported response schema")
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("response dataset hash does not match the frozen task set")
    evaluator_id = str(payload.get("evaluator_id", "")).strip()
    if not evaluator_id:
        raise ValueError("evaluator_id is required")
    responses = payload.get("responses")
    if not isinstance(responses, Mapping) or set(responses) != public_item_ids:
        raise ValueError("responses must contain exactly the frozen public item IDs")
    normalized: dict[str, dict[str, Any]] = {}
    for item_id, row in responses.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"response {item_id} is not an object")
        cannot = bool(row.get("cannot_determine"))
        choice = row.get("intruder_choice")
        if cannot:
            if choice not in (None, ""):
                raise ValueError(f"response {item_id} selects both an intruder and abstention")
            choice = None
        elif str(choice) not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"response {item_id} must select one candidate or abstain")
        coherence = int(row.get("coherence"))
        confidence = int(row.get("confidence"))
        category = str(row.get("pattern_category"))
        if coherence not in range(1, 6) or confidence not in range(1, 6):
            raise ValueError(f"response {item_id} ratings must be 1 through 5")
        if category not in PATTERN_CATEGORIES:
            raise ValueError(f"response {item_id} has an invalid pattern category")
        normalized[str(item_id)] = {
            "intruder_choice": choice,
            "cannot_determine": cannot,
            "coherence": coherence,
            "pattern_category": category,
            "confidence": confidence,
            "explanation": str(row.get("explanation", "")),
        }
    return normalized


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total else math.nan


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"features": 0}
    primary = [row for row in rows if not row.get("repeat_of_parent_uid")]
    accuracy = np.asarray([float(row["correct"]) for row in primary])
    abstention = np.asarray([float(row["abstained"]) for row in primary])
    coherence = np.asarray([float(row["coherence"]) for row in primary])
    activity = np.asarray([float(row["feature_fires"]) for row in primary])
    forced = [row for row in primary if not row["abstained"]]
    return {
        "features": len(primary),
        "intruder_accuracy_intention_to_evaluate": float(accuracy.mean()),
        "abstention_rate": float(abstention.mean()),
        "intruder_accuracy_conditional_forced_choice": (
            float(np.mean([float(row["correct"]) for row in forced]))
            if forced else None
        ),
        "coherence_mean": float(coherence.mean()),
        "coherence_median": float(np.median(coherence)),
        "activity_weighted_intruder_accuracy": _weighted_mean(accuracy, activity),
        "activity_weighted_coherence": _weighted_mean(coherence, activity),
        "category_counts": dict(Counter(str(row["pattern_category"]) for row in primary)),
    }


def bootstrap_group(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, list[float]]:
    """Feature-level stratified bootstrap; repeats are never resampling units."""

    primary = [row for row in rows if not row.get("repeat_of_parent_uid")]
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        by_stratum[str(row["frequency_stratum"])].append(row)
    rng = np.random.default_rng(seed)
    accuracy_draws = np.empty(replicates, dtype=np.float64)
    coherence_draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        sampled: list[Mapping[str, Any]] = []
        for values in by_stratum.values():
            indices = rng.integers(0, len(values), size=len(values))
            sampled.extend(values[int(index)] for index in indices)
        accuracy_draws[draw] = np.mean([float(row["correct"]) for row in sampled])
        coherence_draws[draw] = np.mean([float(row["coherence"]) for row in sampled])
    return {
        "intruder_accuracy_ci95": np.quantile(accuracy_draws, [0.025, 0.975]).tolist(),
        "coherence_mean_ci95": np.quantile(coherence_draws, [0.025, 0.975]).tolist(),
        "intruder_accuracy_draws": accuracy_draws.tolist(),
        "coherence_mean_draws": coherence_draws.tolist(),
    }


def cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / len(left) ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else None


def pairwise_rater_agreement(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_left = {str(row["parent_uid"]): row for row in left if not row.get("repeat_of_parent_uid")}
    by_right = {str(row["parent_uid"]): row for row in right if not row.get("repeat_of_parent_uid")}
    shared = sorted(set(by_left) & set(by_right))
    left_choice = [
        "abstain" if by_left[key]["abstained"] else str(by_left[key]["selected_canonical_index"])
        for key in shared
    ]
    right_choice = [
        "abstain" if by_right[key]["abstained"] else str(by_right[key]["selected_canonical_index"])
        for key in shared
    ]
    return {
        "shared_features": len(shared),
        "intruder_choice_raw_agreement": (
            sum(a == b for a, b in zip(left_choice, right_choice, strict=True)) / len(shared)
            if shared else None
        ),
        "intruder_choice_cohens_kappa": cohens_kappa(left_choice, right_choice),
        "pattern_category_raw_agreement": (
            sum(
                by_left[key]["pattern_category"] == by_right[key]["pattern_category"]
                for key in shared
            ) / len(shared)
            if shared else None
        ),
        "abstention_raw_agreement": (
            sum(by_left[key]["abstained"] == by_right[key]["abstained"] for key in shared)
            / len(shared)
            if shared else None
        ),
        "coherence_exact_agreement": (
            sum(by_left[key]["coherence"] == by_right[key]["coherence"] for key in shared)
            / len(shared)
            if shared else None
        ),
    }
