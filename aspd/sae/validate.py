"""Checks that an SAE directory matches the config and model it is used with."""

import json
from pathlib import Path


def load_report(sae_dir: Path) -> dict[str, dict[str, float]]:
    return json.loads((sae_dir / "sae_report.json").read_text())


_CHECKPOINT_FIELDS = (
    "group_fracs",
    "top_k",
    "top_k_aux",
    "aux_penalty",
    "l1_coeff",
    "threshold_lr",
    "n_batches_to_dead",
    "encoder_init",
)


def require_pair_matches_config(
    sae_dir: Path, sae_cfg: object, sae: object, d_ref: int
) -> None:
    """Refuse to reuse an on-disk pair that the declared config did not produce."""
    import yaml

    diffs: list[str] = []
    want_dtype = sae_cfg.train.torch_dtype  # type: ignore[attr-defined]
    if sae.W_enc.dtype != want_dtype:  # type: ignore[attr-defined]
        diffs.append(f"dtype: on disk {sae.W_enc.dtype}, config says {want_dtype}")  # type: ignore[attr-defined]

    dict_cfg = sae_cfg.dictionary  # type: ignore[attr-defined]
    want_features = dict_cfg.feature_multiplier * d_ref
    if sae.cfg.n_features != want_features:  # type: ignore[attr-defined]
        diffs.append(
            f"n_features: on disk {sae.cfg.n_features}, config's feature_multiplier "  # type: ignore[attr-defined]
            f"{dict_cfg.feature_multiplier} x reference width {d_ref} = {want_features}"
        )
    for field in _CHECKPOINT_FIELDS:
        got, want = getattr(sae.cfg, field), getattr(dict_cfg, field)  # type: ignore[attr-defined]
        # group_fracs round-trips through YAML as a list; compare by value, not by container type.
        if isinstance(want, (tuple, list)):
            got, want = tuple(got), tuple(want)
        if got != want:
            diffs.append(f"{field}: on disk {got}, config says {want}")

    stored_path = Path(sae_dir) / "sae_config.yaml"
    if stored_path.exists():
        stored = yaml.safe_load(stored_path.read_text())["train"]
        for field, want in sae_cfg.train.model_dump().items():  # type: ignore[attr-defined]
            got = stored.get(field)
            if isinstance(want, tuple):
                got, want = tuple(got), tuple(want)
            if got != want:
                diffs.append(f"train.{field}: pair trained with {got}, config says {want}")
    else:
        print(
            f"[sae] WARNING: {stored_path} is absent -- this pair predates config provenance. "
            "n_tokens / sae_batch_tokens / lr leave no trace in the weights and CANNOT be "
            "verified; only dtype and the dictionary fields were checked. Retrain to get a "
            "verifiable pair.",
            flush=True,
        )

    assert not diffs, (
        f"the SAE pair at {sae_dir} was not trained by this config:\n  "
        + "\n  ".join(diffs)
        + f"\nTraining short-circuits on an existing sae_report.json, so re-running would keep "
        f"the OLD extractor while the config claims the new settings. Delete {sae_dir} to "
        "retrain, or point sae_dir at a new path."
    )


def warn_if_unverifiable(sae_dir: Path) -> None:
    """Load-only paths (decomposition, offline eval) carry no SAE config to compare against."""
    if not (Path(sae_dir) / "sae_config.yaml").exists():
        print(
            f"[sae] WARNING: no sae_config.yaml in {sae_dir}. This pair's training settings "
            "(dtype, sae_batch_tokens, n_tokens, lr) are unrecorded, so the run's extractor "
            "cannot be reproduced from the artifacts. Retrain via slurm/sae.sbatch to fix.",
            flush=True,
        )


def summarize(report: dict[str, dict[str, float]]) -> str:
    return "\n".join(
        f"  {site}: fvu={s['fvu']:.4f} l0={s['mean_l0']:.1f} "
        f"dead={s['dead_frac']:.3f} F={int(s['n_features'])}"
        for site, s in report.items()
    )
