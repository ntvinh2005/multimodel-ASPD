"""Reads what an SAE directory records about itself (sites, widths, config)."""

import json
from pathlib import Path


def _sae_config(sae_dir: Path) -> tuple[dict, Path]:
    """`sae_config.yaml` plus the project root its relative paths are stated against."""
    import yaml

    cfg_path = sae_dir / "sae_config.yaml"
    assert cfg_path.exists(), f"{sae_dir} has no sae_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    stated = Path(cfg["sae_dir"])
    roots = [p for p in sae_dir.resolve().parents if (p / stated).resolve() == sae_dir.resolve()]
    assert roots, (
        f"{cfg_path} says sae_dir={stated}, which does not resolve to {sae_dir} from any ancestor; "
        "the pair has been moved away from the project its config describes"
    )
    return cfg, roots[0]


def experiment_config_from_sae_dir(sae_dir: Path) -> Path:
    """The VPD experiment config supplying this pair's target and corpus."""
    prov_path = sae_dir / "provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        assert prov["kind"] == "joint_sae", f"unknown artifact kind {prov['kind']!r} in {prov_path}"
        return Path(prov["run_dir"]) / "experiment_config.yaml"
    cfg, root = _sae_config(sae_dir)
    return root / cfg["experiment_config"]


def module_from_sae_dir(sae_dir: Path) -> str:
    """The decomposed module this pair brackets."""
    for name in ("sites.json", "provenance.json"):
        path = sae_dir / name
        if path.exists():
            return json.loads(path.read_text())["module"]
    cfg, root = _sae_config(sae_dir)

    from aspd.config import LMInterpExperimentConfig

    return LMInterpExperimentConfig.from_file(root / cfg["experiment_config"]).pd.decomposition_targets[0].module_pattern


def load_experiment_config(path: Path | str):
    """`LMInterpExperimentConfig.from_file`, with a project-relative corpus path made absolute."""
    from aspd.config import LMInterpExperimentConfig

    path = Path(path)
    cfg = LMInterpExperimentConfig.from_file(path)
    stated = Path(cfg.data.dataset_name)
    if stated.is_absolute() or stated.is_dir():
        return cfg
    root = next((p for p in path.resolve().parents if (p / stated).is_dir()), None)
    if root is None:
        return cfg
    print(f"[config] corpus {stated} -> {root / stated}", flush=True)
    return cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"dataset_name": str(root / stated)})}
    )
