"""Config gating, artifact immutability and the end-to-end toy pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from aspd.multimodel.cli.posthoc_analysis import main as cli_main
from aspd.multimodel.posthoc.artifacts import load_edges, require_complete
from aspd.multimodel.posthoc.config import PosthocBlocked, load_config
from aspd.multimodel.posthoc.pipeline import run_stages

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[1]
TOY = REPO / "configs/multimodel/posthoc/toy_smoke.yaml"
TEMPLATE = REPO / "configs/multimodel/posthoc/v31_500m_template.yaml"


def _write(tmp_path: Path, edits: dict[str, dict]) -> Path:
    raw = yaml.safe_load(TOY.read_text())
    for section, values in edits.items():
        raw[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_v31_template_is_blocked_for_every_stage() -> None:
    config = load_config(TEMPLATE)
    for stage in ("importance", "representatives", "edges", "compare"):
        assert config.blockers(stage), stage
    assert "representatives.screening_metric" in config.blockers("representatives")
    assert "backend.options.source_dtype" in config.blockers("importance")
    with pytest.raises(PosthocBlocked):
        config.require_ready("compare")


def test_toy_config_is_ready() -> None:
    config = load_config(TOY)
    assert all(not config.blockers(stage) for stage in
               ("importance", "representatives", "edges", "compare"))


def test_null_decision_blocks_only_its_stage(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, {"parents": {"k_par": None}}))
    assert config.blockers("compare") == ["parents.k_par"]
    assert not config.blockers("importance") and not config.blockers("edges")


def test_missing_key_is_an_error_not_a_default(tmp_path: Path) -> None:
    raw = yaml.safe_load(TOY.read_text())
    del raw["parents"]["tau"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="'tau' must be present"):
        load_config(path)


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("posthoc") / "run"
    run_stages(load_config(TOY), root, ("importance", "representatives", "edges", "compare"))
    return root


def test_end_to_end_artifacts_and_provenance(toy_run: Path) -> None:
    for stage in ("importance", "representatives", "edges", "compare"):
        path = require_complete(toy_run, stage)
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["stage"] == stage and manifest["config_sha256"]
        assert "git_sha" in manifest["runtime"]
    edges = load_edges(toy_run / "edges")
    assert edges and all(r.scores.numel() == len(r.upstream) for r in edges)
    assert all(b.matrix.is_upstream_of(r.downstream.matrix) for r in edges for b in r.upstream)
    rows = [json.loads(line) for line in (toy_run / "compare/comparisons.jsonl").open()]
    assert rows and {tuple(r["models"]) for r in rows} == {("alpha", "beta")}
    for row in rows:
        assert row["jaccard"] is None or 0.0 <= row["jaccard"] <= 1.0
        assert row["total_variation"] is None or 0.0 <= row["total_variation"] <= 1.0
    checks = json.loads((toy_run / "edges/ablation_check.json").read_text())["checks"]
    assert checks and all({"taylor", "exact"} <= set(c) for c in checks)


def test_stages_never_overwrite(toy_run: Path) -> None:
    with pytest.raises(FileExistsError):
        run_stages(load_config(TOY), toy_run, ("compare",))


def test_tampered_prerequisite_is_rejected(toy_run: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(toy_run, copy)
    shutil.rmtree(copy / "compare")
    with (copy / "edges/selection.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed after its stage completed"):
        run_stages(load_config(TOY), copy, ("compare",))


def test_compare_rerun_with_new_parent_rule_reuses_edges(toy_run: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "rerun"
    for stage in ("importance", "representatives", "edges"):
        shutil.copytree(toy_run / stage, copy / stage)
    config = load_config(_write(tmp_path, {"parents": {"k_par": 1, "tau": 0.0}}))
    run_stages(config, copy, ("compare",))
    for line in (copy / "compare/comparisons.jsonl").open():
        row = json.loads(line)
        # with k_par=1 each model has at most |R| <= k_rep = 2 parent gates
        assert all(len(s) <= 2 for s in row["parent_gate_sets"])


def test_cli_check_and_blocked_exit_codes(tmp_path: Path, capsys) -> None:
    assert cli_main(["--config", str(TOY), "--check"]) == 0
    capsys.readouterr()
    assert cli_main(["--config", str(TEMPLATE), "--check"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blockers"]["compare"] == ["parents.k_par", "parents.tau"]
    code = cli_main(["--config", str(TEMPLATE), "--output-dir", str(tmp_path / "x"),
                     "--stage", "representatives"])
    assert code == 3 and not (tmp_path / "x").exists()
