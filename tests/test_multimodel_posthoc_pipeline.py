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
    assert config.blockers("representatives") == [
        "representatives.k_rep", "representatives.tau_rep"]
    assert config.blockers("importance") == ["anchors.importance_pools"]
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


# ------------------------------------------------- formerly implicit decisions
def _raw() -> dict:
    return yaml.safe_load(TOY.read_text())


def _dump(tmp_path: Path, raw: dict, name: str = "c.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path


def test_edge_conventions_are_fixed_not_configurable(tmp_path: Path) -> None:
    from aspd.multimodel.posthoc.config import FORMULATION_FIXED_CONVENTIONS

    raw = _raw()
    raw["edges"]["conventions"] = {"jacobian_path": "native_source_model"}
    with pytest.raises(ValueError, match="edges.conventions was removed"):
        load_config(_dump(tmp_path, raw))
    assert FORMULATION_FIXED_CONVENTIONS["upstream_set"].endswith("including_same_gate")


def test_screened_selection_uses_the_screening_step() -> None:
    from aspd.multimodel.posthoc.config import DownstreamGateSelection
    from aspd.multimodel.posthoc.ids import GateId

    fire = {"pre": [5, 0, 9, 3], "mid": [1, 1, 0, 7]}
    support = {GateId("pre", 0): 3, GateId("pre", 1): 3, GateId("pre", 2): 1,
               GateId("pre", 3): 2, GateId("mid", 0): 2, GateId("mid", 3): 3}
    every = DownstreamGateSelection("screened").resolve(fire, support)
    # pre/1 never fires; pre/2 has R in only one model; mid/2 never fires.
    assert every == (GateId("mid", 0), GateId("mid", 3), GateId("pre", 0), GateId("pre", 3))
    capped = DownstreamGateSelection("screened", count=1).resolve(fire, support)
    assert capped == (GateId("mid", 3), GateId("pre", 0))
    floored = DownstreamGateSelection("screened", min_importance_fires=5).resolve(fire, support)
    assert floored == (GateId("mid", 3), GateId("pre", 0))  # pre/3 (3) and mid/0 (1) dropped


def test_seeded_random_sampling_is_reproducible_and_needs_a_seed(tmp_path: Path) -> None:
    from aspd.multimodel.posthoc.ids import GateId
    from aspd.multimodel.posthoc.pipeline import sample_anchor_rows

    raw = _raw()
    random_config = load_config(_dump(tmp_path, raw, "random.yaml"))
    rows = list(range(0, 40, 2))
    gate = GateId("pre", 3)
    first = sample_anchor_rows(rows, 5, random_config, pool_index=7, gate=gate)
    assert first == sample_anchor_rows(rows, 5, random_config, pool_index=7, gate=gate)
    assert first != sample_anchor_rows(rows, 5, random_config, pool_index=8, gate=gate)
    assert len(first) == 5 and set(first) <= set(rows) and first == sorted(first)
    raw["edges"]["anchor_sampling"] = "first_in_pool"
    ordered = load_config(_dump(tmp_path, raw, "first.yaml"))
    assert sample_anchor_rows(rows, 5, ordered, pool_index=7, gate=gate) == rows[:5]
    raw["edges"]["anchor_sampling"] = "seeded_random"
    raw["edges"]["anchor_seed"] = None
    assert load_config(_dump(tmp_path, raw, "noseed.yaml")).blockers("edges") == [
        "edges.anchor_seed"]


@pytest.mark.parametrize(("source", "first_pool"), [
    ("disjoint_after_importance", 3), ("same_as_importance", 0)])
def test_edge_anchor_source_selects_the_pools(tmp_path: Path, source: str,
                                              first_pool: int) -> None:
    raw = _raw()
    raw["anchors"]["edge_anchor_source"] = source
    config = load_config(_dump(tmp_path, raw))
    root = tmp_path / "run"
    run_stages(config, root, ("importance", "representatives", "edges"))
    selection = json.loads((root / "edges/selection.json").read_text())
    pools = {anchor.split("-")[1] for ids in selection["anchors"].values() for anchor in ids}
    assert f"p{first_pool}" in pools
    importance_pools = {f"p{i}" for i in range(3)}
    if source == "disjoint_after_importance":
        assert not pools & importance_pools
    else:
        assert pools <= importance_pools | {"p3", "p4", "p5"} and pools & importance_pools
    manifest = json.loads((root / "edges/manifest.json").read_text())
    assert manifest["conventions"]["edge_anchor_source"] == source
    assert manifest["first_pool_index"] == first_pool


# ------------------------------------------------------------- sharded edge stage
def _edges_by_key(root: Path) -> dict:
    from aspd.multimodel.posthoc.artifacts import load_edge_table

    return {(r.anchor_id, r.downstream.key): r for r in load_edge_table(root)}


def test_sharded_edges_equal_unsharded(toy_run: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "sharded"
    for stage in ("importance", "representatives"):
        shutil.copytree(toy_run / stage, root / stage)
    config = load_config(TOY)
    for index in range(3):
        run_stages(config, root, ("edges",), shard=(index, 3))
    with pytest.raises(FileNotFoundError):  # compare refuses before finalization
        run_stages(config, root, ("compare",))
    shutil.rmtree(root / "compare", ignore_errors=True)
    run_stages(config, root, ("compare",), finalize_shards=3)
    # Discrete structure is identical; scores differ only by float32 rounding because a
    # shard batches different anchors together (different padded lengths).
    sharded, single = _edges_by_key(root), _edges_by_key(toy_run)
    assert sharded.keys() == single.keys()
    for key, record in sharded.items():
        assert record.upstream == single[key].upstream
        torch.testing.assert_close(record.scores, single[key].scores, rtol=1e-4, atol=1e-6)

    def rows(path: Path) -> dict:
        return {(r["gate"], r["anchor_id"], tuple(r["models"])): r
                for r in map(json.loads, path.read_text().splitlines())}

    a, b = rows(root / "compare/comparisons.jsonl"), rows(toy_run / "compare/comparisons.jsonl")
    assert a.keys() == b.keys()
    for key in a:
        assert a[key]["parent_gate_sets"] == b[key]["parent_gate_sets"]
        assert a[key]["jaccard"] == b[key]["jaccard"]
        if a[key]["total_variation"] is not None:
            assert a[key]["total_variation"] == pytest.approx(b[key]["total_variation"], abs=1e-5)
    selection = json.loads((root / "edges/selection.json").read_text())
    reference = json.loads((toy_run / "edges/selection.json").read_text())
    assert selection["anchors"] == reference["anchors"]
    assert json.loads((root / "edges/shards.json").read_text())["shards"][0]["stage"] == (
        "edges-shard-000-of-003")


def test_finalize_rejects_missing_or_tampered_shards(toy_run: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "partial"
    for stage in ("importance", "representatives"):
        shutil.copytree(toy_run / stage, root / stage)
    config = load_config(TOY)
    run_stages(config, root, ("edges",), shard=(0, 2))
    with pytest.raises(FileNotFoundError, match="edges-shard-001-of-002"):
        run_stages(config, root, (), finalize_shards=2)
    run_stages(config, root, ("edges",), shard=(1, 2))
    with (root / "edges-shard-001-of-002/selection.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed after its stage completed"):
        run_stages(config, root, (), finalize_shards=2)


def test_result_files_are_group_readable(toy_run: Path) -> None:
    import stat

    for path in (toy_run / "edges/edges.safetensors", toy_run / "importance/importance.safetensors"):
        assert path.stat().st_mode & stat.S_IRGRP, path
