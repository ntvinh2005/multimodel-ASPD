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


# ------------------------------------------------------------------- profiles
@pytest.fixture(scope="module")
def profiled(toy_run: Path, tmp_path_factory) -> Path:
    import shutil

    root = tmp_path_factory.mktemp("profiles") / "run"
    for stage in ("importance", "representatives", "edges", "compare"):
        shutil.copytree(toy_run / stage, root / stage)
    run_stages(load_config(TOY), root, ("profiles",))
    return root


def test_parents_per_anchor_equal_parent_set(profiled: Path) -> None:
    from safetensors.torch import load_file

    from aspd.multimodel.posthoc.artifacts import load_edge_table
    from aspd.multimodel.posthoc.ids import ComponentId
    from aspd.multimodel.posthoc.parents import parent_set

    rule = load_config(TOY).parent_rule()
    rows = load_file(str(profiled / "profiles/parents_per_anchor.safetensors"))
    vocab = json.loads((profiled / "profiles/parents_vocab.json").read_text())
    stored: dict = {}
    for k in range(rows["E"].numel()):
        key = (vocab["anchor_ids"][int(rows["anchor"][k])], vocab["components"][int(rows["downstream"][k])])
        stored.setdefault(key, []).append(
            (int(rows["rank"][k]), vocab["components"][int(rows["parent"][k])], float(rows["E"][k])))
    table = load_edge_table(profiled)
    checked = 0
    for (anchor, down), entries in stored.items():
        expected = parent_set(table.get(anchor, ComponentId.from_key(down)), rule)
        entries.sort()
        assert [e[1] for e in entries] == [p.component.key for p in expected]
        assert [e[2] for e in entries] == pytest.approx([p.score for p in expected])
        checked += 1
    assert checked > 0


def test_gate_profiles_match_compare_and_pi_sums_to_one(profiled: Path) -> None:
    from aspd.multimodel.posthoc.profiles import GateModelProfile

    lines = (profiled / "profiles/gate_parent_profile.jsonl").read_text().splitlines()
    gates = {row["gate"]: row for row in map(json.loads, lines)}
    selection = json.loads((profiled / "edges/selection.json").read_text())
    assert set(gates) == set(selection["gates"])
    for line in (profiled / "profiles/component_parent_profile.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert 0 < row["anchors"] and all(0 < p["frequency"] <= 1 for p in row["top_parents"])
    # unit check of the pi aggregation: mean pi over valid anchors is a distribution
    from aspd.multimodel.posthoc.ids import ComponentId, MatrixId
    from aspd.multimodel.posthoc.parents import Parent

    a = ComponentId(MatrixId("m", 3, "o"), 0)
    b1, b2 = ComponentId(MatrixId("m", 2, "v"), 1), ComponentId(MatrixId("m", 2, "q"), 2)
    profile = GateModelProfile((a,))
    profile.add({a: (Parent(b1, 1.0), Parent(b2, 3.0))}, {a: 1.0}, {a: 1.0})
    profile.add({a: (Parent(b1, 2.0),)}, {a: 1.0}, {a: 1.0})
    profile.add({}, {a: 1.0}, {a: 1.0})  # invalid anchor: M = 0
    assert (profile.anchors, profile.valid) == (3, 2)
    assert sum(profile.mean_pi().values()) == pytest.approx(1.0)
    assert profile.mean_pi()[b1.gate] == pytest.approx((0.25 + 1.0) / 2)


def test_edge_profile_counts_every_stored_edge(profiled: Path) -> None:
    from safetensors.torch import load_file

    from aspd.multimodel.posthoc.artifacts import load_edge_table

    profile = json.loads((profiled / "profiles/edge_profile.json").read_text())
    stored = sum(len(r) for r in load_edge_table(profiled))
    assert sum(c["count"] for c in profile["all_stored_edges"]["cells"]) == stored
    parents = load_file(str(profiled / "profiles/parents_per_anchor.safetensors"))["E"].numel()
    assert sum(c["count"] for c in profile["parents_only"]["cells"]) == parents
    assert all(c["layer_gap"] >= 0 for c in profile["all_stored_edges"]["cells"])


def test_profile_parent_gate_sets_equal_compare(profiled: Path) -> None:
    """G_{p,i}^{(n)}(alpha) rebuilt from parents_per_anchor equals compare's sets."""

    from safetensors.torch import load_file

    from aspd.multimodel.posthoc.ids import ComponentId

    rows = load_file(str(profiled / "profiles/parents_per_anchor.safetensors"))
    vocab = json.loads((profiled / "profiles/parents_vocab.json").read_text())
    rebuilt: dict = {}
    for k in range(rows["E"].numel()):
        down = ComponentId.from_key(vocab["components"][int(rows["downstream"][k])])
        parent = ComponentId.from_key(vocab["components"][int(rows["parent"][k])])
        key = (down.gate.key, vocab["anchor_ids"][int(rows["anchor"][k])], down.model)
        rebuilt.setdefault(key, set()).add(parent.gate.key)
    compared = 0
    for line in (profiled / "compare/comparisons.jsonl").read_text().splitlines():
        row = json.loads(line)
        for model, gate_set in zip(row["models"], row["parent_gate_sets"], strict=True):
            assert set(gate_set) == rebuilt.get((row["gate"], row["anchor_id"], model), set())
            compared += 1
    assert compared > 0


def test_per_anchor_S_and_pi_match_compare(profiled: Path) -> None:
    from safetensors.torch import load_file

    t = load_file(str(profiled / "profiles/parent_distributions.safetensors"))
    vocab = json.loads((profiled / "profiles/distributions_vocab.json").read_text())
    selection = json.loads((profiled / "edges/selection.json").read_text())
    expected_rows = sum(len(a) for a in selection["anchors"].values()) * len(vocab["models"])
    assert t["header_gate"].numel() == expected_rows
    # pi sums to one on valid rows and S sums to M on every row
    pi_sum = torch.zeros(expected_rows, dtype=torch.float64).index_add_(
        0, t["entry_row"], torch.nan_to_num(t["entry_pi"]))
    s_sum = torch.zeros(expected_rows, dtype=torch.float64).index_add_(0, t["entry_row"], t["entry_S"])
    valid = t["header_valid"]
    torch.testing.assert_close(pi_sum[valid], torch.ones(int(valid.sum()), dtype=torch.float64))
    torch.testing.assert_close(s_sum, t["header_M"])
    strength = {}
    for r in range(expected_rows):
        key = (vocab["gates"][int(t["header_gate"][r])], vocab["anchor_ids"][int(t["header_anchor"][r])],
               vocab["models"][int(t["header_model"][r])])
        strength[key] = float(t["header_M"][r])
    for line in (profiled / "compare/comparisons.jsonl").read_text().splitlines():
        row = json.loads(line)
        for model, m in zip(row["models"], row["strength"], strict=True):
            assert strength[(row["gate"], row["anchor_id"], model)] == pytest.approx(m)


def test_default_pipeline_includes_profiles(tmp_path: Path) -> None:
    root = tmp_path / "all"
    assert cli_main(["--config", str(TOY), "--output-dir", str(root)]) == 0
    for stage in ("importance", "representatives", "edges", "compare", "profiles"):
        assert (root / stage / "COMPLETE").is_file(), stage
