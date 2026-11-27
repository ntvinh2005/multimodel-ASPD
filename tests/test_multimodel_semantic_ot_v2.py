import gzip
import json

import numpy as np
import pytest
import torch

from aspd.multimodel.cli.analyze_aspd_semantic_contributions import independent_example
from aspd.multimodel.cli.collect_aspd_semantic_contributions import (
    gated_coefficients,
    token_offsets,
    verified_json,
)
from aspd.multimodel.cli.install_aspd_semantic_discovery import check_examples
from aspd.multimodel.discovery.semantic_ot_v2 import (
    Anchor,
    comparison,
    coverage,
    diverse,
    matched_controls,
)
from aspd.multimodel.posthoc.artifacts import sha256_file


def anchor(key, doc, text, position=3, token=17, start=None):
    start = len(text) - 1 if start is None else start
    return Anchor(
        key, doc, "wiki", 75, len(text), text, {"gemma": [position, token, start, len(text)]}
    )


def test_native_coefficients_preserve_measured_zeros_and_signs():
    x = torch.tensor([[1.0, -2], [3.0, 2]])
    v = torch.tensor([[1.0, 0], [0.0, 1]])
    support = torch.tensor([[False, True], [True, False]])
    torch.testing.assert_close(
        gated_coefficients(x, v, support), torch.tensor([[0.0, -2], [3.0, 0]])
    )
    with pytest.raises(ValueError, match="axes"):
        gated_coefficients(x, v, support[:1])


def test_token_offsets_validate_the_original_input_ids():
    def fake(*_, **kwargs):
        assert kwargs["return_offsets_mapping"]
        return {"input_ids": [1, 17], "offset_mapping": [(0, 0), (1, 4)]}

    assert token_offsets(fake, " test", (1, 17), 10)[1] == (1, 4)
    with pytest.raises(ValueError, match="different native tokens"):
        token_offsets(fake, " test", (1, 18), 10)


def test_deduplication_and_matched_controls():
    rows = [
        anchor("a", "d1", "copy exact context apple"),
        anchor("b", "d1", "different text from same document apple"),
        anchor("c", "d2", "Copy exact context Apple"),
        anchor("d", "d3", "independent prefix ending in apple"),
        anchor("e", "d4", "Similar short match banana", token=17),
    ]
    assert diverse(rows, np.array([3, 2, 1, 1, 0]), [0, 1, 2, 3]) == [0, 3]
    assert diverse(rows, np.ones(len(rows)), [2, 3], forbidden=[0]) == [3]
    found, tiers = matched_controls(rows, "gemma", [0], [1, 4])
    assert found == [4]
    assert tiers == ["source-native-token-position"]


def test_every_endpoint_coverage_and_catalog_mean():
    def comp(count):
        return {"coverage": {"discovery": 20, "heldout": count}}

    assert coverage({"a": comp(10), "b": comp(30)})["meanHeldout"] == 20
    for data in ({"a": comp(9), "b": comp(31)}, {"a": comp(10), "b": comp(29)}, {}):
        with pytest.raises(ValueError, match="No publishable"):
            coverage(data)


def test_transport_includes_anchor_baseline_and_zero_control():
    vectors = np.eye(2)
    values = np.array([[6, 4], [0, 0]])
    positive = comparison([0], [0], vectors, values, 0, 1)[0]
    negative = comparison([0], [1], vectors, values, 0, 1, mass_b=np.array([4]))[0]
    assert (positive["ot"], negative["ot"]) == (0, 1)
    assert positive["anchorOverlap"] == 1 and positive["distinctContextMass"] == 0


def test_heatmap_records_native_tokens_and_a_gap(tmp_path):
    rows = [
        anchor("p", "doc", "hey", position=1, token=11, start=0),
        anchor("q", "doc", "hey lost A", position=5, token=12, start=9),
    ]
    comp = {"model": "gemma", "uNorm": 4, "sigma": 2}
    example = independent_example(
        1,
        0,
        rows,
        np.array([[0.0], [-3.0]]),
        {"doc": [1, 0]},
        {"doc": "hey lost A"},
        comp,
        contrast=False,
    )
    assert [row["text"] for row in example["sequence"]] == ["hey", " lost ", "A"]
    assert [row["magnitude"] for row in example["sequence"]] == [0, None, 6]
    assert example["sequence"][-1]["coefficient"] == -3
    split = rows[0].split
    other = "discovery" if split == "heldout" else "heldout"
    path = tmp_path / "component-0.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(
            {"runId": "test", "component": "c", "examples": {split: [example], other: []}}, stream
        )
    ref = {"file": path.name, "sha256": sha256_file(path)}
    check_examples(
        tmp_path,
        ref,
        "c",
        {
            "runId": "test",
            "components": {
                "c": {"model": "gemma", "coverage": {split: 1, other: 0}},
            },
        },
    )


def test_pool_hash_tampering_fails_closed(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text('{"pool": 75}')
    (tmp_path / "COMPLETE").write_text(json.dumps({"sha256": {"meta.json": sha256_file(meta)}}))
    assert verified_json(tmp_path, "meta.json") == {"pool": 75}
    meta.write_text('{"pool": 76}')
    with pytest.raises(ValueError, match="Invalid artifact hash"):
        verified_json(tmp_path, "meta.json")


def test_analysis_exports_and_installs_only_complete_coverage(tmp_path, monkeypatch):
    from argparse import Namespace
    from collections import defaultdict

    from aspd.multimodel.cli import analyze_aspd_semantic_contributions as cli
    from aspd.multimodel.cli.collect_aspd_semantic_contributions import CHECKPOINT
    from aspd.multimodel.cli.install_aspd_semantic_discovery import activate, verify

    rows, values, documents, texts = [], [], defaultdict(list), {}
    ids = ["gemma.L13.q/1", "smol.L12.q/2"]
    components = [
        {"id": key, "model": model, "sigma": 1.0, "uNorm": 1.0, "beta": 0.1, "gate": f"pre/{i + 1}"}
        for i, (key, model) in enumerate(zip(ids, ("gemma", "smol"), strict=True))
    ]
    for i in range(850):
        model = i % 2
        active = i % 4 < 2
        doc = f"document-{i}"
        prefix = f"{'aligned' if active else 'control'} case-{i} passage item"
        document = Anchor(
            f"anchor-{i}",
            doc,
            f"source-{model}",
            75,
            len(prefix),
            prefix,
            {key: [5, 22, len(prefix) - 4, len(prefix)] for key in ("gemma", "smol")},
        )
        documents[doc].append(len(rows))
        texts[doc] = prefix
        rows.append(document)
        values.append([float(active and model == 0), -float(active and model == 1)])
    contract = {"checkpoint": CHECKPOINT, "components": components}
    monkeypatch.setattr(
        cli,
        "read_measurements",
        lambda _: (
            contract,
            "9" * 64,
            rows,
            np.array(values),
            documents,
            texts,
            [{}],
        ),
    )
    monkeypatch.setattr(cli, "BOOTSTRAP_DRAWS", 3)

    def test_vectors(prefixes, *_):
        z = np.array([[1, 0] if text.startswith("aligned") else [0, 1] for text in prefixes])
        return {name: z for name in (*cli.VARIANTS, "withoutTerminal")}

    monkeypatch.setattr(cli, "embed_variants", test_vectors)
    output = tmp_path / "output"
    cli.analyze(
        Namespace(measurements=None, snapshot=None, output=output, threads=1, coverage_only=False)
    )
    bundle = output / "frontend"
    result = verify(bundle)
    assert result["summary"]["meanHeldout"] >= 20
    assert result["graphCoverage"] == "unavailable"
    workbench = tmp_path / "workbench"
    workbench.mkdir()
    activate(bundle, workbench)
    pointer = json.loads((workbench / "semantic-discovery-release.json").read_text())
    assert pointer["runId"] == result["runId"]
    target = bundle / result["examples"][ids[1]]["file"]
    target.write_bytes(target.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        activate(bundle, workbench)
    assert json.loads((workbench / "semantic-discovery-release.json").read_text()) == pointer


def test_taylor_selection_uses_only_shared_anchors_and_prefers_independent_sources():
    from aspd.multimodel.cli.measure_semantic_ot_graphs import select_anchors

    common = {
        "a": {"pool": 76, "document": "d1", "source": "s1"},
        "b": {"pool": 77, "document": "d1", "source": "s1"},
        "c": {"pool": 78, "document": "d2", "source": "s1"},
        "d": {"pool": 79, "document": "d3", "source": "s2"},
    }
    examples = {
        "gemma": {"unshared": {"pool": 75, "document": "d4", "source": "s3"}, **common},
        "smol": common,
        "granite": common,
    }

    assert select_anchors(examples, 3) == ["a", "d", "c"]
    with pytest.raises(ValueError, match="Only 3 independent common anchors"):
        select_anchors(examples, 4)


def test_taylor_publication_rejects_different_catalog_or_model(tmp_path):
    from aspd.multimodel.cli.install_semantic_ot_graphs import publish
    from aspd.multimodel.posthoc.artifacts import write_json

    root = tmp_path / "workbench"
    run_id = "d" * 64
    target_root = root / "semantic-discovery-v2" / run_id
    target_root.mkdir(parents=True)
    target = "gemma.L16.mlp_gate/515"
    file = target_root / "catalog.json"
    write_json(
        file,
        {
            "dataset": "natural-validation",
            "checkpoint": "test",
            "components": {target: {}},
        },
    )
    digest = sha256_file(file)
    seal = target_root / "COMPLETE.json"
    write_json(seal, {"catalogSha256": digest})
    write_json(
        root / "semantic-discovery-release.json",
        {
            "runId": run_id,
            "completeSha256": sha256_file(seal),
        },
    )
    graph = {
        "schemaVersion": 1,
        "runId": run_id,
        "catalogSha256": digest,
        "dataset": "natural-validation",
        "checkpoint": "test",
        "anchorOrder": ["abc"],
        "anchors": {"abc": {"pool": 75, "split": "heldout"}},
        "graphs": {
            f"{target}|abc": {
                "anchor": "abc",
                "pool": 75,
                "component": target,
                "model": "gemma",
                "edges": [{"source": "gemma.L13.q/2", "target": target, "E": -0.03, "w": 0.008}],
            }
        },
    }

    def sealed_graph(name):
        dest = tmp_path / name
        dest.mkdir()
        write_json(dest / "graphs.json", graph)
        write_json(
            dest / "COMPLETE.json",
            {
                "sha256": sha256_file(dest / "graphs.json"),
                "anchors": 1,
                "targetGraphs": 1,
            },
        )
        return dest

    publication = publish(sealed_graph("valid"), root)
    with gzip.open(target_root / ("graphs-" + publication["sha256"] + ".json.gz"), "rt") as handle:
        assert json.load(handle)["graphs"][f"{target}|abc"]["edges"][0]["E"] == -0.03
    graph["graphs"][f"{target}|abc"]["edges"][0]["source"] = "granite.L18.q/2"
    with pytest.raises(ValueError, match="Invalid Taylor edge"):
        publish(sealed_graph("wrong-model"), root)
    graph["catalogSha256"] = "not-the-current-catalog"
    with pytest.raises(ValueError, match="does not match"):
        publish(sealed_graph("wrong-catalog"), root)


def test_taylor_replay_preserves_full_native_input_batch_and_padding():
    import torch

    from aspd.multimodel.cli.measure_semantic_ot_graphs import gather_original_batch
    from aspd.multimodel.posthoc.backend import ModelInputs

    ids = torch.arange(21).reshape(3, 7)
    mask = torch.ones_like(ids)
    inputs = ModelInputs(ids, mask, torch.tensor([0, 2, 1]), torch.tensor([0, 5, 3]))
    gathered = gather_original_batch(inputs, [2, 0])
    assert gathered.input_ids is ids and gathered.attention_mask is mask
    assert gathered.anchor_document.tolist() == [1, 0]
    assert gathered.anchor_token.tolist() == [3, 0]
