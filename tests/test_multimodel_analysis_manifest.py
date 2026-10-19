from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aspd.multimodel.cli.analyze import _write_analysis_manifest


def test_analysis_manifest_binds_checkpoint_cache_and_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    for name in ("posthoc.safetensors", "taxonomy.json", "top_activation_examples.json"):
        (output / name).write_bytes(name.encode())
    config = tmp_path / "config.yaml"
    config.write_text("name: fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = SimpleNamespace(
        name="fixture",
        cache=SimpleNamespace(root=tmp_path / "cache"),
        models=[SimpleNamespace(name="base"), SimpleNamespace(name="target")],
        sparsity=SimpleNamespace(n_features=8, top_k=2),
        analysis=SimpleNamespace(example_feature_mode="ranked"),
    )
    cache = SimpleNamespace(token_manifest={"cache_fingerprint": "cache-sha"})

    _write_analysis_manifest(
        output=output,
        config_path=config,
        checkpoint_path=checkpoint,
        checkpoint_state={"step": 200},
        cache=cache,
        cfg=cfg,
    )

    payload = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert payload["checkpoint"]["step"] == 200
    assert payload["cache"]["fingerprint"] == "cache-sha"
    assert set(payload["artifacts"]) == {
        "posthoc.safetensors",
        "taxonomy.json",
        "top_activation_examples.json",
    }
