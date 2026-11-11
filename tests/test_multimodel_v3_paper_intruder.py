"""Tests for the fail-closed ASPD v3.1 paper Intruder adapter."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("param_decomp_lab")

from aspd.multimodel.cli.extract_aspd_v3_paper_intruder import (
    _context_rows_for_feature,
)
from aspd.multimodel.cli.judge_aspd_v3_paper_intruder import _position_audit
from aspd.multimodel.cli.prepare_aspd_v3_paper_intruder import _component
from aspd.multimodel.v3.paper_intruder import PaperIntruderContract, component_key
from aspd.multimodel.v3.production_stream import AlignedAnchorPool, ModelPoolInputs


def _contract() -> PaperIntruderContract:
    return PaperIntruderContract(
        schema_version=1,
        experiment_family="aspd_v3_1_paper_intruder_v1",
        status="RESEARCHER_AUTHORIZED_EVALUATION",
        training_authorized_by_this_contract=False,
        source_config_sha256="a" * 64,
        checkpoint_step=13000,
        checkpoint_manifest_sha256="b" * 64,
        accepted_training_exposures=106_496_000,
        model_order=("gemma", "smol", "granite"),
        forbidden_model_substrings=("qwen",),
        selected_split="test",
        accepted_evaluation_anchors=100_352,
        pool_anchors=2048,
        display_model_key="gemma",
        context_tokens_per_side=20,
        activation_examples_per_target=400,
        minimum_examples_per_component=5,
        n_subsample=200,
        seed=0,
        n_real=4,
        n_trials=10,
        density_tolerance=0.05,
        chance_accuracy=0.2,
        judge_model_id="unsloth/Llama-3.3-70B-Instruct",
        judge_revision="c" * 40,
        judge_quantization="none",
        judge_backend="vllm_openai_compatible",
    )


def test_contract_pins_paper_intruder_defaults_and_v3_gate_budget() -> None:
    contract = _contract()
    assert contract.n_subsample == 200
    assert contract.n_real == 4
    assert contract.n_trials == 10
    assert contract.chance_accuracy == 1 / 5
    assert contract.accepted_evaluation_anchors % contract.pool_anchors == 0
    assert component_key("pre", 7) == "pre/0007"
    with pytest.raises(ValueError, match="paper Intruder protocol changed"):
        contract.model_copy(update={"n_trials": 9}).model_validate(
            contract.model_copy(update={"n_trials": 9}).model_dump()
        )


def test_context_rows_preserve_natural_multi_token_firings() -> None:
    inputs = ModelPoolInputs(
        input_ids=((10, 11, 12, 13, 14, 15),),
        attention_masks=((1, 1, 1, 1, 1, 1),),
        anchor_document_indices=(0, 0, 0),
        anchor_token_indices=(1, 2, 4),
    )
    pool = AlignedAnchorPool(
        anchor_ids=("a", "b", "c"),
        document_ids=("doc",),
        models={"gemma": inputs},
        source_observations=({"source_group": "books"},),
    )
    rows = _context_rows_for_feature(
        pool=pool,
        model_key="gemma",
        feature_positions=np.asarray([0, 2], dtype=np.int32),
        feature_values=np.asarray([0.4, 1.2], dtype=np.float32),
        tokens_per_side=20,
    )
    assert len(rows) == 1
    assert rows[0]["token_ids"] == [10, 11, 12, 13, 14, 15]
    assert rows[0]["firings"] == [False, True, False, False, True, False]
    assert rows[0]["activations"][1] == pytest.approx(0.4)
    assert rows[0]["activations"][4] == pytest.approx(1.2)


def test_harvest_component_keeps_paper_activation_example_schema() -> None:
    row = {
        "component_key": "mid/0042",
        "site": "mid",
        "feature": 42,
        "firing_density": 0.01,
        "contexts": [
            {
                "activation": float(index + 1),
                "token_ids": [1, 2, 3],
                "firings": [False, True, False],
                "activations": [0.0, float(index + 1), 0.0],
            }
            for index in range(5)
        ],
    }
    component = _component(row)
    assert component.component_key == "mid/0042"
    assert len(component.activation_examples) == 5
    assert component.activation_examples[0].activations == {
        "activation": [0.0, 1.0, 0.0]
    }


def test_position_audit_reports_true_and_predicted_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "harvest.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE scores (component_key TEXT, score_type TEXT, score REAL, details TEXT)"
        )
        connection.execute(
            "INSERT INTO scores VALUES (?, ?, ?, ?)",
            (
                "pre/0001",
                "intruder",
                0.5,
                json.dumps(
                    {
                        "trials": [
                            {
                                "correct_answer": 1,
                                "predicted": 1,
                                "is_correct": True,
                            },
                            {
                                "correct_answer": 2,
                                "predicted": 1,
                                "is_correct": False,
                            },
                        ]
                    }
                ),
            ),
        )
    audit = _position_audit(db_path)
    assert audit["n_trials"] == 2
    assert audit["true_position_counts"]["1"] == 1
    assert audit["predicted_position_counts"]["1"] == 2
    assert audit["accuracy_by_true_position"]["2"] == 0.0
