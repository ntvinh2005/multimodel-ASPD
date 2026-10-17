import asyncio
import json
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import load_file, save_file

from aspd.multimodel.config import load_experiment_config
from aspd.multimodel.crosscoder_dashboard import (
    CrosscoderDashboardStore,
    ResearchComparisonStore,
)
from aspd.multimodel.dashboard import AnalysisDashboardStore, build_dashboard_app

ROOT = Path(__file__).parents[1]


def _realistic_fixture(tmp_path: Path) -> tuple[AnalysisDashboardStore, Path]:
    baseline = load_experiment_config(ROOT / "configs/multimodel/qwen3_1_7b/d0_s1_main.yaml")
    cfg = baseline.model_copy(
        update={
            "data": baseline.data.model_copy(update={"validation_tokens": 2}),
            "sparsity": baseline.sparsity.model_copy(update={"n_features": 2, "top_k": 1}),
            "models": [
                model.model_copy(update={"matrices": model.matrices[:2]})
                for model in baseline.models
            ],
        }
    )
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    (run_dir / "experiment_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json")), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"git_commit": "abc123"}), encoding="utf-8"
    )
    (run_dir / "latest_checkpoint.txt").write_text("checkpoint_00000010.pt\n", encoding="utf-8")

    matrices = [matrix.name for matrix in cfg.models[0].matrices]
    tensors = {
        "decoder_norm_NC": torch.tensor([[2.0, 9.0], [2.0, 1.0]]),
        "activation_rho_NC": torch.tensor([[0.5, 0.9], [0.5, 0.1]]),
        "mechanism_beta_NC": torch.tensor([[0.9, 0.8], [0.1, 0.2]]),
        "mechanism_rho_NC": torch.tensor([[0.9, 0.8], [0.1, 0.2]]),
        "fire_count_C": torch.tensor([1.0, 1.0]),
    }
    for model in cfg.models:
        tensors[f"beta/{model.name}/{matrices[0]}"] = torch.tensor([0.7, 0.3])
        tensors[f"beta/{model.name}/{matrices[1]}"] = torch.tensor([0.3, 0.7])
    tensors[f"locus/{cfg.models[0].name}/{matrices[0]}"] = torch.tensor([0.8, 0.2])
    tensors[f"locus/{cfg.models[0].name}/{matrices[1]}"] = torch.tensor([0.2, 0.8])
    tensors[f"locus/{cfg.models[1].name}/{matrices[0]}"] = torch.tensor([0.2, 0.2])
    tensors[f"locus/{cfg.models[1].name}/{matrices[1]}"] = torch.tensor([0.8, 0.8])
    for matrix in matrices:
        tensors[f"pair/{matrix}/component_cosine"] = torch.tensor([0.25, 0.75])
        tensors[f"pair/{matrix}/relative_component_change"] = torch.tensor([1.5, 0.5])
        tensors[f"pair/{matrix}/read_cosine"] = torch.tensor([0.5, 0.8])
        tensors[f"pair/{matrix}/write_cosine"] = torch.tensor([0.5, 0.9])
    save_file(tensors, str(analysis_dir / "posthoc.safetensors"))
    (analysis_dir / "taxonomy.json").write_text(
        json.dumps(
            {
                "shared_activation_concentrated_mechanism": [0],
                "concentrated_activation_shared_mechanism": [],
                "shared_activation_shared_mechanism": [],
                "concentrated_activation_concentrated_mechanism": [1],
                "mixed_or_subset_mass": [],
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "top_activation_examples.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "center_in_window": 1,
                        "center_token": "B",
                        "g_s": 3.25,
                        "text": "A B C",
                        "token_ids": [1, 2, 3],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "vocabulary_effects.json").write_text(
        json.dumps(
            {
                "kind": "direct_projection",
                "description": "decoder direction @ unembedding; not an intervention",
                "features": {
                    "0": {
                        "positive": [
                            {"token": " weak", "token_id": 10, "value": 0.2},
                            {"token": " strong", "token_id": 11, "value": 0.8},
                        ],
                        "negative": [
                            {"token": "less", "token_id": 12, "value": -0.1},
                            {"token": "least", "token_id": 13, "value": -0.9},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return AnalysisDashboardStore(analysis_dir), analysis_dir


def _crosscoder_fixture(tmp_path: Path) -> tuple[CrosscoderDashboardStore, Path, Path, Path]:
    run_dir = tmp_path / "crosscoder"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    rows = [
        {
            "feature_id": 0,
            "classification": "base-specific",
            "relative_decoder_norm": 0.05,
            "decoder_norm_base": 1.0,
            "decoder_norm_finetuned": 0.05,
            "decoder_cosine": 0.1,
            "fire_count": 4,
            "activation_density": 0.4,
            "blind_rank_within_class": 1,
            "blind_rank_base_tail": 1,
            "blind_rank_finetuned_tail": 4,
        },
        {
            "feature_id": 1,
            "classification": "shared",
            "relative_decoder_norm": 0.5,
            "decoder_norm_base": 1.0,
            "decoder_norm_finetuned": 1.0,
            "decoder_cosine": 1.0,
            "fire_count": 5,
            "activation_density": 0.5,
            "blind_rank_within_class": 1,
            "blind_rank_base_tail": 2,
            "blind_rank_finetuned_tail": 3,
        },
        {
            "feature_id": 2,
            "classification": "intermediate",
            "relative_decoder_norm": 0.72,
            "decoder_norm_base": 0.4,
            "decoder_norm_finetuned": 1.0,
            "decoder_cosine": 0.7,
            "fire_count": 1,
            "activation_density": 0.1,
            "blind_rank_within_class": 1,
            "blind_rank_base_tail": 3,
            "blind_rank_finetuned_tail": 2,
        },
        {
            "feature_id": 3,
            "classification": "finetuned-specific",
            "relative_decoder_norm": 0.95,
            "decoder_norm_base": 0.05,
            "decoder_norm_finetuned": 1.0,
            "decoder_cosine": 0.2,
            "fire_count": 0,
            "activation_density": 0.0,
            "blind_rank_within_class": 1,
            "blind_rank_base_tail": 4,
            "blind_rank_finetuned_tail": 1,
        },
    ]
    summary = {
        "n_features": 4,
        "k": 2,
        "tokens": 10,
        "l0": 2.1,
        "fve_total": 0.9,
        "fve_base": 0.89,
        "fve_finetuned": 0.91,
        "dead_fraction": 0.25,
        "relative_decoder_norm_convention": "0=base-specific, 1=finetuned-specific",
        "classification_counts": {
            "base-specific": 1,
            "shared": 1,
            "intermediate": 1,
            "finetuned-specific": 1,
        },
        "classification_fractions": {
            "base-specific": 0.25,
            "shared": 0.25,
            "intermediate": 0.25,
            "finetuned-specific": 0.25,
        },
        "thresholds": {
            "base_specific": [0.0, 0.1],
            "shared": [0.4, 0.6],
            "finetuned_specific": [0.9, 1.0],
        },
        "candidate_groups": {"base-tail": [0], "shared": [1], "finetuned-tail": [2]},
        "candidate_selection": "blind score tails",
        "checkpoint": "checkpoint_10.pt",
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (analysis_dir / "feature_metrics.json").write_text(
        json.dumps({"metadata": summary, "features": rows}), encoding="utf-8"
    )
    (analysis_dir / "top_activation_examples.json").write_text(
        json.dumps(
            {
                "metadata": {"feature_ids": [2]},
                "features": {
                    "2": [
                        {
                            "center_in_window": 1,
                            "center_token": "B",
                            "g_s": 2.0,
                            "position_in_sequence": 8,
                            "text": "A B C",
                            "token_ids": [1, 2, 3],
                            "token_texts": ["A", " B", " C"],
                            "token_strings": ["A", "ĠB", "ĠC"],
                            "token_activations": [0.0, 2.0, None],
                            "validation_sequence_index": 7,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "semantic_annotations.json").write_text(
        json.dumps({"metadata": {}, "annotations": []}), encoding="utf-8"
    )
    (run_dir / "config_resolved.yaml").write_text(
        """name: fixture_crosscoder
models:
  - {name: base, model_id: org/base, grounding: {module: model.layers.1, capture: input}}
  - {name: finetuned, model_id: org/ft, grounding: {module: model.layers.1, capture: input}}
data: {sequence_length: 16}
crosscoder: {max_steps: 10}
""",
        encoding="utf-8",
    )
    (run_dir / "reference.json").write_text(
        json.dumps({"method": "BatchTopKCrosscoder"}), encoding="utf-8"
    )
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 10}) + "\n", encoding="utf-8")
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "annotation_source": "llm_assisted_researcher_review",
                    "limitations": ["qualitative only"],
                },
                "crosscoder": [
                    {
                        "feature_id": 2,
                        "measured_classification": "finetuned-specific",
                        "tentative_label": "turn boundary",
                        "confidence": 0.7,
                        "evidence_validation_sequence_indices": [7],
                    }
                ],
                "aspd": [
                    {
                        "feature_id": 0,
                        "tentative_label": "possible boundary",
                        "confidence": 0.4,
                        "fire_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "metadata": {"not_feature_identity": True, "not_causal": True},
                "global_summary": {"current_finding": "fixture synthesis"},
                "phenomena": [
                    {
                        "id": "boundary",
                        "label": "Boundary",
                        "status": "low_support_hypothesis",
                        "crosscoder_features": [2],
                        "aspd_components": [0],
                        "interpretation": "qualitative only",
                        "limitations": ["not matched"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return (
        CrosscoderDashboardStore(analysis_dir, annotations_path=annotations_path),
        analysis_dir,
        annotations_path,
        comparison_path,
    )


def test_dashboard_maps_real_tensor_indices_to_feature_payload(tmp_path: Path) -> None:
    store, _ = _realistic_fixture(tmp_path)
    detail = store.feature_detail(0)

    assert store.health["overall_status"] == "PASS"
    assert store.model_names == ["base", "finetuned"]
    assert store.matrix_names == ["k_proj", "q_proj"]
    assert detail["activation_rho"] == [0.5, 0.5]
    assert detail["beta_total"] == pytest.approx(1.0)
    assert detail["fire_density"] == 0.5
    assert detail["dominant_locus"] == {"base": "q_proj", "finetuned": "k_proj"}
    assert detail["locus_statement"] == "Dominant locus changed: q_proj → k_proj"
    assert detail["examples"][0]["text"] == "A B C"
    assert detail["examples"][0]["activation_detail_kind"] == "center_only"
    assert detail["examples"][0]["tokens"] == [
        {
            "text": None,
            "exact_token": None,
            "token_id": 1,
            "position": 0,
            "sequence_position": None,
            "activation": None,
            "is_center": False,
            "source": 'top_activation_examples.json → "0"[0]',
        },
        {
            "text": "B",
            "exact_token": None,
            "token_id": 2,
            "position": 1,
            "sequence_position": None,
            "activation": 3.25,
            "is_center": True,
            "source": 'top_activation_examples.json → "0"[0]',
        },
        {
            "text": None,
            "exact_token": None,
            "token_id": 3,
            "position": 2,
            "sequence_position": None,
            "activation": None,
            "is_center": False,
            "source": 'top_activation_examples.json → "0"[0]',
        },
    ]
    assert detail["vocabulary_effects"]["status"] == "available"
    assert detail["vocabulary_effects"]["kind"] == "direct_projection"
    assert detail["vocabulary_effects"]["label"] == "Direct Logit Projection"
    assert [row["value"] for row in detail["vocabulary_effects"]["positive"]] == [0.8, 0.2]
    assert [row["value"] for row in detail["vocabulary_effects"]["negative"]] == [-0.9, -0.1]
    assert detail["context"] == {
        "status": "extracted",
        "fire_count": 1,
        "saved_contexts": 1,
        "source": "top_activation_examples.json",
        "message": "Positive activation examples were extracted and retained.",
    }
    assert detail["sources"]["activation_rho"].endswith("activation_rho_NC[:,0]")


def test_dashboard_exposes_full_per_token_values_only_when_artifact_contains_them(
    tmp_path: Path,
) -> None:
    _, analysis_dir = _realistic_fixture(tmp_path)
    examples_path = analysis_dir / "top_activation_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    examples["0"][0].update(
        {
            "token_texts": ["A", " B", " C"],
            "token_strings": ["A", "ĠB", "ĠC"],
            "token_activations": [0.1, 3.25, 0.4],
            "position_in_sequence": 7,
        }
    )
    examples_path.write_text(json.dumps(examples), encoding="utf-8")

    detail = AnalysisDashboardStore(analysis_dir).feature_detail(0)

    assert detail["examples"][0]["activation_detail_kind"] == "per_token"
    assert [token["text"] for token in detail["examples"][0]["tokens"]] == ["A", " B", " C"]
    assert [token["activation"] for token in detail["examples"][0]["tokens"]] == [
        0.1,
        3.25,
        0.4,
    ]
    assert [token["sequence_position"] for token in detail["examples"][0]["tokens"]] == [
        6,
        7,
        8,
    ]


def test_dashboard_vocabulary_effects_missing_state_does_not_invent_values(tmp_path: Path) -> None:
    _, analysis_dir = _realistic_fixture(tmp_path)
    (analysis_dir / "vocabulary_effects.json").unlink()

    payload = AnalysisDashboardStore(analysis_dir).feature_detail(0)["vocabulary_effects"]

    assert payload["status"] == "missing"
    assert payload["kind"] is None
    assert payload["positive"] == []
    assert payload["negative"] == []
    assert "No direct vocabulary projection" in payload["description"]


def test_dashboard_keeps_model_specific_vocabulary_projections_separate(tmp_path: Path) -> None:
    _, analysis_dir = _realistic_fixture(tmp_path)
    (analysis_dir / "vocabulary_effects.json").write_text(
        json.dumps(
            {
                "kind": "direct_projection",
                "default_model": "finetuned",
                "features": {
                    "0": {
                        "models": {
                            "base": {
                                "positive": [{"token": " base", "value": 0.4}],
                                "negative": [{"token": " no-base", "value": -0.3}],
                            },
                            "finetuned": {
                                "positive": [{"token": " tuned", "value": 0.8}],
                                "negative": [{"token": " no-tuned", "value": -0.7}],
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = AnalysisDashboardStore(analysis_dir).feature_detail(0)["vocabulary_effects"]

    assert payload["status"] == "available"
    assert payload["default_model"] == "finetuned"
    assert [model["model_name"] for model in payload["models"]] == ["base", "finetuned"]
    assert payload["positive"][0]["token"] == " tuned"
    assert payload["models"][0]["positive"][0]["source"].endswith(
        "features.0.models.base.positive[0]"
    )


def test_dashboard_distinguishes_unextracted_from_extracted_empty_contexts(tmp_path: Path) -> None:
    store, analysis_dir = _realistic_fixture(tmp_path)

    unextracted = store.feature_detail(1)
    assert unextracted["fire_count"] == 1
    assert unextracted["examples"] is None
    assert unextracted["context"]["status"] == "not_extracted"
    assert "fired 1 times" in unextracted["context"]["message"]
    assert "does not mean that it was inactive" in unextracted["context"]["message"]
    assert store.health["context_coverage"] == {
        "components": 2,
        "components_with_context_extraction": 1,
        "components_with_at_least_one_saved_context": 1,
        "components_not_extracted": 1,
        "components_with_fire_count_gt_zero_but_no_extracted_context": 1,
        "configured_mode": "ranked",
        "source": "top_activation_examples.json",
    }

    examples_path = analysis_dir / "top_activation_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    examples["1"] = []
    examples_path.write_text(json.dumps(examples), encoding="utf-8")
    complete = AnalysisDashboardStore(analysis_dir)
    extracted_empty = complete.feature_detail(1)
    assert extracted_empty["examples"] == []
    assert extracted_empty["context"]["status"] == "extracted"
    assert extracted_empty["context"]["message"] == (
        "No positive activation examples were retained for this component."
    )
    assert complete.health["context_coverage"]["components_not_extracted"] == 0


def test_dashboard_api_notes_and_exports_do_not_modify_raw_artifacts(tmp_path: Path) -> None:
    store, analysis_dir = _realistic_fixture(tmp_path)
    raw_before = {
        path.name: path.read_bytes()
        for path in analysis_dir.iterdir()
        if path.name != "researcher_notes.json"
    }
    note = {
        "tentative_label": "manual label",
        "notes": "manual notes",
        "semantic_evidence": "repeated context",
        "alternative_interpretation": "formatting",
        "confidence": "medium",
        "why_interesting": "rho mismatch",
        "mentor_notes": "review",
        "candidate_status": "promising",
    }
    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=build_dashboard_app(store))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/verification")).json()["components_discovered"] == 2
            assert (await client.get("/api/overview")).json()["points"][0]["feature_id"] == 0
            assert (await client.get("/api/features/0")).json()["p5"][0]["component_cosine"] == 0.25
            assert (await client.get("/api/compare?ids=0&ids=1")).status_code == 200
            assert (await client.put("/api/notes/0", json=note)).status_code == 200
            assert "# Feature 0" in (await client.get("/api/export/feature/0.md")).text
            assert "feature_id,taxonomy" in (await client.get("/api/export/features.csv")).text

    asyncio.run(exercise_api())
    assert json.loads((analysis_dir / "researcher_notes.json").read_text())["0"] == note
    for name, content in raw_before.items():
        assert (analysis_dir / name).read_bytes() == content


def test_dashboard_health_reports_missing_data_without_zero_fill(tmp_path: Path) -> None:
    store, analysis_dir = _realistic_fixture(tmp_path)
    tensors = load_file(str(analysis_dir / "posthoc.safetensors"))
    del tensors["mechanism_beta_NC"]
    save_file(tensors, str(analysis_dir / "posthoc.safetensors"))
    partial = AnalysisDashboardStore(analysis_dir)

    detail = partial.feature_detail(0)
    assert partial.health["overall_status"] == "FAIL"
    assert detail["beta"] is None
    assert detail["beta_total"] is None
    assert any("mechanism_beta_NC" in diagnostic for diagnostic in detail["missing"])


def test_crosscoder_dashboard_preserves_strict_class_and_separate_tail_group(
    tmp_path: Path,
) -> None:
    store, _, _, _ = _crosscoder_fixture(tmp_path)

    overview = store.overview()
    detail = store.feature_detail(2)

    assert overview["available"] is True
    assert overview["metrics"]["dead_count"] == 1
    assert sum(bin_["count"] for bin_ in overview["relative_norm_histogram"]) == 4
    assert detail["classification"] == "intermediate"
    assert detail["inspection_groups"] == ["finetuned-tail"]
    assert detail["semantic_annotation"]["tentative_label"] == "turn boundary"
    assert detail["semantic_annotation"]["is_measured_ground_truth"] is False
    # The annotation deliberately contains a conflicting class; it must never override measurement.
    assert detail["classification"] != "finetuned-specific"
    assert detail["examples"][0]["tokens"][0]["activation"] == 0.0
    assert detail["examples"][0]["tokens"][2]["activation"] is None


def test_crosscoder_dashboard_exposes_optional_triage_without_overwriting_measurements(
    tmp_path: Path,
) -> None:
    _, analysis_dir, annotations_path, _ = _crosscoder_fixture(tmp_path)
    (analysis_dir / "content_semantic_scan_contexts.json").write_text(
        json.dumps(
            {
                "metadata": {"method": "batchtopk_crosscoder"},
                "features": {
                    "2": [
                        {
                            "center_in_window": 0,
                            "center_token": "topic",
                            "g_s": 3.0,
                            "position_in_sequence": 4,
                            "text": "topic context",
                            "token_ids": [4],
                            "token_texts": ["topic"],
                            "token_strings": ["topic"],
                            "token_activations": [3.0],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "content_semantic_scan.json").write_text(
        json.dumps(
            {
                "metadata": {"status": "post_ranking_semantic_triage"},
                "features": [
                    {
                        "feature_id": 2,
                        "measured": {"strict_classification": "malicious-overwrite"},
                        "deterministic_structural": {
                            "structural_dominated": False,
                            "structural_center_count": 0,
                            "structural_center_fraction": 0.0,
                            "detected_center_categories": [],
                        },
                        "context_support": {"requested": 8, "available": 8, "used": 8},
                        "judge": {
                            "status": "judged",
                            "semantic_type": "topic_content",
                            "tentative_label": "topic discussion",
                            "coherence": 2,
                            "content_level": True,
                            "confidence": 2,
                            "evidence_context_ids": [0],
                            "counterexample_context_ids": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = CrosscoderDashboardStore(analysis_dir, annotations_path=annotations_path)
    detail = store.feature_detail(2)

    assert detail["classification"] == "intermediate"
    assert detail["semantic_triage"]["judge"]["semantic_type"] == "topic_content"
    assert detail["semantic_triage"]["is_measured_ground_truth"] is False
    assert detail["examples"][0]["center_token"] == "topic"
    assert detail["context"]["source"] == "crosscoder/content_semantic_scan_contexts.json"


def test_crosscoder_dashboard_missing_artifacts_are_explicitly_unavailable(tmp_path: Path) -> None:
    store = CrosscoderDashboardStore(tmp_path / "missing")

    assert store.available is False
    assert store.overview()["available"] is False
    assert store.verification_summary()["diagnostics"]
    with pytest.raises(IndexError):
        store.feature_detail(0)


def test_research_comparison_resolves_only_valid_structured_references(tmp_path: Path) -> None:
    aspd_store, _ = _realistic_fixture(tmp_path)
    crosscoder_store, _, annotations_path, comparison_path = _crosscoder_fixture(tmp_path)
    comparison = ResearchComparisonStore(
        comparison_path,
        aspd_store=aspd_store,
        crosscoder_store=crosscoder_store,
        annotations_path=annotations_path,
    )

    payload = comparison.payload()

    assert comparison.available is True
    assert payload["phenomena"][0]["crosscoder_evidence"][0]["classification"] == "intermediate"
    assert payload["phenomena"][0]["aspd_evidence"][0]["taxonomy"] == (
        "shared_activation_concentrated_mechanism"
    )
    assert comparison.related_to_aspd(0)[0]["crosscoder_feature_ids"] == [2]
    assert comparison.related_to_crosscoder(2)[0]["aspd_component_ids"] == [0]

    invalid = json.loads(comparison_path.read_text(encoding="utf-8"))
    invalid["phenomena"][0]["crosscoder_features"] = [999]
    comparison_path.write_text(json.dumps(invalid), encoding="utf-8")
    broken = ResearchComparisonStore(
        comparison_path,
        aspd_store=aspd_store,
        crosscoder_store=crosscoder_store,
        annotations_path=annotations_path,
    )
    assert broken.available is False
    assert broken.phenomena == []
    assert "unresolved references" in broken.diagnostics[-1]


def test_research_comparison_missing_artifact_is_clean_unavailable_state(tmp_path: Path) -> None:
    aspd_store, _ = _realistic_fixture(tmp_path)
    crosscoder_store, _, annotations_path, _ = _crosscoder_fixture(tmp_path)
    comparison = ResearchComparisonStore(
        tmp_path / "missing-comparison.json",
        aspd_store=aspd_store,
        crosscoder_store=crosscoder_store,
        annotations_path=annotations_path,
    )

    assert comparison.available is False
    assert comparison.payload()["phenomena"] == []


def test_dashboard_api_adds_crosscoder_and_comparison_without_changing_aspd_routes(
    tmp_path: Path,
) -> None:
    aspd_store, _ = _realistic_fixture(tmp_path)
    crosscoder_store, _, annotations_path, comparison_path = _crosscoder_fixture(tmp_path)
    comparison = ResearchComparisonStore(
        comparison_path,
        aspd_store=aspd_store,
        crosscoder_store=crosscoder_store,
        annotations_path=annotations_path,
    )

    async def exercise_api() -> None:
        app = build_dashboard_app(
            aspd_store,
            crosscoder_store=crosscoder_store,
            comparison_store=comparison,
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/overview")).status_code == 200
            cc_overview = (await client.get("/api/crosscoder/overview")).json()
            assert cc_overview["metadata"]["model_ids"] == ["org/base", "org/ft"]
            cc_feature = (await client.get("/api/crosscoder/features/2")).json()
            assert cc_feature["classification"] == "intermediate"
            assert cc_feature["related_aspd"][0]["aspd_component_ids"] == [0]
            aspd_feature = (await client.get("/api/features/0")).json()
            assert aspd_feature["related_crosscoder"][0]["crosscoder_feature_ids"] == [2]
            assert (await client.get("/api/research-comparison")).json()["available"] is True

    asyncio.run(exercise_api())
