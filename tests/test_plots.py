"""Figure grouping, and that every figure draws every metric it is given."""

import json

import pytest

from aspd.eval.plots import metrics as M
from aspd.eval.plots.style import MAX_SERIES, chunk_series, compact, ordinal_colors, series_color

CE_KL_METRICS = [
    "ce_ci_masked_nodelta", "ce_ci_masked_wd", "ce_difference_ci_masked",
    "ce_difference_ci_masked_nodelta", "ce_difference_ci_masked_wd",
    "ce_difference_random_masked", "ce_difference_rounded_masked", "ce_difference_stoch_masked",
    "ce_difference_unmasked", "ce_difference_unmasked_nodelta", "ce_difference_unmasked_wd",
    "ce_target", "ce_unmasked_nodelta", "ce_unmasked_wd", "ce_unrecovered_ci_masked",
    "ce_unrecovered_random_masked", "ce_unrecovered_rounded_masked",
    "ce_unrecovered_stoch_masked", "ce_unrecovered_unmasked", "kl_ci_masked",
    "kl_ci_masked_nodelta", "kl_ci_masked_wd", "kl_random_masked", "kl_residual_contribution",
    "kl_rounded_masked", "kl_stoch_masked", "kl_unmasked", "kl_unmasked_nodelta",
    "kl_unmasked_wd", "kl_zero_masked",
]
SCR_METRICS = [f"scr_{d}_threshold_{k}" for d in ("dir1", "dir2", "metric")
               for k in (2, 5, 10, 20, 50, 100, 500)]
TPP_METRICS = [f"tpp_threshold_{k}_{s}" for k in (2, 5, 10, 20, 50, 100, 500)
               for s in ("total_metric", "intended_diff_only", "unintended_diff_only")]


def test_numeric_drops_flags_but_keeps_zero():
    values = M.numeric({"kl": 0.0, "ok": True, "missing": None, "reference_fp32": 1.0, "s": "x"})
    assert values == {"kl": 0.0}, "a zero metric must survive; a flag must not"


def test_numeric_keeps_provenance_on_request():
    assert M.numeric({"n_tokens": 5}, keep_provenance=True) == {"n_tokens": 5.0}


def test_ce_kl_panels_are_all_within_the_categorical_budget():
    """Every panel must fit in the eight validated hues, or `series_color` will assert at draw
    time -- and it does so INSIDE the figure loop, i.e. after some panels are already drawn.
    """
    groups = M.panels_by_family(CE_KL_METRICS)
    for key, members in groups.items():
        for chunk in chunk_series(members):
            assert len(chunk) <= MAX_SERIES, f"{key} chunk {chunk} exceeds the palette"
    drawn = [n for members in groups.values() for c in chunk_series(members) for n in c]
    M.assert_all_covered(CE_KL_METRICS, drawn, "ce_kl")


def test_controls_are_separated_from_learned_masks():
    groups = M.panels_by_family(CE_KL_METRICS)
    assert ("kl", True) in groups and ("kl", False) in groups
    assert set(groups[("kl", True)]) == {
        "kl_random_masked", "kl_rounded_masked", "kl_stoch_masked", "kl_zero_masked"
    }
    # A control token must be a whole word: `kl_zero_masked` is a control, `kl_zeroshot` is not.
    assert not M.is_control("kl_zeroshot_masked")


def test_scr_keys_panels_by_family_and_tpp_by_suffix():
    scr = M.sweep_axis(SCR_METRICS)
    assert set(scr) == {"scr_dir1", "scr_dir2", "scr_metric"}
    assert set(scr["scr_dir1"]) == {""}, "no suffix means no sub-series"
    assert [k for k, _ in scr["scr_dir1"][""]] == [2, 5, 10, 20, 50, 100, 500]
    M.assert_all_covered(SCR_METRICS, M.sweep_names(scr), "scr")

    tpp = M.sweep_axis(TPP_METRICS)
    assert set(tpp) == {"total_metric", "intended_diff_only", "unintended_diff_only"}
    assert set(tpp["total_metric"]) == {"tpp"}
    M.assert_all_covered(TPP_METRICS, M.sweep_names(tpp), "tpp")


def test_sparse_probing_puts_llm_and_sae_on_one_axis():
    """The eval's whole question is SAE-vs-raw at the same k, so they must share a panel."""
    names = [f"{f}_top_{k}_test_accuracy" for f in ("llm", "sae") for k in (1, 2, 5)]
    names += ["llm_test_accuracy", "sae_test_accuracy"]
    panels = M.sweep_axis(names)
    assert list(panels) == ["test_accuracy"]
    assert set(panels["test_accuracy"]) == {"llm", "sae"}
    refs = M.sweep_references(names, panels)
    assert refs == {("test_accuracy", "llm"): "llm_test_accuracy",
                    ("test_accuracy", "sae"): "sae_test_accuracy"}


def test_assert_all_covered_names_what_it_dropped():
    with pytest.raises(AssertionError, match="kl_ci_masked"):
        M.assert_all_covered(["kl_ci_masked", "kl_unmasked"], ["kl_unmasked"], "unit test")


def test_chunk_series_is_balanced_and_lossless():
    names = [f"m{i}" for i in range(9)]
    chunks = chunk_series(names)
    assert [len(c) for c in chunks] == [5, 4], "9 must split 5+4, not 8+1"
    assert [n for c in chunks for n in c] == names
    assert chunk_series([]) == []


def test_series_color_asserts_rather_than_inventing_a_ninth_hue():
    assert series_color(0) != series_color(1)
    with pytest.raises(AssertionError, match="chunk_series"):
        series_color(MAX_SERIES)


def test_ordinal_colors_run_light_to_dark_and_never_repeat():
    colors = ordinal_colors(5)
    assert len(set(colors)) == 5
    assert ordinal_colors(0) == [] and len(ordinal_colors(1)) == 1


def test_compact_step_labels():
    assert compact(250000) == "250k" and compact(2_000_000) == "2M" and compact(1234) == "1,234"


# --------------------------------------------------------------------------- end to end


def _write_ce_kl(path):
    (path / "ce_kl").mkdir(parents=True)
    per_step = {str(s): {m: 0.1 * i for i, m in enumerate(CE_KL_METRICS)} | {"reference_fp32": 1.0}
                for s in (50000, 100000)}
    (path / "ce_kl" / "ce_kl.json").write_text(json.dumps({
        "per_step": per_step,
        "sae_splice_baseline": {"ce_clean": 3.2, "ce_spliced": 3.3, "kl_spliced_vs_clean": 0.01,
                                "n_tokens": 1000},
        "meta": {"module": "transformer.h.0.mlp.c_fc", "n_tokens": 1000, "split": "eval",
                 "seed": 0, "sae_dir": "/tmp/sae"},
    }))


def _write_sweep(path, eval_type, source, metrics):
    (path / "scr_tpp").mkdir(parents=True, exist_ok=True)
    steps = [None] if source == "dictionary" else [50000, 100000]
    reports = [{
        "eval_type": eval_type, "site": "transformer.h.0.mlp.c_fc", "source": source, "step": s,
        "averaged": {m: 0.01 for m in metrics},
        "per_run": {"ds_a": {m: 0.02 for m in metrics}, "ds_b": {m: 0.0 for m in metrics}},
        "clean_accuracies": {"ds_a": {"0": 0.9, "1": 0.8}, "ds_b": {"0": 0.7}},
        "n_degenerate": 0, "n_runs": 2, "tolerance": 0.08,
        "sparse_probing": {"ds_a": {"k_1": {"0": 0.6}, "k_all": {"0": 0.9}}},
        "meta": {},
    } for s in steps]
    (path / "scr_tpp" / f"sweep_{eval_type}_{source}.json").write_text(json.dumps(reports))


def test_matching_figure_survives_a_single_checkpoint(tmp_path):
    """One step is a bar chart, not a one-point line -- and that branch is easy to break."""
    from aspd.eval.plots.matching import headline, plot_dir

    out = tmp_path / "matching"
    out.mkdir()
    (out / "matching_c2o_step50000.json").write_text(json.dumps({
        "meta": {"run_dir": str(tmp_path), "judge_model": "m", "n_subsample": 8,
                 "min_examples": 10, "n_tokens": 1000, "output_site": "s",
                 "scheme": "c2o"},
        "results": [
            {"name": n, "n_pairs": 4, "mean_score": v,
             "score_histogram": {"1": 1, "2": 2, "3": 1}, "pairs": [], "scores": [1, 2, 2, 3]}
            for n, v in (("A_glob", 2.0), ("A_cond", 2.1), ("random", 1.2))
        ],
    }))
    assert [p.name for p in plot_dir(out)] == ["matching_c2o.png"]
    assert headline(out)["A_cond"] == {50000: 2.1}

    i2o = json.loads((out / "matching_c2o_step50000.json").read_text())
    i2o["meta"]["scheme"] = "i2o"
    for r in i2o["results"]:
        r["mean_score"] += 0.5
    (out / "matching_i2o_step60000.json").write_text(json.dumps(i2o))

    assert [p.name for p in plot_dir(out)] == ["matching_c2o.png", "matching_i2o.png"]
    assert headline(out)["A_cond"] == {50000: 2.1}
    assert headline(out, "i2o")["A_cond"] == {60000: 2.6}


def test_safe_plot_swallows_the_failure_and_says_so(capsys):
    from aspd.eval.plots import safe_plot

    def boom():
        raise ValueError("no such metric")

    assert safe_plot(boom) == []
    assert "FAILED" in capsys.readouterr().out


# --------------------------------------------------------------- saebench layouts & comparison


def _saebench_result(root, *relative, sae="pair_in_custom_sae", metrics=None):
    d = root.joinpath(*relative)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sae}_eval_results.json").write_text(json.dumps({
        "eval_result_metrics": metrics or {"sparsity": {"l0": 32.0, "l1": 80.0},
                                           "token_stats": {"total_tokens_eval_reconstruction": 1e6}},
        "eval_result_details": [],
    }))
    return d


