"""Sim: the Jaccard kernel over components' token sets, its mean and its bootstrap interval."""

import sqlite3

import numpy as np
import orjson
import pytest

from aspd.eval.diversity import (
    bootstrap,
    jaccard_kernel,
    load_population,
    mean_off_diagonal,
)


def test_identical_components_have_sim_one():
    """C copies of the same component have Sim = 1."""
    sets = [np.arange(40) for _ in range(20)]
    z = jaccard_kernel(sets)
    assert np.allclose(z, 1.0)
    assert mean_off_diagonal(z) == pytest.approx(1.0)


def test_disjoint_components_have_zero_sim():
    """Mutually disjoint token sets have Sim = 0."""
    sets = [np.arange(40 * i, 40 * (i + 1)) for i in range(20)]
    z = jaccard_kernel(sets)
    assert mean_off_diagonal(z) == pytest.approx(0.0)


def test_jaccard_matches_the_closed_form_at_equal_set_size():
    """With |T_c| = |T_d| = k, Jaccard is i / (2k - i)."""
    k, overlap = 40, 12
    z = jaccard_kernel([np.arange(k), np.arange(k - overlap, 2 * k - overlap)])
    assert z[0, 1] == pytest.approx(overlap / (2 * k - overlap))


def test_bootstrap_brackets_the_estimate_and_reports_a_std():
    rng = np.random.default_rng(3)
    n = 40
    sets = [np.unique(rng.choice(400, size=40, replace=False)) for _ in range(n)]
    z = jaccard_kernel(sets)
    point = mean_off_diagonal(z)
    stats = bootstrap(z, n, n_boot=300, seed=0)
    assert set(stats) == {"z_bar"}
    assert set(stats["z_bar"]) == {"mean", "std", "lo", "hi"}
    assert stats["z_bar"]["lo"] <= point <= stats["z_bar"]["hi"] and stats["z_bar"]["std"] > 0


def test_bootstrap_masks_duplicate_resampled_components():
    """A component resampled twice must not contribute a Z=1 self-pair off the diagonal."""
    sets = [np.arange(40 * i, 40 * (i + 1)) for i in range(10)]  # all disjoint -> true z_bar = 0
    stats = bootstrap(jaccard_kernel(sets), 10, n_boot=200, seed=0)
    assert stats["z_bar"]["mean"] == pytest.approx(0.0, abs=1e-12)


def _write_harvest(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE components (component_key TEXT, firing_density REAL, "
        "n_activation_examples INT, activation_examples TEXT)"
    )
    for key, density, examples in rows:
        conn.execute("INSERT INTO components VALUES (?,?,?,?)",
                     (key, density, len(examples), orjson.dumps(examples).decode()))
    conn.commit()
    conn.close()


def test_min_firings_drops_coincidence_tokens(tmp_path):
    """A token the component fired on once is a coincidence, not a trigger."""
    db = tmp_path / "harvest.db"
    ex = [{"token_ids": [5, 5, 7, 7, 999], "firings": [True] * 5,
           "activations": {"causal_importance": [1.0] * 5}}]
    _write_harvest(db, [("m:0", 1e-4, ex * 5)])

    pop = load_population(db, min_density=5e-5, max_density=1e-3, min_firings=2,
                          n_sample=10, seed=0)
    assert pop.token_sets[0].tolist() == [5, 7, 999]  # 999 fires 5x across the 5 examples
    pop_strict = load_population(db, min_density=5e-5, max_density=1e-3, min_firings=6,
                                 n_sample=10, seed=0)
    assert pop_strict.token_sets[0].tolist() == [5, 7]  # 999 has only 5 firings


def test_band_filter_selects_by_density(tmp_path):
    db = tmp_path / "harvest.db"
    ex = [{"token_ids": [1, 2, 3], "firings": [True] * 3,
           "activations": {"causal_importance": [1.0] * 3}}] * 5
    _write_harvest(db, [("m:in", 1e-4, ex), ("m:dense", 0.5, ex), ("m:dead", 1e-9, ex)])

    pop = load_population(db, min_density=5e-5, max_density=1e-3, min_firings=2,
                          n_sample=10, seed=0)
    assert pop.keys == ["m:in"]
    assert pop.n_band == 1 and pop.n_eligible == 3


# ---- a raised firing threshold (eval_spec.md 8.1) ---------------------------------------------


def _thresholded_db(tmp_path):
    """Two components on a +-2 harvest, and the `ci` stats for them at 0.1."""
    pytest.importorskip("param_decomp_lab")  # the thresholded path needs it; the rest does not
    from aspd.eval.harvest_threshold import threshold_stats

    db = tmp_path / "harvest.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE components (component_key TEXT PRIMARY KEY, firing_density REAL, "
        "n_activation_examples INTEGER, activation_examples TEXT, layer TEXT, component_idx INTEGER, "
        "mean_activations TEXT, input_token_pmi TEXT, output_token_pmi TEXT);"
        "CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);"
        "CREATE TABLE scores (component_key TEXT, score_type TEXT, score REAL, details TEXT, "
        "PRIMARY KEY (component_key, score_type));"
        "CREATE TABLE intruder_prompts (trial_key TEXT PRIMARY KEY, prompt TEXT);"
    )
    conn.execute("INSERT INTO config VALUES ('activation_context_tokens_per_side', '2')")

    def ex(tokens, cis):
        return {"token_ids": tokens, "firings": [c > 0 for c in cis],
                "activations": {"causal_importance": cis, "component_activation": cis}}

    comps = {
        "a:0": [ex([20, 21, 10, 21, 20], [0.05, 0.05, 0.9, 0.05, 0.05]),
                ex([20, 21, 11, 21, 20], [0.05, 0.05, 0.9, 0.05, 0.05])] * 3,
        "a:1": [ex([30, 31, 32, 31, 30], [0.05, 0.05, 0.05, 0.05, 0.05])] * 6,
    }
    pmi = orjson.dumps({"top": [], "bottom": []}).decode()
    for i, (key, exs) in enumerate(comps.items()):
        conn.execute("INSERT INTO components VALUES (?,?,?,?,?,?,?,?,?)",
                     (key, 5e-4, len(exs), orjson.dumps(exs).decode(), "a", i,
                      orjson.dumps({"causal_importance": 0.5}).decode(), pmi, pmi))
    conn.commit()
    conn.close()
    return db, threshold_stats(tmp_path, [0.1], criterion="ci", min_examples=5)


def test_threshold_counts_only_surviving_positions_of_surviving_examples(tmp_path):
    db, stats = _thresholded_db(tmp_path)
    pop = load_population(db, min_density=1e-5, max_density=1e-2, min_firings=2, n_sample=10,
                          seed=0, threshold=0.1, stats=stats)
    assert pop.keys == ["a:0"]
    # Tokens 10 / 11 are the anchors; the 20 / 21 neighbours fired at ci 0.05, below the threshold.
    assert pop.token_sets[0].tolist() == [10, 11]


def test_threshold_moves_the_band_and_eligibility(tmp_path):
    db, stats = _thresholded_db(tmp_path)
    plain = load_population(db, min_density=1e-5, max_density=1e-2, min_firings=2, n_sample=10,
                            seed=0)
    thresholded = load_population(db, min_density=1e-5, max_density=1e-2, min_firings=2,
                                  n_sample=10, seed=0, threshold=0.1, stats=stats)
    # At 0 both components are in the band and a:0 fires on all four tokens.
    assert plain.n_band == 2 and plain.n_eligible == 2
    assert plain.token_sets[plain.keys.index("a:0")].tolist() == [10, 11, 20, 21]
    # At 0.1 a:1 keeps no example (every anchor is 0.05), so it is neither eligible nor in the band.
    assert thresholded.n_band == 1 and thresholded.n_eligible == 1


def test_threshold_requires_its_stats():
    with pytest.raises(AssertionError, match="needs the stats"):
        load_population("unused.db", min_density=0, max_density=1, min_firings=2, n_sample=1,
                        seed=0, threshold=0.1)
