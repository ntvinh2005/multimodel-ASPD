"""The multi-hop total-effect score of the prompt panel, on the `gpt2_all_aspd` run."""


import pytest

pytest.importorskip("param_decomp_lab")

import torch

from aspd.paths import RUNS_DIR as RUNS  # noqa: E402

RUN = "gpt2_all_aspd"
PROMPT = "When Mary and John went to the store, John gave a drink to"
pytestmark = pytest.mark.skipif(not (RUNS / RUN).exists(), reason=f"needs {RUN}")


def target_fn(z):
    from aspd.analysis.circuits.targets import top_k_vs_rest

    return top_k_vs_rest(z, k=10)


@pytest.fixture(scope="module")
def trace():
    from aspd.analysis.prompt import open_prompt

    torch.set_num_threads(16)
    return open_prompt(RUN, PROMPT, runs_root=RUNS, device="cpu")


@pytest.fixture(scope="module")
def totals(trace):
    from aspd.analysis.prompt import total_effect

    return total_effect(trace, target_fn)


@pytest.fixture(scope="module")
def direct(trace):
    from aspd.analysis.prompt import node_scores

    return node_scores(trace, target_fn)


def test_the_one_hop_score_is_identically_zero_away_from_the_last_token(trace, direct):
    module = trace.module_at(9, "attn.o")
    s = direct.components[module]
    assert s[: trace.n_pos - 1].abs().max() == 0.0
    assert s[trace.n_pos - 1].abs().max() > 0.0
    # ... while those same components demonstrably contributed something at that position.
    assert trace.effective(module)[9].abs().max() > 0.1


def test_the_total_score_is_not_zero_there(trace, totals):
    module = trace.module_at(9, "attn.o")
    assert totals.components[module][9].abs().max() > 0.0


def test_every_module_is_scored_including_the_ones_no_logit_target_reaches(trace, totals, direct):
    """`attn.{q,k,v}` and `mlp.in` have no one-hop score at all -- `node_scores` omits them. The
    multi-hop pass reaches them through the module they feed, which is the whole point.
    """
    assert direct.unreachable, "expected this run to have unreachable modules"
    assert set(totals.components) == set(trace.modules)
    for module in direct.unreachable:
        assert totals.components[module].abs().max() > 0.0, f"{module} scored nothing anywhere"


def test_a_component_that_did_not_fire_scores_exactly_zero(trace, totals):
    """`e = g * a`, so a closed gate means the added term is the zero vector and its derivative
    is zero -- a measurement, not a missing entry.
    """
    module = trace.module_at(9, "attn.o")
    pos = trace.n_pos - 1
    fired = set(trace.fired(module, pos).tolist())
    dead = [c for c in range(trace.acts[module].shape[1]) if c not in fired][:50]
    assert totals.components[module][pos, dead].abs().max() == 0.0


# --------------------------------------------------------------------------- against the maths


def test_the_last_decomposed_module_has_nothing_to_compose_through(trace, totals, direct):
    """THE tie between the two scorers."""
    module = trace.module_at(max(trace.layers), "mlp.out")
    pos = trace.n_pos - 1
    fired = trace.fired(module, pos)
    assert fired.numel() > 0
    a, b = totals.components[module][pos, fired], direct.components[module][pos, fired]
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-3), (
        f"total vs direct at {module}: max diff {(a - b).abs().max().item():.3e} "
        f"on a scale of {b.abs().max().item():.3e}"
    )


def test_the_total_score_predicts_what_ablating_the_component_actually_does(trace, totals):
    """Ground truth: scale one component by `1 - eps`, rerun, and measure the target."""
    from aspd.analysis.circuits.replacement import run_replacement

    # The largest multi-hop score at a NON-final position, which is where one hop has nothing.
    best = max(
        ((float(s[p, c].abs()), m, p, int(c))
         for m, s in totals.components.items()
         for p in range(trace.n_pos - 1)
         for c in [int(s[p].abs().argmax())]),
        key=lambda t: t[0],
    )
    _, module, pos, c = best
    score = float(totals.components[module][pos, c])

    def rerun(eps: float) -> float:
        factor = torch.ones_like(trace.gates[module])[None]
        factor[0, pos, c] = 1.0 - eps
        cache = run_replacement(trace.model, trace.tokens, sampling=trace.sampling,
                                error_nodes=True, mask_edits={module: factor})
        return float(target_fn(cache.logits).detach())

    base = totals.target
    err = []
    for eps in (2e-2, 1e-2):
        measured = (rerun(eps) - base) / -eps
        err.append(abs(measured - score))
        assert measured == pytest.approx(score, rel=0.05), (
            f"{module}[{pos}, {c}]: predicted {score:.5f}, ablation says {measured:.5f}"
        )
    assert err[1] < err[0] * 0.6, (
        f"the residual {err} did not shrink with eps -- that is a wrong linearization, not "
        "discretization error"
    )


# ------------------------------------------------------------------------------- the isolation


def test_the_forward_is_untouched_and_the_hooks_are_gone(trace):
    """`(m - 1)` is exactly zero at the operating point, so the pass is the target model's own."""
    from aspd.analysis.prompt import total_effect

    before = {m: list(trace.model.target_model.get_submodule(m)._forward_hooks)
              for m in trace.modules}
    out = total_effect(trace, lambda z: z.sum())
    after = {m: list(trace.model.target_model.get_submodule(m)._forward_hooks)
             for m in trace.modules}
    assert before == after, "a forward hook survived the pass"
    assert out.target == pytest.approx(float(trace.logits.sum()), rel=1e-6)


def test_a_total_effect_pass_does_not_move_the_one_hop_scores(trace):
    """The isolation that matters: this must be readable without changing what anything else reads."""
    from aspd.analysis.prompt import node_scores, total_effect

    was = node_scores(trace, target_fn)
    acts = {m: trace.acts[m].clone() for m in trace.modules}
    total_effect(trace, target_fn)
    now = node_scores(trace, target_fn)

    assert was.target == now.target and was.unreachable == now.unreachable
    for m in was.components:
        assert torch.equal(was.components[m], now.components[m]), m
        assert torch.equal(acts[m], trace.acts[m]), f"{m} activations moved"


def test_two_passes_agree(trace, totals):
    """Nothing accumulates: the encoder's capture is disarmed and the multipliers are rebuilt."""
    from aspd.analysis.prompt import total_effect

    again = total_effect(trace, target_fn)
    for m, s in totals.components.items():
        assert torch.equal(s, again.components[m]), m


# ------------------------------------------------------------------------------------ the route


@pytest.fixture(scope="module")
def client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aspd.analysis.prompt.api import install_prompt_api
    from aspd.analysis.prompt.engine import PromptEngine

    app = fastapi.FastAPI()
    install_prompt_api(app, PromptEngine(RUNS / RUN, device="cpu"), RUN)
    return TestClient(app)


def test_the_browse_route_serves_one_score_and_ranks_by_it(client):
    r = client.get("/api/prompt/browse", params={
        "prompt": PROMPT, "target": "topk", "pos": 9,
        "module": "transformer.h.9.attn.c_proj", "k": 50})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["effect"] == "total" and d["rows"]
    tot = [abs(x["total"]) for x in d["rows"]]
    assert tot == sorted(tot, reverse=True)
    assert tot[0] > 0.0, "the ranking is the thing the one-hop score could not give"
    assert all({"act", "gate", "effective", "density", "role", "layer"} <= set(x)
               for x in d["rows"])
    assert all({"direct", "unreachable"}.isdisjoint(x) for x in d["rows"])
    fired = client.get("/api/prompt/fired", params={
        "prompt": PROMPT, "target": "topk", "pos": 9,
        "module": "transformer.h.9.attn.c_proj", "k": 50}).json()["rows"]
    assert fired and all(x["score"] == 0.0 for x in fired)


def test_the_browse_route_scores_a_module_no_logit_target_reaches(client):
    """`/api/prompt/fired` answers `null` here, by design. This route answers a number."""
    params = {"prompt": PROMPT, "target": "topk", "pos": 13,
              "module": "transformer.h.9.attn.c_attn.q_proj", "k": 20}
    rows = client.get("/api/prompt/browse", params=params).json()["rows"]
    assert rows and max(abs(x["total"]) for x in rows) > 0.0
    # The same module through the one-hop route, for the contrast this route exists to remove.
    fired = client.get("/api/prompt/fired", params=params).json()["rows"]
    assert fired and all(x["score"] is None and x["unreachable"] for x in fired)


def test_the_browse_route_refuses_a_bad_position_or_module(client):
    bad_pos = client.get("/api/prompt/browse", params={"prompt": PROMPT, "pos": 999})
    assert bad_pos.status_code == 404
    bad_mod = client.get("/api/prompt/browse",
                         params={"prompt": PROMPT, "pos": 1, "module": "transformer.h.9.nope"})
    assert bad_mod.status_code == 404


def test_only_the_browse_panel_was_rewired(client):
    """The page side of the same claim: the browse tables call the new route, and every other
    panel still calls what it always called.
    """
    from aspd.analysis.prompt.page import PROMPT_HTML

    # Twice: the per-layer component table, and the `top scoring / at token` table.
    assert PROMPT_HTML.count("/api/prompt/browse") == 2
    for route in ("/api/prompt/trace", "/api/prompt/node", "/api/prompt/features",
                  "/api/prompt/qk_pair", "/api/prompt/atp"):
        assert route in PROMPT_HTML, route
    assert '<option value="total">total score</option>' in PROMPT_HTML
    # One score column in the browse panel, named for what it is.
    assert PROMPT_HTML.count('"total score"') == 2 and "DIRECT" not in PROMPT_HTML


def test_the_routes_that_were_already_there_answer_exactly_as_before(client):
    """The browse panel got a new route; nothing else may have moved. Byte-for-byte, around a
    call to the new one.
    """
    calls = [
        ("/api/prompt/trace", {"prompt": PROMPT, "target": "topk", "k": 25}),
        ("/api/prompt/fired", {"prompt": PROMPT, "target": "topk", "pos": 9,
                               "module": "transformer.h.9.attn.c_proj", "k": 50}),
        ("/api/prompt/node", {"prompt": PROMPT, "target": "topk",
                              "module": "transformer.h.9.attn.c_proj", "idx": 35}),
    ]
    was = [client.get(path, params=p).text for path, p in calls]
    client.get("/api/prompt/browse", params={"prompt": PROMPT, "target": "topk", "pos": 9})
    now = [client.get(path, params=p).text for path, p in calls]
    assert was == now
