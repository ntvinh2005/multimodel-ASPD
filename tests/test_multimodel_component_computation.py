import pytest
import torch
from torch import nn

from aspd.multimodel.posthoc.backend import MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.computation import (
    affine_up_removal,
    attention_probability_delta,
    component_read_write,
    local_mlp,
    replay_native_attention,
)
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId


def test_reading_and_normalized_write_preserve_inactive_reading_and_factor_gauge():
    x, u, v = map(torch.tensor, ([2.0, -1.0], [1.0, 3.0], [0.5, 2.0]))
    result = component_read_write(x, u, v, True, 2)
    changed = component_read_write(x, -u / 3, -v * 3, True, 2)
    assert result["reading"] == -1
    assert result["normalized"] == pytest.approx(changed["normalized"])
    assert [r["value"] for r in result["writingTerms"]] == pytest.approx([
        r["value"] for r in changed["writingTerms"]
    ])
    off = component_read_write(x, u, v, False, 2)
    assert off["reading"] == -1 and off["coefficient"] == 0 and off["magnitude"] == 0


def test_glu_up_is_affine_and_gate_is_nonlinear_at_frozen_inputs():
    a, b, removal, reader = map(torch.tensor, ([1., -1.], [2., 3.], [0.2, 0.4], [1., 2.]))
    h = torch.nn.functional.silu(a) * b
    def run(branch):
        return local_mlp(a, b, torch.nn.functional.silu, removal, branch, reader, True,
                         checked_native_down_input=h)
    result = run("mlp_up")
    gradient = float((torch.nn.functional.silu(a) * removal) @ reader)
    assert result["delta"] == pytest.approx(-gradient, abs=1e-6)
    assert run("mlp_gate")["after"] == pytest.approx(
        float((torch.nn.functional.silu(a - removal) * b) @ reader))
    with pytest.raises(ValueError, match="native Down"):
        local_mlp(a, b, torch.nn.functional.silu, removal, "mlp_gate", reader, True,
                  checked_native_down_input=b)
    down = ComponentId.from_key("smol.L08.mlp_down/7417")
    source = ComponentId.from_key("smol.L08.mlp_up/7417")
    assert affine_up_removal(source, down, result["before"], gradient)["after"] == pytest.approx(result["after"])
    with pytest.raises(ValueError, match="same-layer"):
        affine_up_removal(ComponentId.from_key("smol.L09.mlp_up/7417"), down, 0, 1)


def test_attention_probability_delta_preserves_query_head_mask_and_score_gauge():
    base_logits = torch.tensor([[[0.5, 0.8, -float("inf")]]])
    changed_logits = base_logits + torch.tensor([[[0.1, 0.9, 0.0]]])
    rows = attention_probability_delta(base_logits.softmax(-1), changed_logits.softmax(-1), query=0)
    by_key = {row["key"]: row for row in rows}
    assert by_key.keys() == {0, 1}
    assert by_key[1]["centeredScoreDelta"] - by_key[0]["centeredScoreDelta"] == pytest.approx(0.8)


def test_native_value_replay_preserves_probabilities_and_captures_projected_write():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.v = nn.Identity()

        def forward(self, x):
            logits = (x @ x.transpose(-1, -2)).tril()
            logits[..., 0, 1] = -torch.inf
            probabilities = logits.softmax(-1)
            return 2 * probabilities @ self.v(x), probabilities.unsqueeze(1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

        def forward(self, input_ids, **_kwargs):
            return self.attention(input_ids.float().unsqueeze(-1))

    model = Model()
    ids = torch.tensor([[1, 2]])
    inputs = ModelInputs(ids, torch.ones_like(ids), torch.tensor([0]), torch.tensor([0]))
    spec = MatrixHookSpec(MatrixId("smol", 8, "v"), "attention.v", None, 1, 1)
    result = replay_native_attention(model, inputs, spec, torch.tensor([0.5]))
    assert all(row["delta"] == 0 for row in result["attentionRows"])
    assert result["projectionOutputChanges"][0] == {"index": 0, "value": 1.0}
    assert 0 < result["projectionOutputChanges"][1]["value"] < 1
