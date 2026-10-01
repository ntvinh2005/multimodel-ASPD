"""A minimal transformer-shaped LM for evaluation-pipeline tests."""

import torch
from torch import Tensor, nn

from aspd.sae.sites import SitePair


class ToyMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class ToyBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.rms_2 = nn.LayerNorm(d_model)
        self.mlp = ToyMLP(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.rms_2(x))


class ToyLM(nn.Module):
    def __init__(self, vocab: int, d_model: int, d_ff: int, n_layers: int = 1) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.h = nn.ModuleList([ToyBlock(d_model, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for blk in self.h:
            x = blk(x)
        return self.unembed(self.final_norm(x))


def toy_model_and_sites(
    *, vocab: int = 40, d_model: int = 16, d_ff: int = 24, layer: int = 0, seed: int = 0
) -> tuple[ToyLM, SitePair]:
    torch.manual_seed(seed)
    model = ToyLM(vocab, d_model, d_ff).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sites = SitePair(
        module=f"h.{layer}.mlp.c_fc",
        input_site=f"h.{layer}.rms_2",
        output_site=f"h.{layer}.mlp.c_fc",
    )
    return model, sites
