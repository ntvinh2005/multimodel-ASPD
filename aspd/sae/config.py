"""Config for training an evaluation SAE (width, BatchTopK k, Matryoshka prefixes, AuxK, optimizer)."""

from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from aspd.sae.matryoshka import DEFAULT_GROUP_FRACS

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class SAEDictionaryConfig(BaseModel):
    """Shape and sparsity of the dictionaries -- `MatryoshkaSAEConfig` minus the probed widths."""

    model_config = ConfigDict(extra="forbid")

    feature_multiplier: int = 32  # n_features = feature_multiplier * d_in(input site)
    group_fracs: tuple[float, ...] = DEFAULT_GROUP_FRACS
    top_k: int = 32
    top_k_aux: int = 512
    aux_penalty: float = 1.0 / 32.0
    l1_coeff: float = 0.0  # upstream default; the L1 term is present but inert
    threshold_lr: float = 0.01
    n_batches_to_dead: int = 320
    encoder_init: Literal["reference", "unit_norm"] = "reference"
    """See `MatryoshkaSAEConfig.encoder_init`. `unit_norm` is what a site whose activations are
    not `input_unit_norm`-scaled needs; it is worth 12.8x the alive count on Gemma-2 L13.
    """


class SAETrainingConfig(BaseModel):
    """Optimization of the pair. One Adam over both SAEs, fed by one target forward per batch."""

    model_config = ConfigDict(extra="forbid")

    n_tokens: int = Field(gt=0)
    sae_batch_tokens: int = 2048
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.99)
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    log_every: int = 2000  # in optimizer steps
    eval_batches: int = 20

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]


class TranscoderTrainingConfig(SAETrainingConfig):
    """`SAETrainingConfig` plus the checkpoint axis. One Adam over one dictionary."""

    tokens_per_vpd_step: int = 8192
    """`pd.batch_size * data.max_seq_len` of the arms this transcoder is compared against."""

    checkpoint_every_vpd_steps: int = 50000
    """Matches the arms' `cadence.save_every`, so the checkpoints land on their x-axis exactly."""

    @property
    def checkpoint_every_tokens(self) -> int:
        return self.checkpoint_every_vpd_steps * self.tokens_per_vpd_step

    def vpd_step_at(self, tokens_seen: int) -> int:
        """The arm step a given token count corresponds to, rounded to the checkpoint grid."""
        raw = tokens_seen // self.tokens_per_vpd_step
        grid = self.checkpoint_every_vpd_steps
        return int(round(raw / grid)) * grid


class TranscoderRunConfig(BaseModel):
    """One transcoder pretraining run: which target, which sites, trained how."""

    model_config = ConfigDict(extra="forbid")

    experiment_config: str  # path to the VPD config supplying `target:`/`data:`, repo-relative
    transcoder_dir: str  # where the checkpoints and the report land
    dictionary: SAEDictionaryConfig = SAEDictionaryConfig()
    train: TranscoderTrainingConfig

    input_take: Literal["input", "output"] = "output"
    """Which activation the ENCODER reads, and the only structural difference between the two."""

    @classmethod
    def from_file(cls, path: str) -> "TranscoderRunConfig":
        return cls.model_validate(yaml.safe_load(open(path)))


class SAERunConfig(BaseModel):
    """One SAE pretraining run: which target, which dictionaries, trained how."""

    model_config = ConfigDict(extra="forbid")

    experiment_config: str  # path to the VPD config supplying `target:`/`data:`, repo-relative
    sae_dir: str  # where the frozen pair and its report land
    dictionary: SAEDictionaryConfig = SAEDictionaryConfig()
    train: SAETrainingConfig

    input_take: Literal["input", "output"] = "output"
    """Which activation of the input site the input dictionary is trained on."""

    reuse_output_from: str | None = None
    """Directory whose OUTPUT dictionary this run adopts instead of training its own."""

    @classmethod
    def from_file(cls, path: str) -> "SAERunConfig":
        return cls.model_validate(yaml.safe_load(open(path)))


def dictionary_experiment_config(path: str) -> str:
    """`experiment_config` of a dictionary run config of EITHER kind."""
    raw = yaml.safe_load(open(path))
    assert isinstance(raw, dict), f"{path} is not a mapping"
    experiment_config = raw.get("experiment_config")
    assert isinstance(experiment_config, str), (
        f"{path} declares no `experiment_config`, so it is neither an SAE nor a transcoder run "
        f"config. These tools need one only to resolve `data.tokenizer_name`; pass any "
        f"configs/sae/<target>.yaml or configs/transcoder/<target>.yaml for the same target."
    )
    return experiment_config
