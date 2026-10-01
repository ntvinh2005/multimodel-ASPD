"""Experiment-config schema.

The lab's `ExperimentConfig` with its loss and CI-function unions widened by this package's
entries, so one YAML schema covers VPD, VPD + internal, PD Transcoder and ASPD.
"""

from typing import Annotated, Literal

from aspd.ci import (
    ASPDCiConfig,
    PDTranscoderCiConfig,
)
from aspd.adaptive_l0 import AdaptiveSparsityLossConfig
from aspd.losses import (
    AuxKLossConfig,
    InternalReconLossConfig,
    ComponentAliveTrackerConfig,
    ActivationReconLossConfig,
)
from param_decomp.ci_fns import GlobalCiConfig
from param_decomp.configs import AnyLossMetricConfig, PDConfig
from param_decomp_lab.eval_metrics import AnyEvalMetricConfig
from param_decomp_lab.experiments.lm.run import LMDataConfig, LMTargetConfig
from param_decomp_lab.experiments.utils import EvalConfig, ExperimentConfig
from pydantic import Discriminator, Field

AnyLossMetricConfigShared = Annotated[
    AnyLossMetricConfig
    | ComponentAliveTrackerConfig
    | InternalReconLossConfig
    | AuxKLossConfig
    | ActivationReconLossConfig
    | AdaptiveSparsityLossConfig,
    Discriminator("type"),
]

CiConfigShared = Annotated[
    GlobalCiConfig
    | PDTranscoderCiConfig
    | ASPDCiConfig,
    Discriminator("mode"),
]

AnyEvalMetricConfigShared = AnyEvalMetricConfig


class PDConfigShared(PDConfig):
    """`PDConfig` with both the `loss_metrics` and `ci_config` unions widened."""

    loss_metrics: list[AnyLossMetricConfigShared] = []
    ci_config: CiConfigShared = Field(..., discriminator="mode")


class SharedEvalConfig(EvalConfig):
    """`EvalConfig` over the lab's eval metrics."""

    metrics: list[AnyEvalMetricConfigShared] = []


class LMInterpExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    pd: PDConfigShared
    eval: SharedEvalConfig | None = None

    component_arch: Literal["vpd", "transcoder"] = "vpd"
    """Component parameterization."""

    component_init: Literal["reference", "unit_norm"] = "reference"
    """How the transcoder's encoder `V` is scaled at init. Inert on `component_arch: vpd`."""


def component_arch(cfg: LMInterpExperimentConfig) -> str:
    """`vpd` (core's rank-1 outer products) or `transcoder` (rank-1 plus `b_dec` / `b_out`)."""
    return getattr(cfg, "component_arch", "vpd")


def component_init(cfg: LMInterpExperimentConfig) -> Literal["reference", "unit_norm"]:
    """How the transcoder encoder `V` is drawn. Inert on `component_arch: vpd`."""
    return getattr(cfg, "component_init", "reference")
