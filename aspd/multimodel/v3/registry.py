"""Three-family model, layer, hook, and logical-matrix registry for ASPD v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from torch import nn

from aspd.multimodel.v3.config import V3_LOGICAL_MATRICES, V3ApprovedConfig, V3ApprovedModel

GroundingKind = Literal["residual_pre", "residual_mid"]


@dataclass(frozen=True)
class OutputSlice:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"invalid fused output slice [{self.start}:{self.stop}]")

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class LogicalMatrixTarget:
    model_key: str
    family: str
    layer_index: int
    logical_name: str
    grounding_kind: GroundingKind
    grounding_module_candidates: tuple[str, ...]
    physical_module_candidates: tuple[str, ...]
    output_slice: OutputSlice | None
    d_in: int
    d_out: int
    bias_policy: Literal["physical_if_present", "slice_if_present"]

    @property
    def key(self) -> str:
        return f"{self.model_key}.L{self.layer_index:02d}.{self.logical_name}"


@dataclass(frozen=True)
class ModelFamilyLayout:
    key: str
    family: str
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    layers: tuple[int, ...]
    layer_prefixes: tuple[str, ...]
    residual_pre_suffix: str
    residual_mid_suffix: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    q_out: int
    kv_out: int
    fused_qkv: bool
    fused_gate_up: bool
    supported_precision: tuple[str, ...] = ("bfloat16", "float32")

    def _layer_paths(self, suffix: str, layer: int) -> tuple[str, ...]:
        return tuple(f"{prefix}.{layer}.{suffix}" for prefix in self.layer_prefixes)

    def grounding_paths(self, kind: GroundingKind, layer: int) -> tuple[str, ...]:
        """Return the family-specific module whose input is the requested residual stream."""

        suffix = (
            self.residual_pre_suffix if kind == "residual_pre" else self.residual_mid_suffix
        )
        return self._layer_paths(suffix, layer)

    def targets(self) -> tuple[LogicalMatrixTarget, ...]:
        targets: list[LogicalMatrixTarget] = []
        for layer in self.layers:
            residual_pre = self.grounding_paths("residual_pre", layer)
            residual_mid = self.grounding_paths("residual_mid", layer)
            if self.fused_qkv:
                qkv = self._layer_paths("self_attn.qkv_proj", layer)
                attention = {
                    "q": (qkv, OutputSlice(0, self.q_out), self.hidden_size, self.q_out),
                    "k": (
                        qkv,
                        OutputSlice(self.q_out, self.q_out + self.kv_out),
                        self.hidden_size,
                        self.kv_out,
                    ),
                    "v": (
                        qkv,
                        OutputSlice(self.q_out + self.kv_out, self.q_out + 2 * self.kv_out),
                        self.hidden_size,
                        self.kv_out,
                    ),
                }
            else:
                attention = {
                    name: (
                        self._layer_paths(f"self_attn.{name}_proj", layer),
                        None,
                        self.hidden_size,
                        self.q_out if name == "q" else self.kv_out,
                    )
                    for name in ("q", "k", "v")
                }
            for name, (modules, output_slice, d_in, d_out) in attention.items():
                targets.append(
                    LogicalMatrixTarget(
                        self.key,
                        self.family,
                        layer,
                        name,
                        "residual_pre",
                        residual_pre,
                        modules,
                        output_slice,
                        d_in,
                        d_out,
                        "slice_if_present" if output_slice else "physical_if_present",
                    )
                )
            targets.append(
                LogicalMatrixTarget(
                    self.key,
                    self.family,
                    layer,
                    "o",
                    "residual_mid",
                    residual_mid,
                    self._layer_paths("self_attn.o_proj", layer),
                    None,
                    self.q_out,
                    self.hidden_size,
                    "physical_if_present",
                )
            )
            if self.fused_gate_up:
                gate_up = self._layer_paths("mlp.gate_up_proj", layer)
                mlp_inputs = {
                    "mlp_gate": (gate_up, OutputSlice(0, self.intermediate_size)),
                    "mlp_up": (
                        gate_up,
                        OutputSlice(self.intermediate_size, 2 * self.intermediate_size),
                    ),
                }
            else:
                mlp_inputs = {
                    "mlp_gate": (self._layer_paths("mlp.gate_proj", layer), None),
                    "mlp_up": (self._layer_paths("mlp.up_proj", layer), None),
                }
            for name, (modules, output_slice) in mlp_inputs.items():
                targets.append(
                    LogicalMatrixTarget(
                        self.key,
                        self.family,
                        layer,
                        name,
                        "residual_mid",
                        residual_mid,
                        modules,
                        output_slice,
                        self.hidden_size,
                        self.intermediate_size,
                        "slice_if_present" if output_slice else "physical_if_present",
                    )
                )
            targets.append(
                LogicalMatrixTarget(
                    self.key,
                    self.family,
                    layer,
                    "mlp_down",
                    "residual_mid",
                    residual_mid,
                    self._layer_paths("mlp.down_proj", layer),
                    None,
                    self.intermediate_size,
                    self.hidden_size,
                    "physical_if_present",
                )
            )
        return tuple(targets)


def _layout(model: V3ApprovedModel) -> ModelFamilyLayout:
    if model.key == "qwen":
        return ModelFamilyLayout(
            key=model.key,
            family=model.family,
            model_id=model.model_id,
            revision=model.revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            layers=model.selected_layers,
            layer_prefixes=("model.layers",),
            residual_pre_suffix="input_layernorm",
            residual_mid_suffix="post_attention_layernorm",
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            q_out=4096,
            kv_out=1024,
            fused_qkv=False,
            fused_gate_up=False,
        )
    if model.key == "gemma":
        return ModelFamilyLayout(
            key=model.key,
            family=model.family,
            model_id=model.model_id,
            revision=model.revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            layers=model.selected_layers,
            # First path is the gated checkpoint wrapper; second supports an explicitly
            # extracted text-only Gemma3ForCausalLM without changing logical targets.
            layer_prefixes=("model.language_model.layers", "model.layers"),
            residual_pre_suffix="input_layernorm",
            # Gemma 3 normalizes the attention output before its residual add.  The
            # input of pre_feedforward_layernorm is the actual post-attention residual.
            residual_mid_suffix="pre_feedforward_layernorm",
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=34,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            q_out=2048,
            kv_out=1024,
            fused_qkv=False,
            fused_gate_up=False,
        )
    if model.key == "phi":
        return ModelFamilyLayout(
            key=model.key,
            family=model.family,
            model_id=model.model_id,
            revision=model.revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            layers=model.selected_layers,
            layer_prefixes=("model.layers",),
            residual_pre_suffix="input_layernorm",
            residual_mid_suffix="post_attention_layernorm",
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
            q_out=3072,
            kv_out=1024,
            fused_qkv=True,
            fused_gate_up=True,
        )
    raise AssertionError(f"unhandled approved model: {model.key}")


def build_model_registry(approved: V3ApprovedConfig) -> dict[str, ModelFamilyLayout]:
    registry = {model.key: _layout(model) for model in approved.models}
    if tuple(registry) != ("qwen", "gemma", "phi"):
        raise ValueError("v3 registry order must be qwen, gemma, phi")
    targets = [target for layout in registry.values() for target in layout.targets()]
    if len(targets) != 168 or len({target.key for target in targets}) != 168:
        raise ValueError("v3 registry must contain 168 unique logical matrix targets")
    if {target.logical_name for target in targets} != set(V3_LOGICAL_MATRICES):
        raise ValueError("v3 registry lost a planned logical matrix type")
    return registry


def resolve_submodule(model: nn.Module, candidates: tuple[str, ...]) -> tuple[str, nn.Module]:
    errors: list[str] = []
    for path in candidates:
        try:
            return path, model.get_submodule(path)
        except AttributeError:
            errors.append(path)
    raise ValueError(f"none of the registered module paths exist: {errors}")


def validate_runtime_config(layout: ModelFamilyLayout, config: Any) -> dict[str, Any]:
    """Validate checkpoint metadata without instantiating or mutating source weights."""

    text = getattr(config, "text_config", config)
    observed = {
        "hidden_size": int(text.hidden_size),
        "intermediate_size": int(text.intermediate_size),
        "num_hidden_layers": int(text.num_hidden_layers),
        "num_attention_heads": int(text.num_attention_heads),
        "num_key_value_heads": int(text.num_key_value_heads),
        "head_dim": int(getattr(text, "head_dim", text.hidden_size // text.num_attention_heads)),
    }
    observed["q_out"] = observed["num_attention_heads"] * observed["head_dim"]
    observed["kv_out"] = observed["num_key_value_heads"] * observed["head_dim"]
    expected = {
        "hidden_size": layout.hidden_size,
        "intermediate_size": layout.intermediate_size,
        "num_hidden_layers": layout.num_hidden_layers,
        "num_attention_heads": layout.num_attention_heads,
        "num_key_value_heads": layout.num_key_value_heads,
        "head_dim": layout.head_dim,
        "q_out": layout.q_out,
        "kv_out": layout.kv_out,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if max(layout.layers) >= observed["num_hidden_layers"]:
        mismatches["selected_layers"] = {
            "expected": list(layout.layers),
            "observed_layer_count": observed["num_hidden_layers"],
        }
    if mismatches:
        raise ValueError(f"{layout.key} checkpoint architecture mismatch: {mismatches}")
    return {"model": layout.key, "observed": observed, "status": "PASS"}
