"""ASPD v3.1 three-model checkpoint backend.

Status: ``IMPLEMENTED_NOT_VALIDATED`` -- written against the v3.1 APIs on ``main``
(``main500m_config``, ``ModelOwnedMain2B``, ``ModelOwnedCheckpoint``,
``DiagnosticAlignedAnchorStream``, ``capture_family_pool``) but not yet executed on a
GPU with the real checkpoint.  The 500M checkpoint was trained from commit ``5ca3b43``
(not on ``main`` as of 2026-11-21); confirm checkpoint/code compatibility before an
official run (CONTRIBUTING.md: full evaluations run from a commit merged into ``main``).

Natural gates are computed exactly as the colocated evaluation extractor does
(``extract_aspd_v3_paper_intruder_colocated._encode_colocated`` on the evaluation
branches): every model's local packer + intra-model encoder on its own residuals, one
shared cross-model encoder + readout + z-score + ReLU, then pool-wide BatchTopK with
``update_statistics=False``; ``g = 1[g^s > 0]``.  No support is forced.

Required ``backend.options``: ``training_config``, ``checkpoint``,
``config_file_sha256``, ``dataset_manifest_sha256``, ``split`` (validation|test),
``source_dtype`` (bfloat16|float32), ``device``.  Optional: ``local_files_only``
(default true), ``pool_anchors`` (default: the training pool size).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from aspd.multimodel.posthoc.backend import AnchorPool, MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.ids import MatrixId

SOURCE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _pad_id(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise ValueError("tokenizer has neither a pad nor an eos token")


class V31CheckpointBackend:
    def __init__(self, options: Mapping[str, Any], *, work_dir: Path):
        from aspd.multimodel.v3.hooks import assert_frozen_backbone
        from aspd.multimodel.v3.main500m_config import load_main500m_config
        from aspd.multimodel.v3.model_owned import ModelOwnedMain2B
        from aspd.multimodel.v3.model_owned_checkpoint import ModelOwnedCheckpoint
        from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
        from aspd.multimodel.v3.registry import build_model_registry_2b

        self.options = dict(options)
        if self.options["split"] not in ("validation", "test"):
            raise ValueError("split must be validation or test")
        if self.options["source_dtype"] not in SOURCE_DTYPES:
            raise ValueError(f"source_dtype must be one of {sorted(SOURCE_DTYPES)}")
        self.work_dir = Path(work_dir)
        self.device = torch.device(self.options["device"])
        self.local_files_only = bool(self.options.get("local_files_only", True))
        self.config = load_main500m_config(self.options["training_config"])
        self.registry = build_model_registry_2b(self.config)
        self.model_keys = tuple(self.config.architecture.model_order)
        self.n_features = int(self.config.architecture.dictionary_size)
        self.pool_anchors = int(
            self.options.get("pool_anchors") or self.config.architecture.pool_tokens
        )
        checkpoint = Path(self.options["checkpoint"])
        manager = ModelOwnedCheckpoint(checkpoint.parent, keep_complete=2)

        self.tokenizers: dict[str, Any] = {}
        self.sources: dict[str, nn.Module] = {}
        self.owners: dict[str, Any] = {}
        self.manifest: dict[str, Any] | None = None
        dtype = SOURCE_DTYPES[self.options["source_dtype"]]
        for rank, key in enumerate(self.model_keys):
            layout = self.registry[key]
            self.tokenizers[key] = load_v3_tokenizer(layout, local_files_only=self.local_files_only)
            loaded = load_frozen_v3_model(layout, self.device, local_files_only=self.local_files_only)
            source = loaded.forward_model.to(dtype)
            assert_frozen_backbone(source)
            self.sources[key] = source
            owner = ModelOwnedMain2B(
                self.config, self.registry, key,
                native_biases={t.key: None for t in layout.targets()},
                device=self.device, dtype=torch.float32, sparse_backend="dense",
            )
            _runtime, self.manifest = manager.load_model_only_owner(
                path=checkpoint, model=owner,
                config_fingerprint=self.config.fingerprint(),
                config_file_sha256=self.options["config_file_sha256"],
                dataset_manifest_sha256=self.options["dataset_manifest_sha256"],
                checkpoint_rank=rank,
                expected_checkpoint_world_size=len(self.model_keys),
                expected_model_key=key,
            )
            owner.eval().requires_grad_(False)
            self.owners[key] = owner
        self._check_shared_encoders()
        self._hook_specs = {key: self._build_hook_specs(key) for key in self.model_keys}
        self._factors = {key: self._build_factors(key) for key in self.model_keys}
        self._stream = None
        self._position = 0

    # ---------------------------------------------------------------- loading
    def _check_shared_encoders(self) -> None:
        reference = self.owners[self.model_keys[0]].shared_encoders.state_dict()
        for key in self.model_keys[1:]:
            other = self.owners[key].shared_encoders.state_dict()
            if reference.keys() != other.keys() or any(
                not torch.equal(reference[name], other[name]) for name in reference
            ):
                raise ValueError(f"replicated shared encoder of {key!r} differs from rank 0")

    def _build_hook_specs(self, key: str) -> tuple[MatrixHookSpec, ...]:
        from aspd.multimodel.v3.registry import resolve_submodule

        specs = []
        for target in self.registry[key].targets():
            path, _module = resolve_submodule(self.sources[key], target.physical_module_candidates)
            sliced = target.output_slice
            specs.append(
                MatrixHookSpec(
                    matrix=MatrixId(key, target.layer_index, target.logical_name),
                    module_path=path,
                    output_slice=None if sliced is None else (sliced.start, sliced.stop),
                    d_in=target.d_in,
                    d_out=target.d_out,
                )
            )
        return tuple(specs)

    def _build_factors(self, key: str) -> dict[MatrixId, tuple[Tensor, Tensor]]:
        from aspd.multimodel.v3.reconstruction import _safe_key

        bank = self.owners[key].matrix_bank
        factors = {}
        for target in self.registry[key].targets():
            component = bank.components[_safe_key(target.key)]
            factors[MatrixId(key, target.layer_index, target.logical_name)] = (
                component.U.detach().float(), component.V.detach().float(),
            )
        return factors

    # ---------------------------------------------------------------- protocol
    def hook_specs(self, model: str) -> tuple[MatrixHookSpec, ...]:
        return self._hook_specs[model]

    def factors(self, model: str) -> Mapping[MatrixId, tuple[Tensor, Tensor]]:
        return self._factors[model]

    def source_model(self, model: str) -> nn.Module:
        return self.sources[model]

    def _open_stream(self) -> Any:
        from aspd.multimodel.v3.data import HashSplitPolicy
        from aspd.multimodel.v3.diagnostic_stream import DiagnosticAlignedAnchorStream
        from aspd.multimodel.v3.production_stream import load_pinned_pile_dataset

        data = self.config.data
        return DiagnosticAlignedAnchorStream(
            dataset=load_pinned_pile_dataset(data.dataset_id, data.dataset_revision),
            dataset_id=data.dataset_id,
            dataset_revision=data.dataset_revision,
            tokenizers={key: self.tokenizers[key] for key in self.model_keys},
            tokenizer_revisions={k: self.registry[k].tokenizer_revision for k in self.model_keys},
            split_policy=HashSplitPolicy(
                modulus=data.split_modulus,
                validation_buckets=frozenset(data.validation_buckets),
                test_buckets=frozenset(data.test_buckets),
                seed=data.split_seed,
            ),
            dedup_path=self.work_dir / f"posthoc_stream_dedup_{id(self)}.sqlite",
            selected_split=self.options["split"],
            sequence_length=data.sequence_length,
            model_order=self.model_keys,
        )

    def _gates(self, pool: Any) -> dict[str, Tensor]:
        from aspd.multimodel.v3.training_capture import capture_family_pool

        valid = torch.ones(1, pool.anchor_count, dtype=torch.bool, device=self.device)
        tokens: dict[str, dict[str, Tensor]] = {"pre": {}, "mid": {}}
        autocast = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        for key in self.model_keys:
            family = capture_family_pool(
                self.sources[key], self.registry[key], pool.models[key],
                pad_token_id=_pad_id(self.tokenizers[key]), device=self.device, early_exit=True,
            )
            activations = {"pre": family.residual_pre, "mid": family.residual_mid}
            with torch.no_grad(), autocast:
                for site in ("pre", "mid"):
                    tokens[site][key] = self.owners[key].local_encoders[site](activations[site])
        shared = self.owners[self.model_keys[0]]
        gates = {}
        with torch.no_grad(), autocast:
            for site in ("pre", "mid"):
                scores, _ = shared.shared_encoders[site].forward_with_logits(tokens[site], valid)
                code = shared.gates[site](scores, valid, update_statistics=False)
                gates[site] = (code.values[0] > 0).cpu()
        return gates

    def _to_pool(self, pool: Any) -> AnchorPool:
        from aspd.multimodel.v3.training_capture import padded_model_inputs

        models = {}
        for key in self.model_keys:
            inputs, documents, tokens = padded_model_inputs(
                pool.models[key], pad_token_id=_pad_id(self.tokenizers[key]),
                device=torch.device("cpu"),
            )
            models[key] = ModelInputs(inputs["input_ids"], inputs["attention_mask"],
                                      documents, tokens)
        return AnchorPool(anchor_ids=tuple(pool.anchor_ids), gates=self._gates(pool),
                          models=models)

    def pools(self, count: int, *, skip: int = 0) -> Iterator[AnchorPool]:
        if self._stream is None:
            self._stream = self._open_stream()
        if skip < self._position:
            raise ValueError("the aligned-anchor stream cannot rewind; use a new backend")
        while self._position < skip:
            self._stream.next_pool(pool_anchors=self.pool_anchors)
            self._position += 1
        for _ in range(count):
            pool = self._stream.next_pool(pool_anchors=self.pool_anchors)
            self._position += 1
            yield self._to_pool(pool)

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "v31_checkpoint",
            "scientific_status": "IMPLEMENTED_NOT_VALIDATED",
            "checkpoint": str(self.options["checkpoint"]),
            "checkpoint_step": None if self.manifest is None else int(self.manifest["step"]),
            "config_fingerprint": self.config.fingerprint(),
            "config_file_sha256": self.options["config_file_sha256"],
            "dataset_manifest_sha256": self.options["dataset_manifest_sha256"],
            "split": self.options["split"],
            "source_dtype": self.options["source_dtype"],
            "pool_anchors": self.pool_anchors,
            "gate": "per-anchor z-score -> ReLU -> pool-wide BatchTopK; g = 1[g^s > 0]",
            "models": {
                key: {"model_id": self.registry[key].model_id,
                      "revision": self.registry[key].revision,
                      "tokenizer_revision": self.registry[key].tokenizer_revision}
                for key in self.model_keys
            },
        }

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self.sources.clear()
        self.owners.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
