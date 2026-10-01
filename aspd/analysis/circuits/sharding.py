"""Split dataset attribution across jobs by target layer."""

from collections.abc import Sequence


def install_target_shard(target_layers: Sequence[str]) -> None:
    """Restrict this process's attribution to `target_layers`. Idempotent per layer set."""
    from param_decomp_lab.dataset_attributions import pipeline as pipeline_mod

    wanted = set(target_layers)
    assert wanted, "no target layers assigned to this shard"
    stock = getattr(pipeline_mod.get_sources_by_target, "_aspd_stock", None)
    stock = stock or pipeline_mod.get_sources_by_target

    def _sharded(*args, **kwargs):
        full = stock(*args, **kwargs)
        missing = wanted - set(full)
        assert not missing, (
            f"assigned target layers not present in the model's gradient connectivity: "
            f"{sorted(missing)}. Available: {sorted(full)[:4]}..."
        )
        return {k: v for k, v in full.items() if k in wanted}

    _sharded._aspd_stock = stock  # pyright: ignore[reportFunctionMemberAccess]
    pipeline_mod.get_sources_by_target = _sharded


def install_optional_unembed() -> None:
    """Let a shard that does not own the UNEMBED target still build its accumulators. Idempotent."""
    from param_decomp_lab.dataset_attributions.accumulator import AttributionHarvester as H

    if getattr(H._get_unembed_sources_attr_accumulator, "_aspd_optional", False):
        return
    stock = H._get_unembed_sources_attr_accumulator

    def _get(self, sources_by_target):
        if self.unembed_path not in sources_by_target:
            return {}
        return stock(self, sources_by_target)

    _get._aspd_optional = True  # pyright: ignore[reportFunctionMemberAccess]
    H._get_unembed_sources_attr_accumulator = _get  # pyright: ignore[reportAttributeAccessIssue]


def install_union_merge() -> None:
    """Make `DatasetAttributionStorage.merge` union target keys instead of assuming they match."""
    from param_decomp_lab.dataset_attributions.storage import DatasetAttributionStorage as S

    if getattr(S.merge, "_aspd_union", False):
        return

    def _merge(cls, paths):
        assert paths, "No files to merge"
        merged = cls.load(paths[0])
        for path in paths[1:]:
            other = cls.load(path)
            assert other.ci_threshold == merged.ci_threshold, "CI threshold mismatch"

            for target, sources in other._regular_attr.items():
                if target not in merged._regular_attr:
                    # A target only this shard computed: take its rows wholesale.
                    merged._regular_attr[target] = sources
                    merged._regular_attr_abs[target] = other._regular_attr_abs[target]
                    continue
                for source, tensor in sources.items():
                    if source not in merged._regular_attr[target]:
                        merged._regular_attr[target][source] = tensor
                        merged._regular_attr_abs[target][source] = other._regular_attr_abs[target][
                            source
                        ]
                    else:
                        merged._regular_attr[target][source] += tensor
                        merged._regular_attr_abs[target][source] += other._regular_attr_abs[
                            target
                        ][source]

            for target, tensor in other._embed_attr.items():
                if target not in merged._embed_attr:
                    merged._embed_attr[target] = tensor
                    merged._embed_attr_abs[target] = other._embed_attr_abs[target]
                else:
                    merged._embed_attr[target] += tensor
                    merged._embed_attr_abs[target] += other._embed_attr_abs[target]

            for source, tensor in other._unembed_attr.items():
                if source not in merged._unembed_attr:
                    merged._unembed_attr[source] = tensor
                else:
                    merged._unembed_attr[source] += tensor

            if _same_corpus(merged, other):
                continue

            merged._embed_unembed_attr += other._embed_unembed_attr
            for layer in other._ci_sum:
                merged._ci_sum[layer] += other._ci_sum[layer]
            for layer in other._component_act_sq_sum:
                merged._component_act_sq_sum[layer] += other._component_act_sq_sum[layer]
            merged._logit_sq_sum += other._logit_sq_sum
            merged._embed_token_count += other._embed_token_count
            merged.n_tokens_processed += other.n_tokens_processed
        return merged

    _merge._aspd_union = True  # pyright: ignore[reportFunctionMemberAccess]
    S.merge = classmethod(_merge)  # pyright: ignore[reportAttributeAccessIssue]


def _same_corpus(a, b) -> bool:
    """True when two shards saw the SAME tokens (target shards) rather than a partition (batch)."""
    import torch

    if a.n_tokens_processed != b.n_tokens_processed:
        return False
    return bool(torch.equal(a._embed_token_count, b._embed_token_count))
