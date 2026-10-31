"""Freeze lightweight on-the-fly source/alignment manifests for ASPD v3 main-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from aspd.multimodel.v3.config import load_v3_config_bundle
from aspd.multimodel.v3.data import write_json_immutable
from aspd.multimodel.v3.main_config import load_main_v1_config


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-config",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/main_v1.yaml"),
    )
    parser.add_argument(
        "--approved",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/approved.yaml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/candidates.yaml"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("configs/multimodel/aspd_v3/runtime_template.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    main_config = load_main_v1_config(args.main_config)
    bundle = load_v3_config_bundle(args.approved, args.candidates, args.runtime)
    dataset = bundle.approved.dataset
    info = HfApi().dataset_info(
        dataset.dataset_id,
        revision=dataset.revision,
        files_metadata=True,
    )
    if info.sha != dataset.revision:
        raise RuntimeError(
            f"pinned dataset revision resolved to a different commit: {info.sha}"
        )
    files = []
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if not name.startswith("train/") or not name.endswith(
            (".parquet", ".json", ".jsonl", ".jsonl.zst")
        ):
            continue
        lfs = sibling.lfs
        files.append(
            {
                "path": name,
                "size": sibling.size,
                "blob_id": sibling.blob_id,
                "lfs_sha256": None if lfs is None else lfs.sha256,
            }
        )
    if not files:
        raise RuntimeError("pinned Pile repository exposes no immutable data files")
    manifest = {
        "schema_version": 4,
        "kind": "ASPD_V3_ON_THE_FLY_DATASET_MANIFEST",
        "dataset_id": dataset.dataset_id,
        "requested_revision": dataset.revision,
        "resolved_revision": info.sha,
        "streaming": True,
        "split": "train",
        "ordering": "huggingface_iterable_pinned_repository_order",
        "data_files": sorted(files, key=lambda value: value["path"]),
        "per_anchor_materialization": False,
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    tokenizer_rows = [
        {
            "model_key": model.key,
            "tokenizer_id": model.tokenizer_id,
            "tokenizer_revision": model.tokenizer_revision,
        }
        for model in bundle.approved.models
    ]
    alignment = {
        "schema_version": 2,
        "kind": "ASPD_V3_ON_THE_FLY_ALIGNMENT_MANIFEST",
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "model_order": list(main_config.architecture.model_order),
        "tokenizers": tokenizer_rows,
        "canonical_text": "identity",
        "alignment": "exact_three_way_canonical_prefix",
        "sequence_length_native_tokens_per_model": main_config.data.sequence_length,
        "split": {
            "unit": main_config.data.split_unit,
            "seed": main_config.data.split_seed,
            "modulus": main_config.data.split_modulus,
            "validation_buckets": list(main_config.data.validation_buckets),
            "test_buckets": list(main_config.data.test_buckets),
        },
        "distributed_source": "rank_zero_stream_broadcast_to_four_fsdp_ranks",
        "exact_deduplication": "sqlite_content_sha256_with_checkpoint_high_water",
    }
    alignment["manifest_sha256"] = _fingerprint(alignment)
    dataset_path = Path(main_config.runtime.dataset_manifest)
    alignment_path = Path(main_config.runtime.alignment_manifest)
    write_json_immutable(dataset_path, manifest)
    write_json_immutable(alignment_path, alignment)
    print(
        json.dumps(
            {
                "dataset_manifest": str(dataset_path),
                "dataset_manifest_sha256": manifest["manifest_sha256"],
                "alignment_manifest": str(alignment_path),
                "alignment_manifest_sha256": alignment["manifest_sha256"],
                "data_file_count": len(files),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
