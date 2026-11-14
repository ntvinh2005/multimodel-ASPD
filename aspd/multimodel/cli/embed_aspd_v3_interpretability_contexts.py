"""Embed frozen ASPD v3.1 contexts with a pinned non-judge Mistral representation."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from aspd.multimodel.v3.diagnostic import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _signature(model: torch.nn.Module) -> tuple[int, float, float]:
    count = 0
    total = 0.0
    square = 0.0
    for parameter in model.parameters():
        value = parameter.detach().float()
        count += value.numel()
        total += float(value.sum())
        square += float(value.square().sum())
    return count, total, square


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol["embedding_model_id"] != "mistralai/Mistral-Nemo-Instruct-2407":
        raise ValueError("embedding model changed")
    if any(value in protocol["embedding_model_id"].lower() for value in protocol["forbidden_model_substrings"]):
        raise ValueError("forbidden embedding model")
    if protocol["embedding_pooling"] != "attention_mask_mean_last_hidden_state":
        raise ValueError("embedding pooling changed")
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows or any(not str(row["text"]).strip() for row in rows):
        raise ValueError("embedding input is empty or contains blank text")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    if protocol.get("embedding_tokenizer_fix_mistral_regex") is not True:
        raise ValueError("Mistral tokenizer regex fix is required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        str(args.model),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval().requires_grad_(False)
    before = _signature(model)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = tokenizer(
                [str(row["text"]) for row in rows[start : start + args.batch_size]],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")
            output = model(**batch, return_dict=True).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            chunks.append(pooled.cpu().to(torch.float16).numpy())
    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.shape[0] != len(rows) or not np.isfinite(embeddings).all():
        raise FloatingPointError("invalid context embeddings")
    after = _signature(model)
    if before != after:
        raise RuntimeError("embedding model parameters changed")
    embeddings_path = args.output_dir / "embeddings.npy"
    with embeddings_path.open("xb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
    _write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "status": "ASPD_V3_1_CONTEXT_EMBEDDINGS_COMPLETE",
            "git_sha": _git_sha(),
            "protocol_sha256": sha256_file(args.protocol),
            "input_sha256": sha256_file(args.input),
            "embedding_model_id": protocol["embedding_model_id"],
            "embedding_model_revision": protocol["embedding_model_revision"],
            "embedding_model_path": str(args.model),
            "embedding_pooling": protocol["embedding_pooling"],
            "tokenizer_fix_mistral_regex": True,
            "rows": len(rows),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "embeddings_sha256": sha256_file(embeddings_path),
            "source_weights_unchanged": "PASS",
            "elapsed_seconds": time.perf_counter() - started,
            "hostname": socket.gethostname(),
            "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        },
    )
    (args.output_dir / "COMPLETE").write_text("ASPD_V3_1_CONTEXT_EMBEDDINGS_COMPLETE\n")


if __name__ == "__main__":
    main()
