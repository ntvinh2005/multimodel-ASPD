"""Download trained paper runs from the Hugging Face Hub into `$PARAM_DECOMP_OUT_DIR/runs/<run id>`.

Each run is `<model>_<arm>` (e.g. `gpt2_aspd`, `qwen3_vpd_internal_noablate`, `gpt2_all_aspd`) and holds
`experiment_config.yaml`, the final `model_<N>.pth` (the decomposition; the target model is rebuilt
from the config) and, unless `--no-harvest`, the final harvest `harvest/h-step<N>/`.
`--sae <model>` fetches that model's evaluation SAE (input and output dictionaries, their harvest,
the editing feature samples, the evaluation token stream) into `artifacts/saes/<model>`, the
`--sae-dir` of matching and editing.

    python -m aspd.cli.download --list
    python -m aspd.cli.download gpt2_aspd gpt2_vpd_internal [--no-harvest]
    python -m aspd.cli.download --all
    python -m aspd.cli.download --sae gpt2
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from aspd.paths import RUNS_DIR

REPO_ID = "tueminh/wfd-runs"
PREFIX = "aspd"


SAE_DIR = Path("artifacts/saes")


def available() -> dict[str, int]:
    """Run id (or `saes/<model>`) -> total bytes on the Hub."""
    sizes: dict[str, int] = {}
    for entry in HfApi().list_repo_tree(REPO_ID, path_in_repo=PREFIX, recursive=True):
        parts = entry.path.split("/")
        if len(parts) > 2 and hasattr(entry, "size"):
            key = "/".join(parts[1:3]) if parts[1] == "saes" else parts[1]
            sizes[key] = sizes.get(key, 0) + entry.size
    return dict(sorted(sizes.items()))


def fetch(root: str, patterns: list[str], dest: Path) -> Path:
    """Download `root/<patterns>` from the Hub into `dest`; files already there are kept."""
    staging = dest.parent / f".download-{dest.name}"
    snapshot_download(REPO_ID, allow_patterns=[f"{root}/{p}" for p in patterns], local_dir=staging)
    source = staging / root
    assert source.is_dir(), f"{root} is not on {REPO_ID}; see --list"
    for f in sorted(p for p in source.rglob("*") if p.is_file()):
        target = dest / f.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, target)
    shutil.rmtree(staging)
    return dest


def download(run_id: str, *, harvest: bool = True, runs_dir: Path = RUNS_DIR) -> Path:
    """Fetch one run into `runs_dir/run_id`."""
    from aspd.cli.harvest import link_downstream_id

    patterns = ["experiment_config.yaml", "model_*.pth"] + (["harvest/*"] if harvest else [])
    dest = fetch(f"{PREFIX}/{run_id}", patterns, runs_dir / run_id)
    link_downstream_id(dest)
    return dest


def download_sae(model: str, sae_dir: Path = SAE_DIR) -> Path:
    """Fetch one model's evaluation SAE into `sae_dir/model`."""
    return fetch(f"{PREFIX}/saes/{model}", ["*"], sae_dir / model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run ids, e.g. gpt2_aspd")
    ap.add_argument("--all", action="store_true", help="every run on the Hub")
    ap.add_argument("--list", action="store_true", help="list the runs and their sizes")
    ap.add_argument("--no-harvest", action="store_true", help="config and checkpoint only")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--sae", action="append", default=[], choices=("gpt2", "gemma2", "qwen3"),
                    help="also fetch this model's evaluation SAE into artifacts/saes/<model>")
    args = ap.parse_args()

    sizes = available()
    if args.list:
        for run_id, n in sizes.items():
            print(f"{run_id:32s} {n / 1e9:7.2f} GB")
        return
    runs = [r for r in sizes if not r.startswith("saes/")] if args.all else args.runs
    assert runs or args.sae, "name at least one run or --sae, or pass --all / --list"
    for model in args.sae:
        print(f"[download] saes/{model} -> {download_sae(model)}")
    for run_id in runs:
        dest = download(run_id, harvest=not args.no_harvest, runs_dir=args.runs_dir)
        print(f"[download] {run_id} -> {dest}")


if __name__ == "__main__":
    main()
