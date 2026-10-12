# Multi-model ASPD P1–P5 research dashboard

Local FastAPI + React/TypeScript interface for inspecting real `posthoc.safetensors`, taxonomy,
activation contexts, and separately stored researcher notes. It has no demo-data or random fallback
path. Missing data is rendered as `N/A — data unavailable` and diagnosed in Data Health.

## Inputs

The backend reads these files without modifying them:

```text
analysis/posthoc.safetensors
analysis/taxonomy.json
analysis/top_activation_examples.json
```

It reads `experiment_config.json`, `provenance.json`, and `latest_checkpoint.txt` from the parent run
directory when available. Notes are the only mutable research file and default to
`analysis/researcher_notes.json`.

`top_activation_examples.json` uses key presence as measured provenance:

- a present key with examples means extraction ran and retained positive top contexts;
- a present key with `[]` means extraction ran but found no positive retained activation;
- an absent key means context extraction was not performed for that component. It does **not** mean
  that the component did not fire. Check `fire_count_C[c]` in `posthoc.safetensors` separately.

## Context extraction modes

`analysis.example_feature_mode` controls context coverage during the full P1--P5 analyzer:

| Mode | Components whose contexts are extracted |
|---|---|
| `ranked` | Historical capped sample: half by P1/P2 asymmetry, then fill by total beta mass up to `max_example_features`. |
| `all` | Every latent `c=0,...,C-1`; the JSON contains a key for every component, including `[]`. |
| `explicit` | Only `analysis.example_feature_ids`, for example `[440, 203, 73]`. |

`analysis.example_feature_chunk_size` controls only the feature-axis top-k update. For each
validation batch the checkpoint encoder computes `g^s[B,T,C]` once; chunks do not rerun the model.
`analysis.top_examples_per_feature` remains the number of contexts retained per component.

### Backfill contexts without retraining or recomputing P1--P5

The examples-only CLI loads the existing config, checkpoint, and paired validation cache. It updates
only `top_activation_examples.json`; it does not alter `posthoc.safetensors`, taxonomy membership,
or any P1/P2/P3/P5 value. Updates are merged with existing component keys and atomically replace the
JSON file.

For the existing Qwen3 D0/S1 main run:

```bash
CONFIG=configs/multimodel/qwen3_1_7b/d0_s1_main.yaml
RUN=out/multimodel/runs/qwen3_1_7b_d0_s1_main
CHECKPOINT="${RUN}/$(cat "${RUN}/latest_checkpoint.txt")"

# Produce context coverage for all C=8192 components.
uv run python -m aspd.multimodel.cli.extract_examples \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN/analysis" \
  --all-features \
  --chunk-size 512

# Backfill only component 440, preserving all previously extracted components.
uv run python -m aspd.multimodel.cli.extract_examples \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN/analysis" \
  --feature-ids 440
```

Multiple targeted IDs use one comma-separated argument, such as
`--feature-ids 440,203,73,390,146`. Data Health reports total components, extracted components,
components with at least one saved context, components not extracted, and firing components whose
contexts have not been extracted.

## Verify the real run

From the repository root:

```bash
.venv/bin/python -m aspd.multimodel.cli.serve_analysis \
  out/multimodel/runs/qwen3_1_7b_d0_s1_main/analysis \
  --config configs/multimodel/qwen3_1_7b/d0_s1_main.yaml \
  --verify-only
```

This prints counts and discovered keys from the loaded artifacts. A non-PASS Data Health result must
be investigated before interpretation.

## Build the React frontend

Node is only needed for the frontend build:

```bash
cd multimodel_dashboard
npm install
npm run check
npm run build
cd ..
```

The build writes `multimodel_dashboard/dist/`. The Python server serves that directory and the API
from one origin.

For frontend development, run `npm run dev`; Vite proxies `/api` to
`http://127.0.0.1:8765` by default. Override that with `BACKEND_URL` if needed.

## Run locally

```bash
.venv/bin/python -m aspd.multimodel.cli.serve_analysis \
  out/multimodel/runs/qwen3_1_7b_d0_s1_main/analysis \
  --config configs/multimodel/qwen3_1_7b/d0_s1_main.yaml \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765` on the same machine. From a remote HiPerGator login node, create an SSH
tunnel from the local machine instead of exposing the server publicly:

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<login-host>
```

Then open `http://127.0.0.1:8765` locally.

## Run on Windows with downloaded analysis data

The dashboard can run entirely on a Windows workstation. It does not need a GPU, training cache,
Qwen weights, or the 8-GiB checkpoints. Download or clone this repository so the prebuilt
`multimodel_dashboard/dist/` directory is present, then copy only the analysis metadata and
artifacts into a local run directory:

```text
C:\research\multimodel-ASPD\repo\
├── aspd\
├── multimodel_dashboard\
│   └── dist\
└── local_data\qwen3_1_7b_d0_s1_main\
    ├── experiment_config.json
    ├── provenance.json
    ├── latest_checkpoint.txt              # optional; metadata only
    └── analysis\
        ├── posthoc.safetensors
        ├── taxonomy.json
        └── top_activation_examples.json
```

`experiment_config.json` and `provenance.json` are small but strongly recommended: they provide the
real model names, C, K, layer, matrices, thresholds, token count, and git commit. The actual
checkpoint file referenced by `latest_checkpoint.txt` is not required for viewing P1--P5.

### 1. Install Python 3.13 and uv

Open PowerShell. If `uv` is not installed:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new PowerShell window after installation and verify:

```powershell
uv --version
uv python install 3.13
```

### 2. Create a lightweight dashboard environment

From the repository root:

```powershell
cd C:\research\multimodel-ASPD\repo

uv venv --python 3.13 .venv-dashboard
uv pip install --python .venv-dashboard\Scripts\python.exe `
  torch numpy safetensors fastapi uvicorn pydantic pyyaml
```

This intentionally installs only dashboard/runtime dependencies instead of the full training
environment.

### 3. Verify the downloaded artifacts

```powershell
$ANALYSIS = ".\local_data\qwen3_1_7b_d0_s1_main\analysis"

.\.venv-dashboard\Scripts\python.exe -m aspd.multimodel.cli.serve_analysis `
  $ANALYSIS `
  --verify-only
```

The command should print the discovered component/model counts, tensor keys, matrices, taxonomy
counts, example coverage, and Data Health status. Investigate a `FAIL` before interpreting features.

### 4. Start the local dashboard

```powershell
.\.venv-dashboard\Scripts\python.exe -m aspd.multimodel.cli.serve_analysis `
  $ANALYSIS `
  --host 127.0.0.1 `
  --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in a Windows browser. Keep the PowerShell
window open while using the dashboard; press `Ctrl+C` there to stop it.

The server automatically finds `experiment_config.json` and `provenance.json` in the parent of the
`analysis` directory. If the config is stored elsewhere, pass it explicitly:

```powershell
.\.venv-dashboard\Scripts\python.exe -m aspd.multimodel.cli.serve_analysis `
  $ANALYSIS `
  --config "C:\research\configs\d0_s1_main.yaml" `
  --host 127.0.0.1 `
  --port 8765
```

Researcher notes are written separately to:

```text
local_data\qwen3_1_7b_d0_s1_main\analysis\researcher_notes.json
```

To place notes elsewhere, add:

```powershell
--notes "C:\research\notes\qwen3_d0_s1_researcher_notes.json"
```

### Optional: rebuild the React frontend on Windows

The checked-in `dist/` bundle means Node.js is not needed for normal use. Install Node.js only when
editing the React source, then rebuild with:

```powershell
cd C:\research\multimodel-ASPD\repo\multimodel_dashboard
npm install
npm run check
npm run build
```

## Tests

```bash
.venv/bin/pytest \
  tests/test_multimodel_analysis.py \
  tests/test_multimodel_config.py \
  tests/test_multimodel_dashboard.py
.venv/bin/ruff check \
  aspd/multimodel/analysis.py \
  aspd/multimodel/cli/extract_examples.py \
  aspd/multimodel/dashboard.py \
  aspd/multimodel/cli/serve_analysis.py \
  tests/test_multimodel_analysis.py \
  tests/test_multimodel_config.py \
  tests/test_multimodel_dashboard.py

cd multimodel_dashboard
npm run check
npm run build
```

The Python tests cover ranked/all/explicit extraction, atomic merge semantics, tensor-to-feature
mapping, context coverage, health failures, REST endpoints, exports, and the guarantee that saving
notes does not modify raw artifacts.
