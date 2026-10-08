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

## Tests

```bash
.venv/bin/pytest tests/test_multimodel_dashboard.py
.venv/bin/ruff check \
  aspd/multimodel/dashboard.py \
  aspd/multimodel/cli/serve_analysis.py \
  tests/test_multimodel_dashboard.py

cd multimodel_dashboard
npm run check
npm run build
```

The Python tests cover tensor-to-feature mapping, health failures, REST endpoints, exports, and the
guarantee that saving notes does not modify raw artifacts.
