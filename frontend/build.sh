#!/bin/bash
# Build our fork of the lab's Svelte frontend into dist/, which serve_app.py serves.
#
#   libs/lm_interp/frontend/build.sh
#
# The lab ships this app as a prebuilt dist/ inside its wheel, and there is no node on $PATH, which
# is why every earlier fix to the app had to be backend-side. There IS a node on the cluster --
# /apps/nodejs/22.12.0 -- and the wheel ships the Svelte sources too, so the UI changes that the
# runtime patches cannot express (sortable activating examples, clickable attribution pills, the
# influence-rank readout) live here instead.
#
# src/ started as a verbatim copy of <site-packages>/param_decomp_lab/app/frontend/src. Keep the
# diff against that copy small: bumping the pinned param-decomp rev means re-merging by hand.
#
# npm's cache is redirected off $HOME, which is small on this cluster.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/apps/nodejs/22.12.0/bin:$PATH
REPO_ROOT="$(cd ../../.. && pwd)"
export npm_config_cache="$REPO_ROOT/.cache/npm"
mkdir -p "$npm_config_cache"
test -d node_modules || npm ci
npm run build
echo "built $(pwd)/dist"
