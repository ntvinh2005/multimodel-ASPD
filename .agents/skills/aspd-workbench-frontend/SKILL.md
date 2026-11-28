---
name: aspd-workbench-frontend
description: Implement, review, and verify frontend changes in the Multimodel ASPD Workbench, including measured graph, circuit discovery, and semantic discovery views.
---

# ASPD Workbench frontend

Use black text (`#000`) on white UI surfaces (`#fff`) throughout the Workbench. Explicitly set the
light color scheme, foreground, and background for new views and native form controls. Check
dropdowns and their options, search inputs, notes fields, graph canvases and SVG labels, and
loading and empty states. Legacy dashboard CSS can supply a dark background to unstyled controls.
The Workbench stays light when the browser or operating system prefers dark mode.

Reuse the existing layout, typography, and controls. Reserve color for data such as gate identity,
signed dependencies, measured heatmaps, and activation outlines. Use borders or outlines for
control selection. Preserve the existing sign, scale, and selection semantics.

Build the production frontend and verify computed colors in the rendered Workbench in dark browser
preference. Check the served HTML and CSS on the active demo before reporting deployment.

Follow the [Research Workbench practices](../../../docs/multimodel/research_workbench_practices.md)
for graph provenance, interaction behavior, and release verification.
