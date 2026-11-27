# Research Workbench practices

Use this guide when changing visualization, experiment data, or researcher-facing interpretation in the ASPD Workbench.

## Design and interaction

- Keep each workflow in the correct evaluation view. A Natural Validation result belongs under Natural Validation. A separate research workflow can sit alongside Measured Validation Summary and Measured Graph.
- Reuse the Workbench's light theme, typography, spacing, controls, and graph renderer. Check dropdowns, internal graph labels, inactive rows, legends, and detail panels in the actual browser.
- Put the selected example, token, gate, component, and measured dependencies in one workspace. Keep the final aligned token as the default where that behavior already exists. Update the same inspector on selection. One selection must produce one Research Notes editor.
- Make common tasks direct. Provide searchable PRE/MID gate IDs, sensible initial selections, model comparisons side by side, and document browsing from the relevant component. Preserve the researcher's working context, annotations, and circuit definitions.
- Show graph direction, sign, weight, and model and anchor identity. Base the threshold slider on measured normalized edge weights. Changing the slider filters what the researcher sees. Keep measured parent distributions and their selection rules fixed. Check that zooming and panning preserve the current view.
- Separate the outline of the selected anchor from a measured activation or contribution heatmap. Use a labeled scale that stays fixed across compared examples. Expose exact token identity and raw values on inspection. Give measured zero, inactive support, missing measurement, and display-filtered edge distinct states.

## Research language and evidence

- Write short descriptions grounded in the evidence a researcher can open. Say which components, dataset, model, and anchor produced a result. Omit generic claims about coordination or information processing when the underlying observations do not identify what is computed.
- State the denominator, the unit of analysis, and the limits of a score. For recurrence, say what repeated, how many times, and across how many independent documents. For semantic comparisons, expose diverse strong and contrastive examples, the held-out document counts, and what the matched controls test.
- Show measured observations, researcher hypotheses, counterevidence, and causal validation separately. Shared gate IDs, co-activation, structural recurrence, low OT, and Taylor edge signs provide distinct evidence. A researcher interpretation can remain open when the examples support several explanations.
- For contexts and graphs, carry the exact checkpoint, dataset, split, document, anchor, model, and evaluation pool through each join. A graph at a different position can be inspected as separate evidence. Display unavailable when a dependency was never measured at the selected anchor. Same-anchor attribution and cross-token information flow answer different questions.
- Explain what to look for next directly beside an unfamiliar control or metric. A collapsible walkthrough should use measured examples and identify what remains untested. Keep core comparisons visible without extra navigation.

## Experiments and releases

- Check for existing raw and exported measurements before scheduling compute. Reuse immutable artifacts when their provenance matches. Benchmark missing heavy work, respect current authorizations and shared quotas, and keep pilot output separate from existing studies.
- Define sampling, controls, and qualification before examining held-out scores. Match controls to the scientific question. A plausible match is an unlabeled alternative until it has independent negative evidence. A measured-zero context tests specificity to observed contribution.
- Report source, checksums, split units, exact run revision, and missing coverage. Deduplicate near-identical contexts and count independent documents for coverage claims.
- Test numerical behavior, rendered interactions, production build, and the actual served alias separately. Exercise each route into a shared feature, including search, selection changes, and links from a graph. Verify dataset switches and persistence with existing Research Notes.
- Publish a new measured dataset only after automated integrity and coverage checks. Keep historical artifacts immutable. Use versioned asset paths and update the release pointer only after verification. Record the committed revision, push using the researcher's configured GitHub credentials, and verify the intended branch. Follow the project review process for main.
