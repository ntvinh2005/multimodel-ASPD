import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import ComponentComputationInspector, { normalizedContribution } from "./ComponentComputationInspector";
import { sampleComponentContexts } from "./ComponentContextProbe";
import type { TokenExample, TokenPosition } from "./tokenDiscovery";

it("keeps inactive, absent, and measured magnitudes distinct", () => {
  expect(normalizedContribution(0)).toEqual({ magnitude: 0, relative: null });
  expect(normalizedContribution(undefined, 4, 2)).toEqual({ magnitude: null, relative: null });
  expect(normalizedContribution(-3, 4, 2)).toEqual({ magnitude: 12, relative: 6 });
});

it("rejects coefficients from another anchor or checkpoint", () => {
  const id = "smol.L08.mlp_down/7417";
  const html = renderToStaticMarkup(<ComponentComputationInspector id={id} anchor="selected" checkpoint="actual" measurement={{
    scope: { checkpoint: "actual", anchor: "other", dataset: "data", pool: null, example: null, artifact: "cache" },
    complete: false, targets: [id], edges: [], coefficients: { [id]: 123456 },
  }} />);
  expect(html).not.toContain("123456");
  expect(html).toContain("Component computation");
  expect(html).toContain("The export has no saved model activations for this component at this token.");
});

it("selects natural positive and inactive positions within each exact example", () => {
  const tokens = [0, .6, .2, .1].map((value, index): TokenPosition => ({
    anchor_id: String(index), character_boundary: index + 1, pool_id: "pool", roles_ending: [],
    model_positions: { gemma: index, granite: index, smol: index }, final_query: index === 3,
    pre: { active_gate_ids: [], values: [] }, mid: { active_gate_ids: value ? [7417] : [], values: value ? [value] : [] },
  }));
  const ex: TokenExample = { example_id: "row", group_id: "g", condition: "negative", prompt: "ABCD", positions: tokens };
  const rows = sampleComponentContexts([ex], "MID:7417");
  expect(rows.map(row => [row.position.anchor_id, row.text, row.gate])).toEqual([
    ["3", "D", .1], ["1", "B", .6], ["2", "C", .2], ["0", "A", 0],
  ]);
});
