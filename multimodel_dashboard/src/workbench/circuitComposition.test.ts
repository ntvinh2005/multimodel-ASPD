import { describe, expect, it } from "vitest";
import {
  activationTimeline, analyzeRelationships, compareStructure, componentIdentity, loadCircuitLibrary, mergeCircuits,
  mergeMeasurements, projectCompositionGraph, resolveCircuit, restrictMeasurement, saveCircuitLibrary,
  scopeKey, validateLibrary, validateMeasurement, type Circuit, type CircuitMembership, type CompositionEdge,
  type Measurement, type Scope,
} from "./circuitComposition";
import { measurementFromGraph, measurementFromToken } from "./compositionMeasurements";
import seed from "./data/bottomUpCircuits.json";
import { selectToken, tokenGates, type TokenExample, type TokenGraphAsset } from "./tokenDiscovery";

const checkpoint = "a".repeat(64);
const scope: Scope = {
  checkpoint, dataset: "selection-frame", example: "example", anchor: "query", pool: "frozen-pool", artifact: "a-verified-asset",
};
const a = "gemma.L09.q/1";
const b = "gemma.L10.mlp_up/2";
const c = "gemma.L10.v/3";
const d = "gemma.L11.o/4";
const e = "gemma.L12.k/5";
const f = "gemma.L13.o/6";
const g = "gemma.L14.q/7";
const smolA = "smol.L08.k/1";
const smolB = "smol.L09.mlp_up/2";
const graniteA = "granite.L16.v/1";
const circuit = (id: string, components: string[], children: string[] = []): Circuit => ({
  id, name: id, checkpoint, description: "", hypothesis: "", evidence: "", status: "Unreviewed",
  components, children, snapshots: [], claims: [],
});
const membership = (id: string, ...members: string[]): CircuitMembership => ({ id, name: id, members, children: [] });
const edge = (source: string, target: string, signed: number, weight: number): CompositionEdge => ({ source, target, signed, weight });
const measured = (edges: CompositionEdge[], targets = [...new Set(edges.map(item => item.target))]): Measurement => ({
  scope, complete: true, targets, edges,
});

describe("circuit library", () => {
  it("imports the five archival hypotheses with their raw measurement provenance", () => {
    const library = validateLibrary(seed);
    const snapshots = library.circuits.flatMap(item => item.snapshots);
    expect(library.circuits).toHaveLength(5);
    expect(library.circuits.reduce((sum, item) => sum + item.components.length, 0)).toBe(48);
    expect(library.circuits.every(item => item.status === "Hypothesis proposed")).toBe(true);
    expect(snapshots).toHaveLength(83);
    expect(new Set(snapshots.map(item => scopeKey(item.scope))).size).toBe(snapshots.length);
    const edges = snapshots.flatMap(item => item.edges);
    expect(edges).toHaveLength(494);
    expect(edges.every(item => item.row?.rank !== undefined)).toBe(true);
  });

  it("creates, edits, and restores circuits and claims without overwriting imports by default", () => {
    const storage = new Map<string, string>();
    const local = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => { storage.set(key, value); } };
    expect(loadCircuitLibrary(local, seed).circuits).toHaveLength(5);
    const first = { ...circuit("first", [a, b]), snapshots: [restrictMeasurement(measured([edge(a, b, -1, .2)]), new Set([a, b]))] };
    const second = circuit("second", [a, c], ["first"]);
    const original = validateLibrary({ version: 1, circuits: [first, second] });
    saveCircuitLibrary(local, original);
    expect(loadCircuitLibrary(local, seed)).toEqual(original);
    const edited = { ...first, components: [a], hypothesis: "Researcher hypothesis", claims: [{
      target: "second", statement: "Candidate dependency", evidence: "Existing experiment", status: "Hypothesis proposed" as const,
    }] };
    const update = { version: 1 as const, circuits: [edited] };
    expect(mergeCircuits(original, update)).toEqual(original);
    const restored = mergeCircuits(original, update, true);
    saveCircuitLibrary(local, restored);
    expect(loadCircuitLibrary(local, seed).circuits[0]).toMatchObject({
      components: [a], hypothesis: "Researcher hypothesis", claims: edited.claims, snapshots: first.snapshots,
    });
  });

  it("deduplicates overlaps in nested mechanisms and rejects cycles and cross-checkpoint children", () => {
    const first = circuit("first", [a, b]);
    const second = circuit("second", [b, c]);
    const parent = circuit("parent", [c], ["first", "second"]);
    const grandparent = circuit("grandparent", [d], ["first", "parent"]);
    const byId = new Map([first, second, parent, grandparent].map(item => [item.id, item]));
    expect(resolveCircuit("grandparent", byId)).toEqual({ members: [d, a, b, c], children: ["first", "second", "parent"] });
    expect(() => resolveCircuit("parent", new Map(byId).set("first", { ...first, children: ["parent"] }))).toThrow("cycle");
    expect(() => resolveCircuit("parent", new Map(byId).set("first", { ...first, checkpoint: "b".repeat(64) }))).toThrow("cross-checkpoint");
  });
});

describe("same-anchor circuit relationships", () => {
  it("keeps direction and signs through collapse, bridging, convergence and membership overlap", () => {
    const rows = [edge(a, b, -2, .3), edge(a, c, 1, .2), edge(c, b, 5, 1), edge(a, d, 2, .1), edge(b, d, -4, .5)];
    const circuits = [membership("left", a, g), membership("right", b, g)];
    const analysis = analyzeRelationships(circuits, measured(rows), "gemma", 0);
    expect(analysis.direct).toEqual([{ from: "left", to: "right", edge: rows[0] }]);
    expect(analysis.bridges).toEqual([{ from: "left", to: "right", via: c, incoming: rows[1], outgoing: rows[2] }]);
    expect(analysis.paths[0].edges).toEqual([rows[0]]);
    expect(analysis.convergence).toEqual([{ target: d, arms: [
      { circuit: "left", edge: rows[3] }, { circuit: "right", edge: rows[4] },
    ] }]);
    expect(analysis.shared).toEqual([{ id: g, circuits: ["left", "right"] }]);
    const collapsed = projectCompositionGraph(analysis, []);
    const expanded = projectCompositionGraph(analysis, ["left", "right"]);
    expect(collapsed.nodes.get("circuit:left")?.text).toBe("left");
    expect(collapsed.projected.get(JSON.stringify(["circuit:left", "circuit:right", -1]))?.records).toEqual([rows[0]]);
    expect(expanded.nodes.get(a)?.component).toBe(a);
    expect(expanded.projected.get(JSON.stringify([a, b, -1]))?.records).toEqual([rows[0]]);
    expect(analyzeRelationships(circuits, measured(rows), "gemma", .25).bridges).toEqual([]);
    expect(analyzeRelationships(circuits, measured([edge(g, d, 2, .5)]), "gemma", 0).convergence).toEqual([]);
  });

  it("finds longer sequences from exact component edges, without joining different anchors", () => {
    const left = membership("left", a);
    const right = membership("right", f);
    const rows = [edge(a, c, 2, .1), edge(c, e, -2, .2), edge(e, f, 1, .3)];
    const series = mergeMeasurements([
      measured(rows.slice(0, 2)),
      { ...measured(rows.slice(2)), scope: { ...scope, anchor: "other" } },
      { ...measured(rows.slice(2)), scope: { ...scope, pool: "other" } },
      { ...measured(rows.slice(2)), scope: { ...scope, example: "other" } },
      { ...measured(rows.slice(2)), scope: { ...scope, artifact: "other" } },
    ]);
    expect(series).toHaveLength(5);
    for (const item of series) expect(analyzeRelationships([left, right], item, "gemma", 0).paths).toEqual([]);
    const [compatible] = mergeMeasurements([measured(rows.slice(0, 2)), measured(rows.slice(2))]);
    const analysis = analyzeRelationships([left, right], compatible, "gemma", 0);
    expect(analysis.paths).toEqual([{ from: "left", to: "right", edges: rows }]);
    expect(projectCompositionGraph(analysis, []).nodes.has(e)).toBe(true);
    expect(() => mergeMeasurements([measured(rows), measured([{ ...rows[0], weight: .9 }])])).toThrow("Conflicting scores");
  });

  it("distinguishes a measured empty target, a threshold-hidden edge, and unavailable data", () => {
    const memberships = [membership("members", a, b, c)];
    const row = edge(a, b, -1, .01);
    const result = analyzeRelationships(memberships, measured([row], [b, c]), "gemma", .2);
    expect(result.edges).toHaveLength(0);
    expect(result.coverage).toEqual([{ id: "members", members: 3, measured: 2, empty: 1 }]);
    const unavailable = analyzeRelationships(memberships, undefined, "gemma", .2);
    expect(unavailable.coverage).toEqual([{ id: "members", members: 3, measured: 0, empty: null }]);
    const partial = analyzeRelationships(memberships, { ...measured([row], [b, c]), complete: false }, "gemma", .2);
    expect(partial.coverage[0].empty).toBeNull();
    expect(() => validateMeasurement(measured([edge(a, smolB, -1, .2)]))).toThrow("cross-model");
  });

  it("compares directed shared-gate motifs with distinct model-specific implementations", () => {
    const groups = [membership("early", a, smolA), membership("late", b, smolB)];
    const graph = measured([edge(a, b, -2, .2), edge(smolA, smolB, 2, .5)]);
    const gemma = analyzeRelationships(groups, graph, "gemma", 0);
    const smol = analyzeRelationships(groups, graph, "smol", 0);
    expect(compareStructure(gemma, smol)).toMatchObject({ jaccard: 1, leftOnly: [], rightOnly: [] });
    expect(gemma.direct[0].edge).toMatchObject({ source: a, target: b, signed: -2 });
    expect(smol.direct[0].edge).toMatchObject({ source: smolA, target: smolB, signed: 2 });
  });
});

describe("composition with token discovery", () => {
  const first: TokenExample["positions"][number] = {
    anchor_id: "first", pool_id: "frozen-pool", model_positions: { gemma: 1, smol: 2, granite: 3 },
    character_boundary: 3, final_query: false, roles_ending: [],
    pre: { active_gate_ids: [1], values: [.5] }, mid: { active_gate_ids: [], values: [] },
  };
  const query: TokenExample["positions"][number] = {
    ...first, anchor_id: scope.anchor, character_boundary: 9, final_query: true,
    pre: { active_gate_ids: [1], values: [.1] }, mid: { active_gate_ids: [2], values: [.9] },
  };
  const example: TokenExample = {
    example_id: scope.example!, group_id: "pair", condition: "affirmative", prompt: "The story", positions: [first, query],
  };
  const asset: TokenGraphAsset = {
    anchorId: scope.anchor, checkpointSha256: checkpoint, exampleId: example.example_id, poolId: query.pool_id,
    official: {}, completeComponents: true, componentTargets: { [a]: [1, 2, []], [b]: [.5, 3, [[a, -.2]]] },
    outputScale: { "gemma.L09.q": 1, "gemma.L10.mlp_up": 4 },
  };

  it("targets the clicked aligned token and computes separate component fractions for all models", () => {
    expect(selectToken(example, "")).toBe(query);
    expect(selectToken(example, "first")).toBe(first);
    const members = membership("circuit", a, b, smolA, graniteA);
    const positions = example.positions.map(position => ({ id: position.anchor_id,
      values: new Map(tokenGates(position).map(({ id, value }) => [id, value])),
    }));
    expect(activationTimeline(members, "gemma", positions).map(item => item.fraction)).toEqual([.5, 1]);
    expect(activationTimeline(members, "smol", positions).map(item => item.fraction)).toEqual([1, 1]);
    expect(activationTimeline(members, "granite", positions)[0].rows).toEqual([{ id: graniteA, activation: .5 }]);
    expect(componentIdentity(b)).toMatchObject({ modelId: "gemma", layer: 10, matrix: "up", gateId: "MID:2" });
  });

  it("retains measured coefficients and normalized weights only at the verified token", () => {
    const graph = measurementFromToken(asset, scope, example, query);
    expect(() => validateMeasurement(graph)).not.toThrow();
    expect(graph.targets).toEqual([a, b]);
    expect(graph.coefficients?.[b]).toBe(.5);
    expect(graph.edges[0]).toMatchObject({ source: a, target: b, signed: -.2 });
    expect(graph.edges[0].weight).toBeCloseTo(.15);
    expect(() => measurementFromToken(asset, scope, example, first)).toThrow("different checkpoint, example, pool, or token");
    expect(() => measurementFromToken({ ...asset, poolId: "different" }, scope, example, query)).toThrow("or token");
    expect(() => measurementFromToken({ ...asset, checkpointSha256: "different" }, scope, example, query)).toThrow("or token");
  });

  it("excludes cross-anchor and unnormalized legacy edges in partial graph capture", () => {
    const legacy = { id: "source→target", modelId: "gemma", anchorId: scope.anchor, source: a, target: b,
      signedEffect: -1, normalizedMagnitude: .2, upstreamCoefficient: null, gradient: null,
      normalization: "abs(E) × norm(u) / σ", evidence: "measured-taylor" as const };
    const graph = measurementFromGraph(scope, [legacy, { ...legacy, anchorId: "other" }, { ...legacy, evidence: "taylor-only" }]);
    expect(graph).toMatchObject({ scope, complete: false, targets: [b], edges: [edge(a, b, -1, .2)] });
  });
});
