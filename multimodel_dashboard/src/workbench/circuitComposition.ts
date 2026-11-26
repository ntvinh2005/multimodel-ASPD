import type { Component, MatrixType } from "./types";

export const COMPOSITION_STORAGE = "aspd-circuit-composition-v1";
export const CIRCUIT_STATUSES = ["Unreviewed", "Hypothesis proposed", "Causally validated", "Rejected"] as const;
export type CircuitStatus = typeof CIRCUIT_STATUSES[number];
export type ComponentIdentity = Pick<Component, "id" | "modelId" | "layer" | "matrix" | "gateId">;
export type Scope = {
  checkpoint: string;
  dataset: string;
  example: string | null;
  anchor: string;
  pool: string | null;
  artifact: string;
};
export type CompositionEdge = {
  source: string;
  target: string;
  signed: number;
  weight: number;
  row?: { shard: string; record: number; flat: number; rank: number };
};
export type Measurement = {
  scope: Scope;
  complete: boolean;
  targets: string[];
  edges: CompositionEdge[];
  coefficients?: Record<string, number>;
};
export type CircuitClaim = { target: string; statement: string; status: CircuitStatus; evidence: string };
export type Circuit = {
  id: string;
  checkpoint: string;
  name: string;
  description: string;
  hypothesis: string;
  evidence: string;
  status: CircuitStatus;
  components: string[];
  children: string[];
  snapshots: Measurement[];
  claims: CircuitClaim[];
};
export type CircuitLibrary = { version: 1; circuits: Circuit[] };
export const EMPTY_LIBRARY: CircuitLibrary = { version: 1, circuits: [] };

export function componentIdentity(id: string): ComponentIdentity {
  const match = /^(gemma|smol|granite)\.L(\d+)\.(q|k|v|o|mlp_gate|mlp_up|mlp_down)\/(\d+)$/.exec(id);
  if (!match) throw new Error(`Invalid exact component ID: ${id}`);
  const [, modelId, layerText, matrixText, featureText] = match;
  const layer = Number(layerText);
  const feature = Number(featureText);
  if (!Number.isSafeInteger(layer) || !Number.isSafeInteger(feature)
    || `L${String(layer).padStart(2, "0")}` !== `L${layerText}` || String(feature) !== featureText) {
    throw new Error(`Noncanonical component ID: ${id}`);
  }
  const matrix = (matrixText.startsWith("mlp_") ? matrixText.slice(4) : matrixText.toUpperCase()) as MatrixType;
  const site = ["Q", "K", "V"].includes(matrix) ? "PRE" : "MID";
  return { id, modelId, layer, matrix, gateId: `${site}:${feature}` };
}

export function scopeKey(scope: Scope) {
  return JSON.stringify([scope.checkpoint, scope.dataset, scope.example, scope.anchor, scope.pool, scope.artifact]);
}

export function validateMeasurement(measurement: Measurement, checkpoint = measurement?.scope?.checkpoint) {
  const scope = measurement?.scope;
  if (!scope || !/^[0-9a-f]{64}$/.test(scope.checkpoint) || scope.checkpoint !== checkpoint
    || ![scope.dataset, scope.anchor, scope.artifact].every(value => typeof value === "string" && value.length)
    || ![scope.example, scope.pool].every(value => value === null || typeof value === "string" && value.length)
    || typeof measurement.complete !== "boolean" || !Array.isArray(measurement.targets) || !Array.isArray(measurement.edges)) {
    throw new Error("Invalid dependency provenance.");
  }
  const targets = new Set(measurement.targets);
  if (targets.size !== measurement.targets.length) throw new Error("Duplicate measured target.");
  for (const target of targets) componentIdentity(target);
  const seen = new Set<string>();
  for (const { source, target, signed, weight, row } of measurement.edges) {
    const key = JSON.stringify([source, target]);
    if (source === target || seen.has(key) || componentIdentity(source).modelId !== componentIdentity(target).modelId
      || !targets.has(target) || !Number.isFinite(signed) || !Number.isFinite(weight) || weight < 0
      || (row && (!(typeof row.shard === "string" && row.shard)
        || ![row.record, row.flat, row.rank].every(value => Number.isSafeInteger(value) && value >= 0)))) {
      throw new Error("Invalid, duplicate, or cross-model dependency record.");
    }
    seen.add(key);
  }
  for (const [id, value] of Object.entries(measurement.coefficients ?? {})) {
    if (!targets.has(id) || !Number.isFinite(value)) throw new Error("Invalid measured coefficient.");
  }
}

export function resolveCircuit(id: string, byId: ReadonlyMap<string, Circuit>) {
  const components = new Set<string>();
  const done = new Set<string>();
  const visiting = new Set<string>();
  function visit(key: string, checkpoint?: string) {
    if (visiting.has(key)) throw new Error("Nested circuits must not contain a cycle.");
    const circuit = byId.get(key);
    if (!circuit || (checkpoint && circuit.checkpoint !== checkpoint)) throw new Error("Unknown or cross-checkpoint nested circuit.");
    if (done.has(key)) return;
    visiting.add(key);
    circuit.components.forEach(component => components.add(component));
    circuit.children.forEach(child => visit(child, circuit.checkpoint));
    visiting.delete(key);
    done.add(key);
  }
  visit(id);
  return { members: [...components], children: [...done].filter(key => key !== id) };
}

export function resolveComponents(id: string, byId: ReadonlyMap<string, Circuit>): string[] {
  return resolveCircuit(id, byId).members;
}

export function validateLibrary(value: unknown): CircuitLibrary {
  const library = value as CircuitLibrary;
  if (library?.version !== 1 || !Array.isArray(library.circuits)) throw new Error("Expected a version 1 circuit library.");
  const byId = new Map<string, Circuit>();
  for (const circuit of library.circuits) {
    if (!circuit || ![circuit.id, circuit.name, circuit.checkpoint].every(item => typeof item === "string" && item.trim())
      || !/^[0-9a-f]{64}$/.test(circuit.checkpoint) || byId.has(circuit.id)
      || ![circuit.description, circuit.hypothesis, circuit.evidence].every(item => typeof item === "string")
      || !CIRCUIT_STATUSES.includes(circuit.status)
      || ![circuit.components, circuit.children, circuit.snapshots, circuit.claims].every(Array.isArray)
      || (!circuit.components.length && !circuit.children.length)
      || new Set(circuit.components).size !== circuit.components.length
      || new Set(circuit.children).size !== circuit.children.length) throw new Error("Invalid or duplicate circuit.");
    circuit.components.forEach(componentIdentity);
    circuit.snapshots.forEach(snapshot => validateMeasurement(snapshot, circuit.checkpoint));
    byId.set(circuit.id, circuit);
  }
  for (const circuit of byId.values()) {
    resolveComponents(circuit.id, byId);
    for (const claim of circuit.claims) {
      if (!claim || claim.target === circuit.id || byId.get(claim.target)?.checkpoint !== circuit.checkpoint
        || typeof claim.statement !== "string" || !claim.statement.trim() || typeof claim.evidence !== "string"
        || !CIRCUIT_STATUSES.includes(claim.status)) throw new Error("Invalid circuit relationship claim.");
    }
  }
  return library;
}

export function mergeCircuits(current: CircuitLibrary, incoming: CircuitLibrary, replace = false) {
  const circuits = new Map(current.circuits.map(circuit => [circuit.id, circuit]));
  for (const circuit of incoming.circuits) if (replace || !circuits.has(circuit.id)) circuits.set(circuit.id, circuit);
  return validateLibrary({ version: 1, circuits: [...circuits.values()] });
}

export function loadCircuitLibrary(storage: Pick<Storage, "getItem">, defaults: unknown): CircuitLibrary {
  const raw = storage.getItem(COMPOSITION_STORAGE);
  return validateLibrary(raw === null ? defaults : JSON.parse(raw));
}

export function saveCircuitLibrary(storage: Pick<Storage, "setItem">, library: CircuitLibrary) {
  storage.setItem(COMPOSITION_STORAGE, JSON.stringify(validateLibrary(library)));
}

// A saved membership snapshot is a filtered observation, even if the original asset was complete.
export function restrictMeasurement(measurement: Measurement, members: ReadonlySet<string>): Measurement {
  return {
    scope: measurement.scope,
    complete: false,
    targets: measurement.targets.filter(id => members.has(id)),
    edges: measurement.edges.filter(edge => members.has(edge.source) && members.has(edge.target)),
    coefficients: Object.fromEntries(Object.entries(measurement.coefficients ?? {}).filter(([id]) => members.has(id))),
  };
}

export function mergeMeasurements(measurements: readonly Measurement[]): Measurement[] {
  const groups = new Map<string, Measurement[]>();
  for (const measurement of measurements) {
    const key = scopeKey(measurement.scope);
    const group = groups.get(key) ?? [];
    group.push(measurement);
    groups.set(key, group);
  }
  return [...groups.values()].map(group => {
    const edges = new Map<string, CompositionEdge>();
    const targets = new Set<string>();
    const coefficients = new Map<string, number>();
    for (const measurement of group) {
      measurement.targets.forEach(id => targets.add(id));
      for (const edge of measurement.edges) {
        const key = JSON.stringify([edge.source, edge.target]);
        const previous = edges.get(key);
        if (previous && (previous.signed !== edge.signed || previous.weight !== edge.weight
          || (previous.row && edge.row && JSON.stringify(previous.row) !== JSON.stringify(edge.row)))) {
          throw new Error("Conflicting scores at the same measurement anchor.");
        }
        if (!previous || edge.row) edges.set(key, edge);
      }
      for (const [id, coefficient] of Object.entries(measurement.coefficients ?? {})) {
        if (coefficients.has(id) && coefficients.get(id) !== coefficient) {
          throw new Error("Conflicting coefficients at the same measurement anchor.");
        }
        coefficients.set(id, coefficient);
      }
    }
    return { scope: group[0].scope, complete: group.some(record => record.complete), edges: [...edges.values()],
      targets: [...targets], coefficients: Object.fromEntries(coefficients) };
  });
}

export type CircuitMembership = { id: string; name: string; members: string[]; children: string[] };
export type Directed = { from: string; to: string; edge: CompositionEdge };
export type Bridge = { from: string; to: string; via: string; incoming: CompositionEdge; outgoing: CompositionEdge };
export type Convergence = { target: string; arms: { circuit: string; edge: CompositionEdge }[] };

function shortestCircuitPaths(circuits: readonly { id: string; members: ReadonlySet<string> }[],
  edges: readonly CompositionEdge[]) {
  const bySource = new Map<string, CompositionEdge[]>();
  for (const edge of edges) {
    const outgoing = bySource.get(edge.source) ?? [];
    outgoing.push(edge);
    bySource.set(edge.source, outgoing);
  }
  const paths: { from: string; to: string; edges: CompositionEdge[] }[] = [];
  for (const from of circuits) for (const to of circuits) {
    if (from.id === to.id || !from.members.size || !to.members.size) continue;
    const queue = [...from.members].filter(id => !to.members.has(id));
    const visited = new Set(from.members);
    const previous = new Map<string, CompositionEdge>();
    for (let i = 0; i < queue.length; i++) {
      const component = queue[i];
      if (to.members.has(component)) {
        const path: CompositionEdge[] = [];
        for (let edge = previous.get(component); edge; edge = previous.get(edge.source)) path.push(edge);
        paths.push({ from: from.id, to: to.id, edges: path.reverse() });
        break;
      }
      for (const edge of bySource.get(component) ?? []) {
        if (visited.has(edge.target)) continue;
        visited.add(edge.target);
        previous.set(edge.target, edge);
        queue.push(edge.target);
      }
    }
  }
  return paths;
}

export function analyzeRelationships(circuits: readonly CircuitMembership[], measurement: Measurement | undefined,
  model: string, minimum: number) {
  const scoped = circuits.map(circuit => ({ ...circuit, members: new Set(circuit.members.filter(id => componentIdentity(id).modelId === model)) }));
  const owners = new Map<string, Set<string>>();
  for (const { id, members } of scoped) for (const member of members) {
    const membership = owners.get(member) ?? new Set<string>();
    membership.add(id);
    owners.set(member, membership);
  }
  const byId = new Map(scoped.map(circuit => [circuit.id, circuit.members]));
  const shared = [...owners].filter(([, ids]) => ids.size > 1).map(([id, groups]) => ({ id, circuits: [...groups] }));
  const modelEdges = measurement?.edges.filter(edge => componentIdentity(edge.source).modelId === model
    && componentIdentity(edge.target).modelId === model) ?? [];
  const edges = modelEdges.filter(edge => edge.weight >= minimum);
  const direct: Directed[] = [];
  const into = new Map<string, CompositionEdge[]>();
  const outof = new Map<string, CompositionEdge[]>();
  const converging = new Map<string, Convergence["arms"]>();
  function* ownersOfEdge(source: string, target: string) {
    for (const from of owners.get(source) ?? []) for (const to of owners.get(target) ?? []) {
      if (from !== to && !byId.get(from)?.has(target) && !byId.get(to)?.has(source)) yield { from, to };
    }
  }
  for (const edge of edges) {
    const { source, target } = edge;
    for (const pair of ownersOfEdge(source, target)) direct.push({ ...pair, edge });
    if (owners.has(source)) {
      const incoming = into.get(target) ?? [];
      incoming.push(edge);
      into.set(target, incoming);
      const arms = converging.get(target) ?? [];
      for (const circuit of owners.get(source)!) arms.push({ circuit, edge });
      converging.set(target, arms);
    }
    if (owners.has(target)) {
      const outgoing = outof.get(source) ?? [];
      outgoing.push(edge);
      outof.set(source, outgoing);
    }
  }
  const bridges: Bridge[] = [];
  for (const [via, incoming] of into) {
    for (const left of incoming) for (const right of outof.get(via) ?? []) {
      for (const pair of ownersOfEdge(left.source, right.target)) {
        // A bridge can belong to a third selected circuit; each endpoint remains exact.
        if (via !== left.source && via !== right.target) bridges.push({ ...pair, via, incoming: left, outgoing: right });
      }
    }
  }
  const convergence: Convergence[] = [];
  for (const [target, arms] of converging) {
    const distinct = arms.filter(a => arms.some(b => a.circuit !== b.circuit && a.edge.source !== b.edge.source
      && !byId.get(a.circuit)?.has(b.edge.source) && !byId.get(b.circuit)?.has(a.edge.source)));
    if (distinct.length) convergence.push({ target, arms: distinct });
  }
  const rows = new Set(measurement?.targets ?? []);
  const coverage = scoped.map(({ id, members }) => {
    const measured = [...members].filter(member => rows.has(member));
    const retained = new Set(modelEdges.map(edge => edge.target));
    return { id, members: members.size, measured: measured.length,
      empty: measurement?.complete ? measured.filter(member => !retained.has(member)).length : null };
  });
  const paths = shortestCircuitPaths(scoped, edges);
  return { direct, bridges, convergence, paths, shared, coverage, edges, scoped };
}

export function activationTimeline(circuit: CircuitMembership, model: string, positions: readonly {
  id: string; values: ReadonlyMap<string, number>;
}[]) {
  const components = circuit.members.filter(id => componentIdentity(id).modelId === model);
  return positions.map(position => {
    const rows = components.map(id => ({ id, activation: position.values.get(componentIdentity(id).gateId) ?? 0 }));
    const active = rows.filter(row => row.activation > 0).length;
    return { id: position.id, rows, active, total: rows.length, fraction: rows.length ? active / rows.length : null };
  });
}

export function compareStructure(left: ReturnType<typeof analyzeRelationships>, right: ReturnType<typeof analyzeRelationships>) {
  function signatures(result: ReturnType<typeof analyzeRelationships>) {
    const gate = (id: string) => componentIdentity(id).gateId;
    const byId = new Map(result.scoped.map(circuit => [circuit.id, circuit.members]));
    return new Set([
      ...result.direct.map(({ from, to, edge }) => JSON.stringify(["direct", from, to, gate(edge.source), gate(edge.target)])),
      ...result.bridges.map(({ from, to, via, incoming, outgoing }) => JSON.stringify(
        ["bridge", from, to, gate(incoming.source), gate(via), gate(outgoing.target)])),
      ...result.paths.filter(path => path.edges.length > 2).map(({ from, to, edges }) => JSON.stringify(
        ["sequence", from, to, gate(edges[0].source), ...edges.map(edge => gate(edge.target))])),
      ...result.convergence.flatMap(({ target, arms }) => arms.flatMap((a, i) => arms.slice(i + 1)
        .filter(b => a.circuit !== b.circuit && a.edge.source !== b.edge.source
          && !byId.get(a.circuit)?.has(b.edge.source) && !byId.get(b.circuit)?.has(a.edge.source))
        .map(b => JSON.stringify(["convergence", gate(target), ...[
          [a.circuit, gate(a.edge.source)], [b.circuit, gate(b.edge.source)],
        ].sort((x, y) => JSON.stringify(x).localeCompare(JSON.stringify(y)))])))),
    ]);
  }
  const a = signatures(left);
  const b = signatures(right);
  const shared = [...a].filter(key => b.has(key));
  const union = new Set([...a, ...b]).size;
  const bridgeGates = (result: ReturnType<typeof analyzeRelationships>) => new Set(result.bridges.map(({ via }) => componentIdentity(via).gateId));
  const x = bridgeGates(left);
  const y = bridgeGates(right);
  return { shared, leftOnly: [...a].filter(key => !b.has(key)), rightOnly: [...b].filter(key => !a.has(key)),
    jaccard: union ? shared.length / union : null, sharedBridges: [...x].filter(gate => y.has(gate)),
    leftBridges: [...x].filter(gate => !y.has(gate)), rightBridges: [...y].filter(gate => !x.has(gate)) };
}

export function projectCompositionGraph(analysis: ReturnType<typeof analyzeRelationships>, expanded: readonly string[]) {
  const nodes = new Map<string, { x: number; y: number; text: string; component?: string }>();
  const map = new Map<string, string[]>();
  const memberNodes = new Set(analysis.scoped.flatMap(circuit => [...circuit.members]));
  const span = 232;
  const usefulEdges = new Map<string, CompositionEdge>();
  function add(edge: CompositionEdge) { usefulEdges.set(JSON.stringify([edge.source, edge.target]), edge); }
  for (const edge of analysis.edges) if (memberNodes.has(edge.source) && memberNodes.has(edge.target)) add(edge);
  analysis.bridges.forEach(({ incoming, outgoing }) => { add(incoming); add(outgoing); });
  analysis.paths.forEach(path => path.edges.forEach(add));
  analysis.convergence.forEach(({ arms }) => arms.forEach(({ edge }) => add(edge)));
  for (const [i, circuit] of analysis.scoped.entries()) {
    const x = 116 + i * span;
    const isExpanded = expanded.includes(circuit.id);
    const groupKey = `circuit:${circuit.id}`;
    if (!isExpanded || !circuit.members.size) nodes.set(groupKey, { x, y: 32, text: circuit.name });
    for (const [j, component] of [...circuit.members].entries()) {
      const group = map.get(component) ?? [];
      if (isExpanded) {
        if (!nodes.has(component)) nodes.set(component, { x, y: 32 + j * 38, text: component, component });
        if (!group.includes(component)) group.push(component);
      } else group.push(groupKey);
      map.set(component, group);
    }
  }
  const external = [...new Set([...usefulEdges.values()].flatMap(edge => [edge.source, edge.target]))].filter(id => !map.has(id));
  for (const [i, id] of external.entries()) {
    nodes.set(id, { x: 116 + analysis.scoped.length * span, y: 32 + i * 38, text: id, component: id });
    map.set(id, [id]);
  }
  const projected = new Map<string, { from: string; to: string; records: CompositionEdge[] }>();
  for (const edge of usefulEdges.values()) {
    for (const from of map.get(edge.source) ?? []) for (const to of map.get(edge.target) ?? []) {
      const key = JSON.stringify([from, to, Math.sign(edge.signed)]);
      const row = projected.get(key) ?? { from, to, records: [] };
      row.records.push(edge);
      projected.set(key, row);
    }
  }
  const height = Math.max(110, ...[...nodes.values()].map(node => node.y + 46));
  const width = Math.max(420, span * (analysis.scoped.length + Number(external.length > 0)) + 12);
  return { nodes, projected, width, height };
}
