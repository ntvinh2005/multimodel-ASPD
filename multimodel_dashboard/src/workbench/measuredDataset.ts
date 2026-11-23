import rawDataset from "./data/measuredWorkbenchDataset.json";
import type { WorkbenchDataset } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${path} must be an object`);
  return value;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${path} must be a non-empty string`);
  return value;
}

function requireFiniteOrNull(value: unknown, path: string): void {
  if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error(`${path} must be a finite number or null`);
  }
}

export function validateMeasuredWorkbenchDataset(value: unknown): WorkbenchDataset {
  const root = requireRecord(value, "dataset");
  const manifest = requireRecord(root.manifest, "dataset.manifest");
  if (manifest.schemaVersion !== "1.0") throw new Error("unsupported Workbench schema version");
  if (manifest.synthetic !== false) throw new Error("measured Workbench bundle must set synthetic=false");
  requireString(manifest.datasetId, "dataset.manifest.datasetId");
  requireRecord(manifest.export, "dataset.manifest.export");

  const models = requireArray(root.models, "dataset.models").map((value, index) => requireRecord(value, `dataset.models[${index}]`));
  const modelIds = new Set(models.map((model, index) => requireString(model.id, `dataset.models[${index}].id`)));
  if (modelIds.size !== models.length) throw new Error("duplicate model ids in measured bundle");

  const gates = requireArray(root.gates, "dataset.gates").map((value, index) => requireRecord(value, `dataset.gates[${index}]`));
  const gateIds = new Set(gates.map((gate, index) => requireString(gate.id, `dataset.gates[${index}].id`)));
  if (gateIds.size !== gates.length) throw new Error("duplicate gate ids in measured bundle");

  const anchors = requireArray(root.anchors, "dataset.anchors").map((value, index) => requireRecord(value, `dataset.anchors[${index}]`));
  const anchorIds = new Set(anchors.map((anchor, index) => requireString(anchor.id, `dataset.anchors[${index}].id`)));
  if (anchorIds.size !== anchors.length) throw new Error("duplicate anchor ids in measured bundle");

  const components = requireArray(root.components, "dataset.components").map((value, index) => requireRecord(value, `dataset.components[${index}]`));
  const componentById = new Map<string, Record<string, unknown>>();
  components.forEach((component, index) => {
    const id = requireString(component.id, `dataset.components[${index}].id`);
    if (componentById.has(id)) throw new Error(`duplicate component id ${id}`);
    const modelId = requireString(component.modelId, `dataset.components[${index}].modelId`);
    const gateId = requireString(component.gateId, `dataset.components[${index}].gateId`);
    if (!modelIds.has(modelId)) throw new Error(`component ${id} references unknown model ${modelId}`);
    if (!gateIds.has(gateId)) throw new Error(`component ${id} references unknown gate ${gateId}`);
    requireFiniteOrNull(component.beta, `dataset.components[${index}].beta`);
    requireFiniteOrNull(component.rho, `dataset.components[${index}].rho`);
    componentById.set(id, component);
  });

  const edges = requireArray(root.edges, "dataset.edges").map((value, index) => requireRecord(value, `dataset.edges[${index}]`));
  const edgeIds = new Set<string>();
  edges.forEach((edge, index) => {
    const id = requireString(edge.id, `dataset.edges[${index}].id`);
    if (edgeIds.has(id)) throw new Error(`duplicate edge id ${id}`);
    edgeIds.add(id);
    const sourceId = requireString(edge.source, `dataset.edges[${index}].source`);
    const targetId = requireString(edge.target, `dataset.edges[${index}].target`);
    const source = componentById.get(sourceId);
    const target = componentById.get(targetId);
    if (!source || !target) throw new Error(`edge ${id} references an unknown component`);
    const edgeModel = requireString(edge.modelId, `dataset.edges[${index}].modelId`);
    if (source.modelId !== target.modelId || source.modelId !== edgeModel) {
      throw new Error(`edge ${id} crosses models`);
    }
    if (!anchorIds.has(requireString(edge.anchorId, `dataset.edges[${index}].anchorId`))) {
      throw new Error(`edge ${id} references an unknown anchor`);
    }
    if (edge.evidence !== "measured-taylor") throw new Error(`edge ${id} is not labelled measured-taylor`);
    requireFiniteOrNull(edge.signedEffect, `dataset.edges[${index}].signedEffect`);
    requireFiniteOrNull(edge.normalizedMagnitude, `dataset.edges[${index}].normalizedMagnitude`);
  });
  if (edges.length === 0) throw new Error("measured Workbench bundle has no edges");

  const distributions = requireArray(root.parentDistributions, "dataset.parentDistributions").map((value, index) => requireRecord(value, `dataset.parentDistributions[${index}]`));
  const distributionIds = new Set<string>();
  distributions.forEach((row, index) => {
    const gateId = requireString(row.gateId, `dataset.parentDistributions[${index}].gateId`);
    const modelId = requireString(row.modelId, `dataset.parentDistributions[${index}].modelId`);
    const anchorId = requireString(row.anchorId, `dataset.parentDistributions[${index}].anchorId`);
    if (!modelIds.has(modelId) || !anchorIds.has(anchorId)) throw new Error(`parent distribution ${index} has a broken model or anchor reference`);
    const id = `${gateId}|${modelId}|${anchorId}`;
    if (distributionIds.has(id)) throw new Error(`duplicate parent distribution ${id}`);
    distributionIds.add(id);
    requireFiniteOrNull(row.totalStrength, `dataset.parentDistributions[${index}].totalStrength`);
    if (row.status === "measured") {
      const values = requireRecord(row.values, `dataset.parentDistributions[${index}].values`);
      const total = Object.entries(values).reduce((sum, [key, value]) => {
        if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new Error(`invalid parent mass ${key} in distribution ${id}`);
        return sum + value;
      }, 0);
      if (Math.abs(total - 1) > 1e-8) throw new Error(`parent distribution ${id} does not sum to one`);
    } else if (row.status === "undefined") {
      if (row.values !== null || row.totalStrength !== 0) throw new Error(`undefined parent distribution ${id} must preserve zero total and null values`);
    } else {
      throw new Error(`unsupported parent distribution status for ${id}`);
    }
  });

  const activationExamples = requireArray(root.activationExamples, "dataset.activationExamples").map((value, index) => requireRecord(value, `dataset.activationExamples[${index}]`));
  const exampleGateIds = new Set<string>();
  activationExamples.forEach((row, index) => {
    const base = `dataset.activationExamples[${index}]`;
    const gateId = requireString(row.gateId, `${base}.gateId`);
    if (!gateIds.has(gateId)) throw new Error(`${base} references unknown gate ${gateId}`);
    if (exampleGateIds.has(gateId)) throw new Error(`duplicate activation examples for ${gateId}`);
    exampleGateIds.add(gateId);
    const examples = requireArray(row.examples, `${base}.examples`);
    if (examples.length > 4) throw new Error(`${base} contains more than four saved contexts`);
    examples.forEach((value, exampleIndex) => {
      const example = requireRecord(value, `${base}.examples[${exampleIndex}]`);
      const exampleBase = `${base}.examples[${exampleIndex}]`;
      requireString(example.anchorId, `${exampleBase}.anchorId`);
      requireString(example.documentId, `${exampleBase}.documentId`);
      requireString(example.sourceGroup, `${exampleBase}.sourceGroup`);
      requireString(example.context, `${exampleBase}.context`);
      requireFiniteOrNull(example.activation, `${exampleBase}.activation`);
      const activations = requireArray(example.activations, `${exampleBase}.activations`);
      const firings = requireArray(example.firings, `${exampleBase}.firings`);
      if (activations.length === 0 || activations.length > 65 || firings.length !== activations.length) throw new Error(`${exampleBase} must preserve its clipped window of at most 65 positions`);
      if (typeof example.centerPosition !== "number" || !Number.isInteger(example.centerPosition) || example.centerPosition < 0 || example.centerPosition >= activations.length) throw new Error(`${exampleBase} has invalid center position`);
      activations.forEach((activation, position) => requireFiniteOrNull(activation, `${exampleBase}.activations[${position}]`));
      if (firings.some((firing) => typeof firing !== "boolean")) throw new Error(`${exampleBase}.firings must be boolean`);
      if (firings[example.centerPosition as number] !== true || activations[example.centerPosition as number] !== example.activation) throw new Error(`${exampleBase} does not preserve its natural center firing`);
    });
  });
  if (exampleGateIds.size !== gateIds.size) throw new Error("activation examples do not cover every exported gate");
  return value as WorkbenchDataset;
}

export const measuredWorkbenchDataset = validateMeasuredWorkbenchDataset(rawDataset);
