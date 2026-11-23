import { normalizedDistribution } from "./graphQueries";
import type { AnchorRecord, Component, DependencyEdge, GateStage, MatrixType, WorkbenchDataset } from "./types";

const revision = "5ca3b43326d77d4d46fa8605394c5e726f5b48b9";
const modelSpecs = [
  ["gemma", "Gemma 2 2B IT", "google/gemma-2-2b-it@299a8560", [9, 10, 11, 12, 13, 14, 15, 16]],
  ["smol", "SmolLM2 1.7B Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2e", [8, 9, 10, 11, 12, 13, 14, 15]],
  ["granite", "Granite 3.3 2B Instruct", "ibm-granite/granite-3.3-2b-instruct@707f574c", [16, 17, 18, 19, 20, 21, 22, 23]],
] as const;
const gateSpecs: Array<[GateStage, number]> = [["PRE", 104], ["PRE", 218], ["MID", 532], ["PRE", 871], ["MID", 1440], ["MID", 2001]];
const matrices: MatrixType[] = ["Q", "K", "V", "O", "gate", "up", "down"];
const anchors: AnchorRecord[] = [
  { id: "anchor-geography", label: "Geography · capital", text: "The capital of France is Paris.", start: 25, end: 30, gateActivations: { "PRE:104": .91, "PRE:218": .42, "MID:532": .78, "PRE:871": .18 }, modelTokens: {} },
  { id: "anchor-negation", label: "Negation · correction", text: "Berlin is not the capital of France; Paris is.", start: 39, end: 44, gateActivations: { "PRE:104": .63, "PRE:218": .84, "MID:532": .55, "PRE:871": null }, modelTokens: {} },
  { id: "anchor-weak", label: "Weak-support control", text: "A quiet sentence with limited measured dependency strength.", start: 2, end: 7, gateActivations: { "PRE:104": .08, "PRE:218": .05, "MID:532": .04, "PRE:871": .03 }, modelTokens: {} },
];
for (const anchor of anchors) {
  for (const [modelId] of modelSpecs) {
    const pieces = anchor.text.split(/(\s+)/).filter(Boolean);
    let offset = 0;
    anchor.modelTokens[modelId] = pieces.map((text, position) => {
      const start = offset; offset += text.length;
      return { text, position, isAnchor: start < anchor.end && offset > anchor.start };
    });
  }
}

const components: Component[] = [];
for (let modelIndex = 0; modelIndex < modelSpecs.length; modelIndex += 1) {
  const [modelId, , , layers] = modelSpecs[modelIndex];
  gateSpecs.forEach(([stage, featureIndex], gateIndex) => {
    const count = gateIndex === 2 ? 4 : gateIndex === 3 ? 3 : 2;
    for (let k = 0; k < count; k += 1) {
      const layer = layers[(gateIndex * 2 + k + modelIndex) % layers.length];
      const matrix = matrices[(gateIndex + k * 2 + modelIndex) % matrices.length];
      const beta = Math.max(.018, .84 - gateIndex * .09 - k * .19 + modelIndex * .025);
      components.push({ id: `${modelId}:${layer}:${matrix}:${stage}:${featureIndex}:${k}`, modelId, gateId: `${stage}:${featureIndex}`, layer, matrix, beta, rho: Math.min(.98, beta * (1.04 + k * .08)), representative: k === 0 || (gateIndex === 2 && k === 1) });
    }
  });
}

const edges: DependencyEdge[] = [];
for (const anchor of anchors) {
  for (let modelIndex = 0; modelIndex < modelSpecs.length; modelIndex += 1) {
    const modelId = modelSpecs[modelIndex][0];
    const local = components.filter((component) => component.modelId === modelId);
    local.forEach((target, index) => {
      const candidates = [local[(index + local.length - 2) % local.length], local[(index + local.length - 5) % local.length]];
      candidates.forEach((source, rank) => {
        if (source.layer > target.layer + 1 || source.id === target.id) return;
        const weak = anchor.id === "anchor-weak";
        const magnitude = weak ? 0 : Math.max(.03, .88 - rank * .31 - (index % 5) * .08 + modelIndex * .025);
        const disagreement = target.gateId === "MID:532" && modelId === "granite";
        const sign = disagreement || (index + rank) % 4 === 0 ? -1 : 1;
        edges.push({ id: `${anchor.id}|${source.id}>${target.id}`, modelId, anchorId: anchor.id, source: source.id, target: target.id, signedEffect: sign * magnitude * .73, normalizedMagnitude: magnitude, upstreamCoefficient: sign * (.18 + index * .011), gradient: magnitude * 1.7, normalization: "|E| ||u_target||₂ / sigma(matrix_target)", evidence: index % 11 === 0 ? "intervention-available" : "taylor-only" });
      });
    });
  }
}

const parentDistributions = gateSpecs.flatMap(([stage, featureIndex]) => modelSpecs.flatMap(([modelId], modelIndex) => anchors.map((anchor) => {
  const gateId = `${stage}:${featureIndex}`;
  const targets = new Set(components.filter((component) => component.modelId === modelId && component.gateId === gateId && component.representative).map((component) => component.id));
  const raw: Record<string, number> = {};
  edges.filter((edge) => edge.modelId === modelId && edge.anchorId === anchor.id && targets.has(edge.target)).forEach((edge) => {
    const parent = components.find((component) => component.id === edge.source);
    if (parent && edge.normalizedMagnitude !== null) raw[parent.gateId] = (raw[parent.gateId] ?? 0) + edge.normalizedMagnitude;
  });
  if (targets.size > 0) Object.keys(raw).forEach((parentGate) => { raw[parentGate] /= targets.size; });
  const normalized = normalizedDistribution(raw);
  return { gateId, modelId, anchorId: anchor.id, status: normalized.status, totalStrength: normalized.total, values: normalized.values };
})));

export const syntheticDataset: WorkbenchDataset = {
  manifest: { schemaVersion: "1.0", datasetId: "synthetic-v3.1-workbench-v1", label: "Deterministic synthetic demonstration — not pretrained-model results", synthetic: true, sourceRevision: revision, trainingBranch: "aspd-v3.1-main500m-checkpoint-timeout-fix", trainingConfig: "configs/multimodel/aspd_v3/main500m_preparation_v1.yaml", specification: { status: "local-pdf", path: "docs/Universal_Mechanism.pdf", sha256: "a963c3616a498ca1b299fad6f033a94a2ca27a7ac8029de6dc5c7e67ebf31f27" }, dataset: { id: "monology/pile-uncopyrighted", revision: "3be90335b66f24456a5d6659d9c8d208c0357119", splitPolicy: "document hash buckets; seed 0; validation 0–9; test 10–19" }, screening: { metric: "beta", kRep: 2, tauRep: .2, method: "fixture labels consistent with beta screening" }, edgeScoring: { method: "signed first-order Taylor attribution", normalization: "global within anchor; synthetic fixture" } },
  models: modelSpecs.map(([id, label, checkpoint, selectedLayers]) => ({ id, label, checkpoint, tokenizer: checkpoint, selectedLayers: [...selectedLayers] })),
  gates: gateSpecs.map(([stage, featureIndex]) => ({ id: `${stage}:${featureIndex}`, stage, featureIndex, label: `${stage}:${featureIndex}` })),
  anchors, components, edges, parentDistributions,
};
