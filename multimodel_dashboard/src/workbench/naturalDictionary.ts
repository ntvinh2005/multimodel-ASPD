import { expandedCatalog, type ExpandedSnapshot, type FiringExample } from "./expandedNaturalValidation";
import { searchSharedGates } from "./graphQueries";

export const dictionaryPath = "natural-dictionary-v1";
type Coverage = "i" | "o" | "n";
export const coverageLabel: Record<Coverage, string> = {
  i: "Measured parents and neighborhood",
  o: "Measured as a parent",
  n: "No retained dependency edges",
};

export type DictionaryIndex = {
  schemaVersion: number;
  checkpoint: string;
  checkpointStep: number;
  featureCount: number;
  blockSize: number;
  flags: string;
  blocks: { path: string; sha256: string }[];
  coverage: { incoming: number; outgoingOnly: number; withoutEdges: number; withExamples: number };
  outgoingLimitPerModel: number;
  provenance: { source: string; profileHashes: Record<string, string> };
};

type StoredEdge = [source: number, target: number, E: number, w: number, rank0: number, profileRow: number];
type GateRecord = [anchor: string | null, fireCount: number, modelMask: number, edges: StoredEdge[], examples: FiringExample[]];
export type DictionaryBlock = {
  start: number;
  checkpoint: string;
  components: [id: string, representative: 0 | 1][];
  records: GateRecord[];
};

export function validateDictionary(index: DictionaryIndex) {
  if (index.schemaVersion !== 1 || index.checkpoint !== expandedCatalog.checkpoint || index.checkpointStep !== 61036
      || index.featureCount !== 8192 || index.blockSize < 1
      || index.flags.length !== index.featureCount * 2 || !/^[ion]+$/.test(index.flags)
      || index.blocks.length !== Math.ceil(index.flags.length / index.blockSize)) {
    throw new Error("The gate inventory does not match this Natural Validation checkpoint.");
  }
  return index;
}

export function dictionaryGates(index: DictionaryIndex) {
  return Array.from(index.flags, (status, offset) => ({
    id: `${offset < index.featureCount ? "PRE" : "MID"}:${offset % index.featureCount}`,
    coverage: status as Coverage,
  }));
}

export function searchDictionaryGates<T extends { id: string }>(gates: T[], query: string) {
  const matches = searchSharedGates(gates, query);
  const term = query.trim().toUpperCase().replace(/^(PRE|MID)[\s:/_-]*/, "$1:");
  const exact = matches.filter(({ id }) => id === term || id.split(":")[1] === term);
  return { exact, matches: exact.length ? exact : matches };
}

export function dictionaryLocation(index: DictionaryIndex, id: string) {
  const match = /^(PRE|MID):(\d+)$/.exec(id);
  if (!match || Number(match[2]) >= index.featureCount) throw new Error(`Unknown shared gate ${id}.`);
  const offset = Number(match[2]) + (match[1] === "MID" ? index.featureCount : 0);
  const row = offset % index.blockSize;
  return { ...index.blocks[Math.floor(offset / index.blockSize)], offset, row, coverage: index.flags[offset] as Coverage };
}

export function decodeDictionaryGate(index: DictionaryIndex, id: string, block: DictionaryBlock) {
  const { offset, row, coverage } = dictionaryLocation(index, id);
  const record = block.records[row];
  if (block.checkpoint !== index.checkpoint || block.start + row !== offset || !record) {
    throw new Error("The saved graph block does not match the selected shared gate.");
  }
  const [anchor, fireCount, , edges, examples] = record;
  if ((coverage === "n") !== (!anchor && !edges.length) || Boolean(anchor) !== Boolean(edges.length)) {
    throw new Error("The graph block disagrees with the saved coverage.");
  }
  const snapshot: ExpandedSnapshot | null = anchor ? {
    anchor, kind: "Dictionary neighborhood",
    edges: edges.map(([source, target, E, w, rank0, profileRow]) => ({
      source: block.components[source][0], target: block.components[target][0], E, w, rank0, profileRow,
    })),
    representatives: [...new Set(edges.flatMap(([source, target]) => [source, target]))]
      .map(component => block.components[component]).filter(([, representative]) => representative).map(([id]) => id),
  } : null;
  return { snapshot, fireCount, coverage, examples };
}
