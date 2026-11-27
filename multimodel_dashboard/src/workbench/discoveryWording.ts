import reviewed from "../../../docs/multimodel/results/aspd_v3_1_circuit_discovery_hypotheses.json";
import { CONDITION_LABEL } from "./inductionConditions";
import { FAMILY_LABELS, NEGATION_LABELS } from "./negationWorkbenchData";
import type { DiscoveryCandidate } from "./DiscoveredCircuits";

const reviews: Record<string, (typeof reviewed)[keyof typeof reviewed]> = reviewed;

const datasets: Record<string, string> = {
  "natural-validation": "Natural Validation",
  "induction-smoke": "Induction (smoke)",
  "induction-discovery": "Induction",
  "negation-smoke": "Negation (smoke)",
  "negation-discovery": "Negation",
  "selection-frame": "Selection Frame",
};

const families: Record<string, string> = {
  ...FAMILY_LABELS,
  basic: "Repeated prefix",
  repeated_sequence: "Repeated sequence",
  nonselection_voice_by_polarity: "Carrying a book in active and passive voice",
  selection_voice_by_polarity: "Choosing a book in active and passive voice",
  selection_verb_by_polarity: "“Choose” versus “pick”",
  selection_vs_removal_by_polarity: "“Choose” versus “take”",
  source_phrase_by_polarity: "“Of” versus “from”",
  article_by_polarity: "“The” versus “a”",
  object_article_by_polarity: "“The” versus “a” on the object",
  unrelated_subject_by_polarity: "Unrelated sentence controls",
};

const conditions: Record<string, string> = {
  ...NEGATION_LABELS,
  ...CONDITION_LABEL,
  counterfactual: "Changed continuation",
  repeated_no_association: "Repetition without the expected association",
};

export const discoveryDirections = {
  "bottom-up": "Build from components",
  "top-down": "Trace a target’s inputs",
  composition: "Connect saved circuits",
};

export function datasetLabel(value: string) {
  return datasets[value] ?? value;
}

export function familyLabel(value?: string | null) {
  return value ? families[value] ?? value.replaceAll("_", " ") : undefined;
}

export function selectionFrameFamily(group?: string) {
  return group?.replace(/^selection-frame-(?:discovery|heldout)-/, "");
}

export function splitLabel(value: string) {
  return value === "heldout" ? "Held-out" : value === "discovery" ? "Discovery" : value;
}

export function conditionLabel(dataset: string, value?: string | null, family?: string | null) {
  if (!value) return undefined;
  if (dataset === "selection-frame") {
    const negative = value === "negative" || value === "counterfactual_negative";
    const alternate = value.startsWith("counterfactual_");
    if (family === "selection_voice_by_polarity" || family === "nonselection_voice_by_polarity") {
      return `${alternate ? "Passive" : "Active"}, ${negative ? "negative" : "affirmative"}`;
    }
    if (alternate) return negative ? "Alternate wording, negative" : "Alternate wording, affirmative";
  }
  return conditions[value] ?? value.replaceAll("_", " ");
}

export function discoveryWording(candidate: DiscoveryCandidate) {
  const review = reviews[candidate.id]
    ?? (candidate.interpretation.reviewTitle ? candidate.interpretation : undefined);
  const sources = [...new Set(candidate.witnesses.map(row => familyLabel(row.family)).filter(value => value !== undefined))];
  const heldout = candidate.recurrence.find(row => row.dataset === candidate.dataset && row.split === "heldout");
  const stability = candidate.witnessSignStability;
  return {
    title: review?.reviewTitle ?? candidate.id,
    hypothesis: review?.hypothesis,
    alternative: review?.alternative,
    proposedTest: review?.proposedTest,
    sources,
    heldout: heldout ? `The same components and edges appear at ${heldout.anchors} held-out token positions.`
      : "The catalog records no held-out match for these exact components and edges.",
    signs: stability != null && stability < 1
      ? `${Math.round(100 * stability)}% of edges kept the same sign across the displayed examples.`
      : undefined,
  };
}
