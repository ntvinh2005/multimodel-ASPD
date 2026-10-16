import type { ActivationExample, FeatureDetail, VocabularyEffects } from "../types";

export interface ComponentInspectionData {
  activationExamples: ActivationExample[] | null;
  vocabularyEffects: VocabularyEffects;
}

/** Keep project-specific API naming outside the reusable inspection visualizations. */
export function adaptFeatureDetail(feature: FeatureDetail): ComponentInspectionData {
  return {
    activationExamples: feature.examples,
    vocabularyEffects: feature.vocabulary_effects,
  };
}
