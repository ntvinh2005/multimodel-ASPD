export type NullableNumber = number | null;

export interface Metadata {
  run_name: string | null;
  model_names: string[];
  model_ids: string[] | null;
  n_models: number | null;
  n_features: number | null;
  top_k: number | null;
  selected_layers: string[] | null;
  matrix_names: string[];
  validation_tokens: number | null;
  shared_epsilon: number | null;
  concentrated_threshold: number | null;
  checkpoint: string | null;
  git_commit: string | null;
  low_support_threshold: number;
  sources: Record<string, string | null>;
}

export interface TaxonomyItem {
  category: string;
  count: number | null;
  percentage: number | null;
  component_ids: number[] | null;
  source: string;
}

export interface ResearcherNote {
  tentative_label: string;
  notes: string;
  semantic_evidence: string;
  alternative_interpretation: string;
  confidence: "low" | "medium" | "high";
  why_interesting: string;
  mentor_notes: string;
  candidate_status: "unreviewed" | "promising" | "control" | "reject" | "P6 candidate";
}

export interface FeatureSummary {
  feature_id: number;
  taxonomy: string | null;
  taxonomy_categories: string[];
  activation_rho: number[] | null;
  mechanism_rho: number[] | null;
  beta: number[] | null;
  beta_total: NullableNumber;
  fire_count: number | null;
  fire_density: NullableNumber;
  low_support: boolean | null;
  dominant_locus: Record<string, string | null>;
  locus_shift: NullableNumber;
  rho_gap: NullableNumber;
  p5_min_component_cosine: NullableNumber;
  p5_min_read_cosine: NullableNumber;
  p5_min_write_cosine: NullableNumber;
  p5_max_relative_change: NullableNumber;
  context_extracted: boolean;
  saved_context_count: number;
  has_examples: boolean;
  note: ResearcherNote | null;
}

export interface P5Row {
  matrix: string;
  component_cosine: NullableNumber;
  relative_component_change: NullableNumber;
  read_cosine: NullableNumber;
  write_cosine: NullableNumber;
  sources: Record<string, string>;
}

export interface ActivationExample {
  center_in_window?: number;
  position_in_sequence?: number;
  center_token?: string;
  g_s?: number;
  context_score?: number | null;
  text?: string;
  token_ids?: number[];
  token_strings?: string[];
  token_texts?: string[];
  token_activations?: Array<number | null>;
  tokens: TokenActivation[];
  activation_detail_kind: "center_only" | "per_token";
  activation_detail_label: string;
  source: string;
}

export interface TokenActivation {
  text: string | null;
  exact_token: string | null;
  token_id: number | null;
  position: number;
  sequence_position: number | null;
  activation: number | null;
  is_center: boolean;
  source: string;
}

export type VocabularyEffectKind = "direct_projection" | "logit_difference" | "other";

export interface VocabularyEffect {
  token: string;
  exact_token: string;
  token_id: number | null;
  value: number;
  rank: number;
  source: string;
}

export interface VocabularyEffects {
  status: "available" | "empty" | "missing" | "error";
  kind: VocabularyEffectKind | null;
  label: string;
  description: string;
  positive: VocabularyEffect[];
  negative: VocabularyEffect[];
  source: string;
}

export interface FeatureDetail extends FeatureSummary {
  decoder_norm: number[] | null;
  locus: Record<string, Record<string, NullableNumber>>;
  p5: P5Row[];
  examples: ActivationExample[] | null;
  vocabulary_effects: VocabularyEffects;
  context: {
    status: "extracted" | "not_extracted";
    fire_count: number | null;
    saved_contexts: number;
    source: string;
    message: string;
  };
  locus_statement: string | null;
  sources: Record<string, unknown>;
  missing: string[];
}

export interface OverviewPayload {
  metadata: Metadata;
  taxonomy: TaxonomyItem[];
  points: FeatureSummary[];
  scatter_available: boolean;
  source_templates: Record<string, string>;
}

export interface HealthPayload {
  overall_status: "PASS" | "WARN" | "FAIL";
  startup_diagnostics: string[];
  checks: Array<Record<string, unknown> & { name: string; status: string }>;
  tensor_keys: string[];
  categorized_keys: Record<string, string[]>;
  missing_expected_keys: string[];
  tensor_shapes: Record<string, number[]>;
  context_coverage: ContextCoverage;
}

export interface ContextCoverage {
  components: number | null;
  components_with_context_extraction: number;
  components_with_at_least_one_saved_context: number;
  components_not_extracted: number;
  components_with_fire_count_gt_zero_but_no_extracted_context: number;
  configured_mode: "ranked" | "all" | "explicit" | null;
  source: string;
}
