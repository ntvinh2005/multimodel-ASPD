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

export interface VocabularyEffectModel {
  model_name: string | null;
  positive: VocabularyEffect[];
  negative: VocabularyEffect[];
  source: string;
}

export interface VocabularyEffects {
  status: "available" | "empty" | "missing" | "error";
  kind: VocabularyEffectKind | null;
  label: string;
  description: string;
  positive: VocabularyEffect[];
  negative: VocabularyEffect[];
  models: VocabularyEffectModel[];
  default_model: string | null;
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
  related_crosscoder?: AspdCrosscoderRelationship[];
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

export interface AspdCrosscoderRelationship {
  phenomenon_id: string;
  phenomenon_label: string;
  status: ComparisonStatus;
  status_definition: string;
  crosscoder_feature_ids: number[];
}

export interface CrosscoderAspdRelationship {
  phenomenon_id: string;
  phenomenon_label: string;
  status: ComparisonStatus;
  status_definition: string;
  aspd_component_ids: number[];
}

export interface SemanticAnnotation {
  status?: string;
  tentative_label: string | null;
  interpretation_confidence: number | null;
  annotation_source: string | null;
  evidence_validation_sequence_indices?: number[];
  warning?: string | null;
  notes?: string | null;
  model?: string | null;
  model_version?: string | null;
  limitations?: string[];
  source: string | null;
  is_measured_ground_truth: false;
}

export interface CrosscoderMetadata {
  run_name: string | null;
  model_ids: Array<string | null>;
  sites: Array<string | null>;
  n_features: number | null;
  training_steps: number | null;
  configured_max_steps: number | null;
  train_batch_topk: number | null;
  validation_tokens: number | null;
  sequence_length: number | null;
  checkpoint: string | null;
  config_path: string;
  analysis_dir: string;
  reference: {
    method?: string;
    experiment_repository?: string;
    experiment_commit_audited?: string;
    library_repository?: string;
    library_commit?: string;
    relative_decoder_norm_convention?: string;
  };
}

export interface CrosscoderMetrics {
  fve_total: NullableNumber;
  fve_base: NullableNumber;
  fve_finetuned: NullableNumber;
  validation_l0: NullableNumber;
  dead_count: number | null;
  dead_fraction: NullableNumber;
  relative_norm_min: NullableNumber;
  relative_norm_max: NullableNumber;
}

export interface RelativeNormBin {
  start: number;
  end: number;
  count: number;
}

export interface CrosscoderOverviewPayload {
  available: boolean;
  diagnostics: string[];
  metadata?: CrosscoderMetadata;
  metrics?: CrosscoderMetrics;
  classification_counts?: Record<string, number>;
  classification_fractions?: Record<string, number>;
  thresholds?: {
    base_specific?: [number, number];
    shared?: [number, number];
    finetuned_specific?: [number, number];
  };
  relative_decoder_norm_convention?: string | null;
  relative_norm_histogram?: RelativeNormBin[];
  strict_exclusive_absent?: boolean;
  strict_result?: string;
  candidate_selection?: string | null;
  candidate_groups?: Record<string, number[]>;
  sources?: Record<string, string>;
}

export interface CrosscoderFeatureSummary {
  feature_id: number;
  classification: "base-specific" | "shared" | "intermediate" | "finetuned-specific";
  inspection_groups: string[];
  relative_decoder_norm: number;
  relative_decoder_norm_convention?: string;
  decoder_norm_base: number;
  decoder_norm_finetuned: number;
  decoder_cosine: number | null;
  fire_count: number;
  activation_density: number;
  blind_rank_within_class: number;
  blind_rank_base_tail: number;
  blind_rank_finetuned_tail: number;
  semantic_annotation: SemanticAnnotation | null;
  sources: {
    measured: string;
    inspection_groups: string;
    semantic_annotation: string | null;
  };
}

export interface CrosscoderFeatureDetail extends CrosscoderFeatureSummary {
  examples: ActivationExample[] | null;
  context: {
    status: "extracted" | "not_extracted";
    saved_contexts: number;
    message: string;
    source: string;
  };
  related_aspd?: CrosscoderAspdRelationship[];
}

export interface CrosscoderCandidatesPayload {
  available: boolean;
  candidate_selection: string | null;
  groups: Record<string, number[]>;
  items: CrosscoderFeatureSummary[];
  source: string;
  diagnostics?: string[];
}

export type ComparisonStatus =
  | "semantic_overlap"
  | "weak_semantic_overlap"
  | "low_support_hypothesis"
  | "aspd_only_parameter_evidence"
  | "unresolved"
  | "not_tested";

export interface CrosscoderComparisonEvidence {
  feature_id: number;
  classification: CrosscoderFeatureSummary["classification"];
  inspection_groups: string[];
  relative_decoder_norm: number;
  fire_count: number;
  examples: ActivationExample[] | null;
  semantic_annotation: SemanticAnnotation | null;
  sources: CrosscoderFeatureSummary["sources"];
}

export interface AspdComparisonEvidence {
  component_id: number;
  taxonomy: string | null;
  activation_rho: number[] | null;
  mechanism_rho: number[] | null;
  fire_count: number | null;
  low_support: boolean | null;
  examples: ActivationExample[] | null;
  semantic_annotation: SemanticAnnotation | null;
  sources: Record<string, unknown>;
}

export interface MethodComparisonPhenomenon {
  id: string;
  label: string;
  status: ComparisonStatus;
  status_definition: string;
  crosscoder_features: number[];
  aspd_components: number[];
  interpretation: string;
  limitations: string[];
  crosscoder_evidence: CrosscoderComparisonEvidence[];
  aspd_evidence: AspdComparisonEvidence[];
}

export interface MethodComparisonPayload {
  available: boolean;
  diagnostics: string[];
  metadata?: {
    status?: string;
    date?: string;
    not_feature_identity?: boolean;
    not_causal?: boolean;
    annotation_source?: string;
  };
  global_summary?: {
    crosscoder?: string[];
    aspd?: string[];
    current_finding?: string;
  };
  global_measured?: {
    crosscoder: {
      metrics: CrosscoderMetrics;
      classification_counts: Record<string, number>;
      strict_result: string;
    } | null;
    aspd: {
      taxonomy_counts: Record<string, number | null>;
      n_features: number | null;
      validation_tokens: number | null;
    };
  };
  phenomena: MethodComparisonPhenomenon[];
  source?: string;
}
