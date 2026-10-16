import { useState } from "react";
import type { VocabularyEffect, VocabularyEffectModel, VocabularyEffects } from "../types";

export function vocabularyEffectKicker(data: VocabularyEffects): string {
  if (data.status === "missing") return "No vocabulary/logit artifact available";
  if (data.status === "error") return "Artifact could not be interpreted";
  if (data.kind === "direct_projection") return "Projection proxy — not a causal intervention";
  if (data.kind === "logit_difference") return "Measured intervention-based logit difference";
  return "Artifact-declared vocabulary values";
}

export function LogitEffects({ data, limit = 10 }: { data: VocabularyEffects; limit?: number }) {
  const initialModel = data.default_model ?? data.models?.[0]?.model_name ?? null;
  const [selectedModel, setSelectedModel] = useState<string | null>(initialModel);
  if (data.status === "missing") {
    return <div className="empty-state vocabulary-empty"><strong>Vocabulary data unavailable</strong><p>{data.description}</p></div>;
  }
  if (data.status === "error") {
    return <div className="error-box"><strong>Vocabulary artifact error</strong><p>{data.description}</p></div>;
  }
  if (data.status === "empty") {
    return <div className="empty-state vocabulary-empty"><strong>No vocabulary values retained</strong><p>{data.description}</p></div>;
  }

  const series = modelSeries(data);
  const active = series.find((item) => item.model_name === selectedModel)
    ?? series.find((item) => item.model_name === data.default_model)
    ?? series[0];
  const positive = [...active.positive].sort((a, b) => b.value - a.value).slice(0, limit);
  const negative = [...active.negative].sort((a, b) => a.value - b.value).slice(0, limit);
  const noun = data.kind === "direct_projection" ? "projection" : "logits";
  return <>
    <p className="explain">{data.description}</p>
    {series.length > 1 && <div className="vocabulary-model-tabs" role="group" aria-label="Vocabulary projection model">
      {series.map((item) => <button
        type="button"
        key={item.model_name ?? "single"}
        className={item.model_name === active.model_name ? "active" : ""}
        onClick={() => setSelectedModel(item.model_name)}
      >{item.model_name ?? "single model"}</button>)}
    </div>}
    <div className="logit-columns">
      <LogitColumn title={`Negative ${noun}`} direction="negative" entries={negative} />
      <LogitColumn title={`Positive ${noun}`} direction="positive" entries={positive} />
    </div>
    <div className="source-line">Source: {active.source}</div>
  </>;
}

function modelSeries(data: VocabularyEffects): VocabularyEffectModel[] {
  if (data.models?.length) return data.models;
  return [{
    model_name: null,
    positive: data.positive,
    negative: data.negative,
    source: data.source,
  }];
}

export function LogitColumn({
  title,
  direction,
  entries,
}: {
  title: string;
  direction: "positive" | "negative";
  entries: VocabularyEffect[];
}) {
  return <div className={`logit-column ${direction}`}>
    <h3>{title}</h3>
    {entries.length === 0 ? <p className="na">No {direction} values were retained.</p> :
      <ol>{entries.map((entry) => <li key={`${entry.rank}-${entry.token_id}-${entry.token}`}>
        <span className="logit-rank">{entry.rank}</span>
        <span className="token-chip" title={effectTooltip(entry)}>{entry.token || "\"\""}</span>
        <span className="logit-value">{formatSigned(entry.value)}</span>
      </li>)}</ol>}
  </div>;
}

function effectTooltip(effect: VocabularyEffect): string {
  return [
    `token: ${JSON.stringify(effect.token)}`,
    `exact representation: ${JSON.stringify(effect.exact_token)}`,
    `token id: ${effect.token_id ?? "unavailable"}`,
    `rank: ${effect.rank}`,
    `value: ${effect.value}`,
    `source: ${effect.source}`,
  ].join("\n");
}

function formatSigned(value: number): string {
  const magnitude = Math.abs(value);
  const formatted = magnitude > 0 && magnitude < 1e-4
    ? magnitude.toExponential(3)
    : magnitude.toFixed(4);
  return `${value >= 0 ? "+" : "−"}${formatted}`;
}
