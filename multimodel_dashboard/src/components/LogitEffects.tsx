import type { VocabularyEffect, VocabularyEffects } from "../types";

export function vocabularyEffectKicker(data: VocabularyEffects): string {
  if (data.status === "missing") return "No vocabulary/logit artifact available";
  if (data.status === "error") return "Artifact could not be interpreted";
  if (data.kind === "direct_projection") return "Projection proxy — not a causal intervention";
  if (data.kind === "logit_difference") return "Measured intervention-based logit difference";
  return "Artifact-declared vocabulary values";
}

export function LogitEffects({ data, limit = 10 }: { data: VocabularyEffects; limit?: number }) {
  if (data.status === "missing") {
    return <div className="empty-state vocabulary-empty"><strong>Vocabulary data unavailable</strong><p>{data.description}</p></div>;
  }
  if (data.status === "error") {
    return <div className="error-box"><strong>Vocabulary artifact error</strong><p>{data.description}</p></div>;
  }
  if (data.status === "empty") {
    return <div className="empty-state vocabulary-empty"><strong>No vocabulary values retained</strong><p>{data.description}</p></div>;
  }

  const positive = [...data.positive].sort((a, b) => b.value - a.value).slice(0, limit);
  const negative = [...data.negative].sort((a, b) => a.value - b.value).slice(0, limit);
  return <>
    <p className="explain">{data.description}</p>
    <div className="logit-columns">
      <LogitColumn title="Negative logits" direction="negative" entries={negative} />
      <LogitColumn title="Positive logits" direction="positive" entries={positive} />
    </div>
    <div className="source-line">Source: {data.source}</div>
  </>;
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
