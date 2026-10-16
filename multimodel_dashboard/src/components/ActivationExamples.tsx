import type { CSSProperties } from "react";
import type { ActivationExample, TokenActivation } from "../types";
import { Badge, Metric } from "./Common";

export function ActivationExamples({
  examples,
  missingMessage,
}: {
  examples: ActivationExample[] | null;
  missingMessage: string;
}) {
  if (examples === null) return <p className="na">{missingMessage}</p>;
  if (examples.length === 0) {
    return <p className="na">No positive activation examples were retained for this component.</p>;
  }

  const ordered = [...examples].sort(
    (left, right) => (right.context_score ?? right.g_s ?? 0) - (left.context_score ?? left.g_s ?? 0),
  );
  const hasPerToken = ordered.some((example) => example.activation_detail_kind === "per_token");

  return <>
    <div className="activation-legend" aria-label="Activation highlight legend">
      <span>No measured activation</span>
      <i className="activation-gradient" />
      <span>Strongest measured activation in context</span>
    </div>
    {!hasPerToken && <p className="measurement-note">
      This artifact contains one measured center/firing position per context. Only that position is
      highlighted; surrounding token activations are unavailable and are not inferred.
    </p>}
    <div className="examples-list">{ordered.map((example, index) =>
      <TokenActivationContext example={example} key={`${example.source}-${index}`} />
    )}</div>
  </>;
}

export function TokenActivationContext({ example }: { example: ActivationExample }) {
  const measured = example.tokens
    .map((token) => token.activation)
    .filter((value): value is number => value !== null && Number.isFinite(value) && value > 0);
  const maximum = measured.length > 0 ? Math.max(...measured) : 0;
  const score = example.context_score ?? example.g_s ?? null;

  return <article className="example-card" title={example.source}>
    <div className="example-meta">
      <div>
        <Badge tone="target">Top context</Badge>
        <span className="measurement-kind">{example.activation_detail_kind === "per_token" ? "per-token measured" : "center-only measured"}</span>
      </div>
      <span>max g<sup>s</sup> <Metric value={score} kind="gs" source={`${example.source}.g_s`} /></span>
    </div>
    {example.tokens.length > 0 ?
      <div className="token-activation-context" aria-label="Tokenizer-aligned activation positions">
        {example.tokens.map((token) =>
          <TokenActivationSpan token={token} maximum={maximum} key={`${token.position}-${token.token_id}`} />
        )}
      </div> :
      <p className="na">Tokenizer-aligned positions are unavailable for this context.</p>}
    {example.text && <div className="decoded-context"><span>Decoded context</span><p>{example.text}</p></div>}
    <small>{example.activation_detail_label}</small>
    <details><summary>Measured token metadata</summary>
      <code>{JSON.stringify(example.token_ids ?? "N/A")}</code>
      <div>center_in_window: {example.center_in_window ?? "N/A"}</div>
    </details>
  </article>;
}

export function TokenActivationSpan({
  token,
  maximum,
}: {
  token: TokenActivation;
  maximum: number;
}) {
  const strength = token.activation !== null && token.activation > 0 && maximum > 0
    ? Math.min(1, token.activation / maximum)
    : 0;
  const opacity = strength === 0 ? 0 : 0.12 + Math.sqrt(strength) * 0.66;
  const label = token.text ?? token.exact_token ?? (token.token_id === null ? "?" : `#${token.token_id}`);
  const tooltip = [
    `token text: ${JSON.stringify(token.text)}`,
    `exact tokenizer token: ${JSON.stringify(token.exact_token)}`,
    `token id: ${token.token_id ?? "unavailable"}`,
    `window position: ${token.position}`,
    `sequence position: ${token.sequence_position ?? "unavailable"}`,
    `g^s activation: ${token.activation ?? "unavailable"}`,
  ].join("\n");
  const style = { "--activation-opacity": opacity } as CSSProperties;

  return <span
    className={`activation-token${strength === 1 ? " strongest" : ""}${token.is_center ? " center-token" : ""}`}
    style={style}
    title={tooltip}
  >{label}</span>;
}
