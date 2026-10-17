import { useState, type CSSProperties } from "react";
import type { ActivationExample, TokenActivation } from "../types";
import { Badge, Metric } from "./Common";

type ContextView = "snippet" | "window";

export function ActivationExamples({
  examples,
  missingMessage,
}: {
  examples: ActivationExample[] | null;
  missingMessage: string;
}) {
  const [view, setView] = useState<ContextView>("snippet");
  const [showBounds, setShowBounds] = useState(false);

  if (examples === null) return <p className="na">{missingMessage}</p>;
  if (examples.length === 0) {
    return <p className="na">No positive activation examples were retained for this component.</p>;
  }

  const ordered = [...examples].sort(
    (left, right) => (right.context_score ?? right.g_s ?? 0) - (left.context_score ?? left.g_s ?? 0),
  );
  const perTokenCount = ordered.filter(
    (example) => example.activation_detail_kind === "per_token",
  ).length;
  const sequenceStartCount = ordered.filter((example) =>
    example.activation_detail_kind === "per_token"
    && (example.position_in_sequence
      ?? example.tokens.find((token) => token.is_center)?.sequence_position) === 0
  ).length;
  const singlePositiveCount = ordered.filter((example) =>
    example.activation_detail_kind === "per_token"
    && example.tokens.filter((token) => token.activation !== null && token.activation > 0).length === 1
  ).length;

  return <>
    <div className="activation-controls" aria-label="Activation context controls">
      <div className="segmented-control">
        <button className={view === "snippet" ? "active" : ""} onClick={() => setView("snippet")}>Snippet</button>
        <button className={view === "window" ? "active" : ""} onClick={() => setView("window")}>Saved window</button>
      </div>
      <button onClick={() => setShowBounds((current) => !current)}>
        {showBounds ? "Hide bounds" : "Show bounds"}
      </button>
      <span className="measurement-coverage">
        {perTokenCount}/{ordered.length} contexts contain a measured token window
      </span>
    </div>
    <div className="activation-legend" aria-label="Activation highlight legend">
      <span className="legend-key"><i className="token-state measured-zero" />Measured zero</span>
      <span className="legend-key"><i className="token-state unavailable" />Unavailable / not measured</span>
      <i className="activation-gradient" />
      <span>Positive g<sup>s</sup><sub>t,c</sub> intensity on tokens</span>
    </div>
    {perTokenCount > 0 && <div className="activation-pattern-summary">
      <strong>Measured positional pattern</strong>
      <span>{sequenceStartCount}/{perTokenCount} measured contexts fire at sequence position 0</span>
      <span>{singlePositiveCount}/{perTokenCount} contain exactly one positive token in the saved window</span>
      {sequenceStartCount === perTokenCount && singlePositiveCount === perTokenCount &&
        <p>Every measured top context fires only at the first sequence position. The single highlighted token is therefore the complete measured sparse pattern for this component.</p>}
    </div>}
    {perTokenCount < ordered.length && <p className="measurement-note">
      {perTokenCount === 0
        ? "This older artifact stores only the firing position for each context. Neighboring token activations were not saved, so the dashboard highlights one token and does not infer the rest."
        : `${ordered.length - perTokenCount} context(s) store only the firing position; their neighboring token activations remain uncolored.`}
    </p>}
    <div className="examples-list">{ordered.map((example, index) =>
      <TokenActivationContext
        example={example}
        key={`${example.source}-${index}`}
        showBounds={showBounds}
        view={view}
      />
    )}</div>
  </>;
}

export function TokenActivationContext({
  example,
  showBounds = false,
  view = "window",
}: {
  example: ActivationExample;
  showBounds?: boolean;
  view?: ContextView;
}) {
  const measuredPositive = example.tokens
    .map((token) => token.activation)
    .filter((value): value is number => value !== null && Number.isFinite(value) && value > 0);
  const measuredPositionCount = example.tokens.filter(
    (token) => token.activation !== null && Number.isFinite(token.activation),
  ).length;
  const maximum = measuredPositive.length > 0 ? Math.max(...measuredPositive) : 0;
  const score = example.context_score ?? example.g_s ?? null;
  const { end, start, tokens } = selectVisibleTokens(example, view);
  const center = example.center_in_window ?? example.tokens.findIndex((token) => token.is_center);
  const sequencePosition = example.position_in_sequence
    ?? example.tokens.find((token) => token.is_center)?.sequence_position
    ?? null;

  return <article className="example-card" title={example.source}>
    <div className="example-meta">
      <div>
        <Badge tone="target">Top context</Badge>
        <span className="measurement-kind">
          {example.activation_detail_kind === "per_token" ? "per-token measured" : "center-only measured"}
        </span>
      </div>
      <span>max g<sup>s</sup> <Metric value={score} kind="gs" source={`${example.source}.g_s`} /></span>
    </div>
    <div className="example-measurement-summary">
      <span>{measuredPositionCount}/{example.tokens.length} token positions measured</span>
      <span>{measuredPositive.length} positive · {measuredPositionCount - measuredPositive.length} measured zero</span>
      {center >= 0 && <span>firing position: window {center}{sequencePosition === null ? "" : ` · sequence ${sequencePosition}`}</span>}
    </div>
    {example.tokens.length > 0 ?
      <div className="token-activation-context" aria-label="Tokenizer-aligned activation positions">
        {showBounds && <ContextBoundary
          hiddenCount={start}
          isSavedEdge={start === 0}
          side="left"
          centerAtEdge={center === 0}
        />}
        {tokens.map((token) =>
          <TokenActivationSpan token={token} maximum={maximum} key={`${token.position}-${token.token_id}`} />
        )}
        {showBounds && <ContextBoundary
          hiddenCount={example.tokens.length - end}
          isSavedEdge={end === example.tokens.length}
          side="right"
          centerAtEdge={center === example.tokens.length - 1}
        />}
      </div> :
      <p className="na">Tokenizer-aligned positions are unavailable for this context.</p>}
    {example.text && <details className="decoded-context">
      <summary>Decoded context</summary><p>{example.text}</p>
    </details>}
    <small>{example.activation_detail_label}</small>
    <details><summary>Measured token metadata</summary>
      <code>{JSON.stringify(example.token_ids ?? "N/A")}</code>
      <div>center_in_window: {example.center_in_window ?? "N/A"}</div>
      <div>position_in_sequence: {sequencePosition ?? "N/A"}</div>
    </details>
  </article>;
}

function selectVisibleTokens(example: ActivationExample, view: ContextView) {
  const count = example.tokens.length;
  if (view === "window" || count === 0) return { end: count, start: 0, tokens: example.tokens };

  const activePositions = example.tokens
    .filter((token) => token.activation !== null && token.activation > 0)
    .map((token) => token.position);
  const fallbackCenter = example.center_in_window
    ?? example.tokens.findIndex((token) => token.is_center);
  const anchors = activePositions.length > 0 ? activePositions : [Math.max(0, fallbackCenter)];
  const start = Math.max(0, Math.min(...anchors) - 8);
  const end = Math.min(count, Math.max(...anchors) + 9);
  return { end, start, tokens: example.tokens.slice(start, end) };
}

function ContextBoundary({
  centerAtEdge,
  hiddenCount,
  isSavedEdge,
  side,
}: {
  centerAtEdge: boolean;
  hiddenCount: number;
  isSavedEdge: boolean;
  side: "left" | "right";
}) {
  const label = hiddenCount > 0
    ? `${hiddenCount} saved token${hiddenCount === 1 ? "" : "s"} hidden`
    : centerAtEdge
      ? `firing token is at the ${side} edge of the saved window`
      : `${side} edge of saved window`;
  return <span className={`context-boundary ${isSavedEdge ? "saved-edge" : "snippet-edge"}`}>
    {side === "left" ? `⟨ ${label}` : `${label} ⟩`}
  </span>;
}

function activationOpacity(activation: number | null, maximum: number): number {
  if (activation === null || activation <= 0 || maximum <= 0) return 0;
  const strength = Math.min(1, activation / maximum);
  return 0.12 + Math.sqrt(strength) * 0.66;
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
  const opacity = activationOpacity(token.activation, maximum);
  const measurementClass = token.activation === null
    ? " unavailable-activation"
    : token.activation === 0
      ? " measured-zero-activation"
      : " positive-activation";
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
    className={`activation-token${measurementClass}${strength === 1 ? " strongest" : ""}${token.is_center ? " center-token" : ""}`}
    style={style}
    title={tooltip}
  >{label}</span>;
}
