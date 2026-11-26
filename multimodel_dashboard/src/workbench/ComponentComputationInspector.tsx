import { useEffect, useMemo, useState } from "react";
import { componentIdentity, type Measurement } from "./circuitComposition";
import { MODEL_NAMES, readWorkbenchAsset, type EvalModel } from "./negationWorkbenchData";
import "./componentComputation.css";

type Term = { index: number; value: number };
type FactorRef = { path: string; sha256: string; dtype: string; layout: string };
type Static = {
  uNorm: number; vNorm: number; dIn: number; dOut: number; sigma: number; beta: number | null; rho: number | null;
  representative: boolean; readWeights: Term[]; writeWeights: Term[]; factors: FactorRef;
  nativeActivation?: string;
  heads?: { count: number; width: number; queryHeads: number; vector: "u" | "v"; norms: number[] };
};
type Local = {
  source: string; target: string; before: number; after: number; delta: number; method: string; nativeReplay: boolean;
  changedChannels?: Term[]; readingTerms?: Term[];
};
type Row = { coefficient: number; reading: number; support: boolean };
type NativeRow = Row & {
  anchor: string; pool: string; magnitude: number; normalized: number; readingTerms: Term[]; writingTerms: Term[];
  local?: { before: number; after: number; delta: number; changedChannels: Term[] } | { method: string; pivot: number; projectionOutputChanges: Term[];
    attentionRows: Array<{ head: number; query?: number; key: number; before: number; after: number; centeredScoreDelta: number | null }> };
};
type NativeExport = { schemaVersion: 1; checkpoint: string; component: string; dataset: string; split: string; rows: NativeRow[] };
type Edge = { source: string; target: string; E: number; w: number };
type Witness = { anchor: string; rows: Record<string, Row>; edges: Edge[]; local: Local[] };
type Artifact = {
  schemaVersion: 1; candidate: string; checkpoint: string; static: Record<string, Static>;
  anchors: Record<string, Witness>; provenance: { evidenceSha256: string; extensionsSha256: string };
};
type TokenMeasurement = { activation?: number; support?: boolean; magnitude?: number; scale?: number; beta?: number | null; representative?: boolean;
  dataset?: string; pool?: string };
const DIRECTORY = "component-computation/nvx025";
const format = (value: number | null | undefined) => value == null ? "Unavailable" : value.toPrecision(5);
const model = (id: string) => MODEL_NAMES[id as EvalModel] ?? id;

export function normalizedContribution(coefficient?: number | null, norm?: number | null, sigma?: number | null) {
  const magnitude = coefficient === 0 ? 0
    : coefficient == null || norm == null ? null : Math.abs(coefficient) * norm;
  return { magnitude, relative: magnitude != null && sigma != null && sigma > 0 ? magnitude / sigma : null };
}

function Terms({ values }: { values: Term[] }) {
  return <ul className="ci-terms">{values.map(({ index, value }) => <li key={index}><code>{index}</code> · {format(value)}</li>)}</ul>;
}

export default function ComponentComputationInspector({ id, checkpoint, anchor, measurement, token, candidateId }: {
  id: string; checkpoint: string; anchor: string; measurement?: Measurement; token?: TokenMeasurement; candidateId?: string;
}) {
  const [opened, setOpened] = useState(true);
  const [nativeOpen, setNativeOpen] = useState(false);
  const [replay, setReplay] = useState<NativeExport>();
  const [loaded, setLoaded] = useState<{ artifact?: Artifact; error?: string }>({});
  const [downloadError, setDownloadError] = useState("");
  const identity = componentIdentity(id);
  useEffect(() => {
    if (!opened || candidateId !== "NVX-025") return;
    const abort = new AbortController();
    void readWorkbenchAsset<Artifact>(DIRECTORY + "/index.json", abort.signal).then(artifact => {
      if (artifact.schemaVersion !== 1 || artifact.checkpoint !== checkpoint || artifact.candidate !== candidateId) {
        throw new Error("Computation artifact identity differs from the selected checkpoint or candidate.");
      }
      if (!abort.signal.aborted) setLoaded({ artifact });
    }).catch(error => { if (!abort.signal.aborted) setLoaded({ error: String(error.message ?? error) }); });
    return () => abort.abort();
  }, [opened, checkpoint, candidateId]);
  const artifact = loaded.artifact?.checkpoint === checkpoint && loaded.artifact?.candidate === candidateId ? loaded.artifact : undefined;
  const factors = artifact?.static[id];
  const archived = artifact?.anchors[anchor]?.anchor === anchor ? artifact.anchors[anchor] : undefined;
  const archivedRow = archived?.rows[id];
  const exact = measurement?.scope.checkpoint === checkpoint && measurement?.scope.anchor === anchor ? measurement : undefined;
  const controlledDataset = token?.dataset ?? exact?.scope.dataset;
  const controlledPool = token?.pool ?? exact?.scope.pool;
  useEffect(() => {
    if (!nativeOpen || !controlledPool) return;
    const controller = new AbortController();
    const path = `component-computation/native/${id.replace("/", "--")}.json`;
    void readWorkbenchAsset<NativeExport>(path, controller.signal).then(value => {
      if (!controller.signal.aborted && value.schemaVersion === 1 && value.checkpoint === checkpoint && value.component === id)
        setReplay(value);
    }).catch(() => {});
    return () => controller.abort();
  }, [nativeOpen, checkpoint, id, controlledDataset, controlledPool]);
  const nativeRow = replay?.checkpoint === checkpoint && replay.component === id
    && (replay.dataset === controlledDataset || `${replay.dataset}-${replay.split}` === controlledDataset)
    ? replay.rows.find(row => row.anchor === anchor && row.pool === controlledPool && row.support === token?.support) : undefined;
  const support = token?.support ?? archivedRow?.support;
  const coefficient = archivedRow?.coefficient ?? exact?.coefficients?.[id] ?? nativeRow?.coefficient ?? (support === false ? 0 : undefined);
  const reading = nativeRow?.reading ?? archivedRow?.reading ?? (support === true ? coefficient : undefined);
  const magnitude = token?.magnitude ?? nativeRow?.magnitude ?? normalizedContribution(coefficient, factors?.uNorm).magnitude;
  const sigma = token?.scale ?? factors?.sigma;
  const local = archived?.local.filter(row => row.source === id || row.target === id) ?? [];
  const edges = useMemo(() => {
    const merged = new Map<string, Edge>();
    for (const edge of exact?.edges ?? []) {
      if (edge.source === id || edge.target === id) {
        merged.set(edge.source + "|" + edge.target, { source: edge.source, target: edge.target, E: edge.signed, w: edge.weight });
      }
    }
    for (const edge of archived?.edges ?? []) if (edge.source === id || edge.target === id) merged.set(edge.source + "|" + edge.target, edge);
    return [...merged.values()].sort((a, b) => b.w - a.w);
  }, [exact, archived, id]);
  const peers = local.flatMap(row => archived?.local.filter(peer => peer.source.split(".")[0] !== identity.modelId
    && componentIdentity(peer.target).gateId === componentIdentity(row.target).gateId) ?? []);
  const witnesses = Object.values(artifact?.anchors ?? {}).flatMap(context => {
    const row = context.rows[id];
    return row ? [{ anchor: context.anchor, e: row.coefficient }] : [];
  }).sort((a, b) => Math.abs(b.e) - Math.abs(a.e));
  async function download(ref: FactorRef) {
    try {
      if (!/^factors\/\d+\.f32$/.test(ref.path) || !globalThis.crypto?.subtle) throw new Error("Factor download requires hash verification.");
      const response = await fetch(import.meta.env.BASE_URL + DIRECTORY + "/" + ref.path);
      if (!response.ok) throw new Error("Factor artifact unavailable.");
      const bytes = await response.arrayBuffer();
      const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(x => x.toString(16).padStart(2, "0")).join("");
      if (digest !== ref.sha256 || bytes.byteLength !== 4 * ((factors?.dIn ?? 0) + (factors?.dOut ?? 0))) throw new Error("Factor artifact integrity check failed.");
      const url = URL.createObjectURL(new Blob([bytes]));
      const link = document.createElement("a");
      link.href = url;
      link.download = id.replaceAll("/", "_") + ".f32";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setDownloadError("");
    } catch (error) { setDownloadError(error instanceof Error ? error.message : String(error)); }
  }
  const nativeRole: Record<string, string> = {
    Q: "Writes a query representation before native head reshaping and position transforms.",
    K: "Writes a key representation. Compatibility and actual retrieval require native query and attention data.",
    V: "Writes a value representation. Consumption requires native attention weights and output projection.",
    O: "Reads the attention output and contributes to the residual-stream projection.",
    gate: "Writes to the channel-wise nonlinear MLP gate branch.",
    up: "Writes to the MLP Up branch, multiplied elementwise by the fixed activated Gate branch.",
    down: "Reads the gated MLP hidden channels and contributes to the residual-stream projection.",
  };
  return <details className="ci-inspector" open onToggle={event => setOpened(event.currentTarget.open)}>
    <summary>Component computation · <code>{id}</code></summary>
    <dl><dt>Model · gate · layer · matrix</dt><dd>{model(identity.modelId)} · {identity.gateId} · L{identity.layer} · {identity.matrix}</dd>
      <dt>Anchor</dt><dd><code>{anchor}</code></dd>
      <dt>Shared gˢ · binary g</dt><dd>{format(token?.activation)} · {support == null ? "Unavailable" : Number(support)}</dd>
      <dt>Reference β · ρ</dt><dd>{format(factors?.beta ?? token?.beta)} · {format(factors?.rho)}
        {(factors?.representative ?? token?.representative) ? " · representative" : ""}</dd></dl>
    {loaded.error && candidateId === "NVX-025" && <p role="status">{loaded.error}</p>}
    <details><summary>Input and reading</summary>
      <dl><dt>Ungated reading vᵀx</dt><dd>{format(reading)}</dd><dt>Input dimension · ‖v‖₂</dt><dd>{factors?.dIn ?? "Unavailable"} · {format(factors?.vNorm)}</dd></dl>
      {factors && <><p>Largest checkpoint reader weights by absolute magnitude</p><Terms values={factors.readWeights} /></>}
      {nativeRow && <><p>Largest native reading terms vⱼxⱼ</p><Terms values={nativeRow.readingTerms} /></>}
    </details>
    <details><summary>Contribution and writing</summary>
      <dl><dt>Signed e · ‖eu‖₂ · ‖eu‖₂ / σ</dt><dd>{format(coefficient)} · {format(magnitude)} · {format(magnitude == null || !sigma ? nativeRow?.normalized : magnitude / sigma)}</dd>
        <dt>Output dimension · ‖u‖₂ · σ</dt><dd>{factors?.dOut ?? "Unavailable"} · {format(factors?.uNorm)} · {format(sigma)}</dd></dl>
      <p>Signed coefficients use the checkpoint’s factor basis. Contribution vectors are invariant to reciprocal rescaling of the factors.</p>
      {factors && <><p>{coefficient == null ? "Largest writer weights in u" : "Largest signed contribution coordinates"}</p>
        <Terms values={coefficient == null ? factors.writeWeights : factors.writeWeights.map(term => ({ ...term, value: coefficient * term.value }))} />
        <button type="button" onClick={() => void download(factors.factors)}>Download verified u and v</button>{downloadError && <p role="alert">{downloadError}</p>}</>}
    </details>
    <details open={local.length ? true : undefined} onToggle={event => setNativeOpen(event.currentTarget.open)}><summary>Native computation</summary>
      <p>{nativeRole[identity.matrix]}</p>
      {factors?.nativeActivation && ["gate", "up", "down"].includes(identity.matrix) && <p>Native gate activation · {factors.nativeActivation}</p>}
      {factors?.heads && <details><summary>Native head layout</summary>
        <p>{factors.heads.count} heads of width {factors.heads.width} · {factors.heads.queryHeads} query heads</p>
        <p>Per-head norm of {factors.heads.vector} in logical projection order</p>
        <Terms values={factors.heads.norms.map((value, index) => ({ index, value }))} />
      </details>}
      {local.length ? local.map(row => <p key={row.source}><code>{row.source} → {row.target}</code><br />
        {row.nativeReplay ? "Exact local replay" : "Affine prediction from the archived Taylor derivative"} · Down e {format(row.before)} → {format(row.after)} · Δe {format(row.delta)}</p>)
        : !nativeRow && <p>Native activations for this component and anchor are unavailable in the current export.</p>}
      {nativeRow?.local && ("attentionRows" in nativeRow.local ? <>
        <p>Native attention replay · source position {nativeRow.local.pivot}. Score differences are centered by the softmax normalizer.</p>
        <ul>{nativeRow.local.attentionRows.slice(0, 12).map((row, i) => <li key={i}>
          Head {row.head} · token {row.query ?? (nativeRow.local && "pivot" in nativeRow.local ? nativeRow.local.pivot : "—")} → {row.key}
          {" · probability "}{format(row.before)} → {format(row.after)} · centered Δscore {format(row.centeredScoreDelta)}
        </li>)}</ul>
        <p>Change in output projection by query position</p><Terms values={nativeRow.local.projectionOutputChanges} />
      </> : <><p>Exact local native MLP replay · Down e {format(nativeRow.local.before)} → {format(nativeRow.local.after)} · Δe {format(nativeRow.local.delta)}</p>
        {nativeRow.local.changedChannels && <Terms values={nativeRow.local.changedChannels} />}</>)}
      {peers.length > 0 && <details><summary>Matched cross-model effects</summary>
        <div className="cc-scroll"><table className="cc-table"><thead><tr><th>Exact component pair</th><th>Down e</th>
          <th>Removal Δe</th><th>|Δe/e|</th></tr></thead><tbody>
          {[...local, ...peers].map(row => <tr key={row.source}><th><code>{row.source} → {row.target}</code></th>
            <td>{format(row.before)}</td><td>{format(row.delta)}</td>
            <td>{format(row.before === 0 ? null : Math.abs(row.delta / row.before))}</td></tr>)}
        </tbody></table></div>
        <p>The ratio is within each component’s basis. Component correspondence comes from the archived candidate.</p>
      </details>}
    </details>
    {witnesses.length > 0 && <details><summary>Archived validation anchors · ranked by |e|</summary>
      <p>These anchors have component coefficients. Text and continuous gate scores are unavailable for this archive.</p>
      <ul>{witnesses.map(row => <li key={row.anchor}><code title={row.anchor}>{row.anchor.slice(0, 14)}</code>
        {" · e "}{format(row.e)}{" · ‖eu‖₂ "}{format(factors ? Math.abs(row.e) * factors.uNorm : null)}</li>)}</ul>
    </details>}
    <details><summary>Upstream and downstream effects</summary>
      {!edges.length && <p>No retained dependency records are available for this component at this anchor.</p>}
      {edges.slice(0, 12).map(edge => {
        const baseline = archived?.rows[edge.target]?.coefficient ?? exact?.coefficients?.[edge.target];
        return <p key={edge.source + "|" + edge.target}><b>{edge.target === id ? "Upstream" : "Downstream"}</b>{" "}
          <code>{edge.source} → {edge.target}</code><br />Taylor E {format(edge.E)} · normalized w {format(edge.w)}
          {" · target e "}{format(baseline)}{" · first-order Δe on removal "}{format(-edge.E)}</p>;
      })}
      <p>Each score is a frozen-gate effect at this anchor. Paths are never multiplied or summed here.</p>
    </details>
    {artifact && <details><summary>Provenance</summary><p>{artifact.candidate} · <code>{artifact.provenance.evidenceSha256}</code></p></details>}
  </details>;
}
