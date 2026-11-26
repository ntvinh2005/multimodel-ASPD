import { NotesLocatorCell } from "./ResearchNotes";
import type { NoteTarget } from "./researchNotes";
import { MODEL_IDS, MODEL_NAMES, type MatrixLocator } from "./negationWorkbenchData";

const numeric = (value: number | null | undefined) => value == null ? "—" : value.toPrecision(4);

export default function ComponentBetaLocator({ gate, checkpoint, matrices, onAnnotate, error }: {
  gate: string; checkpoint: string; matrices?: MatrixLocator[]; onAnnotate: (target: NoteTarget) => void; error?: string;
}) {
  const [site, index] = gate.split(":");
  const feature = Number(index);
  return <section className="ig-beta">
    <header><div><span className="uw-eyebrow">IMPORTED NATURAL-VALIDATION β</span><h3>Model-specific component locator</h3></div>
      <p>All component cells for this site. White insets mark the frozen β representatives.</p></header>
    {matrices ? <div className="ig-beta-models">{MODEL_IDS.map((model) => <div key={model}>
      <strong>{MODEL_NAMES[model]}</strong><div className="ig-beta-grid">{matrices.filter((matrix) => matrix.model === model && matrix.site === site).map((row) => {
        const beta = row.beta[feature];
        return <NotesLocatorCell key={`${row.layer}/${row.matrix}`} beta={beta} representative={row.representatives.includes(feature)}
          title={`L${row.layer}/${row.matrix} · β ${numeric(beta)}`}
          target={{ checkpoint, gateId: gate, scope: "component", modelId: model, layer: row.layer, matrix: row.matrix }}
          onSelect={onAnnotate}>
          <small>L{row.layer}</small><b>{row.matrix}</b>
        </NotesLocatorCell>;
      })}</div>
    </div>)}</div> : <p className="ig-graph-status" role="status">{error ?? "Loading reference β…"}</p>}
  </section>;
}
