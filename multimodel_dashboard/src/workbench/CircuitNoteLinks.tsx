import circuitLinks from "./relatedCircuitLinks.json";
import { NOTE_MODELS, type NoteTarget } from "./researchNotes";

const roles: Record<string, string> = {
  P1: "Parent",
  P2: "Grandparent",
  C1: "Child",
  C2: "Grandchild",
};

export default function CircuitNoteLinks({ target, onSelect }: {
  target: NoteTarget;
  onSelect?: (target: NoteTarget) => void;
}) {
  if (target.checkpoint !== circuitLinks.checkpoint || target.scope !== "shared") return null;
  const relations = circuitLinks.relations.filter(({ originGate, relativeGate }) =>
    originGate === target.gateId || relativeGate === target.gateId);
  if (!relations.length) return null;

  return <section className="rn-circuit-links" aria-label="Measured circuit links">
    <h3>Measured circuit links</h3>
    {relations.map((link) => {
      const related = link.originGate === target.gateId;
      const destination = related ? link.relativeGate : link.originGate;
      return <details key={link.id}>
        <summary>
          <b>{link.hop} · {roles[link.hop]}</b> {link.relativeGate} of {link.referenceGate}
        </summary>
        <p>{link.commonHits}/{link.commonOpportunities} shared validation anchors with the original circuit · {
          link.participatingModels.map((model) => NOTE_MODELS[model] ?? model).join(", ")}</p>
        {onSelect && <button type="button" onClick={() => onSelect({ ...target, gateId: destination })}>
          {related ? "Open related note" : "Open original hypothesis"} · {destination}
        </button>}
        <ul>
          {link.examples.map(({ model, anchorId, components, weights }) => <li key={model}>
            <strong>{NOTE_MODELS[model] ?? model}</strong> · <code title={anchorId}>{anchorId.slice(0, 12)}</code>
            <code className="rn-circuit-path">{components.join(" → ")}</code>
            <small>Normalized w · {weights.map((w) => w.toPrecision(3)).join(" → ")}</small>
          </li>)}
        </ul>
      </details>;
    })}
  </section>;
}
