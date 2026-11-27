import { CircuitLibraryButton } from "./CircuitLibrary";
import { ResearchNotesLibrary } from "./ResearchNotes";

export type CircuitView = "natural" | "induction" | "negation";

export default function WorkbenchHeader({ title, view, onChangeView, checkpoint, onOpenCandidate }: {
  title: string;
  view: CircuitView;
  onChangeView: (view: CircuitView) => void;
  checkpoint?: string;
  onOpenCandidate: (id: string) => void;
}) {
  return <header className="uw-header">
    <div><span className="uw-kicker">MULTI-MODEL ASPD</span><h1>{title}</h1></div>
    <div className="uw-header-actions">
      <label className="uw-circuit-select">
        <span>CIRCUIT / EVALUATION VIEW</span>
        <select aria-label="Circuit or evaluation view" value={view} onChange={event => onChangeView(event.target.value as CircuitView)}>
          <option value="natural">Natural Validation</option>
          <option value="induction">Induction</option>
          <option value="negation">Negation</option>
        </select>
      </label>
      <ResearchNotesLibrary buttonLabel="Notes library" onOpenCandidate={onOpenCandidate} />
      <CircuitLibraryButton checkpoint={checkpoint} />
    </div>
  </header>;
}
