import { useEffect, useState, type ReactNode } from "react";
import { api } from "./api";
import CandidateFinder from "./components/CandidateFinder";
import Comparison from "./components/Comparison";
import CrosscoderFeatureExplorer from "./components/CrosscoderFeatureExplorer";
import CrosscoderOverview from "./components/CrosscoderOverview";
import DataHealth from "./components/DataHealth";
import FeatureExplorer from "./components/FeatureExplorer";
import MethodComparison from "./components/MethodComparison";
import Overview from "./components/Overview";
import type {
  CrosscoderOverviewPayload,
  HealthPayload,
  MethodComparisonPayload,
  OverviewPayload,
} from "./types";

type View =
  | "aspd-overview"
  | "aspd-feature"
  | "aspd-candidates"
  | "aspd-compare"
  | "aspd-health"
  | "crosscoder-overview"
  | "crosscoder-feature"
  | "research-comparison";

const unavailableCrosscoder = (message: string): CrosscoderOverviewPayload => ({
  available: false,
  diagnostics: [message],
});
const unavailableComparison = (message: string): MethodComparisonPayload => ({
  available: false,
  diagnostics: [message],
  phenomena: [],
});

export default function App() {
  const [view, setView] = useState<View>("aspd-overview");
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [crosscoder, setCrosscoder] = useState<CrosscoderOverviewPayload | null>(null);
  const [researchComparison, setResearchComparison] = useState<MethodComparisonPayload | null>(null);
  const [featureId, setFeatureId] = useState(0);
  const [crosscoderFeatureId, setCrosscoderFeatureId] = useState(0);
  const [candidateCategory, setCandidateCategory] = useState("shared_activation_concentrated_mechanism");
  const [error, setError] = useState("");
  const [bookmarks, setBookmarks] = useState<number[]>(() => JSON.parse(localStorage.getItem("aspd-bookmarks") || "[]") as number[]);
  const [compareIds, setCompareIds] = useState<number[]>(() => JSON.parse(localStorage.getItem("aspd-compare") || "[]") as number[]);

  useEffect(() => {
    Promise.all([api.overview(), api.health()])
      .then(([aspdOverview, aspdHealth]) => { setOverview(aspdOverview); setHealth(aspdHealth); })
      .catch((reason: Error) => setError(reason.message));
    api.crosscoderOverview()
      .then((payload) => {
        setCrosscoder(payload);
        const firstCandidate = Object.values(payload.candidate_groups ?? {}).flat()[0];
        if (firstCandidate !== undefined) setCrosscoderFeatureId(firstCandidate);
      })
      .catch((reason: Error) => setCrosscoder(unavailableCrosscoder(reason.message)));
    api.researchComparison()
      .then(setResearchComparison)
      .catch((reason: Error) => setResearchComparison(unavailableComparison(reason.message)));
  }, []);
  useEffect(() => localStorage.setItem("aspd-bookmarks", JSON.stringify(bookmarks)), [bookmarks]);
  useEffect(() => localStorage.setItem("aspd-compare", JSON.stringify(compareIds)), [compareIds]);

  function navigate(next: View) { setView(next); window.scrollTo(0, 0); }
  function openAspdFeature(id: number) { setFeatureId(id); navigate("aspd-feature"); }
  function openCrosscoderFeature(id: number) { setCrosscoderFeatureId(id); navigate("crosscoder-feature"); }
  function openCategory(category: string) { setCandidateCategory(category); navigate("aspd-candidates"); }
  function toggleBookmark(id: number) { setBookmarks((old) => old.includes(id) ? old.filter((x) => x !== id) : [...old, id]); }
  function toggleCompare(id: number) {
    setCompareIds((old) => old.includes(id) ? old.filter((x) => x !== id) : old.length < 5 ? [...old, id] : old);
  }

  if (error) return <main className="boot-state"><div className="error-box"><h1>Dashboard could not load real ASPD artifacts</h1>{error}</div></main>;
  if (!overview || !health) return <main className="boot-state"><div className="loading">Loading and validating P1–P5 artifacts…</div></main>;
  const method = view.startsWith("crosscoder") ? "Crosscoder" : view === "research-comparison" ? "Cross-method Comparison" : "ASPD";
  const title = method === "Crosscoder"
    ? crosscoder?.metadata?.model_ids.filter(Boolean).join(" ↔ ") || "Crosscoder artifacts"
    : method === "Cross-method Comparison"
      ? "Representation evidence ↔ parameter-mechanism evidence"
      : overview.metadata.model_names.join(" ↔ ");
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span>MODEL DIFFING</span><strong>Research Inspector</strong><small>Measurement · interpretation · hypothesis</small></div>
      <nav>
        <NavGroup label="ASPD">
          <NavButton view="aspd-overview" current={view} label="Overview" icon="◫" onClick={navigate} />
          <NavButton view="aspd-feature" current={view} label="Feature Explorer" icon="⌕" onClick={navigate} />
          <NavButton view="aspd-candidates" current={view} label="Candidate Finder" icon="≡" onClick={navigate} />
          <NavButton view="aspd-compare" current={view} label={`Comparison (${compareIds.length})`} icon="⇄" onClick={navigate} />
          <NavButton view="aspd-health" current={view} label="Data Health" icon="✓" onClick={navigate} />
        </NavGroup>
        <NavGroup label="Crosscoder">
          <NavButton view="crosscoder-overview" current={view} label="Overview" icon="◒" onClick={navigate} />
          <NavButton view="crosscoder-feature" current={view} label="Feature Explorer" icon="⌕" onClick={navigate} />
        </NavGroup>
        <NavGroup label="Research Comparison">
          <NavButton view="research-comparison" current={view} label="Cross-method Comparison" icon="⋈" onClick={navigate} />
        </NavGroup>
      </nav>
      <div className="sidebar-footer">
        <div><span>Current method</span><strong>{method}</strong></div>
        <div><span>ASPD run</span><strong>{overview.metadata.run_name || "N/A"}</strong></div>
        <div><span>Crosscoder</span><strong>{crosscoder === null ? "Loading" : crosscoder.available ? "Available" : "Unavailable"}</strong></div>
        <div><span>ASPD data health</span><strong className={`status-${health.overall_status.toLowerCase()}`}>{health.overall_status}</strong></div>
        {bookmarks.length > 0 && <div><span>ASPD bookmarks</span><p className="bookmark-list">{bookmarks.map((id) => <button key={id} onClick={() => openAspdFeature(id)}>{id}</button>)}</p></div>}
      </div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><strong>{title}</strong><span>{method}</span></div><div className="truth-label">Measured · derived · post-ranking interpretation · hypothesis · unavailable</div></header>
      <div className="content-wrap">
        {view === "aspd-overview" && <Overview data={overview} onFeature={openAspdFeature} onCategory={openCategory} />}
        {view === "aspd-feature" && <FeatureExplorer featureId={featureId} metadata={overview.metadata} points={overview.points} taxonomy={overview.taxonomy} bookmarked={bookmarks.includes(featureId)} onNavigate={openAspdFeature} onBookmark={toggleBookmark} onCompare={toggleCompare} onCrosscoderFeature={openCrosscoderFeature} />}
        {view === "aspd-candidates" && <CandidateFinder points={overview.points} metadata={overview.metadata} taxonomy={overview.taxonomy} presetCategory={candidateCategory} selected={compareIds} onFeature={openAspdFeature} onToggleSelected={toggleCompare} />}
        {view === "aspd-compare" && <Comparison ids={compareIds} metadata={overview.metadata} onRemove={toggleCompare} onFeature={openAspdFeature} />}
        {view === "aspd-health" && <DataHealth data={health} />}
        {view === "crosscoder-overview" && (crosscoder ? <CrosscoderOverview data={crosscoder} onFeature={openCrosscoderFeature} /> : <div className="loading">Loading Crosscoder artifacts…</div>)}
        {view === "crosscoder-feature" && (crosscoder?.available && crosscoder.metadata ? <CrosscoderFeatureExplorer featureId={crosscoderFeatureId} metadata={crosscoder.metadata} onNavigate={openCrosscoderFeature} onAspdFeature={openAspdFeature} /> : <div className="empty-state">Crosscoder feature data unavailable. Open Crosscoder Overview for diagnostics.</div>)}
        {view === "research-comparison" && (researchComparison ? <MethodComparison data={researchComparison} onCrosscoderFeature={openCrosscoderFeature} onAspdFeature={openAspdFeature} /> : <div className="loading">Loading structured research comparison…</div>)}
      </div>
    </main>
  </div>;
}

function NavGroup({ label, children }: { label: string; children: ReactNode }) {
  return <div className="nav-group"><span className="nav-group-label">{label}</span>{children}</div>;
}

function NavButton({ view, current, label, icon, onClick }: { view: View; current: View; label: string; icon: string; onClick: (view: View) => void }) {
  return <button className={current === view ? "active" : ""} onClick={() => onClick(view)}><span>{icon}</span>{label}</button>;
}
