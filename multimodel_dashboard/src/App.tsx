import { useEffect, useState } from "react";
import { api } from "./api";
import CandidateFinder from "./components/CandidateFinder";
import Comparison from "./components/Comparison";
import DataHealth from "./components/DataHealth";
import FeatureExplorer from "./components/FeatureExplorer";
import Overview from "./components/Overview";
import type { HealthPayload, OverviewPayload } from "./types";

type Tab = "overview" | "feature" | "candidates" | "compare" | "health";

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [featureId, setFeatureId] = useState(0);
  const [candidateCategory, setCandidateCategory] = useState("shared_activation_concentrated_mechanism");
  const [error, setError] = useState("");
  const [bookmarks, setBookmarks] = useState<number[]>(() => JSON.parse(localStorage.getItem("aspd-bookmarks") || "[]") as number[]);
  const [compareIds, setCompareIds] = useState<number[]>(() => JSON.parse(localStorage.getItem("aspd-compare") || "[]") as number[]);

  useEffect(() => {
    Promise.all([api.overview(), api.health()]).then(([o, h]) => { setOverview(o); setHealth(h); }).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(() => localStorage.setItem("aspd-bookmarks", JSON.stringify(bookmarks)), [bookmarks]);
  useEffect(() => localStorage.setItem("aspd-compare", JSON.stringify(compareIds)), [compareIds]);

  function openFeature(id: number) { setFeatureId(id); setTab("feature"); window.scrollTo(0, 0); }
  function openCategory(category: string) { setCandidateCategory(category); setTab("candidates"); window.scrollTo(0, 0); }
  function toggleBookmark(id: number) { setBookmarks((old) => old.includes(id) ? old.filter((x) => x !== id) : [...old, id]); }
  function toggleCompare(id: number) {
    setCompareIds((old) => old.includes(id) ? old.filter((x) => x !== id) : old.length < 5 ? [...old, id] : old);
  }

  if (error) return <main className="boot-state"><div className="error-box"><h1>Dashboard could not load real artifacts</h1>{error}</div></main>;
  if (!overview || !health) return <main className="boot-state"><div className="loading">Loading and validating P1–P5 artifacts…</div></main>;
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span>ASPD</span><strong>P1–P5 Research</strong><small>Model → Parameter Diffing</small></div>
      <nav>{(["overview", "feature", "candidates", "compare", "health"] as Tab[]).map((name) => <button key={name} className={tab === name ? "active" : ""} onClick={() => setTab(name)}><span>{icons[name]}</span>{name === "feature" ? "Feature Explorer" : name === "candidates" ? "Candidate Finder" : name === "compare" ? `Comparison (${compareIds.length})` : name === "health" ? "Data Health" : "Overview"}</button>)}</nav>
      <div className="sidebar-footer"><div><span>Run</span><strong>{overview.metadata.run_name || "N/A"}</strong></div><div><span>Dictionary</span><strong>{overview.metadata.n_features?.toLocaleString() || "N/A"} components</strong></div><div><span>Data health</span><strong className={`status-${health.overall_status.toLowerCase()}`}>{health.overall_status}</strong></div>{bookmarks.length > 0 && <div><span>Bookmarks</span><p className="bookmark-list">{bookmarks.map((id) => <button key={id} onClick={() => openFeature(id)}>{id}</button>)}</p></div>}</div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><strong>{overview.metadata.model_names.join(" ↔ ")}</strong><span>{overview.metadata.checkpoint || "N/A — checkpoint unavailable"}</span></div><div className="truth-label">Measured data only · hover values for source paths</div></header>
      <div className="content-wrap">
        {tab === "overview" && <Overview data={overview} onFeature={openFeature} onCategory={openCategory} />}
        {tab === "feature" && <FeatureExplorer featureId={featureId} metadata={overview.metadata} points={overview.points} taxonomy={overview.taxonomy} bookmarked={bookmarks.includes(featureId)} onNavigate={openFeature} onBookmark={toggleBookmark} onCompare={toggleCompare} />}
        {tab === "candidates" && <CandidateFinder points={overview.points} metadata={overview.metadata} taxonomy={overview.taxonomy} presetCategory={candidateCategory} selected={compareIds} onFeature={openFeature} onToggleSelected={toggleCompare} />}
        {tab === "compare" && <Comparison ids={compareIds} metadata={overview.metadata} onRemove={toggleCompare} onFeature={openFeature} />}
        {tab === "health" && <DataHealth data={health} />}
      </div>
    </main>
  </div>;
}

const icons: Record<Tab, string> = { overview: "◫", feature: "⌕", candidates: "≡", compare: "⇄", health: "✓" };
