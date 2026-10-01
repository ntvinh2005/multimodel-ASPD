"""Read-only access to harvested component data: examples, labels, densities, kappa."""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aspd.analysis.pairs.spaces import parse_role

SortKey = Literal["effective", "activation", "ci"]

PAIR_SCORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_scores (
    metric   TEXT NOT NULL,
    a_module TEXT NOT NULL, a_side TEXT NOT NULL, a_idx INTEGER NOT NULL,
    b_module TEXT NOT NULL, b_side TEXT NOT NULL, b_idx INTEGER NOT NULL,
    head     INTEGER NOT NULL DEFAULT -1,
    score    REAL NOT NULL,
    PRIMARY KEY (metric, a_module, a_side, a_idx, b_module, b_side, b_idx, head)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS pair_metric_coverage (
    metric TEXT NOT NULL,
    a_module TEXT NOT NULL, a_side TEXT NOT NULL,
    b_module TEXT NOT NULL, b_side TEXT NOT NULL,
    n_rows INTEGER NOT NULL, top_k INTEGER NOT NULL, note TEXT,
    PRIMARY KEY (metric, a_module, a_side, b_module, b_side)
);
"""


def canonical_key(module: str, idx: int) -> str:
    role, layer = parse_role(module)
    return f"h.{layer}.{role}:{idx}"


def _connect(path: Path | None) -> sqlite3.Connection | None:
    if path is None or not Path(path).exists():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


@dataclass
class ComponentRecord:
    key: str
    firing_density: float | None
    mean_activations: dict[str, float]
    examples: list[dict[str, object]]
    input_pmi: list[list[object]]
    output_pmi: list[list[object]]
    pmi_available: bool


class HarvestStore:
    """The harvest's `components` table, one component at a time."""

    def __init__(self, harvest_db: Path | None, decode: Callable[[list[int]], list[str]]):
        self.path = Path(harvest_db) if harvest_db else None
        self.decode = decode
        self.available = self.path is not None and self.path.exists()
        self._modules: set[str] | None = None
        self._density: dict[str, object] = {}

    def densities(self, module: str, n: int):
        """Firing density per component of `module` as a `[n]` float tensor, or None."""
        import torch

        if module in self._density:
            return self._density[module]
        con = _connect(self.path)
        if con is None:
            self._density[module] = None
            return None
        with con:
            rows = con.execute(
                "SELECT component_key, firing_density FROM components WHERE component_key LIKE ?",
                (f"{module}:%",),
            ).fetchall()
        if not rows:
            self._density[module] = None
            return None
        out = torch.zeros(n)
        for key, d in rows:
            i = int(str(key).rsplit(":", 1)[1])
            if 0 <= i < n:
                out[i] = float(d or 0.0)
        self._density[module] = out
        return out

    def _row(self, module: str, idx: int) -> sqlite3.Row | None:
        con = _connect(self.path)
        if con is None:
            return None
        with con:
            for key in (f"{module}:{idx}", canonical_key(module, idx)):
                row = con.execute(
                    "SELECT component_key, firing_density, mean_activations, activation_examples,"
                    " input_token_pmi, output_token_pmi FROM components WHERE component_key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    return row
        return None

    def _pmi(self, blob: object) -> tuple[list[list[object]], bool]:
        if not blob:
            return [], False
        data = json.loads(blob)
        top = data.get("top") or []
        if not top:
            return [], False
        return [[self.decode([int(t)])[0], float(v)] for t, v in top], True

    def component(self, module: str, idx: int, *, sort: SortKey, window: int) -> ComponentRecord | None:
        row = self._row(module, idx)
        if row is None:
            return None
        in_pmi, in_ok = self._pmi(row["input_token_pmi"])
        out_pmi, out_ok = self._pmi(row["output_token_pmi"])
        return ComponentRecord(
            key=row["component_key"],
            firing_density=row["firing_density"],
            mean_activations=json.loads(row["mean_activations"] or "{}"),
            examples=self._examples(row["activation_examples"], sort=sort, window=window),
            input_pmi=in_pmi,
            output_pmi=out_pmi,
            pmi_available=in_ok or out_ok,
        )

    def _examples(self, blob: object, *, sort: SortKey, window: int) -> list[dict[str, object]]:
        """Decode, score and sort the stored windows."""
        if not blob:
            return []
        out = []
        for ex in json.loads(blob):
            toks = self.decode([int(t) for t in ex["token_ids"]])
            acts = ex["activations"]
            a = [float(v) for v in acts["component_activation"]]
            g = [float(v) for v in acts["causal_importance"]]
            eff = [gi * ai for gi, ai in zip(g, a, strict=True)]
            series = {"effective": eff, "activation": a, "ci": g}
            key = series[sort]
            center = max(range(len(key)), key=lambda i: abs(key[i])) if key else 0
            lo, hi = max(0, center - window), min(len(toks), center + window + 1)
            out.append(
                {
                    "peak": key[center] if key else 0.0,
                    "center": center,
                    "tokens": toks,
                    "firings": [bool(f) for f in ex["firings"]],
                    "series": series,
                    "window": [lo, hi],
                }
            )
        out.sort(key=lambda e: -abs(float(e["peak"])))  # pyright: ignore[reportArgumentType]
        return out


class InterpStore:
    """Autointerp labels. Read always; write only through `save`, which overwrites in place."""

    def __init__(self, interp_db: Path | None):
        self.path = Path(interp_db) if interp_db else None
        self.available = self.path is not None and self.path.exists()

    def label(self, module: str, idx: int) -> dict[str, str] | None:
        con = _connect(self.path)
        if con is None:
            return None
        with con:
            for key in (f"{module}:{idx}", canonical_key(module, idx)):
                row = con.execute(
                    "SELECT label, reasoning FROM interpretations WHERE component_key = ?", (key,)
                ).fetchone()
                if row is not None:
                    return {"label": row["label"], "reasoning": row["reasoning"] or ""}
        return None

    def save(self, module: str, idx: int, label: str, reasoning: str, raw: str, prompt: str) -> None:
        assert self.path is not None, "no interp db path configured"
        con = sqlite3.connect(self.path)
        with con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS interpretations (component_key TEXT PRIMARY KEY,"
                " label TEXT, reasoning TEXT, raw_response TEXT, prompt TEXT)"
            )
            con.execute(
                "INSERT INTO interpretations (component_key, label, reasoning, raw_response, prompt)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(component_key) DO UPDATE SET"
                " label = excluded.label, reasoning = excluded.reasoning,"
                " raw_response = excluded.raw_response, prompt = excluded.prompt",
                (f"{module}:{idx}", label, reasoning, raw, prompt),
            )
        con.close()
        self.available = True


class PairScoreStore:
    """Data-pass pair metrics, if a sidecar DB has been built. Absent is the normal case today."""

    def __init__(self, db: Path | None):
        self.path = Path(db) if db else None
        self.available = self.path is not None and self.path.exists()

    def coverage(self) -> list[dict[str, object]]:
        con = _connect(self.path)
        if con is None:
            return []
        with con:
            return [dict(r) for r in con.execute("SELECT * FROM pair_metric_coverage")]

    def covered_metrics(self, a_module: str, a_side: str, b_module: str, b_side: str) -> set[str]:
        return {
            str(r["metric"])
            for r in self.coverage()
            if (r["a_module"], r["a_side"], r["b_module"], r["b_side"])
            == (a_module, a_side, b_module, b_side)
        }

    def row(
        self, metric: str, a_module: str, a_side: str, a_idx: int, b_module: str, b_side: str
    ) -> list[dict[str, object]]:
        con = _connect(self.path)
        if con is None:
            return []
        with con:
            return [
                {"idx": int(r["b_idx"]), "score": float(r["score"]), "head": int(r["head"])}
                for r in con.execute(
                    "SELECT b_idx, score, head FROM pair_scores WHERE metric = ? AND a_module = ?"
                    " AND a_side = ? AND a_idx = ? AND b_module = ? AND b_side = ?",
                    (metric, a_module, a_side, a_idx, b_module, b_side),
                )
            ]


class KappaStore:
    """`kappa(c, c') = E_x[g_c g_c' a_c]` per module pair, from `pair_coactivation.pt`."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.available = self.path is not None and self.path.exists()
        self.meta: dict[str, object] = {}
        self._kappa: dict[str, object] = {}
        self._index: dict[str, object] = {}
        self._pairs: dict[str, dict[str, object]] = {}
        self._pos: dict[str, dict[int, int]] = {}
        if not self.available:
            return
        import torch

        blob = torch.load(self.path, weights_only=False, map_location="cpu", mmap=True)
        self.meta = blob["meta"]
        self._kappa = {k: v.float() for k, v in blob["kappa"].items()}
        self._index = blob["index"]
        self._pairs = blob["pairs"]
        self._pos = {m: {int(c): i for i, c in enumerate(ix.tolist())}
                     for m, ix in self._index.items()}

    def covers(self, a_module: str, b_module: str) -> bool:
        return (f"{a_module}|{b_module}" in self._kappa) or (f"{b_module}|{a_module}" in self._kappa)

    def matrix(self, a_module: str, b_module: str):
        """`(kappa[A_pool, B_pool], a_pool, b_pool)` oriented as A x B, or None if uncovered."""
        fwd = self._kappa.get(f"{a_module}|{b_module}")
        if fwd is not None:
            return fwd, self._index[a_module], self._index[b_module]
        rev = self._kappa.get(f"{b_module}|{a_module}")
        if rev is not None:
            return rev.t(), self._index[a_module], self._index[b_module]
        return None

    def row(self, a_module: str, a_idx: int, b_module: str, n_b: int):
        """`(kappa over all `n_b` components of B, coverage mask or None)`, or None."""
        import torch

        got = self.matrix(a_module, b_module)
        if got is None:
            return None
        mat, _, b_pool = got
        pos = self._pos[a_module].get(int(a_idx))
        if pos is None:
            return None
        cols = b_pool.long()
        if int(cols.numel()) == n_b:
            return mat[pos].to(torch.float32), None
        vals = torch.zeros(n_b, dtype=torch.float32)
        keep = torch.zeros(n_b, dtype=torch.bool)
        vals[cols] = mat[pos].to(torch.float32)
        keep[cols] = True
        return vals, keep

    def pool_size(self, module: str) -> int:
        ix = self._index.get(module)
        return 0 if ix is None else int(ix.numel())

    def templates(self, a_module: str, b_module: str) -> list[str]:
        entry = self._entry(a_module, b_module)
        return list(entry["templates"]) if entry else []  # pyright: ignore[reportArgumentType]

    def _entry(self, a_module: str, b_module: str) -> dict[str, object] | None:
        return self._pairs.get(f"{a_module}|{b_module}") or self._pairs.get(f"{b_module}|{a_module}")

    def shared_gate(self, a_module: str, b_module: str) -> bool:
        """Whether these two modules read ONE encoder, which makes kappa measure gate sharing."""
        entry = self._entry(a_module, b_module)
        return bool(entry and entry.get("shared_gate"))

    @property
    def full_pool(self) -> bool:
        return bool(self.meta.get("full_pool"))
