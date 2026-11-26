"""Three bounded searches that only traverse edges inside one measured model/anchor graph."""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from statistics import median

import numpy as np

from aspd.multimodel.discovery.index import C, gate_of

SIZES = (5, 8, 12, 15)
METHODS = ("bottom-up", "top-down", "composition")


@dataclass
class Graph:
    effects: dict
    parents: dict
    children: dict
    degree: Counter

    @classmethod
    def from_rows(cls, rows):
        effects, parents, children, degree = {}, defaultdict(list), defaultdict(list), Counter()
        for source, target, effect, weight in rows.tolist():
            key = source, target
            if key in effects:
                raise ValueError("Duplicate edge inside one exact anchor/model")
            effects[key] = effect, weight
            parents[target].append((weight, source))
            children[source].append((weight, target))
            degree.update(key)
        for index in (parents, children):
            for values in index.values():
                values.sort(reverse=True)
        return cls(effects, parents, children, degree)

    def induced(self, nodes):
        return tuple(
            sorted(
                (source, target)
                for source in nodes
                for _, target in self.children.get(source, ())
                if target in nodes
            )
        )

    def next_nodes(self, nodes, upstream):
        scores = {}
        for node in sorted(nodes):
            pools = (self.parents,) if upstream else (self.parents, self.children)
            for index in pools:
                for weight, other in index.get(node, ())[:4]:
                    if other not in nodes:
                        scores[other] = max(scores.get(other, 0), weight)
        return sorted(scores, key=lambda node: (-scores[node], node))[:8]


def structure(edges):
    nodes = set()
    parents, children = defaultdict(set), defaultdict(set)
    degree_in, degree_out = Counter(), Counter()
    for source, target in edges:
        nodes.update((source, target))
        children[source].add(target)
        parents[target].add(source)
        degree_in[target] += 1
        degree_out[source] += 1
    depths = {}
    for node in sorted(nodes):
        depths[node] = max((depths[parent] + 1 for parent in parents[node]), default=0)
    diamond = 0
    for node in children:
        arms = Counter(end for middle in children[node] for end in children.get(middle, ()))
        diamond += sum(count > 1 for count in arms.values())
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "depth": max(depths.values(), default=0),
        "branching": sum(value > 1 for value in degree_out.values()),
        "convergence": sum(value > 1 for value in degree_in.values()),
        "diamonds": diamond,
    }


def grow(graph, seeds, upstream=False, sizes=SIZES, width=2):
    """Beam growth at fixed node budgets. All vertices come from connected measured edges."""
    beam = [frozenset(seeds)]
    proposals, expanded = set(), 0
    for size in range(len(seeds), max(sizes) + 1):
        scored = {}
        for nodes in beam:
            if size in sizes:
                edges = graph.induced(nodes)
                if structure(edges)["depth"] >= 2:
                    proposals.add(edges)
            if size < max(sizes):
                for new in graph.next_nodes(nodes, upstream):
                    trial = nodes | {new}
                    if trial in scored:
                        continue
                    edges = graph.induced(trial)
                    plausible = structure(edges)
                    # Reward edges across the whole growing graph and penalize hub-only leaves.
                    scored[trial] = (
                        plausible["depth"] >= 2,
                        len(edges) - len(trial),
                        sum(np.log1p(graph.effects[edge][1]) for edge in edges),
                    )
        expanded += len(scored)
        beam = sorted(
            scored,
            key=lambda nodes: (scored[nodes], tuple(-v for v in sorted(nodes))),
            reverse=True,
        )[:width]
        if not beam:
            break
    return proposals, expanded


def bottom_up(graph, seeds):
    # Include recurring archived starts and both high fan-in and high fan-out.
    starts = [node for node in seeds if node in graph.degree][:2]
    for index in (graph.parents, graph.children):
        viable = [node for node, edges in index.items() if len(edges) > 1]
        if viable:
            starts.append(max(viable, key=lambda node: (index[node][1][0], -node)))
    proposals, expanded = {}, 0
    for node in dict.fromkeys(starts):
        found, count = grow(graph, {node})
        expanded += count
        proposals.update((edges, {}) for edges in found)
    return proposals, expanded


CONTRASTS = (
    ("negative", "affirmative"),
    ("counterfactual_negative", "counterfactual_affirmative"),
    ("scope_control", "negative"),
    ("clean", "no_match"),
    ("induction", "no_match"),
    ("original", "no_match"),
)


def estimate_sinks(index, indices):
    """Use matched discovery groups and exact roles to rank internal coefficient contrasts."""
    groups = defaultdict(lambda: defaultdict(list))
    for i in indices:
        row = index.scopes[i]
        if row["split"] != "discovery" or not row.get("role"):
            continue
        key = row["dataset"], row["family"], row["group"], row["role"], row["model"]
        groups[key][row["condition"]].append(i)
    aggregate = defaultdict(list)
    contrasts = 0
    for (data, family, _, role, model), conditions in groups.items():
        for a, b in CONTRASTS:
            if len(conditions[a]) != 1 or len(conditions[b]) != 1:
                continue
            ia, ib = conditions[a][0], conditions[b][0]
            _, left = index.graph(ia)
            _, right = index.graph(ib)
            lhs = {
                int(item["id"]): (float(item["coefficient"]), float(item["normalized"]))
                for item in left
            }
            rhs = {
                int(item["id"]): (float(item["coefficient"]), float(item["normalized"]))
                for item in right
            }
            for node in lhs.keys() | rhs.keys():
                gate = gate_of(node)
                allow_zero_a = (
                    index.scopes[ia]["complete"] and gate not in index.scopes[ia]["active"]
                )
                allow_zero_b = (
                    index.scopes[ib]["complete"] and gate not in index.scopes[ib]["active"]
                )
                x = lhs.get(node, (0.0, 0.0) if allow_zero_a else None)
                y = rhs.get(node, (0.0, 0.0) if allow_zero_b else None)
                if x is None or y is None or not all(np.isfinite([*x, *y])):
                    continue
                aggregate[data, family, role, model, node].append(
                    (x[0] - y[0], x[1] - y[1], (a, b))
                )
            contrasts += 1
    ranked = defaultdict(list)
    for (*key, node), rows in aggregate.items():
        strength = abs(median(value[1] for value in rows))
        if strength <= 0:
            continue
        ranked[tuple(key)].append(
            (strength, node, len(rows), median(item[0] for item in rows), rows[0][2])
        )
    return {key: sorted(values, reverse=True)[:24] for key, values in ranked.items()}, contrasts


def top_down(graph, meta, sinks):
    key = meta["dataset"], meta.get("family"), meta.get("role"), meta["model"]
    proposals, expanded = {}, 0
    for strength, component, comparisons, signed, contrast in [
        item for item in sinks.get(key, ()) if component_in_graph(graph, item[1])
    ][:3]:
        found, count = grow(graph, {component}, upstream=True)
        expanded += count
        for edges in found:
            proposals[edges] = {
                "sink": component,
                "normalizedContrast": strength,
                "signedCoefficientContrast": signed,
                "discoveryComparisons": comparisons,
                "contrastedConditions": contrast,
            }
    return proposals, expanded


def component_in_graph(graph, component):
    return len(graph.parents.get(component, ())) > 0


def shortest_bridge(graph, left, right, other_nodes=()):
    """Find a measured directed path of at most two edges connecting two exact subcircuits."""
    direct = [
        (src, dst)
        for src in left
        for _, dst in graph.children.get(src, ())
        if dst in right and src not in right
    ]
    if direct:
        return (min(direct, key=lambda edge: (-graph.effects[edge][1], edge)),)
    valid = []
    for src in sorted(left - right):
        for _, mid in graph.children.get(src, ())[:10]:
            if mid in left or mid in right or mid in other_nodes:
                continue
            for _, dst in graph.children.get(mid, ())[:10]:
                if dst in right:
                    valid.append(((src, mid), (mid, dst)))
    return max(valid, key=lambda path: min(graph.effects[edge][1] for edge in path), default=())


def composition(graph, templates, by_edge):
    available = Counter()
    for edge in graph.effects:
        for template in by_edge.get(edge, ()):
            available[template] += 1
    active = [templates[i] for i, count in available.items() if count == len(templates[i][1])]
    proposals, considered = {}, 0
    for (left_ids, left), (right_ids, right) in combinations(sorted(active), 2):
        if set(left_ids) & set(right_ids):
            continue
        a, b = (set(value for edge in part for value in edge) for part in (left, right))
        if a <= b or b <= a:
            continue
        considered += 1
        bridge = shortest_bridge(graph, a, b) or shortest_bridge(graph, b, a)
        common = []
        if not (bridge or a & b):
            left_child = {
                dst: src for src in sorted(a) for _, dst in graph.children.get(src, ())[:6]
            }
            right_child = {
                dst: src for src in sorted(b) for _, dst in graph.children.get(src, ())[:6]
            }
            choices = sorted(
                (left_child.keys() & right_child.keys()) - a - b,
                key=lambda value: (
                    -min(
                        graph.effects[left_child[value], value][1],
                        graph.effects[right_child[value], value][1],
                    ),
                    value,
                ),
            )
            if not choices:
                continue
            child = choices[0]
            common = [(left_child[child], child), (right_child[child], child)]
        nodes = a | b | {node for edge in (*bridge, *common) for node in edge}
        if 5 <= len(nodes) <= 24:
            edges = graph.induced(nodes)
            proposals[edges] = {
                "modules": [" / ".join(left_ids), " / ".join(right_ids)],
                "relationship": "overlap" if a & b else "bridge" if bridge else "convergence",
            }
    return proposals, considered


def directed_gate_signature(edges):
    """WL candidates are verified by exact directed isomorphism during cross-model grouping."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, {node: gate_of(node) for node in graph}, "gate")
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="gate", iterations=4), graph


def signature(edges):
    return hashlib.sha256(repr(edges).encode()).hexdigest()[:16]


def measured_paths(index, signature_edges):
    """Inverted postings restricted to shortlisted exact component edges."""
    kept = sorted(
        {(source << 21) | target for edges in signature_edges for source, target in edges}
    )
    keys = np.array(kept, dtype=np.uint64)
    postings = defaultdict(set)
    if not len(keys):
        return postings
    for start in range(0, len(index.edges), 1_000_000):
        block = index.edges[start : start + 1_000_000]
        packed = (block["source"].astype(np.uint64) << 21) | block["target"]
        offsets = np.searchsorted(keys, packed)
        hit = np.flatnonzero(keys[np.minimum(offsets, len(keys) - 1)] == packed)
        scope = np.searchsorted(index.edge_starts, start + hit, side="right") - 1
        for value, scope_id in zip(packed[hit].tolist(), scope.tolist(), strict=True):
            postings[value].add(scope_id)
    return postings


def recurring(postings, edges):
    sets = [postings[(source << 21) | target] for source, target in edges]
    return set.intersection(*sorted(sets, key=len)) if sets else set()


def matched_null(graph, edges, frequency, rng, trials=64):
    """Sample other vertices in the same matrix, degree bin and observed prevalence bin."""
    vertices = {node for edge in edges for node in edge}
    bins = defaultdict(list)
    for node, degree in graph.degree.items():
        key = node // C, int(np.log2(degree)), frequency.get(gate_of(node), 0)
        bins[key].append(node)
    choices = {}
    for node in vertices:
        key = node // C, int(np.log2(graph.degree[node])), frequency.get(gate_of(node), 0)
        choices[node] = [other for other in bins[key] if other not in vertices]
    if min(map(len, choices.values()), default=0) == 0:
        return {
            "trials": 0,
            "hits": None,
            "reason": "no matched substitute for at least one vertex",
        }
    equal = trials_run = 0
    for _ in range(trials):
        shuffled, used = {}, set()
        for node in sorted(vertices, key=lambda n: (len(choices[n]), n)):
            valid = [item for item in choices[node] if item not in used]
            if not valid:
                break
            shuffled[node] = rng.choice(valid)
            used.add(shuffled[node])
        if len(shuffled) != len(vertices):
            continue
        trials_run += 1
        equal += all(
            (shuffled[source], shuffled[target]) in graph.effects for source, target in edges
        )
    return {"trials": trials_run, "hits": equal if trials_run else None}


def pareto(rows, objectives):
    """Return nondomination layers. Missing evidence never gets treated as a positive metric."""
    pending = list(rows)
    ordered = []
    level = 1
    while pending:
        frame = []
        for row in pending:
            value = objectives(row)
            if not any(
                all(a >= b for a, b in zip(objectives(other), value, strict=True))
                and any(a > b for a, b in zip(objectives(other), value, strict=True))
                for other in pending
                if other is not row
            ):
                frame.append(row)
        chosen = {id(row) for row in frame}
        for row in frame:
            row["paretoFront"] = level
        ordered.extend(sorted(frame, key=lambda row: row["id"]))
        pending = [row for row in pending if id(row) not in chosen]
        level += 1
    return ordered
