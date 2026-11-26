import random

import numpy as np

from aspd.multimodel.discovery.index import EDGE, decode, encode, group_split
from aspd.multimodel.discovery.search import (
    Graph,
    bottom_up,
    composition,
    matched_null,
    structure,
    top_down,
)


def make_graph():
    nodes = [encode(f"gemma.L{layer:02}.q/1") for layer in range(9, 16)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6)]
    graph = Graph.from_rows(np.array([
        (nodes[a], nodes[b], -0.4, 0.2) for a, b in edges
    ], dtype=EDGE))
    return graph, nodes


def test_search_finds_connected_growth_and_diamond():
    graph, nodes = make_graph()
    found, count = bottom_up(graph, nodes)
    assert count > 0
    assert any(structure(edges)["diamonds"] == 1 for edges in found)
    assert all(set(edges) <= graph.effects.keys() for edges in found)


def test_top_down_traces_a_context_sensitive_sink():
    graph, nodes = make_graph()
    meta = {"dataset": "negation", "family": "scope", "role": "query", "model": "gemma"}
    found, _ = top_down(graph, meta, {("negation", "scope", "query", "gemma"): [
        (0.4, nodes[-1], 2, -0.4, ("negative", "affirmative"))
    ]})
    assert any(structure(edges)["nodes"] >= 5 for edges in found)
    assert all(record["sink"] == nodes[-1] for record in found.values())


def test_composition_requires_exact_edges_and_same_graph_bridge():
    graph, nodes = make_graph()
    parts = [(("a",), ((nodes[0], nodes[1]), (nodes[1], nodes[3]))),
             (("b",), ((nodes[4], nodes[5]), (nodes[5], nodes[6])))]
    inverted = {edge: [i] for i, (_, edges) in enumerate(parts) for edge in edges}
    result, _ = composition(graph, parts, inverted)
    assert result and next(iter(result.values()))["relationship"] == "bridge"
    missing, _ = composition(Graph({}, {}, {}, {}), parts, inverted)
    assert not missing


def test_index_identity_round_trip_and_missing_matched_null():
    assert decode(encode("granite.L23.mlp_up/8191")) == "granite.L23.mlp_up/8191"
    assert group_split("data", "family", {"x", "y"}, 61036) == group_split("data", "family", ["x", "y"], 61036)
    graph, nodes = make_graph()
    assert matched_null(graph, [(nodes[0], nodes[1])], {}, random.Random(1))["trials"] == 0
