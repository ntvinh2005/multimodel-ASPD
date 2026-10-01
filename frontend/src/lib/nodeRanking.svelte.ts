/**
 * Output-influence rank per graph node, from the `lm_interp` app patches.
 *
 * The graph payload carries no notion of how important a node is: `nodeCiVals` is the CI value,
 * which on a hard-gate arm is 1.0 for every live node. `install_output_influence_pruning` ranks
 * nodes by their path-weighted influence on the logits and `install_node_ranking_api` serves that
 * ranking, so a component can be read as "37th of 6,102 eligible" rather than just "drawn".
 *
 * One ranking is held at a time -- only one graph is active -- keyed implicitly by whatever the
 * last `loadNodeRanking` asked for. The endpoint is absent on a stock lab backend and returns
 * `ranked: false` for a graph small enough to need no pruning; both land here as `null`, and every
 * consumer treats that as "no ranking to show" rather than as an error.
 */

import { apiUrl, fetchJson } from "./api/index";

/** `[rank, influence, density]`; `rank` is null for a node the density filter excluded. */
export type NodeRankEntry = [number | null, number, number | null];

export type GraphRanking = {
    ranked: boolean;
    nAlive: number;
    nEligible: number;
    nDrawn: number;
    maxDensity: number;
    keptInfluenceFrac: number;
    nodes: Record<string, NodeRankEntry>;
};

export const nodeRanking = $state<{ current: GraphRanking | null }>({ current: null });

export async function loadNodeRanking(promptId: number, graphId: number, ciThreshold: number) {
    const url = apiUrl(`/api/lm_interp/node_ranking/${promptId}`);
    url.searchParams.set("ci_threshold", String(ciThreshold));
    try {
        const byGraph = await fetchJson<Record<string, GraphRanking>>(url.toString());
        const entry = byGraph[String(graphId)];
        nodeRanking.current = entry && entry.ranked ? entry : null;
    } catch {
        // A stock backend has no such route. Not having a ranking is not an error condition.
        nodeRanking.current = null;
    }
}

export function rankOf(nodeKey: string): NodeRankEntry | null {
    return nodeRanking.current?.nodes[nodeKey] ?? null;
}
