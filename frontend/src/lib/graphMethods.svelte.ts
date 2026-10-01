/**
 * Which attribution method produced each stored graph.
 *
 * `GraphData` is a pydantic model in the pinned `param_decomp_lab`, so the method cannot ride in
 * the graph payload; `lm_interp.app_methods` serves it alongside instead, keyed by graph id. Fails
 * soft to an empty map, so a stock backend (or one without the method patches) just shows no
 * badges rather than breaking the tab.
 */

import { apiUrl } from "./api/index";

export type GraphMethod = "lab" | "err";

export const METHOD_LABELS: Record<GraphMethod, string> = {
    lab: "lab",
    err: "err",
};

export const METHOD_TITLES: Record<GraphMethod, string> = {
    lab: "param_decomp_lab's attribution: gradients on a replacement forward that is not the target model, no error nodes",
    err: "lm_interp's attribution: exact forward (ŷ + ε = y), with error nodes for what the decomposition cannot explain",
};

export const graphMethods = $state<{ byId: Record<number, GraphMethod> }>({ byId: {} });

export async function loadGraphMethods(promptId: number) {
    try {
        const url = apiUrl(`/api/lm_interp/graph_methods/${promptId}`);
        const res = await fetch(url.toString());
        if (!res.ok) {
            graphMethods.byId = {};
            return;
        }
        const raw = (await res.json()) as Record<string, string>;
        const byId: Record<number, GraphMethod> = {};
        for (const [id, method] of Object.entries(raw)) {
            byId[Number(id)] = method === "err" ? "err" : "lab";
        }
        graphMethods.byId = byId;
    } catch {
        graphMethods.byId = {};
    }
}

export function methodOf(graphId: number): GraphMethod | null {
    return graphMethods.byId[graphId] ?? null;
}
