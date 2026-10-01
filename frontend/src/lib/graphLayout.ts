/**
 * Graph layout utilities for canonical transformer addresses.
 *
 * Canonical address format:
 *   "embed"              — embedding
 *   "output"           — unembed / logits
 *   "{block}.{sublayer}.{projection}" — e.g. "0.attn.q", "2.mlp.down"
 *
 * Node key format:
 *   "{layer}:{seqIdx}:{cIdx}" — e.g. "0.attn.q:3:5", "embed:0:0"
 */

export type LayerInfo = {
    name: string;
    block: number; // -1 for embed, Infinity for output
    sublayer: string; // "attn" | "attn_fused" | "mlp" | "glu" | "embed" | "output"
    projection: string | null; // "q" | "k" | "v" | "o" | "qkv" | "up" | "down" | "gate" | null
    isError: boolean; // "<layer>.err" — the decomposition's residual at that module
};

/**
 * Error-node pseudo-layer, emitted by lm_interp's error-node attribution as "<layer>.err".
 * It is a node per (module, position) carrying what the decomposition could not explain, so it
 * belongs on its own row: aliasing it onto the module's real row (which is what happens if you
 * just split on ".") draws it on top of that module's components.
 */
export const ERROR_SUFFIX = ".err";

export function isErrorLayer(name: string): boolean {
    return name.endsWith(ERROR_SUFFIX);
}

/** "0.attn.q.err" -> "0.attn.q". Returns the input unchanged for a normal layer. */
export function baseLayer(name: string): string {
    return isErrorLayer(name) ? name.slice(0, -ERROR_SUFFIX.length) : name;
}

const SUBLAYER_ORDER = ["attn", "attn_fused", "glu", "mlp"];

// Projections that share a row and get grouped horizontally
const GROUPED_PROJECTIONS: Record<string, string[]> = {
    attn: ["q", "k", "v"],
    glu: ["gate", "up"],
};

// Full projection ordering within each sublayer (grouped inputs first, then outputs)
const PROJECTION_ORDER: Record<string, string[]> = {
    attn: ["q", "k", "v", "o"],
    attn_fused: ["qkv", "o"],
    glu: ["gate", "up", "down"],
    mlp: ["up", "down"],
};

export function parseLayer(name: string): LayerInfo {
    if (name === "embed") return { name, block: -1, sublayer: "embed", projection: null, isError: false };
    if (name === "output") return { name, block: Infinity, sublayer: "output", projection: null, isError: false };

    const isError = isErrorLayer(name);
    const parts = baseLayer(name).split(".");
    return {
        name,
        block: +parts[0],
        sublayer: parts[1],
        projection: parts[2],
        isError,
    };
}

/**
 * Row key: layers that share the same visual row.
 * q/k/v share "0.attn.qkv", gate/up share "0.glu.gate_up".
 * Ungrouped projections (o, down) get their own row.
 */
export function getRowKey(layer: string): string {
    const info = parseLayer(layer);
    if (info.sublayer === "embed" || info.sublayer === "output") return layer;

    // Error nodes never join a q/k/v group: one module's residual is not the same object as
    // another's, and sharing a row would stack them on the components they are the residual OF.
    if (info.isError) return layer;

    const grouped = GROUPED_PROJECTIONS[info.sublayer];
    if (grouped && info.projection && grouped.includes(info.projection)) {
        return `${info.block}.${info.sublayer}.${grouped.join("_")}`;
    }
    return layer;
}

/**
 * Row label for display.
 */
export function getRowLabel(rowKey: string): string {
    if (rowKey === "embed") return "embed";
    if (rowKey === "output") return "output";
    if (isErrorLayer(rowKey)) return `${getRowLabel(baseLayer(rowKey))} ε`;

    const parts = rowKey.split(".");
    const block = parts[0];
    const sublayer = parts[1];
    const projPart = parts[2];

    if (!projPart) return `${block}.${sublayer}`;

    // Grouped projections: show "0.attn.qkv" or "0.glu.gate/up"
    if (projPart.includes("_")) {
        return `${block}.${sublayer}.${projPart.replace(/_/g, "/")}`;
    }
    return rowKey;
}

/**
 * Sort row keys: embed at bottom, output at top, blocks in between.
 * Within a block: sublayers follow SUBLAYER_ORDER, grouped projections before ungrouped.
 */
export function sortRows(rows: string[]): string[] {
    return [...rows].sort((a, b) => {
        // Sort on the BASE address so "0.mlp.down.err" lands beside "0.mlp.down" rather than in
        // a sublayer called "err"; the isError tiebreak below then puts the residual after it.
        const partsA = baseLayer(a).split(".");
        const partsB = baseLayer(b).split(".");

        const blockA = a === "embed" ? -1 : a === "output" ? Infinity : +partsA[0];
        const blockB = b === "embed" ? -1 : b === "output" ? Infinity : +partsB[0];

        if (blockA !== blockB) return blockA - blockB;

        const sublayerA = partsA[1] ?? "";
        const sublayerB = partsB[1] ?? "";
        const sublayerDiff = SUBLAYER_ORDER.indexOf(sublayerA) - SUBLAYER_ORDER.indexOf(sublayerB);
        if (sublayerDiff !== 0) return sublayerDiff;

        // Within same sublayer: order by first projection in the row key
        const projOrder = PROJECTION_ORDER[sublayerA] ?? [];
        const firstProjA = (partsA[2] ?? "").split("_")[0];
        const firstProjB = (partsB[2] ?? "").split("_")[0];
        const projIdxA = projOrder.indexOf(firstProjA);
        const projIdxB = projOrder.indexOf(firstProjB);
        const projDiff = (projIdxA === -1 ? 999 : projIdxA) - (projIdxB === -1 ? 999 : projIdxB);
        if (projDiff !== 0) return projDiff;

        // Same module: its residual row sits directly after its components.
        return Number(isErrorLayer(a)) - Number(isErrorLayer(b));
    });
}

/**
 * Get the grouped projections for a sublayer, if any.
 * Returns null if no grouping (each projection gets its own horizontal space).
 */
export function getGroupProjections(sublayer: string): string[] | null {
    return GROUPED_PROJECTIONS[sublayer] ?? null;
}

/**
 * Check if a specific projection is part of its sublayer's group.
 */
export function isGroupedProjection(sublayer: string, projection: string): boolean {
    const group = GROUPED_PROJECTIONS[sublayer];
    return group !== undefined && group.includes(projection);
}

/**
 * Build the full layer address from block + sublayer + projection.
 */
export function buildLayerAddress(block: number, sublayer: string, projection: string): string {
    return `${block}.${sublayer}.${projection}`;
}
