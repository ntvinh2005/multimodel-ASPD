<script lang="ts">
    import type { NormalizeType } from "../../lib/api";
    import type { Loadable } from "../../lib/index";

    type Props = {
        topK: number;
        componentGap: number;
        layerGap: number;
        filteredEdgeCount: number | null;
        normalizeEdges: NormalizeType;
        ciThreshold: Loadable<number>;
        /** lm_interp display filters; undefined = whatever the server was launched with. */
        maxDensity?: number;
        nodeCap?: number;
        hideUnpinnedEdges?: boolean;
        hideNodeCard?: boolean;
        onTopKChange: (value: number) => void;
        onComponentGapChange: (value: number) => void;
        onLayerGapChange: (value: number) => void;
        onNormalizeChange: (value: NormalizeType) => void;
        onCiThresholdChange: (value: number) => void;
        onMaxDensityChange?: (value: number | undefined) => void;
        onNodeCapChange?: (value: number | undefined) => void;
        onHideUnpinnedEdgesChange?: (value: boolean) => void;
        onHideNodeCardChange?: (value: boolean) => void;
    };

    let {
        topK,
        componentGap,
        layerGap,
        filteredEdgeCount,
        normalizeEdges,
        ciThreshold,
        maxDensity,
        nodeCap,
        hideUnpinnedEdges,
        hideNodeCard,
        onTopKChange,
        onComponentGapChange,
        onLayerGapChange,
        onNormalizeChange,
        onCiThresholdChange,
        onMaxDensityChange,
        onNodeCapChange,
        onHideUnpinnedEdgesChange,
        onHideNodeCardChange,
    }: Props = $props();

    // Local state for inputs (immediate UI feedback, apply on blur)
    let localTopK = $state(topK);
    let localComponentGap = $state(componentGap);
    let localLayerGap = $state(layerGap);
    let localCiThreshold = $derived.by(() => (ciThreshold.status === "loaded" ? ciThreshold.data.toString() : ""));
    // Blank means "unset" for both, which is not the same as 1.0 / a number: it leaves the
    // server's LM_INTERP_MAX_DENSITY / LM_INTERP_NODE_CAP alone rather than overriding them.
    //
    // These hold `number | null`, NOT strings: Svelte coerces `bind:value` on an
    // `<input type="number">` through `to_number`, which yields a number or `null` when the field
    // is empty. Treating them as strings threw `localMaxDensity.trim is not a function` on every
    // blur, so the change callback never fired and the control silently did nothing.
    let localMaxDensity = $state<number | null>(maxDensity ?? null);
    let localNodeCap = $state<number | null>(nodeCap ?? null);
    $effect(() => void (localMaxDensity = maxDensity ?? null));
    $effect(() => void (localNodeCap = nodeCap ?? null));

    /** `null`/blank/NaN all mean "leave the server default alone". */
    function asSetting(v: number | null): number | undefined {
        return v === null || !Number.isFinite(v) ? undefined : v;
    }

    // Sync local state when props change externally
    $effect(() => void (localTopK = topK));
    $effect(() => void (localComponentGap = componentGap));
    $effect(() => void (localLayerGap = layerGap));

    function applyIfChanged<T>(local: T, prop: T, apply: (v: T) => void) {
        if (local !== prop) apply(local);
    }
</script>

<div class="controls-bar">
    <label>
        <span>Edge Norm</span>
        <select value={normalizeEdges} onchange={(e) => onNormalizeChange(e.currentTarget.value as NormalizeType)}>
            <option value="none">None</option>
            <option value="target">L2 by Target Node</option>
            <option value="layer">L2 by Target Layer</option>
        </select>
    </label>
    <label>
        <span>Top K Edges</span>
        <input
            type="number"
            bind:value={localTopK}
            onblur={() => applyIfChanged(localTopK, topK, onTopKChange)}
            onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()}
            min={0}
            max={50_000}
            step={100}
        />
    </label>
    <label class:loading={ciThreshold.status === "loading"}>
        <span>CI Threshold</span>
        <input
            type="number"
            bind:value={localCiThreshold}
            onblur={() => {
                const v = parseFloat(localCiThreshold);
                const currentValue = ciThreshold.status === "loaded" ? ciThreshold.data : null;
                if (!isNaN(v) && v !== currentValue) onCiThresholdChange(v);
            }}
            onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()}
            min={0}
            step={0.1}
            disabled={ciThreshold.status === "loading"}
        />
    </label>
    {#if onMaxDensityChange}
        <label
            title="Hide components that fire on more than this fraction of harvest tokens. A DISPLAY filter: they are still in the graph and still conduct influence to the components behind them. Blank uses the server default."
        >
            <span>Max Density</span>
            <input
                type="number"
                bind:value={localMaxDensity}
                onblur={() => onMaxDensityChange(asSetting(localMaxDensity))}
                onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                min={0}
                max={1}
                step={0.01}
                placeholder="off"
            />
        </label>
    {/if}
    {#if onNodeCapChange}
        <label
            title="How many components to draw, ranked by output-reachable influence. Blank uses the server default."
        >
            <span>Node Cap</span>
            <input
                type="number"
                bind:value={localNodeCap}
                onblur={() => onNodeCapChange(asSetting(localNodeCap))}
                onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                min={1}
                step={100}
                placeholder="default"
            />
        </label>
    {/if}
    <label>
        <span>Horizontal Spread</span>
        <input
            type="number"
            bind:value={localComponentGap}
            onblur={() => applyIfChanged(localComponentGap, componentGap, onComponentGapChange)}
            onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()}
            min={0}
            max={20}
            step={1}
        />
    </label>
    <label>
        <span>Vertical Spread</span>
        <input
            type="number"
            bind:value={localLayerGap}
            onblur={() => applyIfChanged(localLayerGap, layerGap, onLayerGapChange)}
            onkeydown={(e) => e.key === "Enter" && e.currentTarget.blur()}
            min={10}
            max={100}
            step={5}
        />
    </label>
    {#if onHideUnpinnedEdgesChange}
        <label class="checkbox">
            <input
                type="checkbox"
                checked={hideUnpinnedEdges}
                onchange={(e) => onHideUnpinnedEdgesChange(e.currentTarget.checked)}
            />
            <span>Hide unpinned edges</span>
        </label>
    {/if}
    {#if onHideNodeCardChange}
        <label class="checkbox">
            <input
                type="checkbox"
                checked={hideNodeCard}
                onchange={(e) => onHideNodeCardChange(e.currentTarget.checked)}
            />
            <span>Hide component card</span>
        </label>
    {/if}

    {#if filteredEdgeCount !== null}
        <div class="legend">
            <span class="edge-count">Showing {filteredEdgeCount} edges</span>
        </div>
    {/if}
</div>

<style>
    .controls-bar {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
    }

    label {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-sans);
    }

    label span {
        font-weight: 500;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    label.loading {
        opacity: 0.6;
    }

    label.loading input {
        cursor: wait;
    }

    input[type="number"] {
        width: 75px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    select {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        cursor: pointer;
    }

    select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .checkbox {
        cursor: pointer;
    }

    .checkbox input[type="checkbox"] {
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .legend {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        margin-left: auto;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }

    .edge-count {
        font-weight: 500;
        color: var(--text-primary);
    }
</style>
