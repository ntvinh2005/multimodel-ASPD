<script lang="ts">
    import type { Loadable } from "../lib/index";
    import { EXAMPLE_SORT_LABELS, displaySettings, type ExampleSortMode } from "../lib/displaySettings.svelte";
    import TokenHighlights from "./TokenHighlights.svelte";

    export type ActivationExamplesData = {
        tokens: string[][]; // [n_examples, window_size]
        ci: number[][]; // [n_examples, window_size]
        componentActs: number[][]; // [n_examples, window_size]
        maxAbsComponentAct: number;
    };

    interface Props {
        data: Loadable<ActivationExamplesData>;
    }

    let { data }: Props = $props();

    const loading = $derived(data.status !== "loaded");
    const loaded = $derived(data.status === "loaded" ? data.data : null);

    let examplesEl = $state<HTMLDivElement | undefined>(undefined);
    let currentPage = $state(0);
    let pageSize = $state(10);

    let nExamples = $derived(loaded?.tokens.length ?? 0);

    function argmaxAbs(arr: number[]): number {
        let maxIdx = 0;
        for (let i = 1; i < arr.length; i++) {
            if (Math.abs(arr[i]) > Math.abs(arr[maxIdx])) maxIdx = i;
        }
        return maxIdx;
    }

    // Peak |act|, NOT argmax(ci). On a hard-gate arm `ci` is exactly 0 or 1, so `argmax(ci)`
    // returns the FIRST fired token rather than the strongest one, and "Center on peak" centred
    // on the wrong token for every example. `|act|` is what the label claims and it agrees with
    // `argmax(ci)` on a continuous gate anyway.
    let firingPositions = $derived(loaded?.componentActs.map(argmaxAbs) ?? []);

    // Minimum container width (in ch) so that per-row flex centering works without clipping.
    // Each row needs: 2 * max(leftWidth, rightWidth) + centerWidth.
    // Each token adds ~0.3ch overhead for border + margin beyond its character width.
    const TOKEN_OVERHEAD_CH = 0.3;

    let minWidthCh = $derived.by(() => {
        if (!displaySettings.centerOnPeak || !loaded) return 0;
        let max = 0;
        for (let i = 0; i < loaded.tokens.length; i++) {
            const fp = firingPositions[i];
            const tokens = loaded.tokens[i];

            let leftWidth = 0;
            for (let j = 0; j < fp; j++) leftWidth += tokens[j].length + TOKEN_OVERHEAD_CH;

            let rightWidth = 0;
            for (let j = fp + 1; j < tokens.length; j++) rightWidth += tokens[j].length + TOKEN_OVERHEAD_CH;

            const centerWidth = tokens[fp].length + TOKEN_OVERHEAD_CH;
            const required = 2 * Math.max(leftWidth, rightWidth) + centerWidth;
            if (required > max) max = required;
        }
        return Math.ceil(max + 1);
    });

    // Update currentPage when page input changes
    function handlePageInput(event: Event) {
        const { value } = event.target as HTMLInputElement;
        if (value === "") return;
        const valueNum = parseInt(value);
        if (!isNaN(valueNum) && valueNum >= 1 && valueNum <= totalPages) {
            currentPage = valueNum - 1;
        } else {
            throw new Error(`Invalid page number: ${value} (must be 1-${totalPages})`);
        }
    }

    // Stored order is the reservoir's draw order, i.e. a random sample rather than a ranking.
    // Every key below is computed from the payload the card already holds, so sorting costs no
    // fetch -- but it can only reorder the examples the reservoir kept, not surface better ones.
    // NOTE on this project's transcoder arm: `ci` is a hard gate, so `example_ci` is exactly 0 or
    // 1 and the two CI keys tie for every example. Act value is the informative one there.
    function peak(row: number[]): number {
        let max = 0;
        for (const v of row) max = Math.max(max, Math.abs(v));
        return max;
    }

    function mean(row: number[]): number {
        if (row.length === 0) return 0;
        let total = 0;
        for (const v of row) total += Math.abs(v);
        return total / row.length;
    }

    const SORT_KEYS: Record<Exclude<ExampleSortMode, "stored">, ["ci" | "componentActs", (row: number[]) => number]> = {
        peak_act: ["componentActs", peak],
        mean_act: ["componentActs", mean],
        peak_ci: ["ci", peak],
        mean_ci: ["ci", mean],
    };

    // The score each row is sorted by, shown beside it. Without this the only evidence that a sort
    // ran is the order itself, which is unreadable when the values are close -- several components
    // here have ten examples spanning 8.5 to 8.8.
    const sortScores = $derived.by(() => {
        const mode = displaySettings.exampleSort;
        if (mode === "stored" || !loaded) return null;
        const [field, reduce] = SORT_KEYS[mode];
        return loaded[field].map(reduce);
    });

    const sortScoreLabel = $derived(
        displaySettings.exampleSort === "stored" ? "" : EXAMPLE_SORT_LABELS[displaySettings.exampleSort],
    );

    let allIndices = $derived.by(() => {
        const stored = Array.from({ length: nExamples }, (_, i) => i);
        const mode = displaySettings.exampleSort;
        if (mode === "stored" || !loaded) return stored;
        const scores = sortScores!;
        // Descending, ties broken by stored order so the list is stable across re-sorts.
        return stored.sort((a, b) => scores[b] - scores[a] || a - b);
    });

    let paginatedIndices = $derived.by(() => {
        const start = currentPage * pageSize;
        const end = start + pageSize;
        return allIndices.slice(start, end);
    });

    let totalPages = $derived(Math.ceil(allIndices.length / pageSize));

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset to page 0 when data, page size or sort order changes
    $effect(() => {
        void loaded;
        void pageSize;
        void displaySettings.exampleSort;
        currentPage = 0;
    });

    function centerScroll() {
        if (!examplesEl) return;
        examplesEl.scrollLeft = (examplesEl.scrollWidth - examplesEl.clientWidth) / 2;
    }

    $effect(() => {
        if (!displaySettings.centerOnPeak) return;
        void paginatedIndices;
        requestAnimationFrame(centerScroll);
    });
</script>

<div class="container">
    <div class="controls">
        <div class="pagination">
            <button disabled={loading || currentPage === 0} onclick={previousPage}>&lt;</button>
            <input
                type="number"
                min="1"
                max={totalPages}
                value={loading ? "" : currentPage + 1}
                oninput={handlePageInput}
                class="page-input"
                disabled={loading}
            />
            <span>of {loading ? "-" : totalPages}</span>
            <button disabled={loading || currentPage === totalPages - 1} onclick={nextPage}>&gt;</button>
        </div>
        <div class="page-size-control">
            <label for="page-size">Per page:</label>
            <select id="page-size" bind:value={pageSize} disabled={loading}>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
            </select>
        </div>
        <div class="sort-control">
            <label for="example-sort">Sort:</label>
            <select id="example-sort" bind:value={displaySettings.exampleSort} disabled={loading}>
                {#each Object.entries(EXAMPLE_SORT_LABELS) as [mode, label] (mode)}
                    <option value={mode}>{label}</option>
                {/each}
            </select>
        </div>
        <label class="center-toggle">
            <input type="checkbox" bind:checked={displaySettings.centerOnPeak} disabled={loading} />
            Center on peak
        </label>
    </div>
    {#if loading}
        <div class="examples">
            <div class="examples-inner">
                {#each Array(pageSize) as _, i (i)}
                    <div class="skeleton-row"></div>
                {/each}
            </div>
        </div>
    {:else}
        {@const d = loaded!}
        <div class="examples" bind:this={examplesEl}>
            {#if displaySettings.centerOnPeak}
                <div class="examples-inner" style="min-width: {minWidthCh}ch">
                    {#each paginatedIndices as idx (idx)}
                        {@const fp = firingPositions[idx]}
                        <div class="example-row">
                            {#if sortScores}
                                <span class="sort-score" title="{sortScoreLabel} for this example"
                                    >{sortScores[idx].toFixed(2)}</span
                                >
                            {/if}
                            <div class="left-tokens">
                                <TokenHighlights
                                    tokenStrings={d.tokens[idx].slice(0, fp)}
                                    tokenCi={d.ci[idx].slice(0, fp)}
                                    tokenComponentActs={d.componentActs[idx].slice(0, fp)}
                                    maxAbsComponentAct={d.maxAbsComponentAct}
                                />
                            </div>
                            <div class="center-token">
                                <TokenHighlights
                                    tokenStrings={[d.tokens[idx][fp]]}
                                    tokenCi={[d.ci[idx][fp]]}
                                    tokenComponentActs={[d.componentActs[idx][fp]]}
                                    maxAbsComponentAct={d.maxAbsComponentAct}
                                />
                            </div>
                            <div class="right-tokens">
                                <TokenHighlights
                                    tokenStrings={d.tokens[idx].slice(fp + 1)}
                                    tokenCi={d.ci[idx].slice(fp + 1)}
                                    tokenComponentActs={d.componentActs[idx].slice(fp + 1)}
                                    maxAbsComponentAct={d.maxAbsComponentAct}
                                />
                            </div>
                        </div>
                    {/each}
                </div>
            {:else}
                <div class="examples-inner">
                    {#each paginatedIndices as idx (idx)}
                        <div class="example-item">
                            {#if sortScores}
                                <span class="sort-score" title="{sortScoreLabel} for this example"
                                    >{sortScores[idx].toFixed(2)}</span
                                >
                            {/if}
                            <TokenHighlights
                                tokenStrings={d.tokens[idx]}
                                tokenCi={d.ci[idx]}
                                tokenComponentActs={d.componentActs[idx]}
                                maxAbsComponentAct={d.maxAbsComponentAct}
                            />
                        </div>
                    {/each}
                </div>
            {/if}
        </div>
    {/if}
</div>

<style>
    .sort-score {
        flex: 0 0 auto;
        min-width: 5ch;
        margin-right: var(--space-2);
        text-align: right;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .examples {
        padding: var(--space-2);
        overflow-x: auto;
        overflow-y: clip;
    }

    .examples-inner {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        min-width: 100%;
    }

    .example-row {
        display: flex;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }

    .example-item {
        display: flex;
        align-items: baseline;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }

    .left-tokens {
        flex: 1 1 0;
        min-width: 0;
        text-align: right;
    }

    .center-token {
        flex: 0 0 auto;
    }

    .right-tokens {
        flex: 1 1 0;
        min-width: 0;
        text-align: left;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-2);
        border-bottom: 1px solid var(--border-default);
        flex-wrap: wrap;
    }

    .sort-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin-left: auto;
    }

    .sort-control label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        white-space: nowrap;
        font-weight: 500;
    }

    .sort-control select {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
    }

    .sort-control select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .center-toggle {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
        cursor: pointer;
    }

    .center-toggle input {
        cursor: pointer;
    }

    .page-size-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .page-size-control label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        white-space: nowrap;
        font-weight: 500;
    }

    .page-size-control select {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 100px;
    }

    .page-size-control select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .pagination button {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-secondary);
    }

    .pagination button:hover:not(:disabled) {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .pagination button:disabled {
        opacity: 0.5;
    }

    .pagination span {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .page-input {
        width: 50px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        appearance: textfield;
    }

    .page-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .page-input::-webkit-inner-spin-button,
    .page-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .skeleton-row {
        height: calc(var(--text-sm) * 1.8);
        border-radius: var(--radius-sm);
        background: var(--border-default);
        opacity: 0.3;
        animation: skeleton-pulse 1.2s ease-in-out infinite;
    }

    @keyframes skeleton-pulse {
        0%,
        100% {
            opacity: 0.3;
        }
        50% {
            opacity: 0.1;
        }
    }
</style>
