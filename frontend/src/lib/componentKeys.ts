/**
 * Utilities for component key display (e.g. rendering embed/output keys with token strings).
 */

import { isErrorLayer } from "./graphLayout";

export function isTokenNode(key: string): boolean {
    const layer = key.split(":")[0];
    return layer === "embed" || layer === "output";
}

/**
 * An lm_interp error node, "<layer>.err:<seq>:0" — the decomposition's residual at that module.
 * It has no dictionary entry, so nothing that looks a component up by index applies to it:
 * activation contexts, correlations, autointerp labels and interventions are all meaningless.
 */
export function isErrorNode(key: string): boolean {
    return isErrorLayer(key.split(":")[0]);
}

export function formatComponentKey(key: string, tokenStr: string | null): string {
    if (tokenStr && isTokenNode(key)) {
        const layer = key.split(":")[0];
        const label = layer === "embed" ? "input" : "output";
        return `'${tokenStr}' (${label})`;
    }
    return key;
}
