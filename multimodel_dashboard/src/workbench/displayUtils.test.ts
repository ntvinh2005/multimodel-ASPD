import { describe, expect, it } from "vitest";
import { activationOpacity, activationWindow, edgeThresholdScale, panGraphViewBox, zoomGraphViewBox } from "./displayUtils";
import type { ActivationExample, DependencyEdge } from "./types";

const edge = (magnitude: number): DependencyEdge => ({
  id: String(magnitude), modelId: "m", anchorId: "a", source: "s", target: "t",
  signedEffect: magnitude, normalizedMagnitude: magnitude, upstreamCoefficient: null,
  gradient: null, normalization: "test", evidence: "measured-taylor",
});

const example: ActivationExample = {
  activation: .8,
  activations: Array.from({ length: 30 }, (_, position) => position === 5 ? .4 : position === 20 ? .8 : 0),
  anchorId: "a", anchorIndex: 1, centerPosition: 20, context: "context", documentId: "d",
  firings: Array.from({ length: 30 }, (_, position) => position === 5 || position === 20),
  sourceGroup: "source", tokenIds: Array.from({ length: 30 }, (_, position) => position),
  tokenPieces: Array.from({ length: 30 }, (_, position) => `token-${position}`),
  tokenTexts: Array.from({ length: 30 }, (_, position) => ` token-${position}`),
};

describe("graph viewport controls", () => {
  it("zooms around the pointer and clamps runaway zoom-out", () => {
    const centered = zoomGraphViewBox({ x: 0, y: 0, w: 760, h: 510 }, 2, .5, .5);
    expect(centered).toEqual({ x: -380, y: -255, w: 1520, h: 1020 });
    expect(zoomGraphViewBox(centered, 100).w).toBe(3040);
  });

  it("pans from an immutable pointer-down origin", () => {
    expect(panGraphViewBox({ x: 10, y: 20, w: 760, h: 510 }, 76, 51, 760, 510))
      .toEqual({ x: -66, y: -31, w: 760, h: 510 });
  });
});

describe("measured display scales", () => {
  it("derives a fine threshold range from actual edge magnitudes", () => {
    const scale = edgeThresholdScale([edge(.004), edge(.008), edge(.02)]);
    expect(scale.maximum).toBe(.02);
    expect(scale.median).toBe(.008);
    expect(scale.step).toBeLessThan(.001);
  });

  it("keeps snippet and stacked views tied to measured firing positions", () => {
    expect(activationWindow(example, "stacked")).toEqual({ start: 14, end: 27 });
    expect(activationWindow(example, "snippet")).toEqual({ start: 0, end: 29 });
    expect(activationWindow(example, "full")).toEqual({ start: 0, end: 30 });
    expect(activationOpacity(.8, .8)).toBeCloseTo(.86);
  });
});
