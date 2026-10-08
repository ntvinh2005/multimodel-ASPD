import type { FeatureDetail, HealthPayload, OverviewPayload, ResearcherNote } from "./types";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  overview: () => json<OverviewPayload>("/api/overview"),
  health: () => json<HealthPayload>("/api/health"),
  feature: (id: number) => json<FeatureDetail>(`/api/features/${id}`),
  compare: (ids: number[]) =>
    json<{ items: FeatureDetail[] }>(`/api/compare?${ids.map((id) => `ids=${id}`).join("&")}`),
  matchedControl: (id: number, attributes: string[]) =>
    json<Record<string, unknown>>(
      `/api/features/${id}/matched-control?${attributes.map((x) => `attributes=${x}`).join("&")}`,
    ),
  saveNote: (id: number, note: ResearcherNote) =>
    json<{ feature_id: number; note: ResearcherNote }>(`/api/notes/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(note),
    }),
};
