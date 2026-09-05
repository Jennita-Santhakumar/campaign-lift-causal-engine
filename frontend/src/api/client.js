const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8060";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export const api = {
  listCampaigns: () => request("/api/v1/causal/campaigns"),
  seedCampaign: (payload) =>
    request("/api/v1/causal/seed", { method: "POST", body: JSON.stringify(payload) }),
  estimate: (campaign_id, method) =>
    request("/api/v1/causal/estimate", {
      method: "POST",
      body: JSON.stringify({ campaign_id, method }),
    }),
};
