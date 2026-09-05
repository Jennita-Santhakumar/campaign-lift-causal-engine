import { useEffect, useState } from "react";
import { api } from "./api/client.js";
import ATEComparison from "./components/ATEComparison.jsx";
import BalanceTable from "./components/BalanceTable.jsx";
import CATEBreakdown from "./components/CATEBreakdown.jsx";
import SensitivityChart from "./components/SensitivityChart.jsx";

export default function App() {
  const [campaigns, setCampaigns] = useState([]);
  const [campaignId, setCampaignId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [seeding, setSeeding] = useState(false);

  const refreshCampaigns = async () => {
    try {
      const list = await api.listCampaigns();
      setCampaigns(list);
      if (list.length && !campaignId) setCampaignId(list[0].id);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    refreshCampaigns();
  }, []);

  const handleSeed = async () => {
    setSeeding(true);
    setError("");
    try {
      const campaign = await api.seedCampaign({ name: `Campaign ${new Date().toLocaleTimeString()}` });
      await refreshCampaigns();
      setCampaignId(campaign.id);
    } catch (e) {
      setError(e.message);
    } finally {
      setSeeding(false);
    }
  };

  const handleAnalyze = async () => {
    if (!campaignId) return;
    setLoading(true);
    setError("");
    try {
      const data = await api.estimate(campaignId, "psm");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Causal Impact Analysis</h1>
      <p className="subtitle">
        Propensity-score matching, IPW, CATE, panel DiD, and Rosenbaum sensitivity bounds for a
        confounded marketing-campaign dataset.
      </p>

      {error && <div className="error">{error}</div>}

      <div className="card">
        <h2>Campaign</h2>
        <select value={campaignId} onChange={(e) => setCampaignId(e.target.value)}>
          {campaigns.length === 0 && <option value="">No campaigns yet</option>}
          {campaigns.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} (true ATE: {c.true_ate?.toFixed(2)})
            </option>
          ))}
        </select>
        <button onClick={handleSeed} disabled={seeding}>
          {seeding ? "Seeding..." : "Seed new campaign"}
        </button>{" "}
        <button onClick={handleAnalyze} disabled={!campaignId || loading}>
          {loading ? "Analyzing..." : "Run analysis"}
        </button>
      </div>

      {result && (
        <>
          <div className="card">
            <h2>Estimated Average Treatment Effect</h2>
            <div className="grid">
              <div className="stat">
                <div className="label">True ATE (synthetic ground truth)</div>
                <div className="value">${result.true_ate?.toFixed(2)}</div>
              </div>
              <div className="stat">
                <div className="label">PSM-adjusted ATE</div>
                <div className="value">${result.ate.toFixed(2)}</div>
              </div>
              <div className="stat">
                <div className="label">95% CI</div>
                <div className="value">
                  [${result.ci_lower.toFixed(2)}, ${result.ci_upper.toFixed(2)}]
                </div>
              </div>
              <div className="stat">
                <div className="label">Matched pairs</div>
                <div className="value">{result.n_matched_pairs}</div>
              </div>
            </div>
          </div>

          <div className="card">
            <h2>ATE by method (naive vs. adjusted)</h2>
            <ATEComparison allMethods={result.all_methods} trueAte={result.true_ate} />
          </div>

          <div className="card">
            <h2>Covariate balance (SMD before/after matching)</h2>
            <BalanceTable balance={result.balance} />
          </div>

          <div className="card">
            <h2>CATE by engagement segment</h2>
            <CATEBreakdown cate={result.cate} />
          </div>

          <div className="card">
            <h2>Rosenbaum sensitivity curve</h2>
            <SensitivityChart curve={result.sensitivity_curve} />
          </div>
        </>
      )}
    </div>
  );
}
