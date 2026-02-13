// src/pages/Market.jsx
import React, { useState } from "react";
import LiveChart from "../components/LiveChart.jsx";

/**
 * Market page
 * - shows three LiveChart cards (NIFTY, RELIANCE.NS, ADANIENT.NS)
 * - global controls for poll interval and refresh-all
 * - each chart still has its own source selector + refresh
 *
 * Paste to: src/pages/Market.jsx
 */
export default function Market() {
  // poll interval in ms (default 2 minutes)
  const [pollMs, setPollMs] = useState(120000);
  // small nonce used to force remount / refresh of all charts
  const [refreshAllToken, setRefreshAllToken] = useState(0);
  // layout: grid columns
  const [cols, setCols] = useState(3);

  const handleRefreshAll = () => {
    // bump token -> LiveChart components use it indirectly via `key` to remount
    setRefreshAllToken((t) => t + 1);
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6 gap-4">
          <div>
            <h1 className="text-2xl font-bold">Markets</h1>
            <p className="text-sm text-slate-500">Live market charts (NIFTY / Reliance / Adani) — cache-aware and source-selectable.</p>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-slate-600">
              Poll interval:
              <select
                value={pollMs}
                onChange={(e) => setPollMs(Number(e.target.value))}
                className="ml-2 text-sm border rounded px-2 py-1"
                title="How often charts poll the backend"
              >
                <option value={300000}>5m</option>
                <option value={180000}>3m</option>
                <option value={120000}>2m (default)</option>
                <option value={60000}>1m</option>
                <option value={30000}>30s</option>
              </select>
            </label>

            <label className="text-xs text-slate-600">
              Columns:
              <select
                value={cols}
                onChange={(e) => setCols(Number(e.target.value))}
                className="ml-2 text-sm border rounded px-2 py-1"
                title="Number of charts per row"
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </label>

            <button
              onClick={handleRefreshAll}
              className="bg-gray-900 text-white px-3 py-2 rounded-md text-sm shadow"
              title="Force refresh all charts (bypass cache)"
            >
              Refresh all
            </button>
          </div>
        </div>

        <div className={`grid gap-4 ${cols === 1 ? "grid-cols-1" : cols === 2 ? "grid-cols-2" : "grid-cols-3"}`}>
          <div key={`NIFTY-${refreshAllToken}`}>
            <LiveChart
              symbol="NIFTY"
              interval="1min"
              pollMs={pollMs}
              maxPoints={500}
            />
          </div>

          <div key={`RELIANCE.NS-${refreshAllToken}`}>
            <LiveChart
              symbol="RELIANCE.NS"
              interval="1min"
              pollMs={pollMs}
              maxPoints={500}
            />
          </div>

          <div key={`ADANIENT.NS-${refreshAllToken}`}>
            <LiveChart
              symbol="ADANIENT.NS"
              interval="1min"
              pollMs={pollMs}
              maxPoints={500}
            />
          </div>
        </div>

        <div className="mt-6 bg-white p-4 rounded-lg shadow-sm">
          <h2 className="text-lg font-semibold mb-2">Notes</h2>
          <ul className="list-disc ml-6 text-sm text-slate-600">
            <li>Charts default to using the backend's preferred source (yfinance → yahoo → alpha).</li>

            <li>
              Use the chart-level Source selector or the Refresh All button to force alpha or bypass cache. Alpha has strict rate limits (free tier).
            </li>
            <li>Polling too frequently may cause upstream rate-limiting. Prefer 2–5 minute poll intervals for multiple charts.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
