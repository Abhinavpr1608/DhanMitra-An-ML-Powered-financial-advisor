// src/components/LiveChart.jsx
import React, { useEffect, useState, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

/**
 * LiveChart component
 *
 * Props:
 *  - symbol (string) e.g. "RELIANCE.NS" or "NIFTY"
 *  - interval (string) e.g. "1min" (used for alpha fallback)
 *  - pollMs (number) milliseconds between polls (default 120000)
 *  - maxPoints (number) number of points to keep for plotting (default 500)
 */
export default function LiveChart({
  symbol = "RELIANCE.NS",
  interval = "1min",
  pollMs = 120000,
  maxPoints = 500,
}) {
  const [data, setData] = useState([]);
  const [meta, setMeta] = useState({ cached: null, source: null });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // UI controls
  const [sourcePref, setSourcePref] = useState("auto"); // "auto" | "alpha" | "yfinance" | "yahoo"
  const mounted = useRef(true);
  const pollRef = useRef(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // generic fetch that can be forced and can respect sourcePref param
  const fetchOnce = async ({ force = false, useSource = null, isInitial = false } = {}) => {
    if (isInitial) setLoading(true);
    setError(null);
    try {
      let url = `${API_BASE}/api/market?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(
        interval
      )}`;
      if (force) url += `&force=1`;
      const effectiveSource = useSource || (sourcePref !== "auto" ? sourcePref : null);
      if (effectiveSource) url += `&source=${encodeURIComponent(effectiveSource)}`;

      const res = await fetch(url, { cache: "no-store" });
      let j;
      try {
        j = await res.json();
      } catch (e) {
        throw new Error(`Invalid JSON response: ${e.message}`);
      }

      if (!res.ok || j.error) {
        const errMsg = j.error || JSON.stringify(j);
        throw new Error(errMsg);
      }

      // parse and normalize data
      const raw = Array.isArray(j.data) ? j.data.slice(-maxPoints) : [];
      const parsed = raw.map((r) => {
        const timeLabel = typeof r.time === "string" ? r.time : String(r.time || "");
        // label for x-axis: try to show HH:MM (last 5-8 chars) or fallback
        let label = timeLabel;
        if (typeof timeLabel === "string") {
          // formats like "2025-09-12 12:16:00" => show "12:16"
          const mm = timeLabel.match(/(\d{2}:\d{2}):\d{2}$/);
          if (mm) label = mm[1];
          else if (timeLabel.length > 10) label = timeLabel.slice(-8, -3);
        }
        return {
          ...r,
          price: Number(r.price),
          label,
        };
      });

      if (mounted.current) {
        setData(parsed);
        setMeta({ cached: !!j.cached, source: j.source || null, cached_at: j.cached_at, age_seconds: j.age_seconds });
      }
    } catch (e) {
      if (mounted.current) {
        setError(String(e.message || e));
      }
    } finally {
      if (isInitial && mounted.current) setLoading(false);
    }
  };

  // initial fetch + start poller
  useEffect(() => {
    // initial fetch
    fetchOnce({ force: false, isInitial: true });

    // set poll interval (minimum 15s)
    const intervalMs = Math.max(15000, pollMs || 120000);
    pollRef.current = setInterval(() => {
      fetchOnce({ force: false });
    }, intervalMs);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, interval, pollMs, sourcePref]); // restart if symbol or pref changes

  // derived values
  const latest = data.length ? data[data.length - 1].price : null;
  const first = data.length ? data[0].price : null;
  const pctChange =
    latest != null && first != null && first !== 0 ? (((latest - first) / first) * 100).toFixed(2) : null;

  // badge styling
  const badgeClass = meta && meta.cached ? "bg-yellow-100 text-yellow-800" : "bg-green-100 text-green-800";
  const sourceText = meta && meta.source ? ` · ${meta.source}` : "";

  // human readable age
  const ageLabel = meta && typeof meta.age_seconds === "number" ? `${Math.floor(meta.age_seconds / 60)}m ago` : null;

  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs text-slate-500">{symbol}</div>
          <div className="text-2xl font-semibold leading-tight">
            {latest != null ? `₹${Number(latest).toLocaleString("en-IN")}` : "—"}
          </div>
          <div className="text-xs text-slate-500">{pctChange ? `${pctChange}%` : ""}</div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            <select
              value={sourcePref}
              onChange={(e) => setSourcePref(e.target.value)}
              className="text-xs border rounded px-2 py-1"
            >
              <option value="auto">Auto</option>
              <option value="yfinance">yfinance</option>
              <option value="yahoo">yahoo</option>
              <option value="alpha">alpha</option>
            </select>

            <button
              onClick={() => fetchOnce({ force: true })}
              className="text-xs bg-gray-900 text-white px-3 py-1 rounded"
              title="Force refresh (bypass cache)"
            >
              Refresh
            </button>
          </div>

          <div className="text-right text-xs">
            <div className="text-xs text-slate-500">{loading ? "Updating…" : error ? "Error" : `Points: ${data.length}`}</div>
            <div className={`mt-1 inline-block px-2 py-1 rounded text-[10px] ${badgeClass}`}>
              {meta && meta.cached ? "cached" : "live"}
              {sourceText}
              {ageLabel ? ` · updated ${ageLabel}` : ""}
            </div>
            {error && <div className="mt-1 text-red-600 text-xs max-w-xs break-words">{error}</div>}
          </div>
        </div>
      </div>

      <div style={{ width: "100%", height: 240 }} className="mt-4">
        {data && data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={20} />
              <YAxis domain={["dataMin", "dataMax"]} tickFormatter={(v) => Number(v).toFixed(0)} />
              <Tooltip
                formatter={(value) => `₹${Number(value).toLocaleString("en-IN")}`}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <Line type="monotone" dataKey="price" stroke="#4f46e5" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-sm text-slate-500">Loading chart…</div>
        )}
      </div>
    </div>
  );
}
