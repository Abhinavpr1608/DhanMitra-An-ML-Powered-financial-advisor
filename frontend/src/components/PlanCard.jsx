// src/components/PlanCard.jsx
import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

const COLORS = [
  "#4f46e5","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4",
  "#f97316","#84cc16","#ec4899","#3b82f6"
];

const formatCurrency = (n) => {
  try {
    return Number(n).toLocaleString("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 });
  } catch {
    return "₹" + n;
  }
};

export default function PlanCard({ data }) {
  if (!data || !data.recommendation) {
    return (
      <div className="card p-6">
        <h3 className="text-lg font-semibold">No recommendation yet</h3>
        <p className="text-sm text-slate-500 mt-2">Use the form to generate a personalized investment plan.</p>
      </div>
    );
  }

  const rec = data.recommendation;
  const suggested = Number(rec.suggested_monthly_investment) || 0;
  const detailedRaw = (rec.allocation_breakdown && rec.allocation_breakdown.detailed) || [];

  // Build safeDetailed (bucket, fraction, rupees)
  let safeDetailed = [];
  if (Array.isArray(detailedRaw) && detailedRaw.length > 0) {
    safeDetailed = detailedRaw.map((d, i) => ({
      bucket: d.bucket || `Bucket ${i+1}`,
      fraction: Number(d.fraction) || 0,
      rupees: Number(d.rupees != null ? d.rupees : (d.amount || 0)) || Math.round(((Number(d.fraction) || 0) * suggested) * 100) / 100,
      color: COLORS[i % COLORS.length]
    }));
  } else {
    const fr = rec.allocation_breakdown && rec.allocation_breakdown.fractions;
    const br = rec.allocation_breakdown && rec.allocation_breakdown.by_rupees;
    if (fr && Object.keys(fr).length > 0) {
      safeDetailed = Object.keys(fr).map((k, i) => ({
        bucket: k,
        fraction: Number(fr[k]) || 0,
        rupees: br && br[k] !== undefined ? Number(br[k]) || 0 : Math.round(((Number(fr[k]) || 0) * suggested) * 100) / 100,
        color: COLORS[i % COLORS.length]
      }));
    } else if (br && Object.keys(br).length > 0) {
      safeDetailed = Object.keys(br).map((k, i) => ({
        bucket: k,
        fraction: 0,
        rupees: Number(br[k]) || 0,
        color: COLORS[i % COLORS.length]
      }));
    }
  }

  // fallback equal split so pie always has something
  if (!safeDetailed || safeDetailed.length === 0 || safeDetailed.every(d => (!d.rupees && !d.fraction))) {
    const fallback = ["Equity", "Debt", "Gold", "Liquid"];
    safeDetailed = fallback.map((b, i) => ({
      bucket: b,
      fraction: 1 / fallback.length,
      rupees: Math.round((suggested * (1 / fallback.length)) * 100) / 100,
      color: COLORS[i % COLORS.length]
    }));
  }

  // Derive fractions from rupees if necessary and normalize
  const totalRupees = safeDetailed.reduce((s, x) => s + (Number(x.rupees) || 0), 0);
  let totalFraction = safeDetailed.reduce((s, x) => s + (Number(x.fraction) || 0), 0);
  if (totalFraction === 0 && totalRupees > 0) {
    safeDetailed = safeDetailed.map(d => ({ ...d, fraction: Number(d.rupees) / (totalRupees || 1) }));
    totalFraction = safeDetailed.reduce((s, x) => s + (Number(x.fraction) || 0), 0);
  }
  if (totalFraction > 0) {
    safeDetailed = safeDetailed.map(d => ({ ...d, fraction: Number(d.fraction) / totalFraction }));
  }

  // Prepare pieData: prefer rupees if >0, else use fraction
  const useRupees = safeDetailed.some(d => Number(d.rupees) > 0);
  const pieData = useRupees
    ? safeDetailed.map(d => ({ name: d.bucket, value: Number(d.rupees) || 0 }))
    : safeDetailed.map(d => ({ name: d.bucket, value: Number(d.fraction) || 0 }));

  const pieTotal = pieData.reduce((s, p) => s + (Number(p.value) || 0), 0) || 1;
  const centerLabel = useRupees ? formatCurrency(pieTotal) : `${(pieTotal * 100).toFixed(1)}%`;

  // Debug: uncomment if needed
  // console.log("PlanCard pieData:", pieData, "useRupees:", useRupees, "suggested:", suggested);

  return (
    <div className="card p-6 bg-white rounded-2xl shadow">
      <div className="flex gap-6 flex-col lg:flex-row">
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 className="text-xl font-semibold">Personalized Plan</h3>
          <div className="text-3xl font-extrabold mt-3">{formatCurrency(suggested)}</div>
          <div className="text-sm text-slate-500 mt-1">
            Expected CAGR <strong className="text-slate-800">{typeof rec.expected_cagr === "number" ? `${(rec.expected_cagr * 100).toFixed(2)}%` : "N/A"}</strong>
          </div>

          <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 bg-slate-50 rounded-full text-sm text-slate-700">
            Product: <span className="font-medium ml-2">{rec.meta?.product_class || "N/A"}</span>
          </div>

          <div className="mt-6">
            <h4 className="text-sm text-slate-600 mb-2">Detailed split</h4>
            <div className="space-y-2">
              {safeDetailed.map((d, i) => (
                <div key={i} className="flex justify-between items-center">
                  <div className="flex items-center gap-3 min-w-0">
                    <span style={{ width: 12, height: 12, background: d.color, display: "inline-block", borderRadius: 3 }} />
                    <div className="text-sm truncate">{d.bucket}</div>
                  </div>
                  <div className="text-sm text-slate-700">{(Number(d.fraction) * 100).toFixed(1)}% • {formatCurrency(d.rupees)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={{ width: 300, minWidth: 320 }}>
          <div style={{ width: "100%", height: 360, display: "flex", alignItems: "center", justifyContent: "center" }}>
            {pieData && pieData.length > 0 ? (
              <PieChart width={360} height={360}>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx={200}
                  cy={170}
                  outerRadius={130}
                  innerRadius={70}
                  paddingAngle={2}
                  // Use the percent argument Recharts passes to the label renderer
                  label={({ percent }) => `${(percent * 100).toFixed(1)}%`}
                  labelLine={false}
                >
                  {pieData.map((entry, idx) => <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />)}
                </Pie>
                <Tooltip formatter={(val) => (useRupees ? formatCurrency(val) : `${(val * 100).toFixed(2)}%`)} />
                <Legend verticalAlign="bottom" height={36} />
                {/* center label (text inside SVG) */}
                <text x="55%" y="50%" textAnchor="middle" dominantBaseline="middle" style={{ fontSize: 12, fontWeight: 600, transform: "translateY(-6px)" }}>
                  {centerLabel}
                </text>
              </PieChart>
            ) : (
              <div className="text-sm text-slate-400">No allocation</div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6">
        <h4 className="text-sm text-slate-600 mb-2">Top factors (SHAP)</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          {(rec.meta?.explanation || []).length > 0 ? (
            (rec.meta.explanation || []).map((s, i) => (
              <div key={i} className="p-2 rounded bg-slate-50">
                <div className="text-sm text-slate-700 font-medium truncate">{s.feature}</div>
                <div className="text-xs text-slate-500 mt-1">
                  {s.impact_pct != null ? `${s.impact_pct}%` : `${(s.impact || 0) * 100}%`}{" "}
                  {s.direction === "pos" ? "↑" : s.direction === "neg" ? "↓" : ""}
                </div>
              </div>
            ))
          ) : (
            <div className="text-sm text-slate-400">No explanation available</div>
          )}
        </div>
      </div>
    </div>
  );
}
