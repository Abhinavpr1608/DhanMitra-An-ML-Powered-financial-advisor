import React, { useState } from "react";
import PlanCard from "../components/PlanCard";
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function Home() {
  const [form, setForm] = useState({
    income: 60000,
    expenses: 35000,
    goal_amount: 200000,
    duration_months: 24,
    risk_level: "medium",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: name === "risk_level" ? value : value === "" ? "" : Number(value),
    }));
  };

  const submit = async () => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const payload = {
        income: Number(form.income) || undefined,
        expenses: Number(form.expenses) || 0,
        goal_amount: Number(form.goal_amount) || 0,
        duration_months: Math.max(1, Math.floor(Number(form.duration_months) || 1)),
        risk_level: form.risk_level || "medium",
      };

      const r = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await r.text();
      if (!r.ok) {
        try {
          const parsed = JSON.parse(text);
          throw new Error(parsed.error || parsed.message || JSON.stringify(parsed));
        } catch {
          throw new Error(text || `Server returned ${r.status}`);
        }
      }
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
      setResult(data);
    } catch (err) {
      console.error("Recommend failed:", err);
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setForm({ income: 60000, expenses: 35000, goal_amount: 200000, duration_months: 24, risk_level: "medium" });
    setResult(null);
    setError(null);
  };

  return (
    <div className="container-max mx-auto px-4">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 card p-6">
          <h2 className="text-lg font-semibold">Get a Recommendation</h2>
          <p className="text-sm text-slate-500 mt-1">Fill the fields to get a personalised monthly investment plan.</p>

          <div className="mt-4 space-y-3">
            <label className="block">
              <div className="text-sm text-slate-600">Income</div>
              <input name="income" value={form.income} onChange={onChange} type="number" className="w-full border rounded-lg p-2 mt-1" />
            </label>

            <label className="block">
              <div className="text-sm text-slate-600">Expenses</div>
              <input name="expenses" value={form.expenses} onChange={onChange} type="number" className="w-full border rounded-lg p-2 mt-1" />
            </label>

            <label className="block">
              <div className="text-sm text-slate-600">Goal Amount</div>
              <input name="goal_amount" value={form.goal_amount} onChange={onChange} type="number" className="w-full border rounded-lg p-2 mt-1" />
            </label>

            <label className="block">
              <div className="text-sm text-slate-600">Duration (months)</div>
              <input name="duration_months" value={form.duration_months} onChange={onChange} type="number" className="w-full border rounded-lg p-2 mt-1" />
            </label>

            <label className="block">
              <div className="text-sm text-slate-600">Risk level</div>
              <select name="risk_level" value={form.risk_level} onChange={onChange} className="w-full border rounded-lg p-2 mt-1">
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </label>

            <div className="flex gap-3 mt-3">
              <button onClick={submit} disabled={loading} className="btn-primary">{loading ? "Thinking..." : "Recommend"}</button>
              <button onClick={reset} className="px-4 py-2 rounded-lg border">Reset</button>
            </div>

            {error && <div className="text-sm text-red-600 mt-2">{error}</div>}
          </div>
        </div>

        <div className="lg:col-span-2">
          <PlanCard data={result} />
        </div>
      </div>
    </div>
  );
}
