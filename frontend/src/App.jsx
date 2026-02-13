// src/App.jsx
import React, { useState } from "react";
import Markets from "./pages/Markets";
import PlanCard from "./components/PlanCard.jsx";
import Chatbot from "./components/Chatbot.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function RecommendForm() {
  const [form, setForm] = useState({
    income: 60000,
    expenses: 35000,
    goal_amount: 200000,
    duration_months: 24,
    risk_level: "medium",
    age: 30, // optional — helps model decide Gold/FD etc.
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]:
        name === "risk_level"
          ? value
          : value === ""
          ? ""
          : name === "age"
          ? Math.max(18, Math.min(100, Number(value)))
          : Number(value),
    }));
  };

  const formatCurrency = (n) => {
    try {
      const num = Number(n || 0);
      return num.toLocaleString("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 });
    } catch {
      return "₹" + (n || 0);
    }
  };

const submit = async () => {
  setError(null);
  setResult(null);

  // --- validation checks before sending ---
  if (form.income == undefined)
{
  setError("Income filed cannot be blank.");
}
  if (form.income == 0) {
    setError("Income must be greater than 0.");
    return;
  }
  if (form.income < 0) {
    setError("Income cannot be negative.");
    return;
  }
  if (form.income <= form.expenses) {
    setError("Income should be greater than expenses.");
    return;
  }
  if (form.goal_amount <= 0) {
    setError("Goal amount must be greater than 0.");
    return;
  }
  if (form.duration_months <= 0) {
    setError("Duration must be at least 1 month.");
    return;
  }

  setLoading(true);
  try {
    const payload = {
      income: Number(form.income) || undefined,
      expenses: Number(form.expenses) || 0,
      goal_amount: Number(form.goal_amount) || 0,
      duration_months: Math.max(1, Math.floor(Number(form.duration_months) || 1)),
      risk_level: form.risk_level || "medium",
      age: form.age ? Number(form.age) : undefined,
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

    console.info("Received /recommend result:", data);
    window.__APP_RESULT = data;
    setResult(data);
  } catch (err) {
    console.error("Recommend failed:", err);
    setError(err.message || "Request failed");
  } finally {
    setLoading(false);
  }
};


  const reset = () => {
    setForm({ income: 60000, expenses: 35000, goal_amount: 200000, duration_months: 24, risk_level: "medium", age: 30 });
    setResult(null);
    setError(null);
  };

  return (
    <div className="container-max mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-6">
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

            <label className="block">
              <div className="text-sm text-slate-600">Age (optional)</div>
              <input name="age" value={form.age} onChange={onChange} type="number" className="w-full border rounded-lg p-2 mt-1" />
              <div className="text-xs text-slate-400 mt-1">Providing age helps the model recommend age-appropriate products (e.g., Gold/FD).</div>
            </label>

            <div className="flex gap-3 mt-3">
              <button onClick={submit} disabled={loading} className="btn-primary">{loading ? "Thinking..." : "Recommend"}</button>
              <button onClick={reset} className="px-4 py-2 rounded-lg border">Reset</button>
            </div>

            {error && <div className="text-sm text-red-600 mt-2">{error}</div>}
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="card p-6">
            {/* Use PlanCard to render the recommendation (handles pie + SHAP) */}
            {!result ? (
              <div>
                <h3 className="text-lg font-semibold">Plan preview</h3>
                <p className="text-sm text-slate-500 mt-2">After you click Recommend, results will appear here.</p>
              </div>
            ) : (
              <PlanCard data={result} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("recommend");

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b sticky top-0 z-40">
        <div className="container-max mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-600 to-violet-500 flex items-center justify-center text-white font-bold shadow-sm">DM</div>
            <div>
              <div className="text-lg font-bold">DhanMitra</div>
              <div className="text-xs text-slate-500 -mt-1">AI Financial Advisor</div>
            </div>
          </div>

          <nav className="flex items-center gap-3">
            <button onClick={() => setTab("recommend")} className={`px-3 py-2 rounded-lg text-sm ${tab === "recommend" ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}>Recommendation</button>
            <button onClick={() => setTab("markets")} className={`px-3 py-2 rounded-lg text-sm ${tab === "markets" ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}>Markets</button>
            <button onClick={() => setTab("chat")} className={`px-3 py-2 rounded-lg text-sm ${tab === "chat" ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}>Chat</button>
          </nav>
        </div>
      </header>

      <main>
        {tab === "recommend" && <RecommendForm />}
        {tab === "markets" && <Markets />}
        {tab === "chat" && (
          <div className="container-max mx-auto px-4 py-8">
            <div className="card p-6">
              <h2 className="text-lg font-semibold mb-4">Chat with DhanMitra</h2>
              <Chatbot apiUrl={`${API_BASE}/chatbot`} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
