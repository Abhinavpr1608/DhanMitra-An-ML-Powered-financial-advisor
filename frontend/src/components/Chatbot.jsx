import React, { useEffect, useRef, useState } from "react";

// Chatbot.jsx
// Default export React component for DhanMitra chatbot UI
// - Uses Tailwind CSS for styling
// - Stores conversation in localStorage (key: "dhanmitra_chat_history")
// - Sends messages to backend /chatbot
// - Props: apiUrl (optional) - default '/chatbot'

export default function Chatbot({ apiUrl = "/chatbot" }) {
  const [history, setHistory] = useState(() => {
    try {
      const raw = localStorage.getItem("dhanmitra_chat_history");
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  });

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const [warming, setWarming] = useState(true); // new: track warmup
  const bottomRef = useRef(null);

  useEffect(() => {
    // persist history to localStorage
    try {
      localStorage.setItem("dhanmitra_chat_history", JSON.stringify(history));
    } catch (e) {
      console.warn("Could not persist chat history", e);
    }
    // scroll to bottom on new message
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  useEffect(() => {
    // Warm up the backend once on mount (await it and show warming state)
    let mounted = true;
    async function warmup() {
      setWarming(true);
      try {
        // call warmup endpoint (POST /chatbot/warmup)
        const res = await fetch(`${apiUrl}/warmup`, { method: "POST", headers: { "Content-Type": "application/json" } });
        if (!res.ok) {
          console.warn("Warmup failed (non-OK)", res.status);
        } else {
          const j = await res.json();
          if (j && j.loaded === false) {
            console.warn("Warmup returned loaded=false", j);
          }
        }
      } catch (e) {
        console.warn("Warmup error (ignored):", e);
      } finally {
        if (mounted) setWarming(false);
      }
    }
    warmup();
    return () => { mounted = false; };
  }, [apiUrl]);

  function appendMessage(role, text) {
    const msg = {
      id: Date.now() + Math.random().toString(36).slice(2, 8),
      role,
      content: text,
      ts: new Date().toISOString(),
    };
    setHistory((h) => [...h, msg]);
    return msg;
  }

  async function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed || isSending || warming) return;
    setError(null);
    setIsSending(true);

    // append user message locally first (optimistic)
    const userMsg = appendMessage("user", trimmed);
    setInput("");

    // prepare payload: include conversation for context
    const payload = {
      message: trimmed,
      history: history.concat([userMsg]).slice(-12), // send last 12 turns
    };

    // increase timeout — 120s (120000 ms). If you expect longer, set to 300000.
    const TIMEOUT_MS = 120000;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server ${res.status}: ${text}`);
      }
      const j = await res.json();
      const replyText = j.reply || j.answer || j.output || "(no reply)";
      const backend = j.backend ? ` (${j.backend})` : "";
      appendMessage("assistant", replyText + backend);
    } catch (err) {
      console.error("Chat send error", err);
      setError(
        err.name === "AbortError"
          ? `Request timed out after ${TIMEOUT_MS/1000}s — the model may still be loading or CPU generation is slow. Try again or warm up first.`
          : `Chat failed: ${err.message}`
      );
      // on error, append a small assistant error message
      appendMessage("assistant", "⚠️ Sorry — I couldn't reach the model. Try warming up or increasing timeout.");
    } finally {
      setIsSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      // ctrl+enter to insert newline
      setInput((v) => v + "\n");
    } else if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  }

  function clearHistory() {
    setHistory([]);
    try {
      localStorage.removeItem("dhanmitra_chat_history");
    } catch (e) {
      /* ignore */
    }
  }

  return (
    <div className="max-w-3xl mx-auto bg-white shadow-md rounded-2xl overflow-hidden border border-gray-100">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-gradient-to-br from-indigo-500 to-sky-400 flex items-center justify-center text-white font-semibold">DM</div>
          <div>
            <div className="text-sm font-semibold">DhanMitra</div>
            <div className="text-xs text-gray-500">Local finance assistant</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="text-xs text-gray-500">
            {warming ? <span>Warming model…</span> : <span>Ready</span>}
          </div>
          <button
            onClick={clearHistory}
            className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200 text-gray-700"
            title="Clear chat history"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="p-4 h-[60vh] overflow-y-auto" style={{ background: "linear-gradient(180deg,#fbfdff,#ffffff)" }}>
        <div className="space-y-4">
          {history.length === 0 && (
            <div className="text-sm text-gray-500">Ask me about SIPs, FDs, mutual funds or your financial goals.</div>
          )}

          {history.map((m) => (
            <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[78%] p-3 rounded-lg ${m.role === "user" ? "bg-indigo-50 text-right" : "bg-gray-50"}`}>
                <div className="text-sm whitespace-pre-wrap">{m.content}</div>
                <div className="text-[10px] text-gray-400 mt-2">{new Date(m.ts).toLocaleString()}</div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="px-4 py-3 border-t bg-white">
        {error && <div className="text-xs text-red-600 mb-2">{error}</div>}
        <div className="flex items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={warming ? "Warming model — please wait..." : isSending ? "Waiting for reply..." : "Type your question — press Enter to send"}
            rows={2}
            className="flex-1 resize-none rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
            disabled={isSending || warming}
          />
          <div className="flex flex-col gap-2">
            <button
              onClick={sendMessage}
              disabled={isSending || warming}
              className={`px-4 py-2 rounded-md text-sm font-semibold text-white ${isSending || warming ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}
            >
              {warming ? "Warming…" : isSending ? "Sending..." : "Send"}
            </button>
            <button
              onClick={() => { setInput(""); setError(null); }}
              className="text-xs text-gray-500"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
