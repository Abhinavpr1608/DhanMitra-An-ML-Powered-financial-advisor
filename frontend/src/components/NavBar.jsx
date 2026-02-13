import React from "react";
import { Link, useLocation } from "react-router-dom";

const items = [
  { to: "/", label: "Home" },
  { to: "/markets", label: "Markets" },
  { to: "/portfolio", label: "Portfolio" },
  { to: "/chatbot", label: "Chatbot" },
];

export default function NavBar() {
  const loc = useLocation();
  return (
    <header className="bg-white/80 backdrop-blur sticky top-0 z-40 border-b">
      <div className="container-max mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-600 to-violet-500 flex items-center justify-center text-white font-bold shadow-sm">DM</div>
          <div>
            <div className="text-lg font-bold text-slate-900">DhanMitra</div>
            <div className="text-xs text-slate-500 -mt-1">AI Financial Advisor</div>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-3">
          {items.map((it) => (
            <Link
              key={it.to}
              to={it.to}
              className={`px-3 py-2 rounded-lg text-sm ${loc.pathname === it.to ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}
            >
              {it.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <a className="text-sm text-slate-500 hidden sm:inline">Signed out</a>
          <Link to="/chatbot" className="btn-primary text-sm">Chat with assistant</Link>
        </div>
      </div>
    </header>
  );
}
