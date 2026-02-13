# app.py — patched: timestamped cache + yfinance / yahoo / alpha fallback + LOCAL FINANCE CHATBOT
import os
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import torch
torch.set_num_threads(6)  # try 4–8; pick half your logical cores

# ensure .env is loaded from backend folder
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

import requests

# yfinance can be optional — guard import
try:
    import yfinance as yf
except Exception:
    yf = None

# import ML model (adjust if module path differs)
from model import DhanMitraModel

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)

# configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("app")

# >>> finance system prompt used by chatbot (env can override via CHATBOT_SYSTEM_PROMPT)
FINANCE_SYSTEM = (
    "You are DhanMitra, a careful Indian finance assistant. "
    "Explain SIPs, FDs, mutual funds, basic asset allocation, diversification, and goal planning. "
    "Be educational, avoid personalized financial/tax advice, avoid guarantees, and include a short disclaimer."
)

# --- config and env vars (support different env names) ---
ALPHA_KEY = (
    os.environ.get("ALPHA_VANTAGE_KEY")
    or os.environ.get("ALPHA_VANTAGE_API_KEY")
    or os.environ.get("ALPHA_KEY")
)
MARKET_CACHE_TTL = int(os.environ.get("MARKET_CACHE_TTL", "300"))  # seconds
DEBUG_LOG = os.environ.get("DM_DEBUG", "1").lower() in ("1", "true", "yes")

# >>> CHATBOT selection
CHATBOT_BACKEND = (os.environ.get("CHATBOT_BACKEND") or "local-finance").strip().lower()

# --- in-memory cache structure
MARKET_CACHE = {}

# --- load ML model
ML_MODEL = DhanMitraModel()
MODEL_READY = ML_MODEL.load()
MODEL_LOADED_AT = time.time() if MODEL_READY else None

# --- utility converters
def to_plain_python(obj):
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_pd.Series, _pd.Index)):
        return obj.tolist()
    if isinstance(obj, _pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {to_plain_python(k): to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_python(x) for x in obj]
    return obj

# --- helper: build cache key
def _cache_key(symbol, interval, pref_source):
    ps = (pref_source or "auto").lower()
    return f"{symbol}::{interval}::{ps}"

def _cache_get_full(cache_key):
    rec = MARKET_CACHE.get(cache_key)
    if not rec:
        return None
    age = time.time() - rec["ts"]
    if age > MARKET_CACHE_TTL:
        MARKET_CACHE.pop(cache_key, None)
        return None
    return rec

def _cache_set(cache_key, data, source):
    now = time.time()
    MARKET_CACHE[cache_key] = {
        "ts": now,
        "cached_at": datetime.utcnow().replace(tzinfo=timezone.utc),
        "data": data,
        "source": source,
    }

# --- simple fallback/recommendation (unchanged logic) ---
def simple_recommendation(payload):
    try:
        income = float(payload.get("income", 0))
        expenses = float(payload.get("expenses", 0))
        goal_amount = float(payload.get("goal_amount", 0))
        duration_months = int(payload.get("duration_months", 12))
        risk_level = str(payload.get("risk_level", "medium")).lower().strip()
    except Exception as e:
        return {"error": f"Invalid input: {e}"}, 400

    savings = max(income - expenses, 0)
    savings_rate = savings / income if income > 0 else 0.0

    if risk_level in ["low", "low-risk", "conservative"]:
        allocation = {"FD": 0.70, "RD_or_ShortDebtMF": 0.30}
        expected_cagr = 0.06
    elif risk_level in ["high", "high-risk", "aggressive"]:
        allocation = {"EquitySIP": 0.80, "Hybrid_Conservative": 0.20}
        expected_cagr = 0.12
    else:
        allocation = {"Hybrid_Balanced": 0.60, "DebtSIP_or_FD": 0.40}
        expected_cagr = 0.09

    monthly_target = goal_amount / max(duration_months, 1)
    suggested_monthly = max(min(savings, monthly_target * 1.1), 0)
    breakdown = {k: round(v * suggested_monthly, 2) for k, v in allocation.items()}

    plan = {
        "inputs": {
            "income": income,
            "expenses": expenses,
            "savings": savings,
            "savings_rate": round(savings_rate, 3),
            "goal_amount": goal_amount,
            "duration_months": duration_months,
            "risk_level": risk_level,
        },
        "recommendation": {
            "suggested_monthly_investment": round(suggested_monthly, 2),
            "allocation_breakdown": breakdown,
            "expected_cagr": expected_cagr,
            "notes": [
                "Starter logic only. Replace with ML model + live API data.",
                "Tune allocations after adding actual products and risk scoring."
            ]
        }
    }
    return plan, 200

# --- endpoints ---
@app.get("/health")
def health():
    info = {"status": "ok", "ml_model_loaded": MODEL_READY}
    try:
        if MODEL_READY and ML_MODEL and ML_MODEL.classifier is not None:
            try:
                prep, est = ML_MODEL._safe_get_prep_and_model(ML_MODEL.classifier)
            except Exception:
                prep, est = (None, ML_MODEL.classifier)
            try:
                classes = getattr(est, "classes_", None)
                info["model_classes"] = to_plain_python(list(classes)) if classes is not None else None
            except Exception:
                info["model_classes"] = None
            try:
                info["feature_cols"] = to_plain_python(ML_MODEL.feature_cols)
            except Exception:
                info["feature_cols"] = None
            info["model_loaded_at"] = MODEL_LOADED_AT
    except Exception:
        pass
    info["alpha_key_present"] = bool(ALPHA_KEY)
    info["market_cache_ttl"] = MARKET_CACHE_TTL

    # >>> chatbot health
    cb = {"backend": CHATBOT_BACKEND}
    try:
        cb.update(get_chatbot_status())  # defined below
    except Exception:
        cb["loaded"] = False
    info["chatbot"] = cb
    return jsonify(info)

@app.post("/recommend")
def recommend():
    payload = request.get_json(force=True, silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body required"}), 400
    if MODEL_READY:
        try:
            prediction = ML_MODEL.predict(payload)
            clean = to_plain_python(prediction)
            return jsonify(clean), 200
        except Exception:
            app.logger.exception("ML prediction failed")
            out, code = simple_recommendation(payload)
            return jsonify(to_plain_python(out)), code
    else:
        out, code = simple_recommendation(payload)
        return jsonify(to_plain_python(out)), code

@app.post("/reload-model")
def reload_model():
    global MODEL_READY, ML_MODEL, MODEL_LOADED_AT
    try:
        ok = ML_MODEL.load()
        MODEL_READY = bool(ok)
        MODEL_LOADED_AT = time.time() if MODEL_READY else None
        return jsonify({"reloaded": MODEL_READY}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"reloaded": False, "error": str(e)}), 500

@app.get("/api/cache-stats")
def cache_stats():
    out = {}
    for k, v in MARKET_CACHE.items():
        age = int(time.time() - v["ts"])
        out[k] = {"age_seconds": age, "cached_at": v["cached_at"].isoformat(), "source": v["source"], "points": len(v["data"]) }
    return jsonify({"cache": out, "count": len(MARKET_CACHE)}), 200

# --- robust yfinance history helper (tolerant to signature differences) ---
def fetch_yfinance_history(ticker_symbol, period="1d", interval="1m"):
    """
    Try multiple variants of yf.Ticker(...).history(...) to handle differing yfinance signatures.
    Returns DataFrame or None. Raises last exception if needed.
    """
    if yf is None:
        raise RuntimeError("yfinance not available in environment")

    ticker = yf.Ticker(ticker_symbol)
    last_exc = None

    candidates = [
        {"period": period, "interval": interval, "progress": False, "threads": False},
        {"period": period, "interval": interval, "progress": False},
        {"period": period, "interval": interval, "threads": False},
        {"period": period, "interval": interval},
    ]

    for params in candidates:
        try:
            if DEBUG_LOG:
                logger.debug("yfinance.history try with params: %s", params)
            df = ticker.history(**params)
            return df
        except TypeError as te:
            last_exc = te
            if DEBUG_LOG:
                logger.debug("yfinance.history TypeError with params %s: %s", params, te)
            continue
        except Exception as e:
            last_exc = e
            if DEBUG_LOG:
                logger.warning("yfinance.history attempt raised %s with params %s: %s", type(e).__name__, params, e)
            continue

    if last_exc is not None:
        raise last_exc
    return None

# --- market proxy helpers (yfinance/yahoo/alpha) ---
def try_yfinance(symbol):
    if yf is None:
        if DEBUG_LOG:
            logger.debug("yfinance not installed")
        return None, "yfinance_not_installed"

    yf_sym = "^NSEI" if symbol.upper() in ("NIFTY", "NIFTY50", "NSEI", "^NSEI") else symbol
    attempts = [("1d", "1m"), ("5d", "5m"), ("30d", "60m")]
    for period, intr in attempts:
        try:
            df = fetch_yfinance_history(yf_sym, period=period, interval=intr)
            if df is None or getattr(df, "empty", True):
                if DEBUG_LOG:
                    logger.debug("yfinance returned empty for %s %s %s", yf_sym, period, intr)
                continue

            data = []
            try:
                for idx, row in df.iterrows():
                    if hasattr(idx, "tzinfo") and idx.tzinfo is not None:
                        ts = idx.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        ts = idx.strftime("%Y-%m-%d %H:%M:%S")
                    close = None
                    if hasattr(row, "get"):
                        close = row.get("Close") or row.get("close") or row.get("Adj Close") or row.get("AdjClose")
                    else:
                        try:
                            close = float(row[3])
                        except Exception:
                            close = None
                    if close is None:
                        continue
                    try:
                        close_f = float(close)
                        if close_f != close_f:  # NaN
                            continue
                    except Exception:
                        continue
                    data.append({"time": ts, "price": float(close_f)})
            except Exception as e_iter:
                if DEBUG_LOG:
                    logger.debug("yfinance row iteration failed: %s", e_iter)
                continue

            if data:
                return data, None
        except Exception as e:
            if DEBUG_LOG:
                logger.debug("yfinance attempt failed %s %s: %s", period, intr, e)
            continue

    return None, "yfinance_failed"

def try_yahoo(symbol):
    try:
        yahoo_sym = "%5ENSEI" if symbol.upper() in ("NIFTY", "NIFTY50", "NSEI", "^NSEI") else symbol
        end = int(time.time())
        start = end - (60 * 60 * 24)  # last 24 hours
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}?period1={start}&period2={end}&interval=1m"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 429:
            return None, "yahoo_429"
        r.raise_for_status()
        j = r.json()
        chart = j.get("chart", {})
        result = chart.get("result")
        if not result:
            return None, "yahoo_no_result"
        result = result[0]
        timestamps = result.get("timestamp", [])
        indicators = result.get("indicators", {}).get("quote", [])
        if not indicators:
            return None, "yahoo_no_quote"
        closes = indicators[0].get("close", [])
        data = []
        for ts, c in zip(timestamps, closes):
            if c is None:
                continue
            t_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            data.append({"time": t_iso, "price": float(c)})
        return data if data else None, None
    except Exception as e:
        if DEBUG_LOG:
            logger.debug("yahoo exception: %s", str(e))
        return None, "yahoo_exception"

def try_alpha(symbol, interval):
    if not ALPHA_KEY:
        return None, "no_alpha_key"
    try:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "compact",
            "apikey": ALPHA_KEY,
        }
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
        j = r.json()
        if "Note" in j:
            return None, "alpha_note"
        if "Error Message" in j:
            return None, "alpha_error"
        key = f"Time Series ({interval})"
        ts_dict = j.get(key) or j.get("Time Series (1min)") or None
        if ts_dict is None:
            for k, v in j.items():
                if isinstance(v, dict) and v:
                    first_key = next(iter(v.keys()))
                    if isinstance(first_key, str) and (" " in first_key or "-" in first_key):
                        ts_dict = v
                        break
        if not ts_dict:
            return None, "no_timeseries"
        data = []
        for tstamp, vals in sorted(ts_dict.items()):
            close = vals.get("4. close") or vals.get("close")
            if close is None:
                continue
            data.append({"time": tstamp, "price": float(close)})
        return data if data else None, None
    except Exception as e:
        if DEBUG_LOG:
            logger.debug("alpha exception: %s", str(e))
        return None, "alpha_exception"

# --- /api/market endpoint (with timestamped cache fields) ---
@app.get("/api/market")
def proxy_market():
    """
    Query params:
      - symbol (required)
      - interval (optional) default "1min"
      - force=1 to bypass cache
      - source=alpha|yfinance|yahoo to prefer a source
    Response always contains:
      - symbol, data, source, cached (bool), cached_at (ISO UTC), age_seconds (int)
    """
    symbol = (request.args.get("symbol") or "").strip()
    interval = request.args.get("interval", "1min").strip()
    force = request.get_json(silent=True) if False else request.args.get("force", "0") # no-op for IDE hints
    force = request.args.get("force", "0").lower() in ("1", "true", "yes")
    pref_source = (request.args.get("source") or "").lower().strip()  # "alpha"|"yfinance"|"yahoo"|"" 

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    cache_key = _cache_key(symbol, interval, pref_source)
    if not force:
        cached_full = _cache_get_full(cache_key)
        if cached_full:
            age = int(time.time() - cached_full["ts"])
            return jsonify({
                "symbol": symbol,
                "data": cached_full["data"],
                "cached": True,
                "cached_at": cached_full["cached_at"].isoformat(),
                "age_seconds": age,
                "source": cached_full["source"]
            }), 200

    tried = {}

    def set_and_return(data, source):
        _cache_set(cache_key, data, source)
        rec = MARKET_CACHE[cache_key]
        return jsonify({
            "symbol": symbol,
            "data": data,
            "cached": False,
            "cached_at": rec["cached_at"].isoformat(),
            "age_seconds": 0,
            "source": source
        }), 200

    # Decide source order
    if pref_source == "alpha":
        data, err = try_alpha(symbol, interval); tried["alpha"] = err
        if data: return set_and_return(data, "alpha")
        data, err = try_yfinance(symbol); tried["yfinance"] = err
        if data: return set_and_return(data, "yfinance")
        data, err = try_yahoo(symbol); tried["yahoo"] = err
        if data: return set_and_return(data, "yahoo")
    elif pref_source == "yahoo":
        data, err = try_yahoo(symbol); tried["yahoo"] = err
        if data: return set_and_return(data, "yahoo")
        data, err = try_yfinance(symbol); tried["yfinance"] = err
        if data: return set_and_return(data, "yfinance")
        data, err = try_alpha(symbol, interval); tried["alpha"] = err
        if data: return set_and_return(data, "alpha")
    else:
        # default: yfinance -> yahoo -> alpha
        data, err = try_yfinance(symbol); tried["yfinance"] = err
        if data: return set_and_return(data, "yfinance")
        data, err = try_yahoo(symbol); tried["yahoo"] = err
        if data: return set_and_return(data, "yahoo")
        data, err = try_alpha(symbol, interval); tried["alpha"] = err
        if data: return set_and_return(data, "alpha")

    # nothing worked: return stale if available
    stale_full = _cache_get_full(cache_key) or MARKET_CACHE.get(cache_key)
    if stale_full:
        age = int(time.time() - stale_full["ts"])
        return jsonify({
            "symbol": symbol,
            "data": stale_full["data"],
            "cached": True,
            "cached_at": stale_full["cached_at"].isoformat(),
            "age_seconds": age,
            "source": stale_full.get("source"),
            "note": "returned stale cache after failures",
            "tried": tried
        }), 200

    return jsonify({"error": "No data source available", "tried": tried}), 500

# === CHATBOT SECTION ==========================================================
# >>> Local lightweight model: Qwen/Qwen2.5-0.5B-Instruct (fast on CPU)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
HF_MODEL_ENV = os.environ.get("HF_MODEL", "").strip() or "facebook/blenderbot-400M-distill"
HF_MAX_TOKENS = int(os.environ.get("HF_MAX_TOKENS", "256"))

# Lazy loader to avoid long startup if not using local backend
_local_state = {
    "loaded": False,
    "device": None,
    # Default to Qwen 0.5B; allow overriding via .env MODEL_ID
    "model_id": os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
    "tokenizer": None,
    "model": None,
}

def _load_local_finance_model():
    """Load lightweight local model once (Qwen 0.5B) — offline-aware."""
    if _local_state["loaded"]:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.set_num_threads(6)   # tune for your CPU (4–8 usually best)

    logger.info("🧠 Loading local model %s on %s", _local_state["model_id"], device)

    # Respect offline flags: if offline=1 then use only local cache
    local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true", "yes")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(
        _local_state["model_id"],
        local_files_only=local_only,
        use_fast=True
    )

    # ---- model ----
    mdl = AutoModelForCausalLM.from_pretrained(
        _local_state["model_id"],
        torch_dtype=dtype,
        local_files_only=local_only
    )
    if device == "cuda":
        mdl = mdl.to(device)

    _local_state.update({
        "loaded": True,
        "device": device,
        "tokenizer": tok,
        "model": mdl
    })
    logger.info("✅ Local model loaded (device=%s)", device)


def _build_chat_prompt(history, message, system_prompt=None, max_hist=20):
    sys = (system_prompt or FINANCE_SYSTEM).strip()
    lines = [f"System: {sys}"]
    # history can be [{role: user|assistant, content/text/message: "..."}]
    hist = history[-max_hist:] if history else []
    for turn in hist:
        role = (turn.get("role") or "user").lower()
        txt = turn.get("content") or turn.get("text") or turn.get("message") or ""
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {txt}")
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)

def _local_finance_reply(history, message, system_prompt=None, max_new_tokens=140, temperature=0.5):
    _load_local_finance_model()
    tok = _local_state["tokenizer"]; mdl = _local_state["model"]
    device = _local_state["device"]

    prompt = _build_chat_prompt(history, message, system_prompt)
    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # Return only the last assistant segment if present
    return text.rsplit("Assistant:", 1)[-1].strip()

# >>> (Optional) HF fallback kept as-is
def hf_chat_request(message, model_id, max_tokens=HF_MAX_TOKENS, timeout=30, retries=2):
    if not HF_TOKEN:
        raise RuntimeError("HF token not configured (HF_TOKEN)")
    prompt = f"Instruction: {message}\n\nAnswer:"
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "return_full_text": False},
        "options": {"wait_for_model": True}
    }
    backoff = 1.0
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 404:
                raise RuntimeError(f"HF model {model_id} not found (404).")
            if r.status_code in (401, 403):
                raise RuntimeError(f"HF auth error {r.status_code}.")
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else backoff
                time.sleep(wait)
                backoff *= 2
                last_exc = RuntimeError("rate_limited")
                continue
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and j and isinstance(j[0], dict):
                for key in ("generated_text", "text", "summary_text"):
                    if key in j[0]:
                        return j[0][key]
                return str(j[0])
            if isinstance(j, dict):
                for key in ("generated_text", "text", "summary_text"):
                    if key in j:
                        return j[key]
                return str(j)
            return str(j)
        except Exception as e:
            last_exc = e
            logger.debug("hf_chat_request attempt %s failed for model %s: %s", attempt, model_id, e)
            if attempt == retries:
                break
            time.sleep(backoff)
            backoff *= 2
    raise last_exc or RuntimeError("hf_chat_request_failed")

def rule_based_fallback(txt):
    t = txt.lower()
    if any(k in t for k in ("hello", "hi", "hey")):
        return "Hi — I'm DhanMitra. Ask me about SIP vs FD, asset allocation, or your goal planning."
    if "gold" in t:
        return "Gold can hedge inflation/currency risk; many investors keep 5–10% via SGBs or ETFs. (Educational only.)"
    if "sip" in t or "fd" in t:
        return "SIP suits long-term compounding via equity funds; FDs offer capital protection and fixed returns. Match to your horizon & risk. (Educational only.)"
    return "I can help with SIPs, FDs, diversification, and goal planning. Share your goal, horizon, and risk comfort."

def get_chatbot_status():
    if CHATBOT_BACKEND == "local-finance":
        return {"loaded": _local_state["loaded"], "model_id": _local_state["model_id"], "device": _local_state["device"]}
    elif CHATBOT_BACKEND == "hf":
        return {"loaded": bool(HF_TOKEN), "hf_model": HF_MODEL_ENV}
    return {"loaded": True, "note": "rule-based"}

@app.post("/chatbot")
def chatbot():
    payload = request.get_json(force=True, silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body required"}), 400

    # Accept either message + (optional) history, or full "history" with last turn
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    if not message and history:
        # try to extract last user message
        for item in reversed(history):
            if (item.get("role") or "").lower() == "user":
                message = (item.get("content") or item.get("text") or item.get("message") or "").strip()
                if message:
                    break
    if not message:
        return jsonify({"error": "message required"}), 400

    system_prompt = os.environ.get("CHATBOT_SYSTEM_PROMPT", "").strip() or FINANCE_SYSTEM

    # Backend routing
    if CHATBOT_BACKEND == "local-finance":
        try:
            reply = _local_finance_reply(history, message, system_prompt)
            return jsonify({"ok": True, "reply": reply, "backend": "local-finance"}), 200
        except Exception as e:
            logger.exception("local-finance error: %s", e)
            # soft-fallback to HF if token present, else rule-based
            if HF_TOKEN:
                try:
                    reply = hf_chat_request(f"{system_prompt}\n\nUser: {message}\nAssistant:", HF_MODEL_ENV)
                    return jsonify({"ok": True, "reply": reply, "backend": f"hf:{HF_MODEL_ENV}"}), 200
                except Exception as ee:
                    logger.warning("HF fallback failed: %s", ee)
            reply = rule_based_fallback(message)
            return jsonify({"ok": True, "reply": reply, "backend": "rule-fallback"}), 200

    elif CHATBOT_BACKEND == "hf":
        try:
            reply = hf_chat_request(f"{system_prompt}\n\nUser: {message}\nAssistant:", HF_MODEL_ENV)
            return jsonify({"ok": True, "reply": reply, "backend": f"hf:{HF_MODEL_ENV}"}), 200
        except Exception as e:
            logger.warning("HF failed: %s", e)
            return jsonify({"ok": True, "reply": rule_based_fallback(message), "backend": "rule-fallback"}), 200

    # default: rule-based
    return jsonify({"ok": True, "reply": rule_based_fallback(message), "backend": "rule-fallback"}), 200

@app.get("/chatbot/status")
def chatbot_status():
    try:
        return jsonify({"ok": True, **get_chatbot_status()}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/chatbot/warmup")
def chatbot_warmup():
    try:
        if CHATBOT_BACKEND == "local-finance":
            _load_local_finance_model()
        return jsonify({"ok": True, "backend": CHATBOT_BACKEND, **get_chatbot_status()}), 200
    except Exception as e:
        logger.exception("warmup failed: %s", e)
        return jsonify({"ok": False, "backend": CHATBOT_BACKEND, "error": str(e)}), 500

# === END CHATBOT SECTION ======================================================

# --- run app ---
if __name__ == "__main__":
    logger.info(
        "Starting app.py — ALPHA_KEY present: %s, MARKET_CACHE_TTL: %s, CHATBOT_BACKEND: %s",
        bool(ALPHA_KEY), MARKET_CACHE_TTL, CHATBOT_BACKEND
    )
    port = int(os.environ.get("PORT", "8000"))

    # Only preload ONCE (avoid Flask autoreloader double-run)
    if CHATBOT_BACKEND == "local-finance":
        # When debug=True, Werkzeug sets WERKZEUG_RUN_MAIN on the "real" process.
        if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            _load_local_finance_model()
            logger.info("Finance model preloaded on startup.")

    # Run without reloader so it doesn’t start twice
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
