# chatbot_finance.py — robust loader for local small/medium HF models (Qwen recommended)
import os
import torch

# tune this for your CPU (4-8 is usually good)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))

from transformers import AutoModelForCausalLM, AutoTokenizer

# prefer MODEL_ID from .env; default to Qwen lightweight
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# offline flag (set to "1" to require local cache only)
LOCAL_ONLY = os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true", "yes")

# Device/dtype
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE == "cuda" else torch.float32

print(f"🧠 Preparing to load {MODEL_ID} on {_DEVICE} (local_only={LOCAL_ONLY})...")

# load tokenizer (try fast, fallback to slow)
try:
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=LOCAL_ONLY, use_fast=True)
except Exception as e_fast:
    print(f"⚠️ Fast tokenizer load failed: {e_fast}. Trying slow tokenizer...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=LOCAL_ONLY, use_fast=False)
    except Exception as e_slow:
        # common missing dependency: sentencepiece for some tokenizers
        raise RuntimeError(
            "Tokenizer load failed. If this tokenizer uses sentencepiece, install `sentencepiece` and `protobuf`, "
            "or set TRANSFORMERS_OFFLINE=0 to allow first-time download. Original error: %s" % e_slow
        )

# load model
# - Do NOT use device_map='auto' on CPU-only machine; use device_map only when GPUs available & accelerate installed.
load_kwargs = {
    "torch_dtype": _DTYPE,
    "local_files_only": LOCAL_ONLY,
}
# you could enable load_in_8bit via env LOAD_IN_8BIT, but that requires bitsandbytes + GPU
if os.environ.get("LOAD_IN_8BIT", "0").lower() in ("1", "true", "yes"):
    try:
        load_kwargs["load_in_8bit"] = True
    except Exception:
        pass

print("⏳ Loading model (this may take a moment on first download)...")
if _DEVICE == "cuda":
    # device_map='auto' is useful when CUDA + accelerate / multiple GPU are available
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", **load_kwargs)
else:
    # CPU path: keep model on CPU (no device_map)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    _model.to(_DEVICE)

print("✅ Model and tokenizer loaded.")

FINANCE_PROMPT = (
    "You are DhanMitra, a helpful Indian financial assistant. "
    "Explain concepts like SIPs, FDs, mutual funds, diversification, and asset allocation. "
    "Give educational insights only — do not provide personalized financial or tax advice. "
    "Always include a short disclaimer at the end of your answer."
)

def _build_prompt(history, system_prompt=None, max_hist=20):
    sys = system_prompt or FINANCE_PROMPT
    lines = [f"System: {sys}"]
    last = history[-max_hist:] if history else []
    for msg in last:
        role = msg.get("role", "user")
        content = msg.get("content") or msg.get("text") or msg.get("message") or ""
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)

@torch.inference_mode()
def generate_finance_reply(history, system_prompt=None, max_new_tokens=200, temperature=0.6):
    """
    history: list of {role: 'user'|'assistant', content: '...'}
    Returns assistant text (string).
    """
    prompt = _build_prompt(history, system_prompt)
    inputs = _tokenizer(prompt, return_tensors="pt")
    if _DEVICE == "cuda":
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    else:
        # keep tensors on CPU (avoid accidental GPU transfers)
        pass

    out = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=_tokenizer.eos_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )
    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    # Keep only the assistant’s latest part if prompt markers used
    if "Assistant:" in text:
        text = text.rsplit("Assistant:", 1)[-1].strip()
    return text
