"""EXPH unified API generation adapter (Anthropic + OpenAI), modes prefill/instruct.

* Keys come ONLY from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY);
  they are never read from files here, never logged, never written to outputs.
* Concurrency <= 8; 429/5xx/connection errors retried with exponential backoff.
* Every response's token usage is appended to a persistent cost ledger with
  per-model pricing; the ledger enforces the hard budget stop.
"""
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True

MAX_TOKENS = 300
CONCURRENCY = 8
HARD_BUDGET_USD = 150.0  # E10 TOPUP+FABLE run cap; raised from 100.0 per 2026-07-27 run directive (stop-and-report at $100 enforced by the driver, not this constant)

# $ per 1M tokens (input, output). Anthropic: claude-api skill table (2026-06).
# OpenAI list prices (2026-06); if these drift the ledger is still exact in tokens.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),   # standard sticker (intro pricing lower)
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),   # claude-api skill table 2026-06 ($10/$50 per MTok)
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4-0": (15.00, 75.00),
    "claude-sonnet-4-0": (3.00, 15.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5-chat-latest": (1.25, 10.00),
    "gpt-3.5-turbo-instruct": (1.50, 2.00),
    "davinci-002": (2.00, 2.00),
}

# Per-model API parameter policy (design-locked):
#   claude-sonnet-5   : thinking={"type":"disabled"}, NO temperature
#   claude-opus-4-8   : omit thinking entirely, NO temperature
#   claude-haiku-4-5  : temperature=0 allowed
#   claude-sonnet-4-5 : temperature=0 allowed
#   gpt-4.1           : temperature=0
#   gpt-5.1           : reasoning_effort="none" (runtime-verified); no temperature
MODEL_CONFIGS = [
    {"key": "claude-haiku-4-5_prefill", "provider": "anthropic", "model_id": "claude-haiku-4-5",
     "mode": "prefill", "temperature": 0.0, "gold_key": "claude-haiku-4-5"},
    {"key": "claude-haiku-4-5_instruct", "provider": "anthropic", "model_id": "claude-haiku-4-5",
     "mode": "instruct", "temperature": 0.0, "gold_key": "claude-haiku-4-5"},
    {"key": "claude-sonnet-4-5_prefill", "provider": "anthropic", "model_id": "claude-sonnet-4-5",
     "mode": "prefill", "temperature": 0.0, "gold_key": "claude-sonnet-4-5"},
    {"key": "claude-sonnet-5_instruct", "provider": "anthropic", "model_id": "claude-sonnet-5",
     "mode": "instruct", "temperature": None, "thinking": {"type": "disabled"},
     "gold_key": "claude-sonnet-5"},
    {"key": "claude-opus-4-8_instruct", "provider": "anthropic", "model_id": "claude-opus-4-8",
     "mode": "instruct", "temperature": None, "gold_key": "claude-opus-4-8"},
    {"key": "gpt-4.1_instruct", "provider": "openai", "model_id": "gpt-4.1",
     "mode": "instruct", "temperature": 0.0, "gold_key": "gpt-4.1"},
    {"key": "gpt-5.1_instruct", "provider": "openai", "model_id": "gpt-5.1",
     "mode": "instruct", "temperature": None, "reasoning_effort": "none",
     "gold_key": "gpt-5.1"},
]


def get_config(key):
    for c in MODEL_CONFIGS:
        if c["key"] == key:
            return dict(c)
    raise KeyError(key)


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    """Persistent token/cost ledger, thread-safe, hard-stop at HARD_BUDGET_USD.

    extra_paths: other (read-only) ledger files summed into the budget check so
    the hard stop applies to CUMULATIVE spend across experiments."""

    def __init__(self, path, extra_paths=()):
        self.path = path
        self.extra_paths = [p for p in extra_paths if os.path.abspath(p) != os.path.abspath(path)]
        self.lock = threading.Lock()
        self.data = {"models": {}, "hard_budget_usd": HARD_BUDGET_USD}
        if os.path.exists(path):
            with open(path) as fh:
                self.data = json.load(fh)
        self._dirty = 0

    def extra_cost_usd(self):
        total = 0.0
        for p in self.extra_paths:
            try:
                with open(p) as fh:
                    d = json.load(fh)
                for mid, m in d.get("models", {}).items():
                    pin, pout = PRICES.get(mid, (5.0, 25.0))
                    total += m["input_tokens"] / 1e6 * pin + m["output_tokens"] / 1e6 * pout
            except Exception:
                pass
        return total

    def add(self, model_id, in_tok, out_tok, n_req=1):
        with self.lock:
            m = self.data["models"].setdefault(model_id, {
                "requests": 0, "input_tokens": 0, "output_tokens": 0})
            m["requests"] += n_req
            m["input_tokens"] += int(in_tok or 0)
            m["output_tokens"] += int(out_tok or 0)
            self._dirty += 1
            if self._dirty >= 25:
                self._save_locked()

    def cost_usd(self, model_id=None):
        with self.lock:
            return self._cost_locked(model_id)

    def _cost_locked(self, model_id=None):
        total = 0.0
        for mid, m in self.data["models"].items():
            if model_id and mid != model_id:
                continue
            pin, pout = PRICES.get(mid, (5.0, 25.0))
            total += m["input_tokens"] / 1e6 * pin + m["output_tokens"] / 1e6 * pout
        return total

    def check_budget(self):
        extra = self.extra_cost_usd()
        with self.lock:
            total = self._cost_locked() + extra
            if total > HARD_BUDGET_USD:
                self._save_locked()
                raise BudgetExceeded("HARD STOP: cumulative ledger total $%.2f > $%.2f"
                                     % (total, HARD_BUDGET_USD))
            return total

    def save(self):
        with self.lock:
            self._save_locked()

    def _save_locked(self):
        self.data["total_usd"] = round(self._cost_locked(), 4)
        by = {}
        for mid, m in self.data["models"].items():
            pin, pout = PRICES.get(mid, (5.0, 25.0))
            by[mid] = round(m["input_tokens"] / 1e6 * pin + m["output_tokens"] / 1e6 * pout, 4)
        self.data["usd_by_model"] = by
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
        self._dirty = 0


# ------------------------------------------------------------------ clients

_CLIENTS = {}
_CLIENT_LOCK = threading.Lock()


def _client(provider):
    with _CLIENT_LOCK:
        if provider not in _CLIENTS:
            if provider == "anthropic":
                import anthropic
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise SystemExit("ANTHROPIC_API_KEY not set (source the keys env file first)")
                _CLIENTS[provider] = anthropic.Anthropic(max_retries=0)
            elif provider == "openai":
                import openai
                if not os.environ.get("OPENAI_API_KEY"):
                    raise SystemExit("OPENAI_API_KEY not set (source the keys env file first)")
                _CLIENTS[provider] = openai.OpenAI(max_retries=0)
            else:
                raise ValueError(provider)
        return _CLIENTS[provider]


def build_request_messages(cfg, question, target, pre_text=None):
    """Rendered request content, identical across providers for a given mode.

    Returns a payload for call_model: a messages list (chat modes) or, for mode
    "completion", the raw text prompt (TRUE completion: exactly the common.py
    rendering, user string + " " + answer prefix, no chat template)."""
    import exph_common as C
    if cfg["mode"] == "completion":
        prompt = C.gold_user_content(question, target)
        if pre_text is not None:
            prompt += " " + pre_text
        return prompt
    if pre_text is None:  # gold: plain question -> proof, both modes
        return [{"role": "user", "content": C.gold_user_content(question, target)}]
    if cfg["mode"] == "prefill":
        return [{"role": "user", "content": C.gold_user_content(question, target)},
                {"role": "assistant", "content": pre_text}]
    return [{"role": "user", "content": C.instruct_user_content(question, target, pre_text)}]


def _anthropic_call(cfg, messages):
    import anthropic
    client = _client("anthropic")
    kwargs = {"model": cfg["model_id"], "max_tokens": cfg.get("max_tokens", MAX_TOKENS),
              "messages": messages}
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("thinking") is not None:
        kwargs["thinking"] = cfg["thinking"]
    if cfg.get("output_config") is not None:   # effort control (e.g. Fable-5 floor = {"effort":"low"})
        kwargs["output_config"] = cfg["output_config"]
    resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        return {"text": None, "error": "stop_reason_refusal", "thinking": None,
                "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    thinking = "".join(getattr(b, "thinking", "") or "" for b in resp.content
                       if getattr(b, "type", None) == "thinking")
    return {"text": text, "thinking": thinking, "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def _openai_call(cfg, messages):
    client = _client("openai")
    kwargs = {"model": cfg["model_id"], "messages": messages,
              "max_completion_tokens": cfg.get("max_tokens", MAX_TOKENS)}
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    resp = client.chat.completions.create(**kwargs)
    ch = resp.choices[0]
    usage = resp.usage
    return {"text": ch.message.content, "stop_reason": ch.finish_reason,
            "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}


def _openai_completion_call(cfg, prompt):
    """TRUE text-completion endpoint (v1/completions)."""
    client = _client("openai")
    resp = client.completions.create(
        model=cfg["model_id"], prompt=prompt,
        max_tokens=cfg.get("max_tokens", MAX_TOKENS),
        temperature=cfg.get("temperature") if cfg.get("temperature") is not None else 0.0,
        stop=cfg.get("stop", ["\nQ:"]))
    ch = resp.choices[0]
    usage = resp.usage
    return {"text": ch.text, "stop_reason": ch.finish_reason,
            "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}


def _is_retryable(exc):
    name = type(exc).__name__
    if name in ("RateLimitError", "InternalServerError", "APIConnectionError",
                "APITimeoutError", "OverloadedError", "ServiceUnavailableError"):
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 500, 502, 503, 504, 529)


def call_model(cfg, messages, ledger, max_attempts=6):
    """One generation with retry/backoff. Returns raw dict (text may be None on failure)."""
    last = None
    for attempt in range(max_attempts):
        try:
            if isinstance(messages, str):
                out = _openai_completion_call(cfg, messages)
            elif cfg["provider"] == "anthropic":
                out = _anthropic_call(cfg, messages)
            else:
                out = _openai_call(cfg, messages)
            ledger.add(cfg["model_id"], out.get("input_tokens"), out.get("output_tokens"))
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_retryable(e) or attempt == max_attempts - 1:
                return {"text": None, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
            time.sleep(min(60, (2 ** attempt) + random.random()))
    return {"text": None, "error": "%s: %s" % (type(last).__name__, str(last)[:300])}


def generate_batch(cfg, jobs, ledger, raw_path, existing_ids=None, budget_every=20):
    """jobs: list of dicts {run_id, question, target, pre_text(None for gold), **meta}.
    Appends raw rows to raw_path (resume-safe on run_id). Returns n new rows."""
    import exph_common as C
    existing = set(existing_ids or ())
    pending = [j for j in jobs if j["run_id"] not in existing]
    if not pending:
        return 0
    lock = threading.Lock()
    counter = {"done": 0}

    def work(job):
        messages = build_request_messages(cfg, job["question"], job["target"], job.get("pre_text"))
        out = call_model(cfg, messages, ledger)
        row = {k: v for k, v in job.items() if k not in ("question", "target", "pre_text")}
        row.update({
            "model_key": cfg["key"], "provider": cfg["provider"], "model_id": cfg["model_id"],
            "mode": cfg["mode"],
            "failed_generation": out.get("text") is None,
            "continuation": out.get("text"),
            "stop_reason": out.get("stop_reason"),
            "error": out.get("error"),
            "input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens"),
            "decoding": {"max_tokens": MAX_TOKENS, "temperature": cfg.get("temperature"),
                         "thinking": cfg.get("thinking"), "reasoning_effort": cfg.get("reasoning_effort")},
        })
        with lock:
            C.append_jsonl(raw_path, row)
            counter["done"] += 1
            if counter["done"] % budget_every == 0:
                ledger.check_budget()
        return row

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, pending))
    ledger.check_budget()
    ledger.save()
    return len(pending)
