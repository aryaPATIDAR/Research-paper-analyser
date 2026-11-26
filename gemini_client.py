"""
Shared Gemini access for the backend.

One place that owns:
- a single client (created lazily, so a missing key never breaks imports)
- a hard per-request timeout (the SDK default is "wait forever")
- a model fallback chain with a short cool-down for models that are
  overloaded (503/504), rate limited (429) or missing (404)
- tolerant JSON parsing of model output
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS = "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"

_PLACEHOLDER_MARKERS = ("your_", "your-", "changeme", "replace_me", "xxxx", "<")


def is_real_key(value: Optional[str]) -> bool:
    """True when an env value looks like an actual key, not an .env.example placeholder."""
    if not value:
        return False
    v = value.strip().lower()
    if len(v) < 12:
        return False
    return not any(marker in v for marker in _PLACEHOLDER_MARKERS)


class GeminiError(Exception):
    """Gemini could not produce a usable answer.

    kind: not_configured | auth | rate_limited | unavailable | timeout | bad_response | error
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


_client = None
_client_key = None
_client_lock = threading.Lock()

# model name -> unix time until which we skip it
_cooldown: Dict[str, float] = {}
_cooldown_lock = threading.Lock()


def is_configured() -> bool:
    return is_real_key(os.getenv("GEMINI_API_KEY"))


def _get_client():
    global _client, _client_key
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not is_real_key(key):
        raise GeminiError("not_configured", "GEMINI_API_KEY is not set in backend/.env")
    with _client_lock:
        if _client is None or _client_key != key:
            from google import genai
            _client = genai.Client(api_key=key)
            _client_key = key
        return _client


def model_chain() -> List[str]:
    """Primary model followed by fallbacks, de-duplicated, order preserved."""
    primary = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    raw = os.getenv("GEMINI_FALLBACK_MODELS")
    if raw is None:
        raw = DEFAULT_FALLBACK_MODELS
    chain: List[str] = []
    for name in [primary] + [m.strip() for m in raw.split(",")]:
        if name and name not in chain:
            chain.append(name)
    return chain


def _set_cooldown(model: str, seconds: float) -> None:
    with _cooldown_lock:
        _cooldown[model] = time.time() + seconds


def _cooling_down(model: str) -> bool:
    with _cooldown_lock:
        return _cooldown.get(model, 0) > time.time()


def parse_json_lenient(text: str) -> Any:
    """Parse JSON from model output that may be fenced or surrounded by prose.

    Raises ValueError when no JSON value can be recovered.
    """
    if not text or not text.strip():
        raise ValueError("empty response")
    cleaned = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("response is not valid JSON")


def _classify(exc: Exception) -> Tuple[str, str]:
    """Map an SDK/network exception to (kind, short message)."""
    code = getattr(exc, "code", None)
    text = str(exc)
    low = text.lower()
    if code == 429 or "resource_exhausted" in low:
        return "rate_limited", "Gemini rate limit / quota reached (429)"
    if code in (401, 403) or "api key not valid" in low or "api_key_invalid" in low:
        return "auth", "Gemini rejected the API key - check GEMINI_API_KEY"
    if code == 404:
        return "not_found", "model not available for this API key (404)"
    if code in (500, 502, 503, 504):
        return "unavailable", f"Gemini service unavailable ({code})"
    name = type(exc).__name__.lower()
    if "timeout" in name or "timed out" in low or "deadline" in low:
        return "timeout", "Gemini request timed out"
    if "connect" in name or "readerror" in name or "network" in name or "winerror" in low:
        return "unavailable", "network error while contacting Gemini"
    return "error", text[:200]


class _AttemptFailed(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")


def _run_model(client, model: str, prompt: str, max_output_tokens: int, temperature: float,
               per_attempt: float, deadline: float) -> Any:
    """One model: call it (with a single retry on a fast 5xx) and parse the JSON.

    Returns parsed JSON or raises _AttemptFailed. Sets the model's cool-down
    itself, so that also happens when the caller has already moved on.
    """
    from google.genai import types

    attempts = 2
    while True:
        attempts -= 1
        remaining = deadline - time.time()
        if remaining < 3:
            raise _AttemptFailed("timeout", f"{model}: no time left in the budget")
        started = time.time()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=int(min(per_attempt, remaining) * 1000)),
                ),
            )
            data = parse_json_lenient(getattr(response, "text", None) or "")
            logger.info("Gemini %s answered in %.1fs", model, time.time() - started)
            return data
        except ValueError as e:
            logger.warning("Gemini %s returned malformed JSON (%s)", model, e)
            raise _AttemptFailed("bad_response", f"{model}: Gemini returned malformed JSON ({e})")
        except Exception as e:  # SDK / network errors
            kind, message = _classify(e)
            elapsed = time.time() - started
            logger.warning("Gemini %s failed after %.1fs: %s", model, elapsed, message)
            if kind == "auth":
                raise _AttemptFailed(kind, message)
            if kind == "not_found":
                _set_cooldown(model, 3600)
                raise _AttemptFailed("unavailable", f"{model}: {message}")
            if kind == "rate_limited":
                _set_cooldown(model, 60)
                raise _AttemptFailed(kind, f"{model}: {message}")
            if kind == "unavailable" and attempts > 0 and elapsed < 5:
                time.sleep(1.5)  # fast 503: one retry on the same model
                continue
            if kind in ("timeout", "unavailable"):
                _set_cooldown(model, 300)
            raise _AttemptFailed(kind, f"{model}: {message}")


def generate_json(
    prompt: str,
    *,
    max_output_tokens: int = 8192,
    temperature: float = 0.3,
    budget_seconds: Optional[float] = None,
    attempt_timeout: Optional[float] = None,
) -> Tuple[Any, str]:
    """Ask Gemini for a JSON answer.

    Tries the model chain in order. A model that fails hands over to the next
    one immediately; a model that is merely slow (no answer after
    GEMINI_HEDGE_SECONDS) keeps running while the next one is started as
    well, and the first usable answer wins. Overloaded models tend to hang
    for 30s+ before returning 504, so waiting for them one after another
    could take minutes.

    Returns (parsed_json, model_name). Raises GeminiError.
    """
    client = _get_client()
    budget = budget_seconds if budget_seconds is not None else _env_float("GEMINI_BUDGET_SECONDS", 100)
    per_attempt = attempt_timeout if attempt_timeout is not None else _env_float("GEMINI_TIMEOUT_SECONDS", 35)
    hedge = max(2.0, _env_float("GEMINI_HEDGE_SECONDS", 12))
    deadline = time.time() + budget

    chain = model_chain()
    # If everything is cooling down, still try the chain rather than fail outright.
    queue = [m for m in chain if not _cooling_down(m)] or list(chain)

    pending: Dict[Any, str] = {}
    last_kind, last_message = "error", "no Gemini model could be tried"

    def launch() -> None:
        model = queue.pop(0)
        future = _pool.submit(_run_model, client, model, prompt, max_output_tokens,
                              temperature, per_attempt, deadline)
        pending[future] = model

    launch()
    while pending:
        remaining = deadline - time.time()
        if remaining <= 0:
            last_kind, last_message = "timeout", f"no Gemini model answered within {int(budget)}s"
            break
        done, _ = wait(pending, timeout=min(hedge, remaining) if queue else remaining,
                       return_when=FIRST_COMPLETED)
        if not done:
            if queue:
                logger.info("Gemini %s is slow - also trying %s", ", ".join(pending.values()), queue[0])
                launch()
            continue
        for future in done:
            model = pending.pop(future)
            try:
                return future.result(), model
            except _AttemptFailed as e:
                if e.kind == "auth":
                    raise GeminiError(e.kind, e.message)
                last_kind, last_message = e.kind, e.message
            except Exception as e:  # defensive
                last_kind, last_message = "error", f"{model}: {str(e)[:200]}"
        if not pending and queue:
            launch()

    raise GeminiError(last_kind, last_message)
