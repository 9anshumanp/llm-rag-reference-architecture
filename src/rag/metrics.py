from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import os

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

# Simple default pricing (USD per 1M tokens). Override via env if desired.
DEFAULT_PRICING = {
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    # Add/adjust as needed
}

def _env_float(key: str) -> Optional[float]:
    v = os.getenv(key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

@dataclass
class Cost:
    input_usd: float = 0.0
    output_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd

class TokenEstimator:
    def __init__(self, model: str):
        self._model = model
        self._enc = None
        if tiktoken is not None:
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        # Fallback heuristic: ~0.75 words/token
        words = max(1, len(text.split()))
        return int(words / 0.75)

class CostModel:
    def __init__(self, model: str, pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self._model = model
        self._pricing = pricing or DEFAULT_PRICING

    def estimate(self, usage: Usage) -> Cost:
        # Allow env overrides: RAG_PRICE_INPUT_PER_1M, RAG_PRICE_OUTPUT_PER_1M
        in_rate = _env_float("RAG_PRICE_INPUT_PER_1M")
        out_rate = _env_float("RAG_PRICE_OUTPUT_PER_1M")
        if in_rate is None or out_rate is None:
            p = self._pricing.get(self._model, {"input_per_1m": 0.0, "output_per_1m": 0.0})
            in_rate = in_rate if in_rate is not None else p["input_per_1m"]
            out_rate = out_rate if out_rate is not None else p["output_per_1m"]

        return Cost(
            input_usd=(usage.prompt_tokens / 1_000_000.0) * float(in_rate),
            output_usd=(usage.completion_tokens / 1_000_000.0) * float(out_rate),
        )
