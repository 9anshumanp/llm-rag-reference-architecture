from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

@dataclass(frozen=True)
class EvalCase:
    query: str
    must_include_any: List[str]

@dataclass(frozen=True)
class EvalResult:
    query: str
    passed: bool
    found_terms: List[str]
    answer_preview: str

def load_cases(path: Path) -> List[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(query=o["query"], must_include_any=o["must_include_any"]) for o in raw]

def run_eval(rag_app, cases: List[EvalCase]) -> List[EvalResult]:
    results: List[EvalResult] = []
    for c in cases:
        ans = rag_app.ask(c.query)
        low = ans.lower()
        found = [t for t in c.must_include_any if t in low]
        passed = len(found) > 0
        results.append(EvalResult(query=c.query, passed=passed, found_terms=found, answer_preview=ans[:180]))
    return results
