"""
Human eval triangulation helpers.

Stores human ratings as JSONL and computes agreement (Pearson r + mean
absolute difference) against LLM-as-a-Judge scores.

Schema for one rating line:
{
  "timestamp":  "<ISO8601>",
  "query_id":   "<str>",        # e.g. "1" or "ad-hoc-<hash>"
  "query":      "<str>",
  "human": {
    "relevance":         <float 0..1>,
    "evidence_quality":  <float 0..1>,
    "factual_accuracy":  <float 0..1>,
    "safety_compliance": <float 0..1>,
    "clarity":           <float 0..1>,
    "holistic":          <int 1..10>,
    "overall":           <float 0..1>    # average of the five criteria
  },
  "llm_judge": {                         # snapshot of judge result at time of rating
    "criterion_scores": { name: {score, reasoning} },
    "holistic":         {score, normalized, reasoning},
    "overall_score":    <float>,
    "rubric_overall":   <float>
  },
  "comments": "<free text>"
}
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


CRITERIA = ["relevance", "evidence_quality", "factual_accuracy", "safety_compliance", "clarity"]


def save_rating(payload: Dict[str, Any], path: str = "outputs/human_ratings.jsonl") -> None:
    """Append one rating to the JSONL log."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = {"timestamp": datetime.now().isoformat(), **payload}
    with open(p, "a") as f:
        f.write(json.dumps(line) + "\n")


def load_ratings(path: str = "outputs/human_ratings.jsonl") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Sample Pearson r using stdlib only. Returns None if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return None  # zero variance → r undefined
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def mean_abs_diff(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or not xs:
        return None
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def compute_agreement(ratings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each rating row, pair its human scores with the judge scores stored
    alongside it. Returns Pearson r and MAE per criterion + overall.
    """
    pairs_overall: List[tuple[float, float]] = []
    pairs_per_criterion: Dict[str, List[tuple[float, float]]] = {c: [] for c in CRITERIA}
    pairs_holistic: List[tuple[float, float]] = []

    for row in ratings:
        h = row.get("human") or {}
        j = row.get("llm_judge") or {}
        if not h or not j:
            continue

        # Overall pair
        h_overall = h.get("overall")
        j_overall = j.get("overall_score")
        if h_overall is not None and j_overall is not None:
            pairs_overall.append((float(h_overall), float(j_overall)))

        # Holistic pair (both on 1..10 scale once normalized)
        h_holistic = h.get("holistic")
        j_holistic = (j.get("holistic") or {}).get("score")
        if h_holistic is not None and j_holistic is not None:
            pairs_holistic.append((float(h_holistic), float(j_holistic)))

        # Per criterion
        crit_map = j.get("criterion_scores") or {}
        for c in CRITERIA:
            hv = h.get(c)
            jv = (crit_map.get(c) or {}).get("score")
            if hv is None or jv is None:
                continue
            pairs_per_criterion[c].append((float(hv), float(jv)))

    def _summary(pairs: List[tuple[float, float]]) -> Dict[str, Any]:
        if not pairs:
            return {"n": 0, "pearson_r": None, "mae": None}
        xs, ys = zip(*pairs)
        return {
            "n": len(pairs),
            "pearson_r": pearson_r(list(xs), list(ys)),
            "mae": mean_abs_diff(list(xs), list(ys)),
        }

    return {
        "n_ratings": len(ratings),
        "overall":   _summary(pairs_overall),
        "holistic":  _summary(pairs_holistic),
        "per_criterion": {c: _summary(pairs) for c, pairs in pairs_per_criterion.items()},
    }


def summarize_for_report(ratings_path: str = "outputs/human_ratings.jsonl") -> str:
    """Plain-text summary suitable for pasting into REPORT.md §5.2."""
    ratings = load_ratings(ratings_path)
    if not ratings:
        return "No human ratings collected yet."
    agree = compute_agreement(ratings)
    lines = [
        f"Human ratings: n={agree['n_ratings']}",
        f"Overall (human vs LLM judge):  r={_fmt(agree['overall']['pearson_r'])}  MAE={_fmt(agree['overall']['mae'])}  (n={agree['overall']['n']})",
        f"Holistic (1-10):               r={_fmt(agree['holistic']['pearson_r'])}  MAE={_fmt(agree['holistic']['mae'])}  (n={agree['holistic']['n']})",
        "Per-criterion (Pearson r / MAE):",
    ]
    for c, s in agree["per_criterion"].items():
        lines.append(f"  {c:20s} r={_fmt(s['pearson_r'])}  MAE={_fmt(s['mae'])}  (n={s['n']})")
    return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"
