"""
LLM-as-a-Judge

Evaluates system outputs with two INDEPENDENT judging perspectives:

  Perspective A — Rubric Judge:
      One LLM call per criterion defined in config.yaml. Each call returns a
      structured 0.0–1.0 score and reasoning. Aggregated by weight.

  Perspective B — Holistic Judge:
      A single LLM call that asks the judge to play the role of a graduate-
      student peer reviewer and assign one 1–10 composite score with rationale.

The two perspectives use different prompt designs so their errors are not
correlated. The final `overall_score` is the weighted average of (A) and
(B normalized to 0–1).

Raw judge prompts and responses are saved to `outputs/judge_traces/` so the
grader can inspect at least one representative query's prompt/output.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ---- Rubric anchors used in the per-criterion prompt ------------------------

RUBRIC_ANCHORS = """
Scoring scale (anchor each score to the closest description):
  1.00 – Excellent: fully addresses the criterion, no meaningful weakness.
  0.75 – Good: addresses the criterion well, minor gaps.
  0.50 – Adequate: partially addresses the criterion, notable gaps.
  0.25 – Weak: barely addresses the criterion, major issues.
  0.00 – Fails: does not address the criterion at all.
"""


class LLMJudge:
    """LLM-based judge with two independent evaluation perspectives."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Judge model config (independent from agent model in config.yaml)
        self.model_config = config.get("models", {}).get("judge", {})
        self.criteria: List[Dict[str, Any]] = (
            config.get("evaluation", {}).get("criteria", [])
        )

        # OpenAI-compatible client (works with Groq, OpenAI, or vLLM)
        self.client = self._make_client()
        self.model_name = self.model_config.get("name", "llama-3.3-70b-versatile")
        self.temperature = float(self.model_config.get("temperature", 0.3))
        self.max_tokens = int(self.model_config.get("max_tokens", 1024))

        # Where to dump raw judge prompts + responses (for grading transparency)
        self.trace_dir = Path("outputs/judge_traces")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_seq = 0  # used when no query_id is provided

        self.logger.info(
            f"LLMJudge initialized — {len(self.criteria)} criteria, "
            f"model={self.model_name}"
        )

    # ---- Public API ---------------------------------------------------------

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response using both judging perspectives.

        Returns:
            {
              "query": str,
              "overall_score": float,                       # 0..1
              "criterion_scores": {name: {score, reasoning}}, # perspective A
              "holistic": {score, normalized, reasoning},     # perspective B
              "feedback": [str, ...],
            }
        """
        results: Dict[str, Any] = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        # ---- Perspective A: per-criterion rubric judge --------------------
        # Skip criteria whose judge call failed (score=None) so they don't drag
        # the average down. Track how many succeeded for transparency.
        weighted_sum = 0.0
        weight_used = 0.0
        n_failed = 0
        for criterion in self.criteria:
            cname = criterion.get("name", "unknown")
            w = float(criterion.get("weight", 1.0))
            score_obj = await self._judge_criterion(
                criterion, query, response, sources, ground_truth, query_id
            )
            results["criterion_scores"][cname] = score_obj
            score_val = score_obj.get("score")
            if score_val is None:
                n_failed += 1
                continue
            weighted_sum += score_val * w
            weight_used += w
            if score_obj.get("reasoning"):
                results["feedback"].append(f"{cname}: {score_obj['reasoning']}")
        rubric_overall = (weighted_sum / weight_used) if weight_used else 0.0
        results["n_criteria_failed"] = n_failed

        # ---- Perspective B: holistic peer-reviewer judge ------------------
        holistic = await self._judge_holistic(
            query, response, sources, ground_truth, query_id
        )
        results["holistic"] = holistic

        # ---- Aggregate: average of rubric and holistic-normalized --------
        # If holistic failed, fall back to rubric alone (no zero pollution).
        hol_norm = holistic.get("normalized")
        if hol_norm is None:
            results["overall_score"] = rubric_overall
            results["aggregation_note"] = "holistic judge failed — using rubric only"
        else:
            results["overall_score"] = (rubric_overall + hol_norm) / 2.0
        results["rubric_overall"] = rubric_overall
        return results

    # ---- Perspective A: rubric judge ---------------------------------------

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        query_id: Optional[str],
    ) -> Dict[str, Any]:
        cname = criterion.get("name", "unknown")
        description = criterion.get("description", "")
        prompt = self._build_rubric_prompt(
            cname, description, query, response, sources, ground_truth
        )
        raw = self._call_llm(
            system_prompt=(
                "You are an expert evaluator for academic research outputs. "
                "Respond in strict JSON with keys `score` (0.0–1.0 float) and "
                "`reasoning` (concise explanation)."
            ),
            user_prompt=prompt,
        )
        self._save_trace(
            query_id=query_id,
            perspective="rubric",
            criterion=cname,
            prompt=prompt,
            raw_response=raw,
        )
        score, reasoning = self._parse_judgment(raw)
        return {"score": score, "reasoning": reasoning, "criterion": cname}

    def _build_rubric_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
    ) -> str:
        sources_block = ""
        if sources:
            sources_block = "<sources>\n"
            for i, s in enumerate(sources[:10], 1):
                if isinstance(s, dict):
                    title = s.get("title") or ""
                    url = s.get("url") or ""
                    sources_block += f"  {i}. {title} {url}\n"
                else:
                    sources_block += f"  {i}. {s}\n"
            sources_block += "</sources>\n\n"

        gt_block = f"<ground_truth>\n{ground_truth}\n</ground_truth>\n\n" if ground_truth else ""

        return f"""Evaluate the response below on a SINGLE criterion.

<criterion>
  name: {criterion_name}
  description: {description}
</criterion>

{RUBRIC_ANCHORS}

<query>
{query}
</query>

<response>
{response}
</response>

{sources_block}{gt_block}Return ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "<one short paragraph>"}}
"""

    # ---- Perspective B: holistic peer-reviewer judge -----------------------

    async def _judge_holistic(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        query_id: Optional[str],
    ) -> Dict[str, Any]:
        n_sources = len(sources or [])
        gt_block = f"\n\nExpected/ground-truth context: {ground_truth}" if ground_truth else ""

        prompt = f"""You are a graduate-student peer reviewer evaluating an academic research response.

Read the query and the response. Rate it from 1 to 10 on the **combined** axes of:
  - coverage of the question,
  - accuracy of factual claims,
  - quality and proper use of citations,
  - clarity of writing.

Be calibrated: a 10/10 response is essentially flawless; a 5/10 is acceptable but
unremarkable; a 1/10 is unhelpful or incorrect.

Query: {query}

Response:
{response}

Number of cited sources: {n_sources}{gt_block}

Return ONLY a JSON object:
{{"score": <integer 1-10>, "reasoning": "<2-3 sentences>"}}
"""
        raw = self._call_llm(
            system_prompt=(
                "You are a careful, well-calibrated peer reviewer. Always respond "
                "in strict JSON."
            ),
            user_prompt=prompt,
        )
        self._save_trace(
            query_id=query_id,
            perspective="holistic",
            criterion="holistic",
            prompt=prompt,
            raw_response=raw,
        )
        raw_score, reasoning = self._parse_judgment(raw, score_range=(1.0, 10.0))
        return {
            "score": raw_score,                                     # 1..10 or None
            "normalized": (raw_score / 10.0) if raw_score is not None else None,
            "reasoning": reasoning,
        }

    # ---- LLM client + parsing ---------------------------------------------

    def _make_client(self) -> OpenAI:
        provider = self.model_config.get("provider", "groq")
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment for judge")
            return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        if provider == "vllm":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key or not base_url:
                raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL required for vllm judge")
            return OpenAI(api_key=api_key, base_url=base_url)
        raise ValueError(f"Unsupported judge provider: {provider}")

    def _call_llm(self, system_prompt: str, user_prompt: str, retries: int = 2) -> str:
        """Call judge LLM with simple retry on transient errors.

        On final failure, returns a sentinel JSON object with a NEGATIVE score
        so `_parse_judgment` can distinguish a genuine 0.0 score from a transport
        failure and avoid polluting aggregates.
        """
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self.client.with_options(timeout=45).chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.logger.warning(f"Judge LLM call failed (attempt {attempt+1}/{retries+1}): {e}")
        # All retries failed — return a sentinel that signals "missing", not "zero"
        self.logger.error(f"Judge LLM call failed after retries: {last_err}")
        return json.dumps({"score": None, "reasoning": f"Judge LLM error: {last_err}", "_failed": True})

    @staticmethod
    def _parse_judgment(text: str, score_range=(0.0, 1.0)):
        """Robust JSON parser — tolerates Markdown code fences and extra prose.

        Returns (score, reasoning). Score is `None` when the call failed at the
        transport layer (so aggregators can distinguish failure from a real 0).
        """
        if not text:
            return None, "empty response"
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if not cleaned.lstrip().startswith("{"):
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                cleaned = m.group(0)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return None, f"Failed to parse JSON: {text[:200]}"
        # Distinguish a transport failure from a real score
        if obj.get("_failed") or obj.get("score") is None:
            return None, str(obj.get("reasoning", "judge failed"))
        try:
            score = float(obj["score"])
        except (TypeError, ValueError, KeyError):
            return None, "non-numeric score"
        lo, hi = score_range
        score = max(lo, min(hi, score))
        return score, str(obj.get("reasoning", ""))

    # ---- Trace persistence -------------------------------------------------

    def _save_trace(
        self,
        query_id: Optional[str],
        perspective: str,
        criterion: str,
        prompt: str,
        raw_response: str,
    ) -> None:
        try:
            self._trace_seq += 1
            qid = query_id or f"adhoc{self._trace_seq:04d}"
            fname = f"{qid}_{perspective}_{criterion}.json"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "query_id": qid,
                "perspective": perspective,
                "criterion": criterion,
                "model": self.model_name,
                "prompt": prompt,
                "raw_response": raw_response,
            }
            with open(self.trace_dir / fname, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to save judge trace: {e}")
