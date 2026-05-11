"""
NLI-based hallucination detection.

Given a generated response and the list of retrieved sources, this checker:
  1. Asks an LLM to extract the top N atomic factual claims from the response.
  2. For each claim, asks an LLM: "Is this claim entailed by ANY of these
     source snippets? Answer YES/NO with a one-sentence reason."
  3. Returns a list of `{claim, entailed, supporting_source, reasoning}`.

This is stronger than the existing `_check_factual_consistency` in
`src/guardrails/output_guardrail.py`, which only checks whether a
`[Source: X]` marker syntactically matches a retrieved source — it does
not verify semantic entailment of the claim itself.

Design notes:
  - Uses the same OpenAI-compatible client construction as `LLMJudge` so
    the user can swap Groq / OpenAI / vLLM via `models.judge` config.
  - Each LLM call has a sentinel-failure fallback (returns `entailed=None`
    so the caller can distinguish "not supported" from "could not check").
  - Capped at `safety.nli_max_claims` (default 8) to keep token cost bounded.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class NLIChecker:
    """LLM-NLI checker for claim-source entailment."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("safety.nli")
        safety_cfg = config.get("safety", {})
        self.max_claims = int(safety_cfg.get("nli_max_claims", 8))
        # Reuse the judge model so we get the cross-model design for free
        self.model_config = config.get("models", {}).get("judge", {})
        self.model_name = self.model_config.get("name", "openai/gpt-oss-120b")
        self.temperature = float(self.model_config.get("temperature", 0.3))
        self.max_tokens = int(self.model_config.get("max_tokens", 512))
        self.client = self._make_client()

    # ---- Public API ---------------------------------------------------------

    def check_claims(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Extract top claims from `response` and check each against `sources`.

        Returns a list of dicts:
            {
              "claim":            <str>,
              "entailed":         True | False | None,    # None = check failed
              "supporting_source": <url or title or None>,
              "reasoning":        <str>,
            }
        """
        if not response or not sources:
            return []

        claims = self._extract_claims(response)
        if not claims:
            return []

        sources_block = self._format_sources(sources)

        results: List[Dict[str, Any]] = []
        for claim in claims:
            verdict = self._check_one_claim(claim, sources_block)
            results.append(verdict)
        return results

    # ---- Step 1: claim extraction ------------------------------------------

    def _extract_claims(self, response: str) -> List[str]:
        prompt = f"""From the response below, list the TOP {self.max_claims} atomic
factual claims that would need to be supported by external evidence. Skip
opinions, definitions, and meta-commentary. Each claim should be a single
short declarative sentence.

Response:
{response}

Return ONLY a JSON array of strings (no surrounding prose, no markdown):
["claim 1", "claim 2", ...]
"""
        raw = self._call_llm(
            system_prompt=(
                "You extract atomic factual claims from text. Always respond "
                "in strict JSON."
            ),
            user_prompt=prompt,
        )
        return self._parse_claim_list(raw)

    @staticmethod
    def _parse_claim_list(raw: str) -> List[str]:
        if not raw:
            return []
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
        # If wrapped in prose, find first JSON array
        if not text.lstrip().startswith("["):
            m = re.search(r"\[[\s\S]*?\]", text)
            if m:
                text = m.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []

    # ---- Step 2: per-claim entailment --------------------------------------

    def _check_one_claim(self, claim: str, sources_block: str) -> Dict[str, Any]:
        prompt = f"""Determine whether the CLAIM is entailed by ANY of the SOURCE
snippets below.

CLAIM:
{claim}

SOURCES:
{sources_block}

A claim is "entailed" only if a reasonable reader, given just the sources,
would conclude the claim is true. Speculation, partial relation, or
topic-overlap does NOT count as entailment.

Return ONLY a JSON object (no surrounding prose, no markdown):
{{
  "entailed": true | false,
  "supporting_source": "<url or title of the source that entails it, or null>",
  "reasoning": "<one short sentence>"
}}
"""
        raw = self._call_llm(
            system_prompt=(
                "You are a strict NLI verifier. Only answer 'entailed: true' "
                "if the sources clearly support the claim. Always respond in "
                "strict JSON."
            ),
            user_prompt=prompt,
        )
        return self._parse_verdict(claim, raw)

    @staticmethod
    def _parse_verdict(claim: str, raw: str) -> Dict[str, Any]:
        if not raw:
            return {
                "claim": claim,
                "entailed": None,
                "supporting_source": None,
                "reasoning": "empty NLI response",
            }
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
        if not text.lstrip().startswith("{"):
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {
                "claim": claim,
                "entailed": None,
                "supporting_source": None,
                "reasoning": f"failed to parse: {raw[:120]}",
            }
        # Treat _failed sentinel as None
        if obj.get("_failed"):
            return {
                "claim": claim,
                "entailed": None,
                "supporting_source": None,
                "reasoning": str(obj.get("reasoning", "nli failed")),
            }
        entailed_raw = obj.get("entailed")
        if isinstance(entailed_raw, str):
            entailed = entailed_raw.strip().lower() in ("true", "yes", "y")
        elif isinstance(entailed_raw, bool):
            entailed = entailed_raw
        else:
            entailed = None
        return {
            "claim": claim,
            "entailed": entailed,
            "supporting_source": obj.get("supporting_source"),
            "reasoning": str(obj.get("reasoning", "")),
        }

    # ---- Helpers ------------------------------------------------------------

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        lines = []
        for i, s in enumerate(sources, 1):
            if not isinstance(s, dict):
                lines.append(f"  {i}. {s}")
                continue
            title = s.get("title") or ""
            url = s.get("url") or ""
            snippet = (s.get("snippet") or "")[:300]
            head = title or url or f"source {i}"
            body = f" — {snippet}" if snippet else ""
            lines.append(f"  {i}. {head}{body}")
        return "\n".join(lines) if lines else "  (none)"

    def _make_client(self) -> OpenAI:
        provider = self.model_config.get("provider", "groq")
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment for NLI checker")
            return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        if provider == "vllm":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key or not base_url:
                raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL required for vllm NLI checker")
            return OpenAI(api_key=api_key, base_url=base_url)
        raise ValueError(f"Unsupported NLI provider: {provider}")

    def _call_llm(self, system_prompt: str, user_prompt: str, retries: int = 1) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                # Per-request timeout via OpenAI SDK 'timeout' kwarg (seconds).
                resp = self.client.with_options(timeout=30).chat.completions.create(
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
                self.logger.warning(f"NLI LLM call failed (attempt {attempt+1}): {e}")
        self.logger.error(f"NLI LLM call failed after retries: {last_err}")
        return json.dumps({"_failed": True, "reasoning": f"nli error: {last_err}"})
