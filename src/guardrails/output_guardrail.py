"""
Output Guardrail
Checks generated responses for safety violations before they reach the user.

Policy categories handled here:
- pii                : redact (sanitize)
- harmful_content    : refuse (replace with refusal text)
- misinformation_risk: flag (sanitize / annotate)
"""

from typing import Dict, Any, List, Optional
import re


# ---- Pattern banks ----------------------------------------------------------

# PII patterns
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # US-style phone: 555-555-5555, 555.555.5555, (555) 555-5555
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit-card: 13-19 digits, optionally separated by spaces or dashes
    "credit_card": re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    ),
}

# Harmful content keywords applied to outputs (catches model drift / jailbreak success)
HARMFUL_OUTPUT_KEYWORDS = [
    "step-by-step instructions to build a bomb",
    "instructions to synthesize",
    "how to make a bomb", "how to build a bomb",
    "kill yourself", "kill myself",
    "child porn", "csam",
    "ransomware code", "exploit code for cve",
]

BIAS_PATTERNS = [
    r"\ball (men|women|blacks|whites|asians|jews|muslims|christians|hindus) are\b",
    r"\bevery (man|woman|black|white|asian|jew|muslim|christian|hindu) is\b",
]


class OutputGuardrail:
    """Validate generated responses before they are returned to the user."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        safety_cfg = config.get("safety", {})
        self.refusal_message = safety_cfg.get("on_violation", {}).get(
            "message", "I cannot provide this response due to safety policies."
        )
        self._bias_re = [re.compile(p, re.IGNORECASE) for p in BIAS_PATTERNS]

        # NLI-based hallucination detection (optional, controlled by config)
        self.use_nli_check = bool(safety_cfg.get("use_nli_check", False))
        self._nli_checker = None
        if self.use_nli_check:
            try:
                from src.guardrails.nli_check import NLIChecker
                self._nli_checker = NLIChecker(config)
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger("safety.output").warning(
                    f"NLI checker disabled (init failed): {e}"
                )
                self._nli_checker = None

    # ---- Public API ---------------------------------------------------------

    def validate(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a generated response.

        Returns:
            {
                "valid": bool,
                "violations": [...],
                "action": "allow" | "refuse" | "sanitize" | "flag",
                "sanitized_output": str
            }
        """
        violations: List[Dict[str, Any]] = []

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_bias(response))
        if sources:
            violations.extend(self._check_factual_consistency(response, sources))
            # NLI-based hallucination detection (bonus innovation)
            if self.use_nli_check and self._nli_checker is not None:
                violations.extend(self._check_unsupported_claims(response, sources))

        action = self._decide_action(violations)
        sanitized = self._sanitize(response, violations) if violations else response

        # If refusing, replace output with refusal message
        if action == "refuse":
            sanitized = self.refusal_message

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "action": action,
            "sanitized_output": sanitized,
        }

    # ---- Helpers ------------------------------------------------------------

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for pii_type, pat in PII_PATTERNS.items():
            matches = pat.findall(text)
            # Filter false positives for credit card: only keep strings that have
            # at least 13 digits when whitespace/dashes are stripped.
            if pii_type == "credit_card":
                matches = [m for m in matches if len(re.sub(r"[\s-]", "", m)) >= 13]
            if matches:
                violations.append({
                    "category": "pii",
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Output contains {pii_type} value(s).",
                    "severity": "high",
                    "matches": list({m if isinstance(m, str) else m[0] for m in matches}),
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        lowered = text.lower()
        for phrase in HARMFUL_OUTPUT_KEYWORDS:
            if phrase in lowered:
                violations.append({
                    "category": "harmful_content",
                    "validator": "harmful_keywords",
                    "reason": f"Output contains harmful phrase: {phrase!r}",
                    "severity": "high",
                })
        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pat in self._bias_re:
            m = pat.search(text)
            if m:
                violations.append({
                    "category": "bias",
                    "validator": "bias_generalization",
                    "reason": f"Output contains generalization: {m.group(0)!r}",
                    "severity": "medium",
                })
        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Flag uncited claims: if the response uses [Source: X] markers but X
        does not match any retrieved source's title or URL fragment, flag it.

        Source dicts may have keys: title, url, authors, snippet.
        """
        violations: List[Dict[str, Any]] = []
        cited = re.findall(r"\[Source:\s*([^\]]+)\]", response, flags=re.IGNORECASE)
        if not cited:
            return violations

        # Build a lowercase haystack of source titles + URLs
        haystack_parts = []
        for s in sources or []:
            for k in ("title", "url", "authors", "snippet"):
                v = s.get(k) if isinstance(s, dict) else None
                if v:
                    haystack_parts.append(str(v).lower())
        haystack = " || ".join(haystack_parts)

        for c in cited:
            tag = c.strip().lower()
            if not tag:
                continue
            # Match if any non-trivial token (>=4 chars) appears in haystack
            tokens = [t for t in re.split(r"\W+", tag) if len(t) >= 4]
            if not tokens:
                continue
            if not any(tok in haystack for tok in tokens):
                violations.append({
                    "category": "misinformation_risk",
                    "validator": "ungrounded_citation",
                    "reason": f"Citation [{c}] not found in any retrieved source.",
                    "severity": "medium",
                    "citation": c,
                })
        return violations

    def _check_unsupported_claims(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        NLI-based hallucination check (bonus innovation).

        Asks an LLM to extract atomic claims, then verifies each against the
        retrieved sources. Records one violation per non-entailed claim.
        Claims whose check failed (`entailed is None`) are intentionally
        ignored — we never want to over-flag based on transport errors.
        """
        violations: List[Dict[str, Any]] = []
        if self._nli_checker is None:
            return violations
        try:
            verdicts = self._nli_checker.check_claims(response, sources)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger("safety.output").warning(f"NLI check errored: {e}")
            return violations

        for v in verdicts:
            if v.get("entailed") is False:
                violations.append({
                    "category": "unsupported_claim",
                    "validator": "nli_entailment",
                    "reason": v.get("reasoning", "claim not entailed by retrieved sources"),
                    "severity": "medium",
                    "claim": v.get("claim", ""),
                })
        return violations

    def _sanitize(
        self,
        text: str,
        violations: List[Dict[str, Any]],
    ) -> str:
        sanitized = text
        for v in violations:
            if v.get("category") == "pii":
                pii_type = v.get("pii_type", "PII")
                for match in v.get("matches", []):
                    sanitized = sanitized.replace(match, f"[REDACTED-{pii_type.upper()}]")
            elif v.get("category") == "misinformation_risk":
                # Annotate the ungrounded citation
                cit = v.get("citation", "")
                if cit:
                    sanitized = sanitized.replace(
                        f"[Source: {cit}]",
                        f"[UNVERIFIED: {cit}]",
                    )
            elif v.get("category") == "unsupported_claim":
                # Annotate the unsupported claim inline (best-effort literal match)
                claim = v.get("claim", "")
                if claim and claim in sanitized:
                    sanitized = sanitized.replace(
                        claim,
                        f"{claim} [UNSUPPORTED: {v.get('reason', 'no source entails this')[:80]}]",
                    )
        return sanitized

    def _decide_action(self, violations: List[Dict[str, Any]]) -> str:
        """
        Map violations to an action.

        Precedence: refuse (harmful) > sanitize (pii / bias / misinfo) > allow.
        """
        if not violations:
            return "allow"
        cats = {v["category"] for v in violations}
        if "harmful_content" in cats:
            return "refuse"
        # All non-harmful violations (pii, bias, misinformation_risk, unsupported_claim)
        # are sanitization-grade.
        return "sanitize"
