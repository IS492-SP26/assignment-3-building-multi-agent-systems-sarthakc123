"""
Input Guardrail
Checks user inputs for safety violations.

Policy categories:
- prompt_injection : refuse
- harmful_content  : refuse
- pii              : sanitize (redact)
- off_topic        : redirect
- length           : refuse (low severity also returned for too-short queries)
"""

from typing import Dict, Any, List
import re


# ---- Pattern banks ----------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    # Direct override / forget instructions
    r"ignore (all |the )?(previous|prior|above|prior to) (instructions|prompts|messages|rules)",
    r"ignore (everything|all) (above|before|prior)",
    r"disregard (all |the )?(above|previous|prior) (instructions|prompts|rules)",
    r"forget (everything|all|prior|previous|your instructions)",
    # System prompt extraction
    r"reveal( your| the)? (system )?(prompt|instructions|rules)",
    r"show (me )?(your|the) (system )?(prompt|instructions|rules)",
    r"what (are|is) your (system )?(prompt|instructions|rules)",
    r"repeat (the )?(words |text |phrase |message |sentence )?(above|before|preceding)",
    r"print (your |the )?(system )?(prompt|instructions)",
    # Role-play / persona swap
    r"you are now\s+\S+",          # any "you are now X"
    r"act as (a|an|my|the) ",
    r"pretend (to be|you are|that you are) ",
    r"role[- ]?play as ",
    r"simulate (a|an|the) ",
    # Mode / persona aliases (jailbreak personas)
    r"\bDAN\b",                    # Do-Anything-Now
    r"\bSTAN\b",                   # Strive To Avoid Norms
    r"\bAIM\b",                    # Always Intelligent and Machiavellian
    r"\bDUDE\b",
    r"developer mode",
    r"\badmin mode\b",
    r"\bgod mode\b",
    r"unrestricted mode",
    r"jailbreak",
    # Override / disable safety
    r"override (the |your )?(safety|guardrail|filter|rules?|policy)",
    r"bypass (the |your )?(safety|guardrail|filter|rules?|policy|authentication)",
    r"disable (your |the )?(safety|guardrail|filter|rules?|policy)",
    r"turn off (your |the )?(safety|guardrail|filter)",
    r"\bsudo\b",
    # Authority impersonation
    r"i('|\s)?m (an? )?(openai|anthropic|google|developer|admin|engineer|safety researcher)",
    r"as (a|an|the) (developer|admin|openai engineer|safety researcher)",
    # Hypothetical / fiction framing used to extract harmful content
    r"in a hypothetical (world|scenario|setting)",
    r"hypothetically (speaking|if|how)",
    r"for (educational|fictional|hypothetical) purposes",
    r"write (a |an )?(story|fiction|essay) about how to ",
    # Instruction smuggling
    r"translate (this|the following).{0,40}(ignore|reveal|prompt|instruction)",
]

HARMFUL_KEYWORDS = [
    # Violence / self-harm
    "how to kill", "how do i kill", "how to murder", "how to commit suicide",
    "kill myself", "kill yourself", "self-harm",
    # Weapons / explosives
    "how to make a bomb", "how to build a bomb", "build explosive",
    "make explosive", "synthesize explosive", "make napalm", "build napalm",
    "smuggled past airport", "smuggle past airport",
    # Illegal drugs
    "how to cook meth", "synthesize meth",
    # Cyber-offense
    "write malware", "write ransomware", "create malware", "create a virus",
    "make a virus", "ddos attack", "exploit cve", "hack a server",
    "hack into", "crack the password",
    # CSAM / abuse — single-token guard, not exhaustive
    "child porn", "csam",
]

# HCI-domain keywords used for relevance checks. The bar is intentionally low:
# any one match is enough to consider the query on-topic.
HCI_DOMAIN_KEYWORDS = {
    "hci", "human-computer", "human computer", "ux", "ui", "user experience",
    "user interface", "usability", "accessibility", "interaction design",
    "interaction technique", "design pattern", "wireframe", "prototyp",
    "user-centered", "user centered", "user research", "user study",
    "user studies", "user test", "ux research", "design system",
    "explainable ai", "xai", "ar usability", "ar/vr", "augmented reality",
    "virtual reality", "mixed reality", "ai ethics", "ai literacy",
    "conversational ai", "chatbot", "voice interface", "voice ui",
    "screen reader", "wcag", "responsive design", "data visualization",
    "information visualization", "infovis", "visual encoding",
    "visualiz", "uncertainty",  # broader coverage for visualization queries
    "agentic", "ai-driven", "prototyping tool", "design tool",
    "cross-cultural", "cultural", "localization",
    "accessib", "disability", "elderly", "novice user", "end user",
    "interface design", "interaction", "user", "design",  # broad fallbacks
    "research", "study", "survey", "literature review",
    "ethic", "education", "edtech", "learning",
    "artificial intelligence", "machine learning", "ai ",
    "human-ai", "human ai", "agent",
}


class InputGuardrail:
    """Validate user queries before they reach the agent loop."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full top-level config dict (so we can read both `safety`
                and `system.topic`).
        """
        self.config = config
        safety_cfg = config.get("safety", {})
        system_cfg = config.get("system", {})

        self.min_query_length = int(safety_cfg.get("min_query_length", 5))
        self.max_query_length = int(safety_cfg.get("max_query_length", 2000))
        self.topic = (system_cfg.get("topic") or "").lower()

        # Compile injection patterns once
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS
        ]

    # ---- Public API ---------------------------------------------------------

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate an input query.

        Returns:
            {
                "valid": bool,
                "violations": [ {category, validator, reason, severity}, ... ],
                "action": "allow" | "refuse" | "sanitize" | "redirect",
                "sanitized_input": str
            }
        """
        violations: List[Dict[str, Any]] = []
        text = (query or "").strip()
        sanitized = text

        # 1. Length checks
        if len(text) < self.min_query_length:
            violations.append({
                "category": "length",
                "validator": "min_length",
                "reason": f"Query is too short (<{self.min_query_length} chars).",
                "severity": "low",
            })
        if len(text) > self.max_query_length:
            violations.append({
                "category": "length",
                "validator": "max_length",
                "reason": f"Query exceeds max length ({self.max_query_length} chars).",
                "severity": "medium",
            })

        # 2. Prompt injection
        violations.extend(self._check_prompt_injection(text))

        # 3. Harmful content
        violations.extend(self._check_toxic_language(text))

        # 4. Off-topic relevance (only if no higher-severity issues already)
        if not any(v["severity"] == "high" for v in violations):
            violations.extend(self._check_relevance(text))

        action = self._decide_action(violations)
        valid = action == "allow"

        return {
            "valid": valid,
            "violations": violations,
            "action": action,
            "sanitized_input": sanitized,
        }

    # ---- Helpers ------------------------------------------------------------

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pat in self._injection_re:
            m = pat.search(text)
            if m:
                violations.append({
                    "category": "prompt_injection",
                    "validator": "prompt_injection",
                    "reason": f"Matched injection pattern: {m.group(0)!r}",
                    "severity": "high",
                })
        return violations

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        lowered = text.lower()
        for phrase in HARMFUL_KEYWORDS:
            if phrase in lowered:
                violations.append({
                    "category": "harmful_content",
                    "validator": "harmful_keywords",
                    "reason": f"Matched harmful phrase: {phrase!r}",
                    "severity": "high",
                })
        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Off-topic if no HCI-domain keyword overlaps with the query.

        This is intentionally permissive — only fires when we're confident
        the query has nothing to do with HCI/UX/UI/AI/design/research.
        """
        lowered = query.lower()
        if not lowered:
            return []
        for kw in HCI_DOMAIN_KEYWORDS:
            if kw in lowered:
                return []  # at least one HCI keyword present → on-topic
        return [{
            "category": "off_topic",
            "validator": "domain_relevance",
            "reason": (
                f"Query does not appear related to the configured topic "
                f"({self.topic!r}). No HCI/UX/UI/design/AI keywords detected."
            ),
            "severity": "medium",
        }]

    def _decide_action(self, violations: List[Dict[str, Any]]) -> str:
        """
        Map violations to one of: allow / refuse / sanitize / redirect.

        Precedence: refuse > redirect > sanitize > allow.
        """
        if not violations:
            return "allow"

        cats = {v["category"] for v in violations}
        if "harmful_content" in cats or "prompt_injection" in cats:
            return "refuse"
        # Length issues are treated as refuse (we can't process empty/huge)
        if "length" in cats:
            return "refuse"
        if "off_topic" in cats:
            return "redirect"
        if "pii" in cats:
            return "sanitize"
        return "refuse"
