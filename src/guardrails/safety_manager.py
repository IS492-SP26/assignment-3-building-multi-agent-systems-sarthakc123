"""
Safety Manager
Coordinates input/output guardrails and logs every safety event.

Responsibilities:
- Run InputGuardrail on user queries.
- Run OutputGuardrail on model outputs.
- Optionally run NeMoGuardrails as a second-pass layer (degrades gracefully if unavailable).
- Append every check (safe or not) to a JSONL log file with a unique event_id.
- Expose stats and event history for the UI to display.
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from src.guardrails.input_guardrail import InputGuardrail
from src.guardrails.output_guardrail import OutputGuardrail


class SafetyManager:
    """Coordinate safety guardrails for the multi-agent system."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Either the full config dict (preferred) or just the
                `safety` block. We accept both for backwards-compat with
                callers that pass `config["safety"]`.
        """
        # Accept either full config or just the safety subsection
        if "safety" in config:
            self.full_config = config
            safety_cfg = config.get("safety", {})
        else:
            self.full_config = {"safety": config}
            safety_cfg = config

        self.config = safety_cfg
        self.enabled = bool(safety_cfg.get("enabled", True))
        self.log_events_enabled = bool(safety_cfg.get("log_events", True))
        self.use_nemo = bool(safety_cfg.get("use_nemo", False))
        self.logger = logging.getLogger("safety")

        # Default action when category-specific mapping is missing
        self.default_action = safety_cfg.get("on_violation", {}).get("action", "refuse")
        self.default_message = safety_cfg.get("on_violation", {}).get(
            "message", "I cannot process this request due to safety policies."
        )

        # Set up safety log file (JSONL)
        log_path = safety_cfg.get("safety_log_file", "logs/safety_events.log")
        self.safety_log_path = Path(log_path)
        self.safety_log_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory event log (used by UI to render the safety panel)
        self.safety_events: List[Dict[str, Any]] = []

        # Initialize guardrail layers
        self.input_guardrail = InputGuardrail(self.full_config)
        self.output_guardrail = OutputGuardrail(self.full_config)

        # Optional NeMo layer
        self.nemo = None
        if self.use_nemo:
            try:
                from src.guardrails.nemo_adapter import NeMoGuardrailsAdapter
                self.nemo = NeMoGuardrailsAdapter(self.full_config)
                if not getattr(self.nemo, "available", False):
                    self.logger.warning(
                        "NeMo Guardrails configured but adapter unavailable; "
                        "falling back to custom rules only."
                    )
                    self.nemo = None
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"NeMo Guardrails layer disabled: {e}")
                self.nemo = None

        self.logger.info(
            "SafetyManager initialized "
            f"(enabled={self.enabled}, nemo={'on' if self.nemo else 'off'})"
        )

    # ---- Public API ---------------------------------------------------------

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Validate a user query.

        Returns:
            {
                "safe": bool,
                "action": "allow" | "refuse" | "redirect" | "sanitize",
                "violations": [...],
                "policy_categories": [str, ...],
                "message": str,            # user-facing message if not safe
                "query": str,              # possibly sanitized query
                "event_id": str,
            }
        """
        if not self.enabled:
            return {"safe": True, "action": "allow", "query": query, "violations": []}

        result = self.input_guardrail.validate(query)

        # Optional NeMo second-pass
        if self.nemo is not None:
            try:
                nemo_check = self.nemo.check_input(query)
                if not nemo_check.get("safe", True):
                    result["violations"].append({
                        "category": nemo_check.get("category", "harmful_content"),
                        "validator": "nemo_guardrails",
                        "reason": nemo_check.get("reason", "Blocked by NeMo policy"),
                        "severity": "high",
                    })
                    result["action"] = "refuse"
                    result["valid"] = False
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"NeMo input check errored: {e}")

        action = result["action"]
        safe = action == "allow"
        cats = sorted({v["category"] for v in result["violations"]})

        # Build user-facing message
        message = self._build_user_message(action, cats)

        event = self._log_safety_event(
            event_type="input",
            content=query,
            violations=result["violations"],
            safe=safe,
            action=action,
            policy_categories=cats,
        )

        return {
            "safe": safe,
            "action": action,
            "violations": result["violations"],
            "policy_categories": cats,
            "message": message,
            "query": result.get("sanitized_input", query),
            "event_id": event["event_id"],
        }

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a generated response.

        Returns:
            {
                "safe": bool,
                "action": "allow" | "refuse" | "sanitize",
                "violations": [...],
                "policy_categories": [str, ...],
                "response": str,    # sanitized or refused
                "event_id": str,
            }
        """
        if not self.enabled:
            return {"safe": True, "action": "allow", "response": response, "violations": []}

        result = self.output_guardrail.validate(response, sources)

        # Optional NeMo second-pass
        if self.nemo is not None:
            try:
                nemo_check = self.nemo.check_output(response)
                if not nemo_check.get("safe", True):
                    result["violations"].append({
                        "category": nemo_check.get("category", "harmful_content"),
                        "validator": "nemo_guardrails",
                        "reason": nemo_check.get("reason", "Blocked by NeMo policy"),
                        "severity": "high",
                    })
                    result["action"] = "refuse"
                    result["valid"] = False
                    result["sanitized_output"] = self.default_message
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"NeMo output check errored: {e}")

        action = result["action"]
        safe = action == "allow"
        cats = sorted({v["category"] for v in result["violations"]})

        event = self._log_safety_event(
            event_type="output",
            content=response,
            violations=result["violations"],
            safe=safe,
            action=action,
            policy_categories=cats,
        )

        return {
            "safe": safe,
            "action": action,
            "violations": result["violations"],
            "policy_categories": cats,
            "response": result["sanitized_output"],
            "event_id": event["event_id"],
        }

    def get_safety_events(self) -> List[Dict[str, Any]]:
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        unsafe = sum(1 for e in self.safety_events if not e["safe"])
        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": unsafe,
            "violation_rate": (unsafe / total) if total else 0.0,
        }

    def clear_events(self) -> None:
        self.safety_events = []

    # ---- Internals ----------------------------------------------------------

    def _build_user_message(self, action: str, categories: List[str]) -> str:
        if action == "allow":
            return ""
        if action == "refuse":
            cat_label = ", ".join(categories) if categories else "policy violation"
            return (
                f"{self.default_message} "
                f"(Triggered category: {cat_label})"
            )
        if action == "redirect":
            return (
                "Your query appears off-topic for this assistant "
                "(scope: HCI / user-experience / interaction-design research). "
                "Please rephrase to focus on UX/UI/HCI topics."
            )
        if action == "sanitize":
            return "The response has been sanitized to remove sensitive content."
        return self.default_message

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        safe: bool,
        action: str,
        policy_categories: List[str],
    ) -> Dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": safe,
            "action": action,
            "policy_categories": policy_categories,
            "violations": violations,
            "content_preview": (
                content[:200] + "..." if len(content or "") > 200 else (content or "")
            ),
        }

        self.safety_events.append(event)

        if not safe:
            self.logger.warning(
                f"Safety event ({event_type}): action={action} "
                f"categories={policy_categories}"
            )

        if self.log_events_enabled:
            try:
                with open(self.safety_log_path, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Failed to write safety log: {e}")

        return event
