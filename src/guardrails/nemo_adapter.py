"""
Optional NeMo Guardrails adapter.

Acts as a SECONDARY safety layer — the custom rule-based input/output
guardrails always run first. If NeMo Guardrails fails to import or
initialize, the adapter sets `available = False` and the SafetyManager
silently skips it. This keeps the system usable even on machines where
NeMo cannot be installed or the LLM endpoint is unreachable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


class NeMoGuardrailsAdapter:
    """Thin wrapper around `nemoguardrails.LLMRails`."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("safety.nemo")
        self.available = False
        self.rails = None

        try:
            from nemoguardrails import LLMRails, RailsConfig  # type: ignore
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"nemoguardrails not importable: {e}")
            return

        config_dir = (
            Path(__file__).parent / "nemo_config"
        ).resolve()
        if not (config_dir / "config.yml").exists():
            self.logger.warning(
                f"NeMo config dir missing: {config_dir}. Adapter disabled."
            )
            return

        # NeMo's OpenAI engine reads OPENAI_API_KEY/OPENAI_API_BASE from env.
        # When using Groq, point it at the Groq OpenAI-compatible endpoint
        # using a temporary env override so we don't clobber the global env.
        original_api_key = os.environ.get("OPENAI_API_KEY")
        original_api_base = os.environ.get("OPENAI_API_BASE")
        groq_key = os.environ.get("GROQ_API_KEY")
        try:
            if groq_key:
                os.environ["OPENAI_API_KEY"] = groq_key
                os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

            rails_config = RailsConfig.from_path(str(config_dir))
            self.rails = LLMRails(rails_config)
            self.available = True
            self.logger.info("NeMo Guardrails initialized successfully.")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"NeMo Guardrails init failed: {e}. Adapter disabled.")
            self.available = False
        finally:
            # Restore original env so other components (like the agent client)
            # use whatever provider was configured for them.
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            elif "OPENAI_API_KEY" in os.environ and groq_key:
                # Only delete if WE set it
                del os.environ["OPENAI_API_KEY"]
            if original_api_base is not None:
                os.environ["OPENAI_API_BASE"] = original_api_base
            elif "OPENAI_API_BASE" in os.environ and groq_key:
                del os.environ["OPENAI_API_BASE"]

    # ---- Public API ---------------------------------------------------------

    def check_input(self, text: str) -> Dict[str, Any]:
        """Returns {'safe': bool, 'reason': str, 'category': str}."""
        if not self.available or self.rails is None:
            return {"safe": True, "reason": "nemo_unavailable", "category": "n/a"}
        try:
            response = self.rails.generate(messages=[
                {"role": "user", "content": text},
            ])
            content = (response or {}).get("content", "") if isinstance(response, dict) else str(response)
            # NeMo's self-check refuses by emitting "I'm sorry"-style refusals
            if self._looks_like_refusal(content):
                return {
                    "safe": False,
                    "reason": content[:200],
                    "category": "harmful_content",
                }
            return {"safe": True, "reason": "ok", "category": "n/a"}
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"NeMo check_input errored: {e}")
            return {"safe": True, "reason": f"nemo_error:{e}", "category": "n/a"}

    def check_output(self, text: str) -> Dict[str, Any]:
        """Returns {'safe': bool, 'reason': str, 'category': str}."""
        if not self.available or self.rails is None:
            return {"safe": True, "reason": "nemo_unavailable", "category": "n/a"}
        try:
            # Use generate with a synthetic conversation so we exercise output rails.
            response = self.rails.generate(messages=[
                {"role": "user", "content": "Please review the following assistant output."},
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Was that response acceptable?"},
            ])
            content = (response or {}).get("content", "") if isinstance(response, dict) else str(response)
            if self._looks_like_refusal(content):
                return {
                    "safe": False,
                    "reason": content[:200],
                    "category": "harmful_content",
                }
            return {"safe": True, "reason": "ok", "category": "n/a"}
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"NeMo check_output errored: {e}")
            return {"safe": True, "reason": f"nemo_error:{e}", "category": "n/a"}

    # ---- Helpers ------------------------------------------------------------

    @staticmethod
    def _looks_like_refusal(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        triggers = (
            "i'm sorry, i can't",
            "i cannot help with that",
            "i can't help with that",
            "i'm not able to",
            "i won't be able to",
            "violates",
            "against policy",
        )
        return any(t in lowered for t in triggers)
