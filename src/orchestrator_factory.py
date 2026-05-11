"""
Orchestrator Factory
Returns the orchestrator implementation selected by `system.orchestrator` in config.yaml.

Supported values:
- "autogen"   → AutoGenOrchestrator (RoundRobinGroupChat)
- "langgraph" → LangGraphOrchestrator (StateGraph)
"""

from __future__ import annotations

import logging
from typing import Any, Dict


def create_orchestrator(config: Dict[str, Any]):
    """
    Build the orchestrator chosen in config['system']['orchestrator'].

    Both implementations expose the same public surface:
      - process_query(query: str, max_rounds: int = 20) -> dict
      - visualize_workflow() -> str
      - get_agent_descriptions() -> dict
      - safety_manager attribute (for UI to read safety events/stats)

    Defaults to "autogen" if unspecified.
    """
    name = (config.get("system", {}).get("orchestrator") or "autogen").lower()
    logger = logging.getLogger("orchestrator_factory")
    logger.info(f"Selected orchestrator: {name}")

    if name == "autogen":
        from src.autogen_orchestrator import AutoGenOrchestrator
        return AutoGenOrchestrator(config)
    if name == "langgraph":
        from src.langgraph_orchestrator import LangGraphOrchestrator
        return LangGraphOrchestrator(config)
    raise ValueError(
        f"Unknown orchestrator: {name!r}. "
        "Set system.orchestrator in config.yaml to 'autogen' or 'langgraph'."
    )
