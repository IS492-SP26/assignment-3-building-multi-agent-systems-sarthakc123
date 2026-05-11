"""
LangGraph-based Orchestrator (parallel implementation to AutoGen).

This orchestrator implements the same agent topology
(Planner → Researcher → Writer → Critic) using LangGraph's StateGraph
instead of AutoGen's RoundRobinGroupChat.

Why this exists:
- LangGraph gives explicit, visualizable control flow (vs. AutoGen's implicit
  round-robin) and makes the revision loop ("critic asks for revision → back
  to writer") a first-class graph edge.
- The Researcher node calls the search tools directly rather than relying on
  the LLM's tool-calling capability, which lets it work on backends that
  don't support function calls (e.g. the assignment vLLM endpoint).

Public surface matches `AutoGenOrchestrator` so the CLI / Streamlit / evaluator
can swap implementations via `system.orchestrator` in config.yaml.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional, TypedDict

from openai import OpenAI

from src.guardrails.safety_manager import SafetyManager
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


# Default tool-call timeout (seconds). Prevents Tavily / Semantic Scholar
# hangs from blocking the orchestrator indefinitely.
TOOL_TIMEOUT_S = 30


def _call_with_timeout(fn, *args, timeout: int = TOOL_TIMEOUT_S, **kwargs):
    """Run `fn(*args, **kwargs)` in a worker thread, kill it after `timeout`s."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout)


def _safe_diff_for_ui(state_diff: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate state diffs so the UI doesn't choke on giant strings."""
    out: Dict[str, Any] = {}
    for k, v in state_diff.items():
        if isinstance(v, str):
            out[k] = v[:400] + ("…" if len(v) > 400 else "")
        elif isinstance(v, list):
            # show count + last item preview if it's messages
            if k == "messages" and v:
                last = v[-1] if isinstance(v[-1], dict) else {"content": str(v[-1])}
                out[k] = {"count": len(v), "last_source": last.get("source", "?"),
                          "last_preview": (last.get("content") or "")[:300]}
            else:
                out[k] = {"count": len(v)}
        else:
            out[k] = v
    return out


# ---- Default system prompts (kept in sync with autogen_agents.py) ----------

DEFAULT_PLANNER_PROMPT = """You are a Research Planner. Break down the query into 3–5 numbered, actionable research steps. Identify key concepts, recommended sources (academic vs. web), and specific search queries the Researcher should run. Be concise."""

DEFAULT_WRITER_PROMPT = """You are a Research Writer. Synthesize the research findings into a clear, well-organized response that directly answers the original query. Use inline citations in the form [Source: Title or URL]. Include a brief References section at the end. Avoid copying source text verbatim — paraphrase and synthesize."""

DEFAULT_CRITIC_PROMPT = """You are a Research Critic. Evaluate the draft on Relevance, Evidence Quality, Completeness, Accuracy, and Clarity. If approved, end your message with the literal token "APPROVED". If improvements are needed, end with "NEEDS REVISION" and list specific suggestions."""


# ---- State ------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    query: str
    plan: str
    research_findings: List[str]
    sources: List[Dict[str, Any]]
    draft: str
    critique: str
    final_response: str
    iteration_count: int
    messages: List[Dict[str, str]]  # ordered transcript [{source, content}]


# ---- Orchestrator -----------------------------------------------------------

class LangGraphOrchestrator:
    """Multi-agent research orchestration using LangGraph."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("langgraph_orchestrator")

        # Always-present safety manager (matches AutoGen path)
        self.safety_manager = SafetyManager(config)

        # LLM client — reuse the same provider/model as the AutoGen agents
        self.client = self._make_client(config)
        model_cfg = config.get("models", {}).get("default", {})
        self.model_name = model_cfg.get("name", "llama-3.3-70b-versatile")
        self.temperature = float(model_cfg.get("temperature", 0.7))
        self.max_tokens = int(model_cfg.get("max_tokens", 2048))

        # Iteration cap
        self.max_iterations = int(config.get("system", {}).get("max_iterations", 2))

        # Build the graph (lazy — defer import so AutoGen-only users don't pay)
        self.graph = self._build_graph()

    # ---- Public API ---------------------------------------------------------

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """Run the LangGraph and return a result dict shaped like AutoGenOrchestrator."""
        self.logger.info(f"Processing query: {query}")

        # 1. Input safety check
        input_check = self.safety_manager.check_input_safety(query)
        if not input_check["safe"]:
            self.logger.warning(
                f"Query blocked by input guardrail: {input_check['policy_categories']}"
            )
            return {
                "query": query,
                "response": input_check["message"],
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0,
                    "num_sources": 0,
                    "agents_involved": [],
                    "blocked_by_safety": True,
                    "safety_events": [{
                        "type": "input",
                        "action": input_check["action"],
                        "policy_categories": input_check["policy_categories"],
                        "violations": input_check["violations"],
                        "event_id": input_check["event_id"],
                    }],
                    "orchestrator": "langgraph",
                },
            }

        # 2. Run the graph
        initial_state: ResearchState = {
            "query": query,
            "plan": "",
            "research_findings": [],
            "sources": [],
            "draft": "",
            "critique": "",
            "final_response": "",
            "iteration_count": 0,
            "messages": [{"source": "user", "content": query}],
        }

        try:
            # LangGraph invoke is synchronous; node-level errors bubble up here
            final_state: ResearchState = self.graph.invoke(initial_state)
        except Exception as e:
            self.logger.error(f"LangGraph execution error: {e}", exc_info=True)
            return {
                "query": query,
                "response": f"An error occurred while processing your query: {e}",
                "conversation_history": [],
                "metadata": {"error": True, "orchestrator": "langgraph"},
            }

        # 3. Output safety check
        sources = final_state.get("sources", [])
        output_check = self.safety_manager.check_output_safety(
            final_state.get("final_response", ""),
            sources=sources,
        )

        return {
            "query": query,
            "response": output_check["response"],
            "conversation_history": final_state.get("messages", []),
            "metadata": {
                "num_messages": len(final_state.get("messages", [])),
                "num_sources": len(sources),
                "agents_involved": list({m["source"] for m in final_state.get("messages", [])}),
                "plan": final_state.get("plan", ""),
                "research_findings": final_state.get("research_findings", []),
                "critique": final_state.get("critique", ""),
                "iteration_count": final_state.get("iteration_count", 0),
                "sources": sources,
                "orchestrator": "langgraph",
                "safety_events": [
                    {
                        "type": "input",
                        "action": input_check["action"],
                        "policy_categories": input_check["policy_categories"],
                        "violations": input_check["violations"],
                        "event_id": input_check["event_id"],
                    },
                    {
                        "type": "output",
                        "action": output_check["action"],
                        "policy_categories": output_check["policy_categories"],
                        "violations": output_check["violations"],
                        "event_id": output_check["event_id"],
                    },
                ],
            },
        }

    def process_query_stream(self, query: str) -> Iterator[Dict[str, Any]]:
        """
        Streaming variant of `process_query` — yields lifecycle events so the UI
        can render live per-node updates.

        Events:
          {"type": "input_check", "action": ..., "policy_categories": [...], "event_id": ...}
          {"type": "blocked",     "result": <final dict>}            # if blocked at input
          {"type": "node_start",  "node": "planner"|"researcher"|"writer"|"critic", "started_at": <epoch>}
          {"type": "node_end",    "node": ..., "elapsed_s": ..., "state_diff": {...}}
          {"type": "output_check","action": ..., "policy_categories": [...], "event_id": ...}
          {"type": "done",        "result": <final dict>}            # always last
        """
        self.logger.info(f"[stream] processing query: {query}")

        # 1. Input safety check
        input_check = self.safety_manager.check_input_safety(query)
        yield {
            "type": "input_check",
            "action": input_check["action"],
            "policy_categories": input_check["policy_categories"],
            "violations": input_check["violations"],
            "event_id": input_check["event_id"],
            "safe": input_check["safe"],
        }
        if not input_check["safe"]:
            self.logger.warning(
                f"[stream] blocked at input: {input_check['policy_categories']}"
            )
            yield {
                "type": "blocked",
                "result": {
                    "query": query,
                    "response": input_check["message"],
                    "conversation_history": [],
                    "metadata": {
                        "num_messages": 0,
                        "num_sources": 0,
                        "agents_involved": [],
                        "blocked_by_safety": True,
                        "safety_events": [{
                            "type": "input",
                            "action": input_check["action"],
                            "policy_categories": input_check["policy_categories"],
                            "violations": input_check["violations"],
                            "event_id": input_check["event_id"],
                        }],
                        "orchestrator": "langgraph",
                    },
                },
            }
            return

        # 2. Stream the graph node-by-node
        initial_state: ResearchState = {
            "query": query,
            "plan": "",
            "research_findings": [],
            "sources": [],
            "draft": "",
            "critique": "",
            "final_response": "",
            "iteration_count": 0,
            "messages": [{"source": "user", "content": query}],
        }
        final_state: Dict[str, Any] = dict(initial_state)
        # Track per-node timing using a stack — graph.stream doesn't emit a
        # node_start event, only node_end (as the dict key). So we emit a
        # synthetic "node_start" before the LangGraph stream call and rely on
        # the user-facing timing being approximate.
        last_emit_t = time.time()
        try:
            for event in self.graph.stream(initial_state):
                # event is {<node_name>: <updated_state_subset>}
                for node_name, state_diff in event.items():
                    now = time.time()
                    elapsed = now - last_emit_t
                    last_emit_t = now
                    # Merge into final_state
                    for k, v in (state_diff or {}).items():
                        if k == "messages" and isinstance(v, list):
                            # Replace (LangGraph already merged appends)
                            final_state["messages"] = v
                        else:
                            final_state[k] = v
                    yield {
                        "type": "node_end",
                        "node": node_name,
                        "elapsed_s": round(elapsed, 1),
                        "state_diff": _safe_diff_for_ui(state_diff or {}),
                    }
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"[stream] graph error: {e}", exc_info=True)
            yield {
                "type": "done",
                "result": {
                    "query": query,
                    "response": f"An error occurred while processing your query: {e}",
                    "conversation_history": final_state.get("messages", []),
                    "metadata": {"error": True, "orchestrator": "langgraph"},
                },
            }
            return

        # 3. Output safety check
        sources = final_state.get("sources", []) or []
        output_check = self.safety_manager.check_output_safety(
            final_state.get("final_response", "") or final_state.get("draft", ""),
            sources=sources,
        )
        yield {
            "type": "output_check",
            "action": output_check["action"],
            "policy_categories": output_check["policy_categories"],
            "violations": output_check["violations"],
            "event_id": output_check["event_id"],
        }

        # 4. Final assembled result
        yield {
            "type": "done",
            "result": {
                "query": query,
                "response": output_check["response"],
                "conversation_history": final_state.get("messages", []),
                "metadata": {
                    "num_messages": len(final_state.get("messages", []) or []),
                    "num_sources": len(sources),
                    "agents_involved": list({
                        m.get("source", "Unknown") for m in (final_state.get("messages") or [])
                    }),
                    "plan": final_state.get("plan", ""),
                    "research_findings": final_state.get("research_findings", []),
                    "critique": final_state.get("critique", ""),
                    "iteration_count": final_state.get("iteration_count", 0),
                    "sources": sources,
                    "orchestrator": "langgraph",
                    "safety_events": [
                        {
                            "type": "input",
                            "action": input_check["action"],
                            "policy_categories": input_check["policy_categories"],
                            "violations": input_check["violations"],
                            "event_id": input_check["event_id"],
                        },
                        {
                            "type": "output",
                            "action": output_check["action"],
                            "policy_categories": output_check["policy_categories"],
                            "violations": output_check["violations"],
                            "event_id": output_check["event_id"],
                        },
                    ],
                },
            },
        }

    def visualize_workflow(self) -> str:
        return """LangGraph Research Workflow:

START → Planner → Researcher → Writer → Critic → (decision)

decision:
  - critique contains "NEEDS REVISION" AND iteration < max → back to Writer
  - else → END
"""

    def get_agent_descriptions(self) -> Dict[str, str]:
        return {
            "Planner":    "Decomposes query into research steps",
            "Researcher": "Calls web_search/paper_search to gather evidence",
            "Writer":     "Synthesizes findings with inline citations",
            "Critic":     "Evaluates quality; can request a revision",
        }

    # ---- Internals ----------------------------------------------------------

    def _make_client(self, config: Dict[str, Any]) -> OpenAI:
        """Return an OpenAI-compatible client pointed at the configured provider."""
        model_cfg = config.get("models", {}).get("default", {})
        provider = model_cfg.get("provider", "groq")

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        if provider == "vllm":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key or not base_url:
                raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL required for vllm provider")
            return OpenAI(api_key=api_key, base_url=base_url)
        raise ValueError(f"Unsupported provider for LangGraph: {provider}")

    def _build_graph(self):
        """Compile the StateGraph. Imported here to keep module-level lightweight."""
        from langgraph.graph import StateGraph, END

        sg = StateGraph(ResearchState)
        sg.add_node("planner", self._planner_node)
        sg.add_node("researcher", self._researcher_node)
        sg.add_node("writer", self._writer_node)
        sg.add_node("critic", self._critic_node)

        sg.set_entry_point("planner")
        sg.add_edge("planner", "researcher")
        sg.add_edge("researcher", "writer")
        sg.add_edge("writer", "critic")
        sg.add_conditional_edges(
            "critic",
            self._critic_decision,
            {"revise": "writer", "end": END},
        )
        return sg.compile()

    # ---- Nodes --------------------------------------------------------------

    def _planner_node(self, state: ResearchState) -> Dict[str, Any]:
        prompt = self._agent_prompt("planner", DEFAULT_PLANNER_PROMPT)
        plan = self._call_llm(prompt, f"Research query: {state['query']}\n\nProduce the plan now.")
        msgs = state.get("messages", []) + [{"source": "Planner", "content": plan}]
        return {"plan": plan, "messages": msgs}

    def _researcher_node(self, state: ResearchState) -> Dict[str, Any]:
        # Honor per-tool config flags (`tools.web_search.enabled` /
        # `tools.paper_search.enabled`). Disabled tools are skipped entirely
        # so they cannot block on rate-limited backoffs.
        tools_cfg = (self.config.get("tools") or {})
        web_enabled = (tools_cfg.get("web_search") or {}).get("enabled", True)
        paper_enabled = (tools_cfg.get("paper_search") or {}).get("enabled", True)
        web_results = "(web_search disabled)"
        paper_results = "(paper_search disabled)"

        if web_enabled:
            try:
                web_results = _call_with_timeout(web_search, state["query"], max_results=5)
            except concurrent.futures.TimeoutError:
                self.logger.warning(f"web_search timed out after {TOOL_TIMEOUT_S}s")
                web_results = "(web_search timed out)"
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"web_search failed: {e}")
                web_results = "(web_search unavailable)"

        if paper_enabled:
            try:
                paper_results = _call_with_timeout(paper_search, state["query"], max_results=5)
            except concurrent.futures.TimeoutError:
                self.logger.warning(f"paper_search timed out after {TOOL_TIMEOUT_S}s")
                paper_results = "(paper_search timed out)"
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"paper_search failed: {e}")
                paper_results = "(paper_search unavailable)"

        # Extract URLs as structured sources
        urls = list(set(re.findall(r"https?://[^\s<>\"{}|\\^\[\]]+", f"{web_results}\n{paper_results}")))
        sources = [{"url": u, "title": "", "snippet": ""} for u in urls]

        # Use the LLM to digest raw search output into organized findings
        digest_prompt = (
            "You are a Research Assistant. Summarize the following raw search "
            "results into 5–8 bullet points of evidence relevant to the query. "
            "For each bullet, include a `[Source: URL]` marker."
        )
        digest_input = (
            f"Query: {state['query']}\n\n"
            f"--- WEB RESULTS ---\n{web_results}\n\n"
            f"--- PAPER RESULTS ---\n{paper_results}\n"
        )
        findings = self._call_llm(digest_prompt, digest_input)

        msgs = state.get("messages", []) + [
            {"source": "Researcher", "content": f"raw web results:\n{web_results[:500]}"},
            {"source": "Researcher", "content": f"raw paper results:\n{paper_results[:500]}"},
            {"source": "Researcher", "content": findings},
        ]
        return {"research_findings": [findings], "sources": sources, "messages": msgs}

    def _writer_node(self, state: ResearchState) -> Dict[str, Any]:
        prompt = self._agent_prompt("writer", DEFAULT_WRITER_PROMPT)
        critique_block = ""
        if state.get("critique") and state.get("iteration_count", 0) > 0:
            critique_block = (
                f"\n\nCRITIC FEEDBACK FROM PREVIOUS DRAFT:\n{state['critique']}\n\n"
                "Address the feedback above in this revision."
            )
        user_msg = (
            f"Original query: {state['query']}\n\n"
            f"Research plan:\n{state.get('plan','')}\n\n"
            f"Research findings:\n" + "\n".join(state.get("research_findings", []))
            + critique_block
            + "\n\nWrite the final answer now."
        )
        draft = self._call_llm(prompt, user_msg)
        msgs = state.get("messages", []) + [{"source": "Writer", "content": draft}]
        return {"draft": draft, "final_response": draft, "messages": msgs}

    def _critic_node(self, state: ResearchState) -> Dict[str, Any]:
        prompt = self._agent_prompt("critic", DEFAULT_CRITIC_PROMPT)
        user_msg = (
            f"Original query: {state['query']}\n\n"
            f"Draft:\n{state.get('draft','')}\n\n"
            "Evaluate. Conclude with APPROVED or NEEDS REVISION."
        )
        critique = self._call_llm(prompt, user_msg)
        msgs = state.get("messages", []) + [{"source": "Critic", "content": critique}]
        return {
            "critique": critique,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": msgs,
        }

    def _critic_decision(self, state: ResearchState) -> str:
        critique = (state.get("critique") or "").upper()
        if "NEEDS REVISION" in critique and state.get("iteration_count", 0) < self.max_iterations:
            self.logger.info("Critic requested revision; routing back to writer.")
            return "revise"
        return "end"

    # ---- Helpers ------------------------------------------------------------

    def _agent_prompt(self, name: str, default: str) -> str:
        custom = self.config.get("agents", {}).get(name, {}).get("system_prompt", "") or ""
        return custom.strip() if custom.strip() else default

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        try:
            # 60s per-request timeout — agent calls are larger than NLI calls.
            resp = self.client.with_options(timeout=60).chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"LLM call failed: {e}")
            return f"(LLM error: {e})"
