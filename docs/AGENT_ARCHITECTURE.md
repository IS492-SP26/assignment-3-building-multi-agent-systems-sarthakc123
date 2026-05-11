# Agent Architecture

This document describes how the multi-agent research system is organized: the
agent roles, the two interchangeable orchestrators (AutoGen and LangGraph), the
safety pipeline, and the data flow between components.

---

## 1. High-level overview

```
                     ┌─────────────────────────────┐
   user query ─────▶ │      SafetyManager          │  (input guardrail)
                     └──────────────┬──────────────┘
                                    │ allow / refuse / redirect
                                    ▼
                     ┌─────────────────────────────┐
                     │       Orchestrator          │
                     │  (AutoGen  |  LangGraph)    │
                     └──────────────┬──────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
           ┌────────┐         ┌──────────┐        ┌────────┐
           │Planner │  ───▶   │Researcher│  ───▶  │ Writer │
           └────────┘         └────┬─────┘        └────┬───┘
                                   │ tools             │
                                   ▼                   ▼
                       ┌─────────────────────┐    ┌────────┐
                       │ web_search()        │    │ Critic │
                       │ paper_search()      │    └────┬───┘
                       └─────────────────────┘         │
                                                       │ APPROVED?
                                                       ▼
                     ┌─────────────────────────────┐
   final answer ◀── │      SafetyManager          │  (output guardrail)
                     └─────────────────────────────┘
```

Two execution modes for the same agent topology:

- **AutoGen** (`src/autogen_orchestrator.py`) — uses
  [`autogen_agentchat.teams.RoundRobinGroupChat`][1] with a
  [`TextMentionTermination`][2] condition. Each agent gets a turn in fixed
  order; the team stops when any message contains `TERMINATE`.
- **LangGraph** (`src/langgraph_orchestrator.py`) — explicit `StateGraph` with
  conditional edges and an iteration counter for revision loops.

[1]: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/teams.html
[2]: https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.conditions.html

Both implementations expose the same public surface:

```python
class Orchestrator:
    def __init__(self, config: dict): ...
    def process_query(self, query: str, max_rounds: int = 20) -> dict: ...
    def visualize_workflow(self) -> str: ...
```

`config.yaml` selects which one is active via `system.orchestrator` ∈
`{"autogen", "langgraph"}`.

---

## 2. Agent roles

| Agent       | Tools available              | Primary responsibility                               | Termination signal       |
|-------------|------------------------------|------------------------------------------------------|--------------------------|
| **Planner** | none                         | Decompose query → numbered research steps            | `PLAN COMPLETE` (advisory) |
| **Researcher** | `web_search()`, `paper_search()` | Gather evidence; emit titles, URLs, snippets        | `RESEARCH COMPLETE` (advisory) |
| **Writer**  | none                         | Synthesize findings; inline citations `[Source: …]` | `DRAFT COMPLETE` (advisory) |
| **Critic**  | none                         | Evaluate quality on relevance / evidence / clarity  | `TERMINATE` (hard stop)  |

**Why these four?** The pattern follows the AutoGen literature-review example:
plan → gather → synthesize → critique. The Critic is what enforces termination
— without it, the round-robin team would loop indefinitely.

System prompts live in
[`src/agents/autogen_agents.py`](../src/agents/autogen_agents.py) and can be
overridden per-agent via `agents.<name>.system_prompt` in `config.yaml`. Empty
string → use the built-in default.

---

## 3. AutoGen execution path

```
┌─────────────────────────────────────────────────────────────┐
│ AutoGenOrchestrator.process_query(query)                    │
│                                                             │
│   1. SafetyManager.check_input_safety(query)                │
│      ├── refuse / redirect → return refusal payload          │
│      └── allow → continue                                    │
│                                                             │
│   2. team = RoundRobinGroupChat([Planner, Researcher,        │
│                                  Writer, Critic],            │
│                                 termination=TERMINATE)       │
│      result = await team.run(task=task_message)              │
│                                                             │
│   3. _extract_results(messages) → final_response, sources    │
│                                                             │
│   4. SafetyManager.check_output_safety(response, sources)    │
│      ├── refuse  → replace response with refusal             │
│      ├── sanitize → redact PII / annotate ungrounded cites   │
│      └── allow → unchanged                                   │
│                                                             │
│   5. return {response, conversation_history, metadata{       │
│        num_messages, num_sources, agents_involved,           │
│        plan, research_findings, critique, sources,           │
│        safety_events: [{type:"input"…},{type:"output"…}]     │
│      }}                                                      │
└─────────────────────────────────────────────────────────────┘
```

Notable implementation detail at
[`src/autogen_orchestrator.py`](../src/autogen_orchestrator.py): we handle both
list and async-iterable forms of `result.messages`, because different AutoGen
versions return different types — older versions returned an async stream,
0.4+ returns a plain list.

---

## 4. LangGraph execution path

```
       ┌─────────┐
       │  START  │
       └────┬────┘
            ▼
       ┌─────────┐
       │ planner │ ─── writes state.plan
       └────┬────┘
            ▼
      ┌──────────────┐
      │  researcher  │ ─── calls web_search / paper_search,
      └──────┬───────┘     writes state.sources
             ▼
       ┌──────────┐
       │  writer  │ ─── writes state.draft
       └────┬─────┘
            ▼
       ┌──────────┐
       │  critic  │ ─── writes state.critique
       └────┬─────┘
            │
       ┌────┴──────────────────────┐
       ▼                           ▼
  "NEEDS REVISION"               default
  AND iter < 2                     │
       │                           ▼
       └────▶ writer            ┌────────┐
       (revision loop)          │  END   │
                                └────────┘
```

State (`TypedDict` / dataclass):

```python
class ResearchState:
    query: str
    plan: str
    research_findings: list[str]
    sources: list[dict]
    draft: str
    critique: str
    final_response: str
    iteration_count: int
    safety_events: list[dict]
```

Differences from AutoGen path:

- **Explicit graph**: every transition is visible in code, not implicit in the
  RoundRobin order.
- **Conditional revision**: the critic node can route back to the writer if
  the critique contains "NEEDS REVISION" and we haven't exceeded
  `max_iterations`.
- **No tool-calling agents**: tools are invoked from the researcher *node*
  directly (so the graph works on LLMs that don't support function-calling).
- **Same SafetyManager**: input check before invoking the graph, output check
  on `final_response` before returning.

---

## 5. Tool integration

Tools are plain Python functions in [`src/tools/`](../src/tools/), wrapped as
`autogen_core.tools.FunctionTool` for the AutoGen path, called directly for the
LangGraph path:

| Tool                  | Provider               | Returns                                          |
|-----------------------|------------------------|--------------------------------------------------|
| `web_search(query)`   | Tavily / Brave         | List of `{title, url, snippet, score, date}`     |
| `paper_search(query)` | Semantic Scholar       | List of `{title, authors, abstract, citations, url, year}` |
| `CitationTool`        | local (formatting)     | APA / MLA strings; bibliography; deduplication   |

Tools degrade gracefully when API keys are missing — the Researcher gets an
empty result list rather than an exception, and the Writer falls back to
parametric knowledge with a flagged "no sources retrieved" notice.

---

## 6. Safety pipeline

Three components in [`src/guardrails/`](../src/guardrails/):

```
                 ┌─────────────────────┐
                 │  InputGuardrail     │   regex + keyword + topic relevance
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  SafetyManager      │   coordinates, logs JSONL, builds
                 │  (always present)   │   user-facing messages
                 └──────────┬──────────┘
                            │ optional
                            ▼
                 ┌─────────────────────┐
                 │  NeMoGuardrails     │   LLM self-check (Colang config)
                 │  (use_nemo: true)   │   degrades if init fails
                 └─────────────────────┘
```

Custom rules run first because they are deterministic and dependency-free
(important when the NeMo LLM endpoint is unreachable). NeMo runs as a
secondary content-moderation pass when enabled. **Five policy categories**:

| Category              | Examples                                                | Action                |
|-----------------------|---------------------------------------------------------|-----------------------|
| `prompt_injection`    | "ignore all previous instructions", "reveal system prompt", DAN/STAN/AIM personas, hypothetical framing, instruction smuggling | refuse |
| `harmful_content`     | Weapons, self-harm, illegal-drug synthesis, malware     | refuse                |
| `pii`                 | Email / phone / SSN / credit-card regex                 | sanitize (redact)     |
| `off_topic`           | No HCI / UX / interaction / accessibility keyword       | redirect              |
| `misinformation_risk` | `[Source: X]` not matching any retrieved source         | sanitize (annotate)   |

Output guardrail also covers PII redaction, harmful keywords (in case the
agent hallucinates them) and bias generalization patterns. See
[`src/guardrails/input_guardrail.py`](../src/guardrails/input_guardrail.py)
and [`src/guardrails/output_guardrail.py`](../src/guardrails/output_guardrail.py)
for the full pattern banks.

Every check (safe and unsafe) is logged to `logs/safety_events.log` as JSONL
with a unique `event_id`, `timestamp`, `policy_categories`, and content
preview, so the UI and the report can show the full audit trail.

---

## 7. Data flow / message format

**Result returned by `process_query`:**

```python
{
  "query": "<original query>",
  "response": "<final synthesized answer with inline citations, possibly sanitized>",
  "conversation_history": [
    {"source": "Planner",    "content": "..."},
    {"source": "Researcher", "content": "[FunctionCall(...)]"},
    {"source": "Researcher", "content": "..."},
    {"source": "Writer",     "content": "..."},
    {"source": "Critic",     "content": "...TERMINATE"},
  ],
  "metadata": {
    "num_messages": 5,
    "num_sources": 7,
    "agents_involved": ["Planner", "Researcher", "Writer", "Critic"],
    "plan": "...",
    "research_findings": ["...", "..."],
    "critique": "...",
    "sources": [{"title": "...", "url": "..."}],
    "safety_events": [
      {"type": "input",  "action": "allow",    "policy_categories": [], "violations": [], "event_id": "..."},
      {"type": "output", "action": "sanitize", "policy_categories": ["pii"], "violations": [...], "event_id": "..."},
    ],
    "blocked_by_safety": false,
  }
}
```

The CLI ([`src/ui/cli.py`](../src/ui/cli.py)) and Streamlit UI
([`src/ui/streamlit_app.py`](../src/ui/streamlit_app.py)) both render from
this same dictionary.

---

## 8. Configuration knobs

All in [`config.yaml`](../config.yaml):

| Key                              | Effect                                                    |
|----------------------------------|-----------------------------------------------------------|
| `system.orchestrator`            | `"autogen"` or `"langgraph"`                              |
| `system.topic`                   | Used by `_check_relevance` for off-topic detection        |
| `system.max_iterations`          | LangGraph revision-loop cap                               |
| `models.default.{provider,name}` | Agent model — `"groq"` / `"openai"` / `"vllm"`            |
| `models.judge.{provider,name}`   | Judge model used by `LLMJudge`                            |
| `agents.<name>.system_prompt`    | Override default agent prompt (empty → use default)       |
| `safety.enabled`                 | Master toggle                                             |
| `safety.use_nemo`                | Activate NeMo Guardrails second-pass layer                |
| `safety.use_llm_classifier`      | Run an extra LLM call for harmful-content classification  |
| `safety.prohibited_categories`   | Documented list (informational)                           |
| `safety.on_violation.action`     | Default action when category-specific not given            |
| `safety.safety_log_file`         | JSONL audit log path                                      |
| `evaluation.criteria`            | Names + weights for the LLM judge's per-criterion scores  |

---

## 9. Quick reference — file paths

| Component        | File                                                                                              |
|------------------|---------------------------------------------------------------------------------------------------|
| Agent factories  | [src/agents/autogen_agents.py](../src/agents/autogen_agents.py)                                   |
| AutoGen path     | [src/autogen_orchestrator.py](../src/autogen_orchestrator.py)                                     |
| LangGraph path   | [src/langgraph_orchestrator.py](../src/langgraph_orchestrator.py) *(planned)*                     |
| Web search       | [src/tools/web_search.py](../src/tools/web_search.py)                                             |
| Paper search     | [src/tools/paper_search.py](../src/tools/paper_search.py)                                         |
| Citations        | [src/tools/citation_tool.py](../src/tools/citation_tool.py)                                       |
| Input guardrail  | [src/guardrails/input_guardrail.py](../src/guardrails/input_guardrail.py)                         |
| Output guardrail | [src/guardrails/output_guardrail.py](../src/guardrails/output_guardrail.py)                       |
| Safety manager   | [src/guardrails/safety_manager.py](../src/guardrails/safety_manager.py)                           |
| NeMo adapter     | [src/guardrails/nemo_adapter.py](../src/guardrails/nemo_adapter.py)                               |
| NeMo Colang      | [src/guardrails/nemo_config/config.yml](../src/guardrails/nemo_config/config.yml)                 |
| LLM Judge        | [src/evaluation/judge.py](../src/evaluation/judge.py)                                             |
| Batch evaluator  | [src/evaluation/evaluator.py](../src/evaluation/evaluator.py)                                     |
| CLI              | [src/ui/cli.py](../src/ui/cli.py)                                                                  |
| Streamlit        | [src/ui/streamlit_app.py](../src/ui/streamlit_app.py)                                              |
| Adversarial set  | [data/safety_test_queries.json](../data/safety_test_queries.json)                                  |
| Eval queries     | [data/example_queries.json](../data/example_queries.json)                                          |
