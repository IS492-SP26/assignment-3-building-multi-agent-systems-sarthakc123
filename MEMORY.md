# Project Memory — Assignment 3 (Multi-Agent Research System)

> Running progress log. Update this whenever a step finishes, a decision is made, or context might reset.
> Source of truth for plan: [`/Users/amd/.claude/plans/read-the-asssignment-github-clever-wand.md`](file:///Users/amd/.claude/plans/read-the-asssignment-github-clever-wand.md)
> Source of truth for assignment: [ASSIGNMENT_INSTRUCTIONS.md](ASSIGNMENT_INSTRUCTIONS.md)

---

## Decisions (locked-in)

| Decision | Choice | Rationale |
|---|---|---|
| Orchestration | **Both AutoGen + LangGraph** | AutoGen already scaffolded; LangGraph adds bonus credit. Swappable via `system.orchestrator` in `config.yaml`. |
| Guardrails | **Custom rules primary + NeMo Guardrails secondary** | Custom rules are deterministic and dependency-free; NeMo as optional second-pass. Custom must work standalone if NeMo install fails. |
| LLM provider | **Groq `llama-3.3-70b-versatile`** | User has Groq API key. vLLM endpoint kept as commented alternative in config.yaml. |
| Topic | **HCI Research** (default in config) | Matches assignment guidance and existing 10 example queries. |
| Report | Draft `REPORT.md` at end | All sections; user will edit before submission. |
| Artifacts | **All four** | session JSON, answer MD, judge traces, eval report — required by assignment notes. |

---

## Progress

### ✅ Done
- Read assignment + starter code (full inventory in plan file).
- Built plan + got user approval.
- **Step 1: Config + .env updates**
  - `config.yaml`: switched `models.default` and `models.judge` to Groq `llama-3.3-70b-versatile`. Added `system.orchestrator: "autogen"` flag. Expanded `safety` block with 5 policy categories, `safety_log_file`, `use_llm_classifier`, `use_nemo`, length thresholds. vLLM block preserved as commented alternative.
  - `.env.example`: reorganized — Groq primary, OpenAI/vLLM alternative documented, `DEFAULT_MODEL`/`JUDGE_MODEL` updated to `llama-3.3-70b-versatile`.

### ✅ Done since last update
- Step 2 smoke test passed on Groq (Planner→Researcher→Writer→Critic all ran).
- Fixed AutoGen 0.4 `result.messages` async-iter bug in `autogen_orchestrator.py`.
- `InputGuardrail` complete — 14 attack families covered (direct override, persona jailbreak DAN/STAN/AIM, mode swap, authority impersonation, grandma attack, instruction smuggling, context appending, hypothetical/fiction framing, etc.).
- `OutputGuardrail` complete — PII/harmful/bias/ungrounded-citation.
- `SafetyManager` complete — JSONL logging, NeMo as optional second layer.
- `NeMoGuardrailsAdapter` + `nemo_config/config.yml` — degrades gracefully if init fails.
- SafetyManager wired into `AutoGenOrchestrator.process_query` (input check pre, output check post; safety_events attached to metadata).
- `data/safety_test_queries.json` — **14 adversarial queries** across attack families. **14/14 pass expected behavior.** safety_13 (leetspeak) intentionally documents a partial-defense limitation.
- `docs/AGENT_ARCHITECTURE.md` created — full architecture writeup with diagrams, agent table, safety pipeline, data flow, config knobs, file paths.

### ✅ Recent (all done)
- LangGraph orchestrator built — `src/langgraph_orchestrator.py`. StateGraph, conditional revision edge, uses OpenAI SDK pointed at Groq.
- Orchestrator factory — `src/orchestrator_factory.py`. CLI, Streamlit, main, evaluator all use it.
- CLI improved — safety panel, `export`/`traces`/`stats` commands, safety + system metrics in `stats`.
- Streamlit improved — safety panel always visible, two-perspective judge button, JSON/MD download buttons, agent transcript expander, real safety stats from manager.
- LLMJudge rewritten — two perspectives (rubric + holistic peer reviewer), cross-model design (judge = `openai/gpt-oss-120b`, agents = Llama 3.x), raw traces persisted to `outputs/judge_traces/`.
- Evaluator improved — sync/async orchestrator detection, error-analysis section (lowest/highest criterion, weakest category, failed-query list).
- main.py — added `--mode demo` end-to-end runner. `run_demo.sh` wrapper.
- Tools async fix — both `web_search` and `paper_search` now handle running event loops via thread-pool fallback.
- Sample artifacts generated and committed: `outputs/sample_session.json`, `outputs/sample_answer.md`, `outputs/sample_judge.json`, `outputs/judge_traces/sample_q1_*`. Final sample has 16 real Tavily sources, 25 messages, 4 agents, NIST/Algolia/Cohere citations.
- README.md rewritten with quickstart, artifact map, safety policy table, orchestrator comparison.
- REPORT.md drafted — 4-page write-up with abstract, system design, safety design, evaluation results (representative run table), discussion (AutoGen vs LangGraph trade-off, judge bias, leetspeak limitation, rate-limit reflection), references (APA), appendix table of 14/14 adversarial results.

### 🗒️ Step 2 reference
- **Smoke test baseline**
  - ✅ Dependencies installed via `pip3 install -r requirements.txt`.
  - ✅ vLLM endpoint reachable (`test_openai_api.py` returned a valid completion).
  - ❌ **AutoGen agent loop fails on vLLM endpoint** — server does not have `--enable-auto-tool-choice` / `--tool-call-parser` enabled, so tool-calling Researcher agent gets HTTP 400.
  - Decision: switch back to Groq path. User getting a free Groq API key.
  - Config reverted to `provider: "groq"`, `model: "llama-3.3-70b-versatile"`. vLLM block kept commented as alternative.
  - Pending: user pastes `GROQ_API_KEY` into `.env`, then re-run smoke test.

### 💡 Insights for REPORT.md (Discussion / Limitations / Reflections)

- **AutoGen RoundRobinGroupChat amplifies context**: every agent sees the full accumulated conversation, so by the 4th turn (Critic) the input tokens often exceed a flat-tier per-minute LLM quota (Groq 6000 TPM on `llama-3.1-8b-instant`). LangGraph nodes only see the state they need (no accumulation), so the same workflow comfortably fits the quota. → **Trade-off:** AutoGen's natural multi-agent conversation feels more "agentic" but is token-hungry; LangGraph is more disciplined but loses the emergent back-and-forth.
- **Judge bias avoided by cross-model design**: agents use Meta Llama 3.3 70B (and llama-3.1-8b for free-tier headroom); judge uses OpenAI gpt-oss-120b via Groq. Two different model families → reduces correlated errors.
- **Two-perspective judging matters**: per-criterion rubric judge + holistic peer-reviewer disagree on edge cases (e.g., a response with perfect citation but thin coverage scores high on `evidence_quality` but 5/10 holistic). Average smooths individual perspective bias.
- **Guardrail false-positive vs false-negative trade-off**: tight keyword/regex rules block 13/14 adversarial queries but the off-topic redirect is broad (HCI keyword overlap). Documented limitation: leetspeak evasion (`m4k3 a b0mb`) — caught only because the leetspeak text contains no HCI keyword, so the off-topic redirect fires; a leetspeak harmful query *with* an HCI keyword would slip through.
- **Source grounding via citation matching**: OutputGuardrail compares `[Source: X]` markers to retrieved source titles/URLs. Works for direct quotes; fails to catch paraphrased hallucinations not tagged with a source.
- **Free-tier rate limits dominate iteration speed**: developing a multi-agent system on Groq's free tier is fundamentally throttled by per-minute and per-day token budgets. Plan for ~75s waits between full demo runs during development.
- **Unbounded external calls = silent UI hangs**: discovered live during Streamlit testing — the orchestrator stalled for 4 minutes at 0% CPU with two ESTABLISHED TCP connections to Cloudflare-fronted endpoints (Tavily / Semantic Scholar). No request timeouts anywhere meant a single stuck dependency blocked the whole pipeline indefinitely. Fixed by adding per-hop timeouts: `concurrent.futures` around sync tool wrappers (30 s), `client.with_options(timeout=…)` on every OpenAI call (60 s agents / 45 s judge / 30 s NLI). Documented in REPORT.md §4. **Generalizable lesson**: in a multi-agent system orchestrating *N* third-party services per query, every hop needs an explicit timeout — otherwise the slowest dependency's failure mode becomes the system's failure mode.

### Known gotchas (lessons from this step)
- **AutoGen 0.4+ tool-calling is incompatible with the assignment vLLM server.** Do not try to use `provider: vllm` while any agent has tools attached. The fix on the server side is `--enable-auto-tool-choice --tool-call-parser hermes` (or similar) — out of our control.
- The system Python (`/Library/Frameworks/Python.framework/Versions/3.12`) had old `autogen` package installed but not `autogen-agentchat`. Always verify the new AutoGen 0.4+ packages (`autogen-agentchat`, `autogen-ext`, `autogen-core`) are importable before running orchestrator code.

### ⬜ Next (in order)
1. Implement `src/guardrails/input_guardrail.py` — 5 policy categories.
2. Implement `src/guardrails/output_guardrail.py` — PII + harmful + factual.
3. Implement `src/guardrails/safety_manager.py` — JSONL logging + coordination.
4. Create `src/guardrails/nemo_adapter.py` + `nemo_config/` (optional).
5. Wire SafetyManager into `AutoGenOrchestrator.process_query`.
6. Create `data/safety_test_queries.json` + test guardrails.
7. Build `src/langgraph_orchestrator.py` (parallel to AutoGen).
8. Add orchestrator factory; wire into CLI/UI/main/evaluator.
9. Improve `LLMJudge` — 2 perspectives + save raw traces to `outputs/judge_traces/`.
10. Improve `evaluator` — error analysis + sync/async orchestrator handling.
11. Wire `SystemEvaluator` into `main.py --mode evaluate`.
12. Update `cli.py` — safety display + `export` command.
13. Update `streamlit_app.py` — safety panel + judge button + download buttons.
14. Add `main.py --mode demo` + `run_demo.sh`.
15. Generate committed sample artifacts.
16. Update `README.md`.
17. Draft `REPORT.md`.

---

## Key paths cheat sheet

| What | Path |
|---|---|
| AutoGen agents (done) | [src/agents/autogen_agents.py](src/agents/autogen_agents.py) |
| AutoGen orchestrator | [src/autogen_orchestrator.py](src/autogen_orchestrator.py) |
| LangGraph orchestrator | `src/langgraph_orchestrator.py` (TBD) |
| Tools (done) | [src/tools/](src/tools/) — web_search, paper_search, citation_tool |
| Guardrails | [src/guardrails/](src/guardrails/) — input, output, safety_manager |
| Judge | [src/evaluation/judge.py](src/evaluation/judge.py) |
| Evaluator | [src/evaluation/evaluator.py](src/evaluation/evaluator.py) |
| CLI | [src/ui/cli.py](src/ui/cli.py) |
| Streamlit | [src/ui/streamlit_app.py](src/ui/streamlit_app.py) |
| Test queries (10) | [data/example_queries.json](data/example_queries.json) |
| Adversarial test queries | `data/safety_test_queries.json` (TBD) |
| Config | [config.yaml](config.yaml) |
| Env template | [.env.example](.env.example) |
| Sample artifacts dir | `outputs/` (TBD; only sample_*.json + sample_answer.md + judge_traces/* committed) |
| Logs dir | `logs/` (runtime — gitignored) |

---

## Safety policy (5 categories)

| Category | Trigger | Default action |
|---|---|---|
| `prompt_injection` | "ignore previous instructions", "act as", "reveal system prompt", role confusion | refuse |
| `harmful_content` | violence, self-harm, illegal-activity instructions | refuse |
| `pii` | email / phone / SSN / credit-card regex | sanitize (redact) |
| `off_topic` | low keyword overlap with HCI domain (UX, UI, accessibility, interaction, design…) | redirect |
| `misinformation_risk` | `[Source: …]` citations not matching any retrieved source title/URL | sanitize / flag |

---

## Models / providers

- **Default** (config.yaml `models.default`): `groq` + `llama-3.3-70b-versatile`, temp 0.7, max_tokens 2048.
- **Judge** (config.yaml `models.judge`): same model, temp 0.3, max_tokens 1024.
- **Alternate**: vLLM `Qwen/Qwen3-8B` at `https://vllm.salt-lab.org/v1` (commented block in config.yaml).

---

## Verification gates (must all pass before "done")

1. `python test_openai_api.py` — model reachable.
2. `python main.py --mode autogen` — single-query AutoGen returns synthesized answer with sources.
3. `python main.py --mode autogen` with `system.orchestrator: langgraph` — same query works on LangGraph path.
4. `python main.py --mode cli` — interactive CLI; `export` writes session JSON + answer MD.
5. `python main.py --mode web` — Streamlit; safety panel + Run Judge button + downloads work.
6. Adversarial queries trigger expected category + action; events in `logs/safety_events.log`.
7. `python main.py --mode evaluate` — runs all 10 queries, writes `outputs/evaluation_<ts>.json`.
8. `bash run_demo.sh` — single-command end-to-end produces all 4 artifact types.
9. `outputs/sample_session.json`, `outputs/sample_answer.md`, `outputs/judge_traces/q1_*.json` committed.

---

## Notes / gotchas

- `src/autogen_orchestrator.py::_process_query_async` uses `async for message in result.messages` — assumes AutoGen returns an async iterable. May need adjustment for some AutoGen versions.
- `groq` package is imported in `judge.py` — tied to Groq SDK pattern.
- `safety_test_queries.json` does **not** exist yet; will be created in step 6.
- vLLM model name in original config (`openai/gpt-oss-20b`) didn't match the assignment's stated `Qwen/Qwen3-8B` — both are now alternatives, primary is Groq.
- Author git config: `sarthakc123` / `sarthak.chandarana99@gmail.com`.
