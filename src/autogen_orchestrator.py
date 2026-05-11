"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
import re
from typing import Dict, Any, List, Optional

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        # Initialize safety manager (always present so the UI can read events;
        # it no-ops when safety.enabled is false).
        self.safety_manager = SafetyManager(config)

        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)

        self.logger.info("Research team created successfully")

        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")

        # ---- 1. Input safety check (block before agent loop) ----
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
                    "safety_events": [
                        {
                            "type": "input",
                            "action": input_check["action"],
                            "policy_categories": input_check["policy_categories"],
                            "violations": input_check["violations"],
                            "event_id": input_check["event_id"],
                        }
                    ],
                },
            }

        try:
            # Run the async query processing
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._process_query_async(query, max_rounds)
                    ).result()
            else:
                result = loop.run_until_complete(self._process_query_async(query, max_rounds))

            # ---- 2. Output safety check on the final synthesized response ----
            sources = self._collect_sources(result.get("conversation_history", []))
            output_check = self.safety_manager.check_output_safety(
                result.get("response", ""),
                sources=sources,
            )
            result["response"] = output_check["response"]
            result.setdefault("metadata", {})
            result["metadata"]["safety_events"] = [
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
            ]
            result["metadata"]["sources"] = sources

            self.logger.info("Query processing complete")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }
    
    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async implementation of query processing.
        
        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            Dictionary containing results
        """
        # Create task message
        task_message = f"""Research Query: {query}

Please work together to answer this query comprehensively:
1. Planner: Create a research plan
2. Researcher: Gather evidence from web and academic sources
3. Writer: Synthesize findings into a well-cited response
4. Critic: Evaluate the quality and provide feedback"""
        
        # Run the team
        result = await self.team.run(task=task_message)

        # Extract conversation history. Handle both sync (list) and async-iterable
        # forms returned by different AutoGen versions.
        messages = []
        raw_messages = getattr(result, "messages", []) or []
        if hasattr(raw_messages, "__aiter__"):
            async for message in raw_messages:
                messages.append({
                    "source": getattr(message, "source", "Unknown"),
                    "content": getattr(message, "content", str(message)),
                })
        else:
            for message in raw_messages:
                messages.append({
                    "source": getattr(message, "source", "Unknown"),
                    "content": getattr(message, "content", str(message)),
                })
        
        # Extract final response. Prefer the Writer's last message (the actual
        # synthesized answer). Fall back to Critic if no Writer message, then to
        # the most recent message of any kind.
        final_response = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("source") == "Writer":
                    final_response = msg.get("content", "")
                    break
            if not final_response:
                for msg in reversed(messages):
                    if msg.get("source") == "Critic":
                        final_response = msg.get("content", "")
                        break
            if not final_response:
                final_response = messages[-1].get("content", "")
        
        return self._extract_results(query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
            
            elif source == "Critic":
                critique = content
        
        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")
        
        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()
        
        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
            }
        }

    @staticmethod
    def _collect_sources(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pull source titles/URLs out of the Researcher's tool-result messages
        so the OutputGuardrail can ground citations against them.
        """
        sources: List[Dict[str, Any]] = []
        url_re = re.compile(r"https?://[^\s<>\"{}|\\^\[\]]+")
        for msg in messages:
            content = msg.get("content") or ""
            if not isinstance(content, str):
                continue
            for url in url_re.findall(content):
                if not any(s.get("url") == url for s in sources):
                    sources.append({"url": url, "title": "", "snippet": content[:200]})
        return sources

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

