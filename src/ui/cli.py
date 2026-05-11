"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any
import yaml
import logging
from dotenv import load_dotenv

from src.orchestrator_factory import create_orchestrator

# Load environment variables
load_dotenv()

class CLI:
    """Command-line interface for the research assistant."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize orchestrator (chosen via system.orchestrator in config.yaml)
        try:
            self.orchestrator = create_orchestrator(self.config)
            self.logger = logging.getLogger("cli")
            orch_name = self.config.get("system", {}).get("orchestrator", "autogen")
            self.logger.info(f"{orch_name} orchestrator initialized successfully")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0
        self._last_result = None

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """Main CLI loop."""
        self._print_welcome()

        while self.running:
            try:
                # Get user input
                query = input("\nEnter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue
                elif query.lower() == 'export':
                    self._export_last_result()
                    continue
                elif query.lower() == 'traces':
                    self.config.setdefault("ui", {})["verbose"] = not self.config.get("ui", {}).get("verbose", False)
                    print(f"\nTraces {'ON' if self.config['ui']['verbose'] else 'OFF'}")
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("Processing your query...")
                print("=" * 70)

                try:
                    # Process through orchestrator (synchronous call, not async)
                    result = self.orchestrator.process_query(query)
                    self.query_count += 1
                    self._last_result = result

                    # Display result
                    self._display_result(result)

                except Exception as e:
                    print(f"\nError processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Error in CLI loop")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear the screen")
        print("  stats   - Show system statistics + safety stats")
        print("  traces  - Toggle verbose agent traces")
        print("  export  - Save last result as JSON + Markdown in outputs/")
        print("  quit    - Exit the application")
        print("\nOr enter a research query to get started!")

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThank you for using the Multi-Agent Research Assistant!")
        print("Goodbye!\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    def _print_stats(self):
        """Print system + safety statistics."""
        print("\nSystem Statistics:")
        print(f"  Queries processed:  {self.query_count}")
        print(f"  System:             {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic:              {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Orchestrator:       {self.config.get('system', {}).get('orchestrator', 'autogen')}")
        print(f"  Model:              {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")
        # Safety stats
        sm = getattr(self.orchestrator, "safety_manager", None)
        if sm:
            stats = sm.get_safety_stats()
            print(f"\nSafety Statistics:")
            print(f"  Total checks:       {stats['total_events']}")
            print(f"  Input checks:       {stats['input_checks']}")
            print(f"  Output checks:      {stats['output_checks']}")
            print(f"  Violations:         {stats['violations']}")
            print(f"  Violation rate:     {stats['violation_rate']:.1%}")

    def _export_last_result(self):
        """Save the last query result as JSON + Markdown in outputs/."""
        if not self._last_result:
            print("\nNo result to export yet. Run a query first.")
            return
        from datetime import datetime
        from pathlib import Path
        import json
        outdir = Path("outputs")
        outdir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = outdir / f"session_{ts}.json"
        md_path = outdir / f"answer_{ts}.md"

        # JSON: full session
        with open(json_path, "w") as f:
            json.dump(self._last_result, f, indent=2, default=str)

        # Markdown: query + answer + citations + safety summary
        md = self._render_markdown(self._last_result)
        with open(md_path, "w") as f:
            f.write(md)

        print(f"\n✅ Exported:")
        print(f"   {json_path}")
        print(f"   {md_path}")

    def _render_markdown(self, result):
        """Build a clean Markdown view of a result for export."""
        from datetime import datetime
        md = []
        md.append(f"# Research Answer\n")
        md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
        md.append(f"## Query\n{result.get('query', '')}\n")
        md.append(f"## Answer\n{result.get('response', '')}\n")
        metadata = result.get("metadata", {}) or {}
        # Citations
        citations = self._extract_citations(result)
        if citations:
            md.append("## Citations\n")
            for i, c in enumerate(citations, 1):
                md.append(f"{i}. {c}")
            md.append("")
        # Metadata
        md.append("## Metadata\n")
        md.append(f"- Messages exchanged: {metadata.get('num_messages', 0)}")
        md.append(f"- Sources gathered:   {metadata.get('num_sources', 0)}")
        md.append(f"- Agents involved:    {', '.join(metadata.get('agents_involved', []))}")
        md.append(f"- Orchestrator:       {metadata.get('orchestrator', self.config.get('system', {}).get('orchestrator', 'autogen'))}")
        md.append("")
        # Safety
        events = metadata.get("safety_events", [])
        if events:
            md.append("## Safety Events\n")
            for ev in events:
                cats = ev.get("policy_categories") or ["none"]
                md.append(f"- **{ev.get('type', '').upper()}** action=`{ev.get('action', '')}` categories=`{', '.join(cats)}` event_id=`{ev.get('event_id', '')[:8]}`")
            md.append("")
        return "\n".join(md)

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        print("\n" + "=" * 70)
        print("RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return

        # Display response
        response = result.get("response", "")
        print(f"\n{response}\n")

        # Extract and display citations from conversation
        citations = self._extract_citations(result)
        if citations:
            print("\n" + "-" * 70)
            print("📚 CITATIONS")
            print("-" * 70)
            for i, citation in enumerate(citations, 1):
                print(f"[{i}] {citation}")

        # Display metadata
        metadata = result.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("📊 METADATA")
            print("-" * 70)
            print(f"  • Messages exchanged: {metadata.get('num_messages', 0)}")
            print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
            print(f"  • Agents involved: {', '.join(metadata.get('agents_involved', []))}")
            orch_label = metadata.get('orchestrator') or self.config.get('system', {}).get('orchestrator', 'autogen')
            print(f"  • Orchestrator: {orch_label}")

        # Display safety events
        safety_events = (metadata or {}).get("safety_events", [])
        if safety_events:
            print("\n" + "-" * 70)
            print("🛡️  SAFETY")
            print("-" * 70)
            for ev in safety_events:
                cats = ev.get("policy_categories") or []
                cat_str = ", ".join(cats) if cats else "none"
                icon = "✅" if ev.get("action") == "allow" else "⚠️ "
                print(f"  {icon} {ev.get('type', '').upper():6s}  action={ev.get('action', ''):9s}  categories=[{cat_str}]")
                # Surface short reasons for any violations
                for v in (ev.get("violations") or [])[:3]:
                    print(f"        - {v.get('category', '')}: {v.get('reason', '')[:80]}")
            if metadata.get("blocked_by_safety"):
                print(f"\n  ❌ This query was blocked at input — no agent ran.")

        # Display conversation summary if verbose mode
        if self._should_show_traces():
            self._display_conversation_summary(result.get("conversation_history", []))

        print("=" * 70 + "\n")
    
    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Extract citations/URLs from conversation history."""
        citations = []
        
        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")
            
            # Find URLs in content
            import re
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
            
            for url in urls:
                if url not in citations:
                    citations.append(url)
        
        return citations[:10]  # Limit to top 10

    def _should_show_traces(self) -> bool:
        """Check if agent traces should be displayed."""
        # Check config for verbose mode
        return self.config.get("ui", {}).get("verbose", False)

    def _display_conversation_summary(self, conversation_history: list):
        """Display a summary of the agent conversation."""
        if not conversation_history:
            return
            
        print("\n" + "-" * 70)
        print("🔍 CONVERSATION SUMMARY")
        print("-" * 70)
        
        for i, msg in enumerate(conversation_history, 1):
            agent = msg.get("source", "Unknown")
            content = msg.get("content", "")
            
            # Truncate long content
            preview = content[:150] + "..." if len(content) > 150 else content
            preview = preview.replace("\n", " ")
            
            print(f"\n{i}. {agent}:")
            print(f"   {preview}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run CLI
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
