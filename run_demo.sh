#!/usr/bin/env bash
# Single-command end-to-end demo.
# Runs one HCI research query through the multi-agent system, then through the
# LLM judge, and exports all artifacts to outputs/.
#
# Usage:
#   bash run_demo.sh                     # default HCI query
#   bash run_demo.sh "your custom query"

set -e

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "❌ .env not found. Copy .env.example to .env and add your GROQ_API_KEY first."
    exit 1
fi

QUERY="${1:-What are the key principles of accessible user interface design?}"

echo "======================================================================"
echo " MULTI-AGENT RESEARCH SYSTEM — END-TO-END DEMO"
echo "======================================================================"
echo " Query: $QUERY"
echo ""

python3 main.py --mode demo --query "$QUERY"

echo ""
echo "======================================================================"
echo " Demo complete. Inspect outputs/ for:"
echo "   - session_*.json    (full agent transcript + judge scores)"
echo "   - answer_*.md       (final synthesized answer + citations)"
echo "   - judge_*.json      (LLM-as-a-Judge per-criterion scores)"
echo "   - judge_traces/     (raw judge prompts + responses)"
echo "======================================================================"
