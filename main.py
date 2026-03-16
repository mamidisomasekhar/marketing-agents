"""Command-line interface for the marketing agents swarm."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.orchestrator import Pipeline
from agents.event_discovery import EventDiscoveryAgent
from agents.event_qualification import EventQualificationAgent
from agents.event_website_scraper import EventWebsiteScraperAgent
from agents.event_intelligence import EventIntelligenceAgent
from agents.event_prioritization import EventPrioritizationAgent
from agents.outreach_email import OutreachEmailAgent
from agents.excel_table_generator import ExcelTableGeneratorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_pipeline(provider: str = "tavily") -> Pipeline:
    """Create and configure the event research pipeline.
    
    Pipeline flow:
    1. Event Discovery - Find events globally (with filters)
    2. Event Qualification - Score and tier events
    3. Event Website Scraper - Extract website details
    4. Event Intelligence - Strategic analysis
    5. Event Prioritization - Sort and recommend
    6. Outreach Email - Generate sponsorship emails
    7. Excel Table Generator - Output Excel-ready format
    
    Returns:
        Configured Pipeline instance
    """
    pipeline = Pipeline()
    
    # Add all event research agents in sequence
    pipeline.add_agent(EventDiscoveryAgent(max_events=50, provider=provider))
    pipeline.add_agent(EventQualificationAgent())
    pipeline.add_agent(EventWebsiteScraperAgent())
    pipeline.add_agent(EventIntelligenceAgent())
    pipeline.add_agent(EventPrioritizationAgent())
    pipeline.add_agent(OutreachEmailAgent())
    pipeline.add_agent(ExcelTableGeneratorAgent())
    
    return pipeline


def run_command(args):
    """Run the pipeline with input parameters."""
    provider = getattr(args, 'provider', 'tavily')
    pipeline = create_pipeline(provider=provider)
    
    if not pipeline.agents:
        logger.error("No agents configured. Add agents before running.")
        sys.exit(1)
    
    logger.info(f"Using search provider: {provider}")
    
    # Build parameters from arguments
    params = {}
    
    if args.industry:
        params["industry"] = args.industry
        logger.info(f"Industry: {args.industry}")
    
    if args.region:
        params["region"] = args.region
        logger.info(f"Region: {args.region}")
    
    if args.theme:
        params["theme"] = args.theme
        logger.info(f"Theme: {args.theme}")
    
    # Build query from inputs
    query = args.prompt or f"{args.industry or 'Technology'} events"
    if args.region:
        query += f" in {args.region}"
    
    logger.info(f"Running with query: {query}")
    
    try:
        # Pass parameters to pipeline
        result = pipeline.execute(query, initial_context={"parameters": params})
        
        print("\n" + "=" * 60)
        print("PIPELINE RESULT")
        print("=" * 60)
        print(f"Agent: {result.agent_name}")
        print(f"Status: {result.status}")
        
        # Get the events from findings
        findings = result.findings
        events = findings.get("events", [])
        
        print(f"\nTotal Events: {len(events)}")
        
        # Print summary table
        if events:
            print("\n" + "-" * 60)
            print("EVENT SUMMARY")
            print("-" * 60)
            print(f"{'Event Name':<45} {'Score':<8} {'Tier':<25}")
            print("-" * 60)
            for event in events[:10]:  # Show top 10
                name = event.get("event_name", "N/A")[:43]
                score = event.get("overall_score", "N/A")
                tier = event.get("priority_tier", "N/A")[:23]
                print(f"{name:<45} {score:<8} {tier:<25}")
            
            if len(events) > 10:
                print(f"... and {len(events) - 10} more events")
        
        # Print CSV output if available
        if findings.get("csv"):
            print("\n" + "=" * 60)
            print("CSV OUTPUT (for Excel)")
            print("=" * 60)
            print(findings["csv"])
        
        # Save full results to file
        output_file = args.output or "event_pipeline_results.json"
        with open(output_file, "w") as f:
            json.dump(findings, f, indent=2)
        print(f"\nFull results saved to: {output_file}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def list_agents_command(args):
    """List all available agents."""
    pipeline = create_pipeline()
    
    print("\nEvent Research Pipeline Agents:")
    print("=" * 50)
    
    if not pipeline.agents:
        print("  (No agents configured)")
    else:
        for i, agent in enumerate(pipeline.agents, 1):
            print(f"\n  {i}. {agent.name}")
            print(f"     {agent.description}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Marketing Agents Swarm - Event Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find Payments events in Middle East (using Search1API - free)
  python -m marketing_agents run --industry Payments --region "Middle East" --provider search1api
  
  # Find AI conferences in APAC (using DuckDuckGo - free)
  python -m marketing_agents run --industry "Artificial Intelligence" --region APAC --provider duckduckgo
  
  # Find FinTech events globally (using default Tavily)
  python -m marketing_agents run --industry FinTech --region global
  
  # With custom theme and Serper
  python -m marketing_agents run --industry Payments --theme "digital payments" --region USA --provider serper
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run the event research pipeline")
    run_parser.add_argument(
        "--prompt", "-p",
        help="The research prompt/query (optional if industry is specified)"
    )
    run_parser.add_argument(
        "--industry", "-i",
        required=True,
        help="Industry focus (e.g., FinTech, Payments, Artificial Intelligence, Technology)"
    )
    run_parser.add_argument(
        "--region", "-r",
        help="Region (e.g., USA, Europe, APAC, Middle East, Asia, global)"
    )
    run_parser.add_argument(
        "--theme", "-t",
        help="Theme or topic (e.g., digital payments, machine learning, blockchain)"
    )
    run_parser.add_argument(
        "--output", "-o",
        default="event_pipeline_results.json",
        help="Output file name (default: event_pipeline_results.json)"
    )
    run_parser.add_argument(
        "--provider", "-v",
        default="auto",
        choices=["auto", "tavily", "serper", "search1api", "duckduckgo"],
        help="Search provider (auto=tavily→serper→search1api→duckduckgo)"
    )
    run_parser.set_defaults(func=run_command)
    
    # list-agents command
    list_parser = subparsers.add_parser("list-agents", help="List available agents")
    list_parser.set_defaults(func=list_agents_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
