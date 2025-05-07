import asyncio
import time
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.planner_agent import init_planner_agent, ReportPlan
from src.llm_config import LLMConfig

async def main():
    """Run the planner agent example."""
    print("=== Report Planner Agent Example ===")
    
    llm_config = LLMConfig()

    print(f"Creating planner agent with model: {llm_config.model}")
    agent = init_planner_agent(llm_config)
    
    # Define the research query
    query = """QUERY: Write a comprehensive report on the current state of AI safety research, 
    focusing on the main challenges and proposed solutions."""
    
    print(f"\nStarting planning process for query: {query}")
    print("\nThe agent will search for background context and create a structured report outline.")
    
    # Record start time
    start_time = time.time()
    
    # Run the planner and get the report plan
    result = await agent.run(query)
    plan: ReportPlan = result.output
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\nPlanning completed in {int(minutes)} minutes and {int(seconds)} seconds")
    print("\n=== REPORT PLAN ===\n")
    print(f"Report Title: {plan.report_title}")
    print("\nBackground Context:")
    print(plan.background_context)
    print("\nReport Outline:")
    for i, section in enumerate(plan.report_outline, 1):
        print(f"\nSection {i}:")
        print(f"Title: {section.title}")
        print(f"Key Question: {section.key_question}")

if __name__ == "__main__":
    asyncio.run(main()) 