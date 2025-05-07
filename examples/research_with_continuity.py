#!/usr/bin/env python
"""
Example script that demonstrates using message history for agent communication
in an iterative research project.

This example shows:
1. How to start a research project
2. How to save the project state including agent message history
3. How to load and continue a research project
"""

import asyncio
import os
from dotenv import load_dotenv
from src.iterative_researcher import IterativeResearcher
from src.llm_config import create_default_config

# Load environment variables
load_dotenv()

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def example_research_with_history():
    """Example of using message history in research."""
    print("Starting initial research phase...")
    
    # Create a default config
    config = create_default_config()
    
    # Create an iterative agent
    agent = IterativeResearcher(max_iterations=2, max_minutes=30, config=config, verbose=True)
    
    # Define a research query
    query = "What are the environmental impacts of electric vehicles compared to traditional vehicles?"
    background = "I'm researching for a paper on sustainable transportation."
    
    # Run initial research (limited to 2 iterations)
    report = await agent.run_research(query, background_context=background)
    
    print("\n--- INITIAL RESEARCH REPORT ---")
    print(report)
    
    # Save the research state to a file
    print("\nSaving research state...")
    session_file = "research_session.json"
    agent.export_session(agent.research_state, filepath=session_file)
    
    print(f"Research state saved to {session_file}")
    print("The saved state includes all message histories, allowing for continuity in conversations between agents.")
    
    # Now demonstrate loading the saved state and continuing
    print("\n--- CONTINUING RESEARCH FROM SAVED STATE ---")
    
    # Load the research state
    loaded_agent, loaded_state = IterativeResearcher.load_session_from_file(session_file, config=config)
    
    # Continue the research for 1 more iteration
    continued_report = await loaded_agent.continue_research(loaded_state, max_additional_iterations=1)
    
    print("\n--- CONTINUED RESEARCH REPORT ---")
    print(continued_report)
    
    # You can compare the two reports to see how the research evolved
    print("\nDemonstration complete. The key differences are:")
    print("1. The agent maintained memory of previous conversations through message history")
    print("2. The research could be paused and continued with full context preservation")

if __name__ == "__main__":
    asyncio.run(example_research_with_history()) 