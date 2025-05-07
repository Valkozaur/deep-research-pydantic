#!/usr/bin/env python
"""
Example script that demonstrates using the DeepResearcher for in-depth
research with automatically structured reports.

The DeepResearcher:
1. Creates a structured report plan with sections
2. Conducts iterative research on each section in parallel
3. Combines all sections into a polished final report
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.deep_researcher import DeepResearcher
from src.llm_config import create_default_config


async def main():
    """Run the deep researcher example."""
    print("=== Deep Researcher Example ===")
    
    # Create the deep researcher with default settings
    # - max_iterations: 3 (for faster example)
    # - max_minutes: 5 (time limit of 5 minutes per section)
    max_iterations = 3
    max_minutes = 5
    
    print(f"Creating deep researcher with max_iterations={max_iterations}, max_minutes={max_minutes}")
    researcher = DeepResearcher(
        max_iterations=max_iterations, 
        max_minutes=max_minutes, 
        verbose=True
    )
    
    # Define the research query
    query = "Create a comprehensive report on renewable energy technologies, their current adoption rates, and future prospects."
    
    print(f"\nStarting deep research on query: {query}")
    print("\nThis process will:")
    print("1. Create a structured report plan with multiple sections")
    print("2. Conduct iterative research on each section in parallel")
    print("3. Combine all sections into a polished final report")
    print("\nThis may take several minutes as the agents perform research on multiple sections.")
    
    # Record start time
    start_time = time.time()
    
    # Run the deep research and get the final report
    report = await researcher.run(query)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\nDeep research completed in {int(minutes)} minutes and {int(seconds)} seconds")
    print("\n=== FINAL RESEARCH REPORT ===\n")
    print(report)


if __name__ == "__main__":
    asyncio.run(main()) 