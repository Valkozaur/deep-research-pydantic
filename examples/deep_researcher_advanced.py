#!/usr/bin/env python
"""
Advanced example script that demonstrates using the DeepResearcher with
additional options like saving the report to a file and configuring the LLM.

This example shows:
1. How to configure the LLM model
2. How to run a deep research query
3. How to save the final report to a Markdown file
"""

import asyncio
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.deep_researcher import DeepResearcher
from src.llm_config import LLMConfig


async def main():
    """Run the advanced deep researcher example."""
    print("=== Advanced Deep Researcher Example ===")
    
    # Create a custom LLM configuration
    # You can specify a different model here if desired
    llm_config = LLMConfig(
        model="openai:gpt-4o",  # Change this to your preferred model
        temperature=0.2,  # Lower temperature for more factual responses
        max_tokens=2000,  # Adjust token limit as needed
    )
    
    # Create the deep researcher with custom settings
    researcher = DeepResearcher(
        max_iterations=3,  # For faster example; increase for more thorough research
        max_minutes=7,     # Time limit per section
        verbose=True,      # Print detailed progress
        config=llm_config  # Use our custom LLM configuration
    )
    
    # Define the research query
    query = "Analyze the impact of artificial intelligence on healthcare, including current applications, challenges, and future prospects."
    
    print(f"\nStarting deep research with model: {llm_config.model}")
    print(f"Query: {query}")
    print("\nThis may take several minutes...")
    
    # Record start time
    start_time = time.time()
    
    # Run the deep research and get the final report
    report = await researcher.run(query)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\nDeep research completed in {int(minutes)} minutes and {int(seconds)} seconds")
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_query = query.split(".")[0][:30].replace(" ", "_").lower()
    filename = f"{timestamp}_{sanitized_query}.md"
    filepath = reports_dir / filename
    
    # Save the report to a file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {filepath}")
    print("\nReport preview (first 500 characters):")
    print("----------------------------------------")
    print(report[:500] + "...")
    print("----------------------------------------")
    print(f"\nView the full report in the file: {filepath}")


if __name__ == "__main__":
    asyncio.run(main()) 