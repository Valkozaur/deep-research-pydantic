import asyncio
import time
from src.iterative_research import IterativeAgent
from src.llm_config import create_default_config

async def main():
    """Run the iterative research agent example."""
    print("=== Iterative Research Agent Example ===")
    
    # Create the iterative agent with default settings
    # - max_iterations: 5 (for faster example)
    # - max_minutes: 10 (time limit of 10 minutes)
    max_iterations = 5
    max_minutes = 10
    
    print(f"Creating iterative agent with max_iterations={max_iterations}, max_minutes={max_minutes}")
    agent = IterativeAgent(max_iterations=max_iterations, max_minutes=max_minutes, verbose=True)
    
    # Define the research query
    query = "Write a report on Plato - who was he, what were his main works and what are the main philosophical ideas he's known for"
    
    # Optional background context
    background_context = "Some initial context about ancient Greek philosophy."
    
    print(f"\nStarting iterative research on query: {query}")
    print(f"Background context: {background_context}")
    print("\nThis may take several minutes as the agent performs multiple iterations of research.")
    print("The agent will automatically stop after 5 iterations or 10 minutes, whichever comes first.")
    
    # Define a progress callback function to print progress updates
    def progress_callback(message: str):
        print(f"Progress: {message}")
    
    # Record start time
    start_time = time.time()
    
    # Run the research and get the final report
    report = await agent.run_research(
        query=query, 
        background_context=background_context,
        progress_callback=progress_callback
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\nResearch completed in {int(minutes)} minutes and {int(seconds)} seconds")
    print("\n=== FINAL RESEARCH REPORT ===\n")
    print(report)

if __name__ == "__main__":
    asyncio.run(main()) 