import asyncio
from src.agents.thinking_agent import init_thinking_agent
from src.agents.knowledge_gap_agent import init_knowledge_gap_agent, KnowledgeGapOutput
from src.agents.tool_selector_agent import init_tool_selector_agent
from src.llm_config import create_default_config

async def run_iterative_research():
    # Create the default LLM configuration
    config = create_default_config()
    
    # Initialize the agents with the configuration
    thinking_agent = init_thinking_agent(config)
    knowledge_gap_agent = init_knowledge_gap_agent(config)
    tool_selector_agent = init_tool_selector_agent(config)

    # Your research query
    original_query = (
        "Write a report on Plato - who was he, what were his main works "
        "and what are the main philosophical ideas he's known for"
    )

    # Sample input string for the Thinking Agent
    # This matches the format expected by the thinking agent
    agent_input_string = f"""
===========================================================
ORIGINAL QUERY: {original_query}

BACKGROUND CONTEXT: Some initial context about ancient Greek philosophy.

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
Iteration 1: Initial thoughts - need to find biographical info, list of works, and major philosophical concepts. Planned to start with a web search. Findings: Found birth/death dates, key works like 'Republic', 'Symposium', and concepts like Forms and Idealism.
===========================================================
"""

    print("Running Thinking Agent...")
    print("Input:")
    print(agent_input_string)

    # Run the thinking agent asynchronously
    try:
        thinking_result = await thinking_agent.run(agent_input_string)
        print("\nThinking Agent Output:")
        print(thinking_result.output)
        
        # Prepare input for the knowledge gap agent
        # We'll use the original query and include the thinking agent's output
        # in the history of actions
        knowledge_gap_input = f"""
===========================================================
ORIGINAL QUERY: {original_query}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
Iteration 1: Initial thoughts - need to find biographical info, list of works, and major philosophical concepts. Planned to start with a web search. Findings: Found birth/death dates, key works like 'Republic', 'Symposium', and concepts like Forms and Idealism.

Iteration 2: {thinking_result.output}
===========================================================
"""
        
        # Run the knowledge gap agent
        print("\nRunning Knowledge Gap Agent...")
        knowledge_gap_result = await knowledge_gap_agent.run(knowledge_gap_input)
        print("\nKnowledge Gap Agent Output:")
        print(knowledge_gap_result.output)
        
        # The output is typed as KnowledgeGapOutput due to our agent configuration
        gaps = knowledge_gap_result.output
        
        print("\nResearch Complete:", gaps.research_complete)
        print("Outstanding Gaps:")
        
        if not gaps.research_complete:
            for i, gap in enumerate(gaps.outstanding_gaps, 1):
                print(f"{i}. {gap}")
                
                # For each gap, run the tool selector agent
                tool_selector_input = f"""
                ===========================================================
                ORIGINAL QUERY: {original_query}

                KNOWLEDGE GAP TO ADDRESS: {gap}

                BACKGROUND CONTEXT: Some initial context about ancient Greek philosophy.

                HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
                Iteration 1: Initial thoughts - need to find biographical info, list of works, and major philosophical concepts. Planned to start with a web search. Findings: Found birth/death dates, key works like 'Republic', 'Symposium', and concepts like Forms and Idealism.

                Iteration 2: {thinking_result.output}
                ===========================================================
                """
                print(f"\nRunning Tool Selector Agent for Gap {i}...")
                tool_selector_result = await tool_selector_agent.run(tool_selector_input)
                
                print(f"\nTool Selector Output for Gap {i}:")
                print(tool_selector_result.output)
                
        else:
            print("Research complete - no outstanding gaps")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

# To run the async function
if __name__ == "__main__":
    asyncio.run(run_iterative_research())