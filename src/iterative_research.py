"""
Agent class that manages the iterative research process.

This agent coordinates the various specialized agents (thinking, knowledge gap, 
tool selector, search, writer) to perform iterative research and produce a final report.

The IterativeAgent takes the following parameters:
- max_iterations: The maximum number of iterations to perform
- max_minutes: The maximum number of minutes to spend on research
- config: The LLM configuration to use for all agents

The IterativeAgent's main method is run_research(), which takes a query and optional background context
and returns a final report.
"""
import asyncio
import time
import logging
from typing import List, Dict, Optional, Any, Coroutine, Callable
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai import Agent

from .llm_config import LLMConfig
from .agents.thinking_agent import init_thinking_agent
from .agents.knowledge_gap_agent import init_knowledge_gap_agent, KnowledgeGapOutput
from .agents.tool_selector_agent import init_tool_selector_agent, AgentSelectionPlan, AgentTask
from .agents.tool_agents.search_agent import init_search_agent
from .agents.writer_agent import init_writer_agent
from .agents.tool_agents import ToolAgentOutput, init_tool_agents


# Configure logging
logger = logging.getLogger("iterative_agent")

# Callback type for progress updates
ProgressCallback = Callable[[str], None]


class ResearchState(BaseModel):
    """Tracks the state of the iterative research process"""
    original_query: str
    background_context: str
    all_findings: List[str]
    history_of_actions: str
    start_time: float
    iteration_count: int
    research_complete: bool


class IterativeAgent:
    """
    Agent that manages the iterative research process by coordinating specialized agents.
    """
    
    def __init__(self, max_iterations: int = 10, max_minutes: int = 30, 
                 config: Optional[LLMConfig] = None, verbose: bool = False):
        """
        Initialize the iterative agent.
        
        Args:
            max_iterations: The maximum number of iterations to perform
            max_minutes: The maximum number of minutes to spend on research
            config: The LLM configuration to use for all agents
            verbose: Whether to log detailed progress information
        """
        self.max_iterations = max_iterations
        self.max_minutes = max_minutes
        self.verbose = verbose
        
        # Create the default LLM configuration if not provided
        if config is None:
            from .llm_config import create_default_config
            self.config = create_default_config()
        else:
            self.config = config
        
        # Initialize the specialized agents
        self.thinking_agent = init_thinking_agent(self.config)
        self.knowledge_gap_agent = init_knowledge_gap_agent(self.config)
        self.tool_selector_agent = init_tool_selector_agent(self.config)
        self.writer_agent = init_writer_agent(self.config)
        
        # Initialize tool agents dictionary
        self.tool_agents = init_tool_agents(self.config)
    
    async def run_research(self, query: str, background_context: str = "", 
                         progress_callback: Optional[ProgressCallback] = None) -> str:
        """
        Run the iterative research process on the given query.
        
        Args:
            query: The research query to investigate
            background_context: Optional background context for the query
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            The final research report as a string
        """
        # Initialize the research state
        state = ResearchState(
            original_query=query,
            background_context=background_context,
            all_findings=[],
            history_of_actions="",
            start_time=time.time(),
            iteration_count=1,
            research_complete=False
        )
        
        # Function to log progress
        def log_progress(message: str):
            if self.verbose:
                logger.info(message)
            if progress_callback:
                progress_callback(message)
        
        log_progress(f"Starting research on query: {query}")
        log_progress(f"Maximum iterations: {self.max_iterations}, maximum minutes: {self.max_minutes}")
        
        # Start with the thinking agent to get initial thoughts
        log_progress("Running thinking agent for initial thoughts...")
        thinking_input = self._format_thinking_input(state)
        thinking_result = await self.thinking_agent.run(thinking_input)
        
        # Record the initial thoughts
        initial_thought = f"Iteration {state.iteration_count}: {thinking_result.output}"
        state.all_findings.append(initial_thought)
        state.history_of_actions += f"{initial_thought}\n\n"
        log_progress(f"Initial thoughts: {thinking_result.output}")
        
        # Main research loop
        while not state.research_complete and state.iteration_count < self.max_iterations:
            # Check if we've exceeded the time limit
            elapsed_minutes = (time.time() - state.start_time) / 60
            if elapsed_minutes >= self.max_minutes:
                message = f"Reached maximum time limit of {self.max_minutes} minutes. Research may not self-complete."
                state.all_findings.append(message)
                log_progress(message)
                break
            
            # Increment iteration counter
            state.iteration_count += 1
            log_progress(f"\n--- Starting Iteration {state.iteration_count} ---")
            
            # Run the knowledge gap agent to identify gaps
            log_progress("Running knowledge gap agent to identify gaps...")
            gaps = await self._run_knowledge_gap_agent(state)
            
            # Update the research state
            state.research_complete = gaps.research_complete
            
            if gaps.research_complete:
                log_progress("Knowledge gap agent determined that research is complete!")
            else:
                log_progress(f"Knowledge gap agent identified {len(gaps.outstanding_gaps)} gaps:")
                for i, gap in enumerate(gaps.outstanding_gaps):
                    log_progress(f"  Gap {i+1}: {gap}")
                
                # Process each knowledge gap
                await self._process_knowledge_gaps(gaps.outstanding_gaps, state, log_progress)
            
        # Generate the final report
        log_progress("\nResearch complete. Generating final report...")
        final_report = await self._generate_final_report(state)
        log_progress("Final report generated successfully.")
        
        return final_report
    
    def _format_thinking_input(self, state: ResearchState) -> str:
        """Format the input for the thinking agent."""
        return f"""
        ===========================================================
        ORIGINAL QUERY: {state.original_query}

        BACKGROUND CONTEXT: {state.background_context}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {state.history_of_actions}
        ===========================================================
        """
    
    async def _run_knowledge_gap_agent(self, state: ResearchState) -> KnowledgeGapOutput:
        """Run the knowledge gap agent to identify gaps in the research."""
        knowledge_gap_input = f"""
        ===========================================================
        ORIGINAL QUERY: {state.original_query}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {state.history_of_actions}
        ===========================================================
        """
        
        knowledge_gap_result = await self.knowledge_gap_agent.run(knowledge_gap_input)
        return knowledge_gap_result.output
    
    async def _process_knowledge_gaps(self, gaps: List[str], state: ResearchState, 
                                     log_progress: ProgressCallback):
        """Process each knowledge gap by selecting and running appropriate tools."""
        # Create tasks for processing all gaps
        tasks = []
        task_details = []  # To keep track of task details for logging
        
        for gap_index, gap in enumerate(gaps):
            # For each gap, run the tool selector agent
            log_progress(f"Running tool selector for Gap {gap_index + 1}...")
            
            tool_selector_input = f"""
            ===========================================================
            ORIGINAL QUERY: {state.original_query}

            KNOWLEDGE GAP TO ADDRESS: {gap}

            BACKGROUND CONTEXT: {state.background_context}

            HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
            {state.history_of_actions}
            ===========================================================
            """
            
            tool_selector_result = await self.tool_selector_agent.run(tool_selector_input, output_type=AgentSelectionPlan)
            plan = tool_selector_result.output
            
            log_progress(f"Tool selector recommended {len(plan.tasks)} tasks for Gap {gap_index + 1}")
            
            # Create a task for each agent in the plan
            for task_index, task in enumerate(plan.tasks):
                log_progress(f"  Task {task_index + 1}: Using {task.agent} to search for '{task.query}'")
                
                # Check if the agent exists in our tool agents dictionary
                if task.agent in self.tool_agents:
                    # Create a task for the agent
                    coro = self._run_tool_agent(gap_index, task_index, task, gap)
                    tasks.append(coro)
                    task_details.append((gap_index, task_index, task.agent, task.query))
                else:
                    # Log that the agent is not implemented
                    message = f"Gap {gap_index + 1}, Task {task_index + 1} - Agent '{task.agent}' not implemented"
                    log_progress(f"  Warning: {message}")
                    state.all_findings.append(message)
                    state.history_of_actions += f"\nIteration {state.iteration_count}, {message}\n"
        
        # Run all agent tasks concurrently
        if tasks:
            log_progress(f"Running {len(tasks)} agent tasks concurrently...")
            results = await asyncio.gather(*tasks)
            
            # Process the results
            for i, result in enumerate(results):
                if result:
                    gap_idx, task_idx, agent_name, query = task_details[i]
                    log_progress(f"  Completed: Gap {gap_idx + 1}, Task {task_idx + 1} - {agent_name} for '{query}'")
                    state.all_findings.append(result)
                    state.history_of_actions += f"\nIteration {state.iteration_count}, {result}\n"
        else:
            log_progress("No tasks to run for current gaps.")
    
    async def _create_message_coro(self, message: str) -> str:
        """Create a coroutine that returns a fixed message."""
        return message
    
    async def _run_tool_agent(self, gap_index: int, task_index: int, task: AgentTask, gap: str) -> str:
        """Run a tool agent for a specific task."""
        try:
            # Get the agent from the tool agents dictionary
            agent = self.tool_agents[task.agent]
            
            # Run the agent with the query
            agent_output = await agent.run(task.query)
            
            # Format the agent results
            # Note: Tool agents should return ToolAgentOutput objects
            if hasattr(agent_output, 'output') and hasattr(agent_output.output, 'output'):
                # Handle the nested output structure from tool agents
                output_text = agent_output.output.output
                sources = agent_output.output.sources if hasattr(agent_output.output, 'sources') else []
                
                # Format the output with sources
                if sources:
                    source_text = ", ".join(sources)
                    return f"Gap {gap_index + 1}, Task {task_index + 1} - {task.agent} for '{task.query}': {output_text} (Sources: {source_text})"
                else:
                    return f"Gap {gap_index + 1}, Task {task_index + 1} - {task.agent} for '{task.query}': {output_text}"
            else:
                # Fallback for unexpected output format
                return f"Gap {gap_index + 1}, Task {task_index + 1} - {task.agent} for '{task.query}': {str(agent_output)}"
                
        except Exception as e:
            return f"Gap {gap_index + 1}, Task {task_index + 1} - Error when running {task.agent} for '{task.query}': {str(e)}"
    
    async def _run_search_agent(self, gap_index: int, task_index: int, task: AgentTask, gap: str) -> str:
        """
        Run the search agent for a specific task.
        
        Note: This is kept for backward compatibility, but _run_tool_agent should be used instead.
        """
        return await self._run_tool_agent(gap_index, task_index, task, gap)
    
    async def _generate_final_report(self, state: ResearchState) -> str:
        """Generate the final research report."""
        writer_input = f"""
        ===========================================================
        QUERY: {state.original_query}

        FINDINGS: 
        {"\n".join(state.all_findings)}
        ===========================================================
        """
        
        writer_result = await self.writer_agent.run(writer_input)
        return writer_result.output 