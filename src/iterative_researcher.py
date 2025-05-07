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
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python, to_json

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
    # New field to store message history for each agent
    agent_messages: Dict[str, List[ModelMessage]] = {}


class IterativeResearcher:
    """
    Agent that manages the iterative research process by coordinating specialized agents.
    """
    
    def ensure_agent_message_keys(self, state: ResearchState):
        """Ensure all required agent message keys exist in the state."""
        required_keys = [
            "thinking_agent",
            "knowledge_gap_agent",
            "tool_selector_agent",
            "writer_agent"
        ]
        for key in required_keys:
            if key not in state.agent_messages:
                state.agent_messages[key] = []

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
        self.research_state = None  # Store the current research state
        
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
            research_complete=False,
            agent_messages={
                "thinking_agent": [],
                "knowledge_gap_agent": [],
                "tool_selector_agent": [],
                "writer_agent": []
            }
        )
        
        # Store the research state
        self.research_state = state
        
        # Ensure all required agent message keys exist
        self.ensure_agent_message_keys(state)
        
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
        
        # Format thinking agent input using message history
        thinking_input = self._format_thinking_input(state)
        
        # Run thinking agent using message history if available
        thinking_result = await self.thinking_agent.run(
            thinking_input,
            message_history=state.agent_messages["thinking_agent"] if state.agent_messages["thinking_agent"] else None
        )
        
        # Store message history for this agent
        state.agent_messages["thinking_agent"] = thinking_result.all_messages()
        
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
        
        # Optional: Save all agent messages for future use
        self._save_agent_messages(state)
        
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
        
        knowledge_gap_result = await self.knowledge_gap_agent.run(
            knowledge_gap_input, 
            message_history=state.agent_messages["knowledge_gap_agent"] if state.agent_messages["knowledge_gap_agent"] else None
        )
        
        # Store message history for this agent
        state.agent_messages["knowledge_gap_agent"] = knowledge_gap_result.all_messages()
        
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
            
            # If there's no message history for this specific gap, use an empty list
            gap_key = f"tool_selector_agent_gap_{gap_index}"
            if gap_key not in state.agent_messages:
                state.agent_messages[gap_key] = []
            
            tool_selector_result = await self.tool_selector_agent.run(
                tool_selector_input, 
                output_type=AgentSelectionPlan,
                message_history=state.agent_messages[gap_key] if state.agent_messages[gap_key] else None
            )
            
            # Store message history for this agent + gap
            state.agent_messages[gap_key] = tool_selector_result.all_messages()
            
            plan = tool_selector_result.output
            
            log_progress(f"Tool selector recommended {len(plan.tasks)} tasks for Gap {gap_index + 1}")
            
            # Create a task for each agent in the plan
            for task_index, task in enumerate(plan.tasks):
                log_progress(f"  Task {task_index + 1}: Using {task.agent} to search for '{task.query}'")
                
                # Check if the agent exists in our tool agents dictionary
                if task.agent in self.tool_agents:
                    # Create a task for the agent
                    coro = self._run_tool_agent(gap_index, task_index, task, gap, state)
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
    
    async def _run_tool_agent(self, gap_index: int, task_index: int, task: AgentTask, gap: str, state: ResearchState) -> str:
        """Run a tool agent for a specific task."""
        try:
            # Get the agent from the tool agents dictionary
            agent = self.tool_agents[task.agent]
            
            # Create a unique key for this agent + task combination
            task_key = f"{task.agent}_gap_{gap_index}_task_{task_index}"
            if task_key not in state.agent_messages:
                state.agent_messages[task_key] = []
            
            # Run the agent with the query, passing previous message history if available
            agent_output = await agent.run(
                task.query,
                message_history=state.agent_messages[task_key] if state.agent_messages[task_key] else None
            )
            
            # Store message history for this agent
            state.agent_messages[task_key] = agent_output.all_messages()
            
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
        return await self._run_tool_agent(gap_index, task_index, task, gap, state=None)
    
    async def _generate_final_report(self, state: ResearchState) -> str:
        """Generate the final research report."""
        writer_input = f"""
        ===========================================================
        QUERY: {state.original_query}

        FINDINGS: 
        {"\n".join(state.all_findings)}
        ===========================================================
        """
        
        writer_result = await self.writer_agent.run(
            writer_input,
            message_history=state.agent_messages["writer_agent"] if state.agent_messages["writer_agent"] else None
        )
        
        # Store message history for this agent
        state.agent_messages["writer_agent"] = writer_result.all_messages()
        
        return writer_result.output 
    
    def _save_agent_messages(self, state: ResearchState) -> None:
        """Save all agent messages to JSON for potential future use."""
        try:
            # Convert all messages to JSON-serializable objects
            serialized_messages = {}
            for key, messages in state.agent_messages.items():
                if messages:
                    # Use Pydantic's to_jsonable_python to convert to JSON-serializable format
                    serialized_messages[key] = to_jsonable_python(messages)
            
            # This could be expanded to save to disk if needed
            # For now, we just keep it in memory
            state.serialized_messages = serialized_messages
            
        except Exception as e:
            logger.error(f"Failed to serialize agent messages: {str(e)}")
    
    def load_agent_messages(self, serialized_messages: Dict[str, Any]) -> Dict[str, List[ModelMessage]]:
        """Load agent messages from serialized format."""
        loaded_messages = {}
        try:
            for key, messages in serialized_messages.items():
                # Use the TypeAdapter to convert back to ModelMessage objects
                loaded_messages[key] = ModelMessagesTypeAdapter.validate_python(messages)
            return loaded_messages
        except Exception as e:
            logger.error(f"Failed to load agent messages: {str(e)}")
            return {}
            
    def export_session(self, state: ResearchState, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the research session to a dictionary or a JSON file.
        
        Args:
            state: The current research state
            filepath: Optional filepath to save the JSON export
            
        Returns:
            Dictionary representation of the research session
        """
        try:
            # First, ensure all messages are serialized
            self._save_agent_messages(state)
            
            # Create a serializable version of the state
            session_export = {
                "original_query": state.original_query,
                "background_context": state.background_context,
                "all_findings": state.all_findings,
                "history_of_actions": state.history_of_actions,
                "start_time": state.start_time,
                "iteration_count": state.iteration_count,
                "research_complete": state.research_complete,
                "serialized_messages": getattr(state, "serialized_messages", {})
            }
            
            # If filepath is provided, save to file
            if filepath:
                import json
                with open(filepath, 'w') as f:
                    json.dump(session_export, f, indent=2)
                    logger.info(f"Research session exported to {filepath}")
                    
            return session_export
            
        except Exception as e:
            logger.error(f"Failed to export research session: {str(e)}")
            return {"error": str(e)}
    
    def import_session(self, session_data: Dict[str, Any]) -> ResearchState:
        """
        Import a previously exported research session.
        
        Args:
            session_data: Dictionary containing the research session data
            
        Returns:
            Reconstructed ResearchState object
        """
        try:
            # Create a new ResearchState
            state = ResearchState(
                original_query=session_data.get("original_query", ""),
                background_context=session_data.get("background_context", ""),
                all_findings=session_data.get("all_findings", []),
                history_of_actions=session_data.get("history_of_actions", ""),
                start_time=session_data.get("start_time", time.time()),
                iteration_count=session_data.get("iteration_count", 1),
                research_complete=session_data.get("research_complete", False)
            )
            
            # Load the serialized messages
            if "serialized_messages" in session_data and session_data["serialized_messages"]:
                agent_messages = self.load_agent_messages(session_data["serialized_messages"])
                state.agent_messages = agent_messages
            
            # Ensure all required agent message keys exist
            self.ensure_agent_message_keys(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to import research session: {str(e)}")
            # Return an empty state
            return ResearchState(
                original_query="",
                background_context="",
                all_findings=[],
                history_of_actions="",
                start_time=time.time(),
                iteration_count=1,
                research_complete=False
            )
            
    @classmethod
    def load_session_from_file(cls, filepath: str, config: Optional[LLMConfig] = None) -> tuple["IterativeResearcher", ResearchState]:
        """
        Load a research session from a file.
        
        Args:
            filepath: Path to the session file
            config: Optional LLM configuration
            
        Returns:
            Tuple of (IterativeAgent, ResearchState)
        """
        try:
            import json
            with open(filepath, 'r') as f:
                session_data = json.load(f)
                
            # Create a new IterativeAgent
            agent = cls(config=config)
            
            # Import the session
            state = agent.import_session(session_data)
            
            # Ensure all required agent message keys exist
            agent.ensure_agent_message_keys(state)
            
            return agent, state
            
        except Exception as e:
            logger.error(f"Failed to load session from file: {str(e)}")
            # Return a new agent and empty state
            agent = cls(config=config)
            state = ResearchState(
                original_query="",
                background_context="",
                all_findings=[],
                history_of_actions="",
                start_time=time.time(),
                iteration_count=1,
                research_complete=False
            )
            return agent, state
            
    async def continue_research(self, state: ResearchState, max_additional_iterations: int = 5, 
                              progress_callback: Optional[ProgressCallback] = None) -> str:
        """
        Continue research from a previous session.
        
        Args:
            state: The research state from a previous session
            max_additional_iterations: Maximum number of additional iterations to perform
            progress_callback: Optional callback function to receive progress updates
            
        Returns:
            The final research report as a string
        """
        # Store the research state
        self.research_state = state
        
        # Ensure all required agent message keys exist
        self.ensure_agent_message_keys(state)
        
        # Function to log progress
        def log_progress(message: str):
            if self.verbose:
                logger.info(message)
            if progress_callback:
                progress_callback(message)
                
        log_progress(f"Continuing research on query: {state.original_query}")
        log_progress(f"Starting from iteration {state.iteration_count}")
        log_progress(f"Maximum additional iterations: {max_additional_iterations}")
        
        # Calculate the new max iterations
        max_total_iterations = state.iteration_count + max_additional_iterations
        
        # Main research loop
        while not state.research_complete and state.iteration_count < max_total_iterations:
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
            
            # Continue from the knowledge gap agent
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
        
        # Save all agent messages
        self._save_agent_messages(state)
        
        return final_report 