"""
A class that manages the deep research process by coordinating specialized agents.
Breaks down a research query into a report plan with sections, runs iterative research for each section,
and compiles a final comprehensive report.
"""
import asyncio
import time
from typing import List, Optional

from pydantic import BaseModel

from .iterative_researcher import IterativeResearcher
from .agents.planner_agent import init_planner_agent, ReportPlan, ReportPlanSection
from .agents.proofreader_agent import ReportDraftSection, ReportDraft, init_proofreader_agent
from .agents.long_writer_agent import init_long_writer_agent, write_report
from .llm_config import LLMConfig, create_default_config


class DeepResearcher:
    """
    Manager for the deep research workflow that breaks down a query into a report plan with sections 
    and then runs an iterative research loop for each section.
    """
    def __init__(
            self, 
            max_iterations: int = 5,
            max_minutes: int = 10,
            verbose: bool = True,
            config: Optional[LLMConfig] = None
        ):
        """
        Initialize the deep researcher.
        
        Args:
            max_iterations: The maximum number of iterations for each section research
            max_minutes: The maximum number of minutes to spend on each section research
            verbose: Whether to log detailed progress information
            config: The LLM configuration to use for all agents
        """
        self.max_iterations = max_iterations
        self.max_minutes = max_minutes
        self.verbose = verbose
        self.config = create_default_config() if not config else config
        
        # Initialize specialized agents
        self.planner_agent = init_planner_agent(self.config)
        self.proofreader_agent = init_proofreader_agent(self.config)
        self.long_writer_agent = init_long_writer_agent(self.config)

    async def run(self, query: str) -> str:
        """
        Run the deep research workflow
        
        Args:
            query: The research query to investigate
            
        Returns:
            The final research report as a string
        """
        start_time = time.time()
        
        # First build the report plan
        self._log_message("=== Building Report Plan ===")
        report_plan = await self._build_report_plan(query)
        
        # Run the independent research loops for each section
        self._log_message("=== Initializing Research Loops ===")
        research_results = await self._run_research_loops(report_plan)
        
        # Create the final report
        self._log_message("=== Building Final Report ===")
        final_report = await self._create_final_report(query, report_plan, research_results)
        
        elapsed_time = time.time() - start_time
        self._log_message(f"DeepResearcher completed in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds")
        
        return final_report

    async def _build_report_plan(self, query: str) -> ReportPlan:
        """
        Build the initial report plan including the report outline and background context
        
        Args:
            query: The research query to investigate
            
        Returns:
            A ReportPlan object containing the report outline and background context
        """
        user_message = f"QUERY: {query}"
        result = await self.planner_agent.run(user_message)
        report_plan = result.output
        
        if self.verbose:
            num_sections = len(report_plan.report_outline)
            message_log = '\n\n'.join(f"Section: {section.title}\nKey question: {section.key_question}" 
                                     for section in report_plan.report_outline)
            if report_plan.background_context:
                message_log += f"\n\nThe following background context has been included for the report build:\n{report_plan.background_context}"
            else:
                message_log += "\n\nNo background context was provided for the report build.\n"
            self._log_message(f"Report plan created with {num_sections} sections:\n{message_log}")
        
        return report_plan

    async def _run_research_loops(self, report_plan: ReportPlan) -> List[str]:
        """
        For a given ReportPlan, run a research loop concurrently for each section
        
        Args:
            report_plan: The ReportPlan object containing the report outline
            
        Returns:
            A list of section drafts (research results)
        """
        async def run_research_for_section(section: ReportPlanSection):
            iterative_researcher = IterativeResearcher(
                max_iterations=self.max_iterations,
                max_minutes=self.max_minutes,
                verbose=self.verbose,
                config=self.config
            )
            
            return await iterative_researcher.run_research(
                query=section.key_question,
                background_context=report_plan.background_context
            )
        
        # Run all research loops concurrently
        research_results = await asyncio.gather(
            *(run_research_for_section(section) for section in report_plan.report_outline)
        )
        return research_results

    async def _create_final_report(
        self, 
        query: str, 
        report_plan: ReportPlan, 
        section_drafts: List[str],
        use_long_writer: bool = True
    ) -> str:
        """
        Create the final report from the original report plan and the drafts of each section
        
        Args:
            query: The original research query
            report_plan: The ReportPlan object containing the report outline
            section_drafts: A list of section drafts (research results)
            use_long_writer: Whether to use the long writer agent
            
        Returns:
            The final research report as a string
        """
        # Build a ReportDraft object
        report_draft = ReportDraft(sections=[])
        
        for i, section_draft in enumerate(section_drafts):
            report_draft.sections.append(
                ReportDraftSection(
                    section_title=report_plan.report_outline[i].title,
                    section_content=section_draft
                )
            )

        if use_long_writer:
            final_output = await write_report(
                self.long_writer_agent, 
                query, 
                report_plan.report_title, 
                report_draft
            )
        else:
            user_prompt = f"QUERY:\n{query}\n\nREPORT DRAFT:\n{report_draft.model_dump_json()}"
            # Run the proofreader agent to produce the final report
            final_report = await self.proofreader_agent.run(user_prompt)
            final_output = final_report.output

        self._log_message("Final report completed")
        return final_output

    def _log_message(self, message: str) -> None:
        """Log a message if verbose is True"""
        if self.verbose:
            print(message)