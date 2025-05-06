from pydantic import BaseModel, Field

class ToolAgentOutput(BaseModel):
    """Standard output for all tool agents"""
    output: str
    sources: list[str] = Field(default_factory=list)

from .search_agent import init_search_agent
from ...llm_config import LLMConfig

def init_tool_agents(config: LLMConfig) -> dict[str]:
    search_agent = init_search_agent(config)

    return {
        "WebSearchAgent": search_agent
    }
