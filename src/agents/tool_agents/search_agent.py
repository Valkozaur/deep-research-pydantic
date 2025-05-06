"""
Agent used to perform web searches and summarize the results.

The SearchAgent takes as input a string in the format of AgentTask.model_dump_json(), or can take a simple query string as input

The Agent then:
1. Uses the web_search tool to retrieve search results
2. Analyzes the retrieved information
3. Writes a 3+ paragraph summary of the search results
4. Includes citations/URLs in brackets next to information sources
5. Returns the formatted summary as a string

The agent can use either OpenAI's built-in web search capability or a custom
web search implementation based on environment configuration.
"""

from pydantic_ai import Agent
from ...llm_config import LLMConfig
from . import ToolAgentOutput
from pydantic_ai.common_tools.tavily import tavily_search_tool

INSTRUCTIONS = f"""You are a research assistant that specializes in retrieving and summarizing information from the web.

OBJECTIVE:
Given an AgentTask, follow these steps:
- Convert the 'query' into an optimized SERP search term for Google, limited to 3-5 words
- If an 'entity_website' is provided, make sure to include the domain name in your optimized Google search term
- Enter the optimized search term into the web_search tool
- After using the web_search tool, write a 3+ paragraph summary that captures the main points from the search results

GUIDELINES:
- In your summary, try to comprehensively answer/address the 'gap' provided (which is the objective of the search)
- The summary should always quote detailed facts, figures and numbers where these are available
- If the search results are not relevant to the search term or do not address the 'gap', simply write "No relevant results found"
- Use headings and bullets to organize the summary if needed
- Include citations/URLs in brackets next to all associated information in your summary
- Do not make additional searches

OUTPUT FORMAT:
Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only. Ensure you are escaping any characters that are not valid JSON or can break the JSON parser:
{ToolAgentOutput.model_json_schema()}
```
"""

def init_search_agent(config: LLMConfig) -> Agent:
    """
    Initialize a search agent that can perform web searches and summarize results.
    
    Args:
        config: LLM configuration
        
    Returns:
        An initialized Agent
    """
    return Agent(
        name="Web Search Agent",
        model=config.model,
        tools=[tavily_search_tool(config.tavily_api_key)],
        description="An agent that can search the web and summarize results",
        system_prompt=INSTRUCTIONS,
        output_type=ToolAgentOutput
    )
