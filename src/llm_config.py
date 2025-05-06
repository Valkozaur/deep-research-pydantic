from typing import Union
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def get_env_with_prefix(key: str, default: str = None) -> str:
    """Get environment variable with optional prefix."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not found")
    return value

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = get_env_with_prefix("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = get_env_with_prefix("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = get_env_with_prefix("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = get_env_with_prefix("AZURE_OPENAI_API_VERSION")
TAVILY_API_KEY = get_env_with_prefix("TAVILY_API_KEY")

class LLMConfig:
    def __init__(self):
        # Initialize Azure OpenAI model using Pydantic AI
        azure_provider = AzureProvider(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
        )
        
        self.model = OpenAIModel(
            AZURE_OPENAI_DEPLOYMENT,  # Using deployment name as model name
            provider=azure_provider,
        )

        self.tavily_api_key = TAVILY_API_KEY

def create_default_config() -> LLMConfig:
    return LLMConfig()

def get_base_url(model: OpenAIModel) -> str:
    """Utility function to get the base URL for a given model"""
    return str(model._provider.azure_endpoint)

def model_supports_structured_output(model: OpenAIModel) -> bool:
    """Utility function to check if a model supports structured output"""
    # Azure OpenAI models support structured output
    return True
