from typing import Union, Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
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

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class LLMConfig:
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        use_azure: bool = True,
        use_google: bool = False
    ):
        self.tavily_api_key = TAVILY_API_KEY
        
        if use_google:
            # Use Google Gemini models with GLA provider (requires API key)
            google_provider = GoogleGLAProvider(
                api_key=api_key or GOOGLE_API_KEY
            )
            self.model = GeminiModel(
                model_name or "gemini-2.0-flash",
                provider=google_provider
            )
        elif use_azure and not any([model_name, base_url, api_key, api_version]):
            # Default to Azure OpenAI if no custom parameters provided
            azure_provider = AzureProvider(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
                api_key=AZURE_OPENAI_API_KEY,
            )
            
            self.model = OpenAIModel(
                AZURE_OPENAI_DEPLOYMENT,  # Using deployment name as model name
                provider=azure_provider,
            )
        else:
            # Use custom OpenAI configuration
            if use_azure:
                # Custom Azure configuration
                provider = AzureProvider(
                    azure_endpoint=base_url or AZURE_OPENAI_ENDPOINT,
                    api_version=api_version or AZURE_OPENAI_API_VERSION,
                    api_key=api_key or AZURE_OPENAI_API_KEY,
                )
                self.model = OpenAIModel(
                    model_name or AZURE_OPENAI_DEPLOYMENT,
                    provider=provider,
                )
            else:
                # Use standard OpenAI provider (could be OpenAI API or Ollama, etc.)
                provider = OpenAIProvider(
                    base_url=base_url,
                    api_key=api_key,
                )
                self.model = OpenAIModel(
                    model_name,
                    provider=provider,
                )

def create_default_config() -> LLMConfig:
    return LLMConfig()

def create_custom_config(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    api_version: str = None,
    use_azure: bool = False,
    use_google: bool = False
) -> LLMConfig:
    """Create a custom LLM configuration.
    
    Args:
        model_name: The name of the model to use
        base_url: Base URL for the API (required for Ollama, etc.)
        api_key: API key (optional for some providers like Ollama)
        api_version: API version (mostly used for Azure)
        use_azure: Whether to use Azure OpenAI provider
        use_google: Whether to use Google Gemini models
    
    Returns:
        An LLMConfig object with the specified parameters
    """
    return LLMConfig(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        api_version=api_version,
        use_azure=use_azure,
        use_google=use_google
    )

def create_google_config(
    model_name: str = "gemini-2.0-flash",
    api_key: str = None
) -> LLMConfig:
    """Create a configuration for Google Gemini models using the Generative Language API.
    
    Args:
        model_name: The name of the Gemini model to use
        api_key: Google API key for the Generative Language API
        
    Returns:
        An LLMConfig object configured for Google Gemini models
    """
    return LLMConfig(
        model_name=model_name,
        api_key=api_key,
        use_azure=False,
        use_google=True
    )

def get_base_url(model: Union[OpenAIModel, GeminiModel]) -> str:
    """Utility function to get the base URL for a given model"""
    if isinstance(model, GeminiModel):
        if hasattr(model._provider, 'region'):
            return f"https://{model._provider.region}-aiplatform.googleapis.com"
        return "https://generativelanguage.googleapis.com"
    if hasattr(model._provider, 'azure_endpoint'):
        return str(model._provider.azure_endpoint)
    return model._provider.base_url

def model_supports_structured_output(model: Union[OpenAIModel, GeminiModel]) -> bool:
    """Utility function to check if a model supports structured output"""
    # Most OpenAI-compatible models and Gemini 2.0 models support structured output
    return True
