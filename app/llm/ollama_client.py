import requests
from typing import Optional
from langchain_community.llms import Ollama
from app.config.settings import settings
from app.utils.logger import logger
from app.utils.exceptions import LLMQueryError

SUPPORTED_MODELS = ["llama3", "mistral", "deepseek"]

class OllamaClient:
    """Ollama LLM integration client supporting llama3, mistral, and deepseek models."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self._validate_model(self.model)

    def _validate_model(self, model_name: str):
        """Validate if the model is among supported architectures."""
        base_model_name = model_name.split(":")[0].lower()
        if not any(supported in base_model_name for supported in SUPPORTED_MODELS):
            logger.warning(f"Model '{model_name}' is not in standard list {SUPPORTED_MODELS}. Proceeding with configuration.")

    def get_llm(self, model_override: Optional[str] = None, temperature: float = 0.1) -> Ollama:
        """Returns initialized LangChain Ollama LLM instance."""
        target_model = model_override or self.model
        logger.info(f"Initializing Ollama client [Model: '{target_model}', Base URL: '{self.base_url}']")
        try:
            return Ollama(
                model=target_model,
                base_url=self.base_url,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Failed to instantiate Ollama client: {str(e)}", exc_info=True)
            raise LLMQueryError(f"Failed to connect to Ollama model '{target_model}': {str(e)}")

    def check_health(self) -> bool:
        """Ping Ollama service endpoint to check connectivity."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed at {self.base_url}: {str(e)}")
            return False

# Global singleton client
ollama_client = OllamaClient()
