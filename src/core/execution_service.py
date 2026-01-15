""" Factory for LLM clients."""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import Dict, Literal, Optional
from schemas.agent_schemas import AgentConfig
from pydantic import SecretStr

logger = logging.getLogger(__name__)

class ExecutionService:
    """Factory for LLM clients.
    Receives and processes only validated Pydantic models from ConfigLoader.
    """
    def __init__(self, agent_configs: Optional[Dict[str, AgentConfig]] = None):
        """
        Initializes the ExecutionService and verifies the API key exists.
        Parameters:
            agent_config (Optional[Dict[str, AgentConfig]]):
                Dictionary of agent configurations.
                    Key: Agent name as found in YAML config.
                    Value: Validated AgentConfig Pydantic model.
                Can be None if this service instance is only used for embeddings
                or generic model creation.
        """
        self.agent_configs = agent_configs or {}
    def _validate_api_key(self, api_key_name: str = "GEMINI_API_KEY"):
        """
        Internal helper to fetch and validate the API Key.

        Raises: 
            ValueError:
                If the specified api key is not set in environment variables.
        """
        api_key = os.getenv(api_key_name)
        if not api_key:
            logger.error(f"{api_key_name} not found in environment variables.")
            raise ValueError(f"{api_key_name} not found in environment variables.")
        logger.info(f"Successfully retrieved {api_key_name} from environment.")
        return api_key

    def get_gemini_client(self, agent_name: str):
        """
        Factory method to create a configured ChatGoogleGenerativeAI client based on a specific agent configuration.

        Parameters:
            agent_name (str):
                The name of the agent whose configuration will be used to set up the client.
        
        Raises:
            ValueError: 
                If the agent_name is not found in loaded configurations.
        Returns: 
            Configured ChatGoogleGenerativeAI client instance.
        """
        logger.info(f"Creating Gemini client for agent: {agent_name}")

        if not self.agent_configs:
            raise ValueError("No agent configurations were loaded into ExecutionService.")

        api_key = self._validate_api_key("GEMINI_API_KEY")
        
        agent_specific_config = self.agent_configs.get(agent_name)

        if not agent_specific_config:
            raise ValueError(f" Agent '{agent_name}' not found in loaded configurations. Check YAML files against Pydantic models.")
        
        model_name = agent_specific_config.model.name
        model_temperature = agent_specific_config.model.temperature

        try:
            gemini_model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=model_temperature,
                api_key=api_key
            )

            logger.info(f"Gemini client created for agent '{agent_name}' with model '{model_name}' and temperature {model_temperature}.")
        
        except Exception as e:
            logger.error(f"Error creating Gemini client for agent '{agent_name}': {e}")
            raise
                
        return gemini_model
    def get_embedding_client(self, model_name: str, task_type: Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]):
        """
        Factory method to create a configured Embedding client.

        Note: 
            GoogleGenerativeAIEmbeddings class automatically checks for the existence of the GEMINI_API_KEY environment variable.
        """
        logger.info(f"Creating Embedding client for model: {model_name}")

        # Verifies the existence of the GEMINI API Key.
        raw_api_key = self._validate_api_key("GEMINI_API_KEY")

        try:
            embedding_client = GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type = task_type,
                api_key = SecretStr(raw_api_key), # Casting required for type-checker.,
                output_dimensionality = 768 # Set for compatibility with Pinecone Vector DB
            )
            logger.info(f"Embedding client created for model '{model_name}' with task type '{task_type}'.")
            return embedding_client
        except Exception as e:
            logger.error(f"Error creating Embedding client: {e}")
            raise





