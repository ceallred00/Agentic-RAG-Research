""" Factory for LLM clients."""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict
from configs.schemas.agent_schemas import AgentConfig

logger = logging.getLogger(__name__)

class ExecutionService:
    """Factory for LLM clients.
    Receives and processes only validated Pydantic models from ConfigLoader.
    """
    def __init__(self, agent_configs: Dict[str, AgentConfig]):
        """
        Parameters:
            agent_config [Dict]:
                Key: Agent name as found in YAML config.
                Value: Validated AgentConfig Pydantic model.
        """
        self.agent_configs = agent_configs

    def get_gemini_client(self, agent_name: str):
        """
        Factory method to create a configured ChatGoogleGenerativeAI client.
        Parameters:
            agent_name (str):
                The name of the agent whose configuration will be used to set up the client.
        
        Raises:
            ValueError: 
                If the agent_name is not found in loaded configurations.
                If the GEMINI_API_KEY is not set in environment variables.
        Returns: 
            Configured ChatGoogleGenerativeAI client instance.
            """
        logger.info(f"Creating Gemini client for agent: {agent_name}")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        logger.info(f"Successfully retrieved GEMINI_API_KEY from environment.")
        
        agent_specific_config = self.agent_configs.get(agent_name)

        if not agent_specific_config:
            raise ValueError(f" Agent '{agent_name}' not found in loaded configurations. Check YAML files against Pydantic models.")
        
        model_data = agent_specific_config.model
        model_name = model_data.name
        model_temperature = model_data.temperature

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
