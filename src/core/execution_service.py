"""Factory for LLM clients."""

import os
import logging
from typing import Dict, Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import AsyncOpenAI
from schemas.agent_schemas import AgentConfig

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
        Factory method to create a configured ChatGoogleGenerativeAI client based on a
        specific agent configuration.

        Parameters:
            agent_name (str):
                The name of the agent whose configuration will be used to set up the
                client.

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
            raise ValueError(
                f"""Agent '{agent_name}' not found in loaded configurations.
                Check YAML files against Pydantic models."""
            )

        model_name = agent_specific_config.model.name
        model_temperature = agent_specific_config.model.temperature

        try:
            gemini_model = ChatGoogleGenerativeAI(model=model_name, temperature=model_temperature, api_key=api_key)

            logger.info(
                f"""Gemini client created for agent '{agent_name}' with
                model '{model_name}' and temperature {model_temperature}."""
            )

        except Exception as e:
            logger.error(f"Error creating Gemini client for agent '{agent_name}': {e}")
            raise

        return gemini_model

    def get_embedding_client(
        self,
        model_name: str,
        task_type: Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"],
    ):
        """
        Factory method to create a configured Dense Embedding client.

        Note:
            GoogleGenerativeAIEmbeddings class automatically checks for the existence of
            the GEMINI_API_KEY environment variable.
        """
        logger.info(f"Creating Embedding client for model: {model_name}")

        # Verifies the existence of the GEMINI API Key.
        raw_api_key = self._validate_api_key("GEMINI_API_KEY")

        try:
            embedding_client = GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type=task_type,
                api_key=SecretStr(raw_api_key),  # Casting required for type-checker.,
                output_dimensionality=768,  # Set for compatibility with Pinecone Vector DB
            )
            logger.info(f"Embedding client created for model '{model_name}' with task type '{task_type}'.")
            return embedding_client
        except Exception as e:
            logger.error(f"Error creating Embedding client: {e}")
            raise

    def get_pinecone_client(self) -> Pinecone:
        """Factory method to create a Pinecone client.

        Returns:
            Configured Pinecone (GRPC) client instance.
        """

        logger.info("Creating Pinecone client.")

        pinecone_api_key = self._validate_api_key("PINECONE_API_KEY")

        try:
            pc = Pinecone(api_key=pinecone_api_key)
            logger.info("Pinecone client created successfully.")
            return pc
        except Exception as e:
            logger.error(f"Error creating Pinecone client: {e}")
            raise

    def get_eden_ai_client(self, model_name: str = "openai/gpt-4o") -> ChatOpenAI:
        """
        Factory method to create an Eden AI client, using the
        ChatOpenAI proxy.

        The proxy was explicitly chosen because the native Eden AI
        integration (ChatEdenAI) is maintained by the LangChain community.
        Updates may lag behind, thus utilizing the ChatOpenAI proxy is
        a better choice, ensuring stability.

        This client can be used with any provider offered through EdenAI.

        Args:
            model_name (str):
                The name of the model to use.

                Format: provider/model-name

                Ex: "openai/gpt-4"
                    "anthropic/claude-3-5-sonnet-20241022"

                Available models can be found at this link:
                https://docs.edenai.co/v3/how-to/llm/chat-completions#available-models

        Returns:
            Configured ChatOpenAI client instance.
        """
        logger.info(f"Creating Eden AI client for model: {model_name}")

        eden_api_key = self._validate_api_key("EDEN_AI_API_KEY")

        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key=SecretStr(eden_api_key),
                base_url="https://api.edenai.run/v3/llm",
                streaming=True,
            )
            logger.info(f"Eden AI client created for model '{model_name}'.")
            return llm
        except Exception as e:
            logger.error(f"Error creating Eden AI client: {e}")
            raise
    
    def get_eden_ai_async_client(self) -> AsyncOpenAI:
        """
        Factory method to create an Eden AI async client.

        Returns:
            Configured AsyncOpenAI client instance.
        
        NOTE: This method is specifically designed to create an async client for Eden AI
        for use in streaming scenarios within the rag evaluation graph.
        The RAG evaluation graph is not compatible with the ChatOpenAI proxy for Eden AI
        thus necessitating the use of the AsyncOpenAI client directly configured for 
        Eden AI's API.

        NOTE: This client does not speicfy a model at the client level. 
        The model name is specified when passed to ragas.llm.llm_factory():
            
            llm_factory(model="gpt-4o",
                        provider = "openai",
                        client=async_client)
        """
        logger.info(f"Creating Eden AI async connection client.")

        eden_api_key = self._validate_api_key("EDEN_AI_API_KEY")

        try:
            async_client = AsyncOpenAI(
                api_key=eden_api_key,
                base_url="https://api.edenai.run/v3/llm",
            )
            logger.info(f"Eden AI async connection client created.")
            return async_client
        except Exception as e:
            logger.error(f"Error creating Eden AI async client: {e}")
            raise