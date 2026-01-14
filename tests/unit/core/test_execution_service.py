
import pytest
from src.core.execution_service import ExecutionService
from src.schemas.agent_schemas import AgentConfig
from unittest.mock import patch

@pytest.fixture
def mock_gemini_client():
    with patch("src.core.execution_service.ChatGoogleGenerativeAI") as MockClient:
        yield MockClient

@pytest.fixture
def sample_agent_configs_objects(sample_agent_config_dict):
    """ Converts sample agent config dicts to AgentConfig Pydantic models. """
    agent_configs = {name: AgentConfig(**config) for name, config in sample_agent_config_dict.items()}
    return agent_configs

@pytest.fixture
def gemini_api_key_env(monkeypatch):
    key_value = "GEMINI_API_KEY"
    monkeypatch.setenv("GEMINI_API_KEY", key_value)
    yield key_value

@pytest.fixture
def instance_execution_service(sample_agent_configs_objects):
    """ 
    Creates an instance of ExecutionService with sample agent configs objects.
    """
    return ExecutionService(agent_configs=sample_agent_configs_objects)

class TestExecutionService:
    """ Test suite for ExecutionService class. """
    class TestGetGeminiClient:
        """ Test cases for get_gemini_client method. """

        def test_happy_path(self, gemini_api_key_env, instance_execution_service, mock_gemini_client):
            """ Test creating Gemini client with valid agent configuration.

            Happy path test. Test should succeed with valid agent configuration.
            """

            # Attempt to create Gemini client for a valid agent
            gemini_client = instance_execution_service.get_gemini_client(agent_name="good_test_agent_2")

            mock_gemini_client.assert_called_once_with(
                model="gemini-3-small",
                temperature=0.7,
                api_key=gemini_api_key_env
            )
            assert gemini_client == mock_gemini_client.return_value
        
        def test_invalid_agent_name(self, gemini_api_key_env, instance_execution_service):
            """ Test behavior when an invalid agent name is provided.
            
            The method should raise a ValueError indicating the agent is not found.
            """
            # Attempt to create Gemini client for an invalid agent
            with pytest.raises(ValueError) as ve:
                instance_execution_service.get_gemini_client(agent_name="non_existent_agent")

            assert "Agent 'non_existent_agent' not found in loaded configurations" in str(ve.value)
        
        def test_missing_gemini_api_key(self, instance_execution_service, monkeypatch):
            """ Test behavior when GEMINI_API_KEY environment variable is missing.
            
            The method should raise a ValueError indicating the missing API key.
            """

            # Remove GEMINI_API_KEY if it exists
            monkeypatch.delenv("GEMINI_API_KEY", raising=False)

            # Attempt to create Gemini client without GEMINI_API_KEY
            with pytest.raises(ValueError) as ve:
                instance_execution_service.get_gemini_client(agent_name="good_test_agent_1")

            assert "GEMINI_API_KEY not found in environment variables." in str(ve.value)

        def test_gemini_client_creation_failure(self, gemini_api_key_env, instance_execution_service, mock_gemini_client):
            """ Test behavior when Gemini client creation fails.
            
            The method should log an error and raise the exception.
            """

            # Configure the mock to raise an exception when instantiated
            mock_gemini_client.side_effect = Exception("Client creation failed")

            # Attempt to create Gemini client, expecting an exception
            with pytest.raises(Exception) as excinfo:
                instance_execution_service.get_gemini_client(agent_name="good_test_agent_1")

            assert "Client creation failed" in str(excinfo.value)
        
        def test_missing_config_file(self, instance_execution_service):
            #TODO: Write missing_config_file test
            pass
    class TestGetEmbeddingClient:
        """ Test cases for get_embedding_client method. """
        #TODO: Write test cases for the embedding client.

        def test_happy_path(self, instance_execution_service):
            pass



            