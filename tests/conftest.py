"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import patch
from pydantic import SecretStr
from src.schemas.agent_schemas import AgentConfig
from src.core.execution_service import ExecutionService

# ==============================================================================
# 1. CONSTANTS & DATA FIXTURES
#    - Basic dictionaries and configuration data used across tests.
# ==============================================================================

@pytest.fixture
def base_agent_config():
    """
    Returns a base agent configuration dictionary.
    Configuration is modified in individual tests as needed.
    """
    return {
        "version": "1.0",
        "agent_metadata": {
            "name": "base_test_agent",
            "description": "A test agent configuration following valid schema."
        },
        "model": {
            "provider": "google",
            "name": "gemini-3-pro-preview",
            "temperature": 0.5
        },
        "system_prompt": "You are a helpful assistant."
    }

@pytest.fixture
def sample_agent_config_dict():
    """Returns a dictionary containing multiple valid agent configurations."""
    return {
        'good_test_agent_1': {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_1",
                "description": "A test agent configuration following valid schema."
            },
            "model": {
                "provider": "google",
                "name": "gemini-3-pro-preview",
                "temperature": 0.5
            },
            "system_prompt": "You are a helpful assistant."
        },
        'good_test_agent_2': {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_2",
                "description": "Another test agent configuration following valid schema."
            },
            "model": {
                "provider": "google",
                "name": "gemini-3-small",
                "temperature": 0.7
            },
            "system_prompt": "You are a creative assistant."
        }
    }

@pytest.fixture
def sample_agent_configs_objects(sample_agent_config_dict):
    """Converts sample agent config dicts to AgentConfig Pydantic models."""
    return {
        name: AgentConfig(**config) 
        for name, config in sample_agent_config_dict.items()
    }


# ==============================================================================
# 2. ENVIRONMENT & FILESYSTEM
#    - Environment variables (API keys) and temporary directories.
# ==============================================================================

@pytest.fixture
def valid_agents_dir(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    return agents_dir

@pytest.fixture
def valid_architectures_dir(tmp_path):
    architectures_dir = tmp_path / "architectures"
    architectures_dir.mkdir()
    return architectures_dir

@pytest.fixture
def gemini_api_key_env(monkeypatch):
    """Sets the Gemini API key in os.environ and returns the raw string."""
    key_value = "TEST_GEMINI_API_KEY"
    monkeypatch.setenv("GEMINI_API_KEY", key_value)
    yield key_value

@pytest.fixture
def secret_gemini_api_key_env(gemini_api_key_env):
    """Returns the Gemini API key as a SecretStr object."""
    yield SecretStr(gemini_api_key_env)

@pytest.fixture
def pinecone_api_key_env(monkeypatch):
    """Sets the Pinecone API key in os.environ and returns the raw string."""
    key_value = "TEST_PINECONE_API_KEY"
    monkeypatch.setenv("PINECONE_API_KEY", key_value)
    yield key_value


# ==============================================================================
# 3. MOCKS
#    - Mock objects for external dependencies (LLMs, Databases).
# ==============================================================================

@pytest.fixture
def mock_gemini_client():
    with patch("src.core.execution_service.ChatGoogleGenerativeAI") as MockClient:
        yield MockClient

@pytest.fixture
def mock_gemini_dense_embedding_client():
    with patch("src.core.execution_service.GoogleGenerativeAIEmbeddings") as MockEmbeddingClient:
        yield MockEmbeddingClient

@pytest.fixture
def mock_pinecone_client():
    with patch("src.core.execution_service.Pinecone") as MockPineconeClient:
        yield MockPineconeClient


# ==============================================================================
# 4. SERVICE INSTANCES
#    - Initialized instances of your application's core services.
# ==============================================================================

@pytest.fixture
def instance_execution_service():
    """Creates an instance of ExecutionService with NO agent configs."""
    return ExecutionService()

@pytest.fixture
def instance_agent_config_execution_service(sample_agent_configs_objects):
    """Creates an instance of ExecutionService PRE-LOADED with sample agent configs."""
    return ExecutionService(agent_configs=sample_agent_configs_objects)