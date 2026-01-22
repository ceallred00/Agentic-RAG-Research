"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import patch
from src.schemas.agent_schemas import AgentConfig
from pydantic import SecretStr
from src.core.execution_service import ExecutionService


@pytest.fixture
def base_agent_config():
    """
    Fixture that returns a base agent configuration dictionary.

    Base configuration follows the schema defined in AgentConfig.
    Configuration is modifed in individual tests as needed.

    """
    return {
            "version": "1.0",
            "agent_metadata": {
                "name": "base_test_agent",
                "description": "A test agent configuration following valid schema."
            },
            "model": {
                "provider": "google",
                "name" : "gemini-3-pro-preview",
                "temperature": 0.5
            },
            "system_prompt": "You are a helpful assistant."
        }


@pytest.fixture
def valid_agents_dir(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    return agents_dir

@pytest.fixture
def sample_agent_config_dict():
    sample_agent_dict = {'good_test_agent_1': {
            "version": "1.0",
            "agent_metadata": {
                "name": "good_test_agent_1",
                "description": "A test agent configuration following valid schema."
            },
            "model": {
                "provider": "google",
                "name" : "gemini-3-pro-preview",
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
                "name" : "gemini-3-small",
                "temperature": 0.7
            },
            "system_prompt": "You are a creative assistant."
    }}

    return sample_agent_dict

@pytest.fixture
def valid_architectures_dir(tmp_path):
    architectures_dir = tmp_path / "architectures"
    architectures_dir.mkdir()

    return architectures_dir

#TODO: Add a fixture for architecture configs once those are defined.

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

@pytest.fixture
def sample_agent_configs_objects(sample_agent_config_dict):
    """ Converts sample agent config dicts to AgentConfig Pydantic models. """
    agent_configs = {name: AgentConfig(**config) for name, config in sample_agent_config_dict.items()}
    return agent_configs

@pytest.fixture
def gemini_api_key_env(monkeypatch):
    key_value = "TEST_GEMINI_API_KEY"
    monkeypatch.setenv("GEMINI_API_KEY", key_value)
    yield key_value

@pytest.fixture
def secret_gemini_api_key_env(gemini_api_key_env):
    yield SecretStr(gemini_api_key_env)

@pytest.fixture
def pinecone_api_key_env(monkeypatch):
    key_value = "TEST_PINECONE_API_KEY"
    monkeypatch.setenv("PINECONE_API_KEY", key_value)
    yield key_value

@pytest.fixture
def instance_agent_config_execution_service(sample_agent_configs_objects):
    """ 
    Creates an instance of ExecutionService with sample agent configs objects.
    """
    return ExecutionService(agent_configs=sample_agent_configs_objects)

@pytest.fixture
def instance_execution_service():
    """ 
    Creates an instance of ExecutionService with no agent configs.
    """
    return ExecutionService()