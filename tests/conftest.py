"""Pytest configuration and shared fixtures."""

import pytest


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

