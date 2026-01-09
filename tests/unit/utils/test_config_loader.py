"""
Unit tests for the ConfigLoader utility class.

Scope:

"""
import yaml
import copy
import pytest
from src.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """ Test suite for ConfigLoader class. """
    class TestLoadAgents:
        """ Test cases for load_agents method. """

        def test_happy_path(self, base_agent_config, valid_agents_dir):
            """ Test loading agent configurations from valid YAML file.

            Configurations are valid YAML syntactically and semantically according to Pydantic model. 
            
            Happy path test. Test should succeed with valid YAML files."""
            loader = ConfigLoader(base_path=valid_agents_dir.parent)
            agent_dir = valid_agents_dir

            # Create a valid YAML file
            valid_yaml_path = agent_dir / "good_test_agent.yaml"

            # Create good agent
            # Deepcopy to avoid mutating the fixture data
            agent_config = copy.deepcopy(base_agent_config)
            agent_config["agent_metadata"]["name"] = "good_test_agent"

            # Convert dict to YAML string
            valid_yaml_content = yaml.dump(agent_config)
            
            # Write the YAML content to the file
            valid_yaml_path.write_text(valid_yaml_content)
            # Execute the method
            loaded_agents = loader.load_agents()

            #Assertions
            assert loaded_agents is not None
            assert loaded_agents["good_test_agent"] is not None
            assert loaded_agents["good_test_agent"].agent_metadata.name == "good_test_agent"
            assert loaded_agents["good_test_agent"].model.name == "gemini-3-pro-preview"
        
        def test_invalid_agent_config(self, base_agent_config, valid_agents_dir, caplog):
            """ Test behavior when an agent YAML file has invalid configuration according to Pydantic model.
            
            The invalid agent configuration should be skipped. An error should be logged.
            """
            loader = ConfigLoader(base_path=valid_agents_dir.parent)
            agent_dir = valid_agents_dir

            # Set up logging capture
            caplog.set_level("ERROR")

            invalid_yaml_path = agent_dir / "bad_test_agent.yaml"

            # Create bad agent with missing required fields

            bad_agent_config = copy.deepcopy(base_agent_config)
            del bad_agent_config["model"]["name"]  # Remove required field

            invalid_yaml_content = yaml.dump(bad_agent_config)
            invalid_yaml_path.write_text(invalid_yaml_content)

            # Execute the method
            loaded_agents = loader.load_agents()

            # Assertions
            # The invalid agent should not be loaded. Only loading valid agents.
            assert len(loaded_agents) == 0
            assert "Validation error for agent 'bad_test_agent'" in caplog.text
        
        def test_partial_validity_agent_config(self, base_agent_config, valid_agents_dir, caplog):
            """ Test behavior when directory contains both valid and invalid agent YAML files. 
            
            The good_agent_config should be loaded successfully. 
            The bad_agent_config should be skipped due to validation errors with the Pydantic model.
            The error should be logged.

            """
            loader = ConfigLoader(base_path=valid_agents_dir.parent)
            agent_dir = valid_agents_dir

            # Set up logging capture
            caplog.set_level("ERROR")

            # Create a valid YAML file
            valid_yaml_path = agent_dir / "good_test_agent.yaml"
            good_agent_config = copy.deepcopy(base_agent_config)
            good_agent_config["agent_metadata"]["name"] = "good_test_agent"
            valid_yaml_content = yaml.dump(good_agent_config)
            valid_yaml_path.write_text(valid_yaml_content)

            # Create an invalid YAML file
            invalid_yaml_path = agent_dir / "bad_test_agent.yaml"
            bad_agent_config = copy.deepcopy(base_agent_config)
            del bad_agent_config["model"]["name"]  # Remove required field
            invalid_yaml_content = yaml.dump(bad_agent_config)
            invalid_yaml_path.write_text(invalid_yaml_content)

            # Execute the method
            loaded_agents = loader.load_agents()

            # Assertions
            assert len(loaded_agents) == 1
            assert "good_test_agent" in loaded_agents
            assert loaded_agents["good_test_agent"].agent_metadata.name == "good_test_agent"
            assert "Validation error for agent 'bad_test_agent'" in caplog.text
        
        def test_empty_agent_directory(self, valid_agents_dir):
            """ Test behavior when the agents directory is empty. 
            
            Expecting a ValueError indicating no configuration files found.
            """
            loader = ConfigLoader(base_path=valid_agents_dir.parent)

            # Execute the method and expect ValueError
            with pytest.raises(ValueError) as excinfo:
                loader.load_agents()
            
            assert "No configuration files found" in str(excinfo.value)
        
        def test_nonexistent_agent_directory(self, tmp_path):
            """ Test behavior when the agents directory does not exist.
             
            Expecting a FileNotFoundError indicating the directory is missing."""
            loader = ConfigLoader(base_path=tmp_path)

            # Execute the method and expect FileNotFoundError
            with pytest.raises(FileNotFoundError) as excinfo:
                loader.load_agents()
            
            assert "Configuration directory not found" in str(excinfo.value)
        
        def test_mixed_file_type_in_agent_directory(self, base_agent_config, valid_agents_dir, caplog):
            """ Test behavior when the agents directory contains non-YAML files.
            
            The non-YAML file should be skipped, adding only valid agent configurations.

            """
            loader = ConfigLoader(base_path=valid_agents_dir.parent)
            agent_dir = valid_agents_dir

            # Create a valid YAML file
            valid_yaml_path = agent_dir / "good_test_agent.yaml"
            good_agent_config = copy.deepcopy(base_agent_config)
            good_agent_config["agent_metadata"]["name"] = "good_test_agent"
            valid_yaml_content = yaml.dump(good_agent_config)
            valid_yaml_path.write_text(valid_yaml_content)

            # Create a non-YAML file
            non_yaml_path = agent_dir / "not_a_yaml.txt"
            non_yaml_path.write_text("This is not a YAML file.")

            # Execute the method
            loaded_agents = loader.load_agents()

            # Assertions
            assert len(loaded_agents) == 1
            assert "good_test_agent" in loaded_agents
            assert loaded_agents["good_test_agent"].agent_metadata.name == "good_test_agent"
        
        def test_malformed_yaml_in_agent_directory(self, base_agent_config, valid_agents_dir, caplog):
            """ Test behavior when an agent YAML file is malformed. 
            
            The malformed YAML file should be skipped. An error should be logged.
            The valid YAML file should be loaded successfully.
            """
            loader = ConfigLoader(base_path=valid_agents_dir.parent)
            agent_dir = valid_agents_dir

            # Set up logging capture
            caplog.set_level("ERROR")

            malformed_yaml_path = agent_dir / "malformed_agent.yaml"
            
            # Write malformed YAML content
            malformed_yaml_content = """agent_metadata: \n\tname: bad_indent"""
            malformed_yaml_path.write_text(malformed_yaml_content)

            valid_yaml_path = agent_dir / "base_test_agent.yaml"
            base_agent_config = copy.deepcopy(base_agent_config)
            valid_yaml_content = yaml.dump(base_agent_config)
            valid_yaml_path.write_text(valid_yaml_content)

            loaded_agents = loader.load_agents()

            # Assertions
            # The malformed YAML should be logged and skipped. No agents loaded.
            assert len(loaded_agents) == 1
            assert "Error loading config file" in caplog.text

        
    class TestLoadArchitectures:
        """ Test cases for load_architectures method. """
        def test_load_architectures(self, tmp_path):
            """ Test loading architecture configurations from YAML files. """
            pass

