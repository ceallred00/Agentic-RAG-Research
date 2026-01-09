import logging
import yaml
from pathlib import Path
from typing import Union, Dict, Any
from pydantic import ValidationError

# Import Pydantic models for validation
from configs.schemas.agent_schemas import AgentConfig
# from confid.schemas.architecture_schemas import ArchitectureConfig

from src.constants import CONFIGS_DIR

logger=logging.getLogger(__name__)

class ConfigLoader:
    """Helper to parse YAML file and validate them against Pydantic models.
    
    This class manages the retrieval and validation of configuration files from
    a specified directory structure."""

    def __init__(self, base_path: Union[str, Path] = CONFIGS_DIR, encoding: str = "utf-8"):
        """
        Parameters:
            base_path:
                Base directory where configuration subfolders are located.
                Defaults to CONFIGS_DIR, specified in the .env file.
            encoding:
                File encoding to use when reading YAML files. Defaults to 'utf-8'.
        """
        self.base_path = Path(base_path)
        self.encoding = encoding

    def load_agents(self) -> Dict[str, AgentConfig]:
        """
        Load YAMLs and validate against AgentConfig model.

        Scans the 'agents' subfolder, parses all YAML files found, 
        and validates them against the AgentConfig Pydantic model.

        Returns:
            A dictionary mapping agent names to their validated AgentConfig models.

        Note:
            Files that fail validation are logged and skipped. The process does 
            not halt on individual file errors.
        """
        logging.info("Loading agent configurations...")

        raw_configs = self._load_from_directory("agents")
        validated_configs = {}

        for agent_name, raw_config_data in raw_configs.items():
            try:
                validated_config = AgentConfig(**raw_config_data)
                validated_configs[agent_name] = validated_config
            except ValidationError as ve:
                logging.error(f"Validation error for agent '{agent_name}': {ve}. Check YAML file against AgentConfig schema.")
        return validated_configs

    # def load_architectures(self) -> Dict[str, Any]:
    #     """Public convenience method specifically for architectures."""
    #     logging.info("Loading architecture configurations...")

    #     raw_configs = self._load_from_directory("architectures")
    #     validated_configs = {}

    #     for arch_name, raw_config_data in raw_configs.items():
    #         try:
    #             validated_config = ArchitectureConfig(**raw_config_data)
    #             validated_configs[arch_name] = validated_config
    #         except ValidationError as ve:
    #             logging.error(f"Validation error for architecture '{arch_name}': {ve}")
    #     return validated_configs

    def _load_from_directory(self, subfolder: str) -> Dict[str, Any]:
        """
        Internal helper: Reads all YAML files in the specified subfolder.

        Returns:
            A dictionary mapping file names to their parsed content.
        
        Raises:
            FileNotFoundError:
                If the constructed directory path does not exist.
            ValueError:
                If no configuration files are found in the subfolder.
        """
        config_dict = {}
        target_dir = self.base_path / subfolder

        logging.info(f"Loading configurations from directory: {target_dir}")
        
        if not target_dir.exists():
            # Agent cannot load without configs
            raise FileNotFoundError(f"Configuration directory not found: {target_dir}")

        # Iterate over all YAML files in the target directory
        # .glob returns full path objects ending in .yaml
        for file_path in target_dir.glob("*.yaml"):
            try:
                # Read and parse each YAML file
                # read_text handles opening/closing automatically
                content = file_path.read_text(encoding=self.encoding)
                data = yaml.safe_load(content)

                # If the file is empty, yaml.safe_load returns None
                # Skips empty files
                if data is None:
                    logger.warning(f"Empty configuration file: {file_path.name}")
                    continue
                    
                # Use filename (stem) as the key 
                config_dict[file_path.stem] = data

            except Exception as e:
                logger.error(f"Error loading config file {file_path.name}: {e}")
        
        if not config_dict:
            # Agent cannot load without configs
            raise ValueError(f"No configuration files found in: {target_dir}")
                
        return config_dict