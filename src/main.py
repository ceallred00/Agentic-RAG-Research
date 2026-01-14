import logging
from utils.config_loader import ConfigLoader
from core.logging_setup import setup_logging
from core.execution_service import ExecutionService

logger = logging.getLogger()

def run():
    # Set up logging at the start of the application
    setup_logging()

    try:
        config_loader = ConfigLoader()
        
        agent_configs = config_loader.load_agents()
        
        if not agent_configs:
            logger.critical("No agent configurations loaded. Exiting application.")
            return
        
        # architecture_configs = config_loader.load_architectures()
        # if not architecture_configs:
        #     logger.critical("No architecture configurations loaded. Exiting application.")
        #     return
        
        # Create the ExecutionService with loaded configurations
        # Should have access to all available agents (assuming accurate configurations)
        execution_service = ExecutionService(agent_configs=agent_configs)
        
        # Example usage: create a Gemini client for a specific agent
        try:
            gemini_client = execution_service.get_gemini_client(agent_name="base_agent")
            logger.info("Gemini client created successfully.")
        
        except ValueError as ve:
            logger.error(f"Error creating Gemini client: {ve}")
    
    except Exception as e:
        logger.critical(f"Application crashed: {e}")


if __name__ == "__main__":
    run()