import logging
import sys
from constants import LOG_FILE_PATH
from pathlib import Path


def setup_logging(log_level=logging.INFO,
                  log_file_path=LOG_FILE_PATH,
                  console_log_level=logging.INFO):
    """Sets up logging configuration for the application.

    Parameters:
        log_level:
            Logging level for the root logger and file handler
            (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        log_file_path:
            Path to the log file. Defaults to LOG_FILE_PATH from constants.py.
        console_log_level:
            Logging level for the console handler. Defaults to logging.INFO.
    """
    # Convert to Path object if a string is provided
    log_file_path_obj = Path(log_file_path)

    # Create logs directory if it doesn't exist
    log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Log message should use (Time - Logger Name - Log Level - Message) format
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path_obj)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(log_format)

    if not root_logger.hasHandlers():
        # Avoid adding multiple handlers if already set up
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    logging.info(f"Logging configuration set up successfully at log_file_path: {log_file_path}")
