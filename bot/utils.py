import logging
import sys
import os

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Sets up and returns a logger with a custom format and stream handler.

    Args:
        name: The name for the logger, typically __name__ of the calling module.
        level: The default logging level string (e.g., "INFO", "DEBUG").
               Can be overridden by the LOG_LEVEL environment variable.

    Returns:
        A configured logging.Logger instance.
    """
    log_level_env = os.getenv("LOG_LEVEL", level).upper()
    numeric_level = getattr(logging, log_level_env, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Prevent logs from propagating to the root logger if it has handlers,
    # to avoid duplicate messages if other parts of the application configure the root logger.
    logger.propagate = False 

    # Add handler only if the logger doesn't already have one.
    # This prevents adding multiple handlers if setup_logger is called multiple times
    # for the same logger name (though typically it should be called once per module).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger 