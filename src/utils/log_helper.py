"""Helper functions for logging"""

import logging
import os
import sys
from datetime import datetime

# Configure the root logger to log debug and higher level messages
logging.basicConfig(level=logging.DEBUG)

# Configure application's logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)

# Get the loggers for httpx and httpcore libraries
httpx_logger = logging.getLogger("httpx")
openai_logger = logging.getLogger("openai")
httpcore_logger = logging.getLogger("httpcore")
watchfiles_logger = logging.getLogger("watchfiles")
gtts_logger = logging.getLogger("gtts")
urllib_logger = logging.getLogger("urllib3")
sql_alchemy_logger = logging.getLogger("sqlalchemy")
py_dub_logger = logging.getLogger("pydub")
gcp_logger = logging.getLogger("google.cloud.storage._opentelemetry_tracing")


httpx_logger.setLevel(logging.WARN)
httpcore_logger.setLevel(logging.INFO)
watchfiles_logger.setLevel(logging.WARN)
openai_logger.setLevel(logging.INFO)
gtts_logger.setLevel(logging.INFO)
urllib_logger.setLevel(logging.INFO)
sql_alchemy_logger.setLevel(logging.WARN)
py_dub_logger.setLevel(logging.INFO)
gcp_logger.setLevel(logging.INFO)

# StreamHandler to output logs to stdout
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


def create_log_file(service_name):
    """Create log file"""

    log_dir = "logs"  # This folder will be in the path from where you run the code
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f'{service_name}_{datetime.now().strftime("%d%b%Y_%H%M%S")}.log'
    log_location = os.path.join(log_dir, log_file_name)
    print("log_location: ", log_location)

    # FileHandler to output logs to a file
    fh = logging.FileHandler(log_location, mode="a")
    fh.setFormatter(formatter)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(fh)
    logger.propagate = False  # Prevent duplication of log messages

    # Get uvicorn's logger and add the file handler to it
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.propagate = False  # Prevent duplication of log messages
    if not any(isinstance(h, logging.FileHandler) for h in uvicorn_logger.handlers):
        uvicorn_logger.addHandler(fh)

    # Add file handler to uvicorn's access logger as well
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    if not any(
        isinstance(h, logging.FileHandler) for h in uvicorn_access_logger.handlers
    ):
        uvicorn_access_logger.addHandler(fh)
    uvicorn_access_logger.setLevel(logging.DEBUG)  # Set the desired log level
    uvicorn_access_logger.propagate = False
