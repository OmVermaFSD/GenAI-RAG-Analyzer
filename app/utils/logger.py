import logging
import sys
import time
from contextlib import contextmanager

# Configure logging format and handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("DocuMind-AI")

@contextmanager
def log_execution_time(operation_name: str):
    """Context manager to measure and log execution duration of operations (e.g. LLM responses)."""
    start_time = time.time()
    logger.info(f"Starting operation: '{operation_name}'")
    try:
        yield
    finally:
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Completed operation: '{operation_name}' in {elapsed:.2f}ms")
