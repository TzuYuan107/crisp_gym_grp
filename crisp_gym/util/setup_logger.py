"""Setup logging for the project scripts."""

import atexit
import logging
import os
import queue
from logging.handlers import QueueListener

from rich.logging import RichHandler

# MODIFIED: was the bare relative path "crisp.log" (landed wherever the script happened to be
# launched from). Moved under outputs/logs/ to match this repo's existing gitignored outputs/
# convention (see .gitignore) -- same directory vlsa_lerobot_policy.py's DEBUG_LOG_DIR already
# uses for obs_log.txt, instead of a stray file at the repo root.
LOG_FILE_PATH = "outputs/logs/crisp.log"


def setup_logging(level=logging.INFO):  # noqa: ANN001
    """Recommended logging setup for the project."""
    console_formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(console_formatter)

    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]"
    )
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(file_formatter)

    log_queue = queue.Queue()
    # queue_handler = QueueHandler(log_queue)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    # FIXED: file_handler was built above but never actually attached to the logger, so
    # crisp.log stayed permanently empty (0 bytes) regardless of how much ran -- every log line
    # only ever reached the console. Attaching it here is what makes crisp.log real.
    logger.addHandler(file_handler)

    handlers = [
        console_handler,
        file_handler,
    ]

    listener = QueueListener(log_queue, *handlers)
    listener.start()

    atexit.register(listener.stop)
