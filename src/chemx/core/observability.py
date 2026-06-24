"""Application-wide logging configuration.

Library modules emit records through standard ``logging`` loggers but do not
configure handlers themselves. Frontends own presentation policy and call
``configure_logging`` once during startup.
"""

import logging
import sys

LOGGER_NAMESPACE = "chemx"


def configure_logging(verbosity: int = 0) -> int:
    """Configure stderr logging and return the selected logging level.

    ``verbosity`` follows the conventional command-line counting model:

    - ``0``: errors only;
    - ``1``: high-level lifecycle information;
    - ``2`` or greater: detailed diagnostic information.

    Repeated calls replace handlers installed on the application logger. This
    keeps tests deterministic and avoids duplicate records when a frontend is
    initialized more than once in one process.
    """
    level = logging.ERROR
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logger = logging.getLogger(LOGGER_NAMESPACE)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return level
