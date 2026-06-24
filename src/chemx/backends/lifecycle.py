"""Optional backend startup and readiness lifecycle."""

import logging
from typing import Protocol

from .base import ModelBackend

logger = logging.getLogger(__name__)


class PreparedBackend(Protocol):
    """Backend that requires setup before its first request."""

    def prepare(self) -> None:
        """Prepare the backend for model requests."""


def prepare_backend(backend: ModelBackend) -> None:
    """Run provider-specific setup when the backend supports it."""
    prepare = getattr(backend, "prepare", None)
    if callable(prepare):
        logger.info("preparing model backend type=%s", type(backend).__name__)
        prepare()
        logger.info("model backend ready type=%s", type(backend).__name__)
    else:
        logger.debug(
            "model backend requires no preparation type=%s",
            type(backend).__name__,
        )
