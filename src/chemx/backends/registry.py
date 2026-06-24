"""Registration and construction of model backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .base import ModelBackend

logger = logging.getLogger(__name__)

BackendFactory = Callable[..., ModelBackend]


@dataclass(frozen=True)
class BackendRegistration:
    """Immutable construction metadata for one provider implementation."""

    name: str
    default_model: str
    factory: BackendFactory


_BACKENDS: dict[str, BackendRegistration] = {}


def register_backend(
    name: str,
    *,
    default_model: str,
) -> Callable[[BackendFactory], BackendFactory]:
    """Register a backend factory under a normalized provider name.

    Registration occurs during package import. Duplicate names are rejected so
    provider selection remains deterministic.
    """
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Backend name cannot be empty.")

    def decorator(factory: BackendFactory) -> BackendFactory:
        if normalized_name in _BACKENDS:
            raise ValueError(f"Backend is already registered: {normalized_name}")
        _BACKENDS[normalized_name] = BackendRegistration(
            name=normalized_name,
            default_model=default_model,
            factory=factory,
        )
        logger.debug("backend registered provider=%s", normalized_name)
        return factory

    return decorator


def available_backends() -> tuple[str, ...]:
    """Return registered backend names in deterministic order."""
    return tuple(sorted(_BACKENDS))


def get_backend_registration(name: str) -> BackendRegistration:
    """Return registration metadata for a backend."""
    normalized_name = name.strip().lower()
    try:
        return _BACKENDS[normalized_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model provider: {name}") from error


def create_backend(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> ModelBackend:
    """Create a model backend using its registered factory."""
    registration = get_backend_registration(provider)
    logger.info(
        "creating model backend provider=%s model=%s",
        registration.name,
        model,
    )
    return registration.factory(
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
