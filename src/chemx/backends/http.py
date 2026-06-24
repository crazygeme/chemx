"""Shared HTTP utilities for model backends."""

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ModelError

logger = logging.getLogger(__name__)


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POST a JSON payload and return a JSON object.

    Request bodies and authorization headers are intentionally excluded from
    logs. Diagnostic records contain only endpoint, timeout, response status,
    and response size.
    """
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )

    try:
        logger.debug("model HTTP request started url=%s timeout=%s", url, timeout)
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            logger.debug(
                "model HTTP request completed url=%s status=%s bytes=%d",
                url,
                getattr(response, "status", "unknown"),
                len(response_body),
            )
            result = json.loads(response_body.decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        logger.error("model HTTP request failed url=%s status=%d", url, error.code)
        raise ModelError(
            f"Model request failed with HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        logger.error("model HTTP connection failed url=%s", url)
        raise ModelError(
            f"Could not connect to the model provider: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise ModelError("The model provider returned invalid JSON.") from error

    if not isinstance(result, dict):
        raise ModelError("The model provider returned an unexpected response.")
    return result
