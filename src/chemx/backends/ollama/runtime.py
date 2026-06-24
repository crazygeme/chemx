"""Local Ollama service discovery and startup."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..base import ModelError

logger = logging.getLogger(__name__)

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def ensure_ollama_running(base_url: str, *, startup_timeout: float = 15.0) -> None:
    """Ensure an Ollama API is available, starting a local service if needed."""
    logger.info("checking Ollama service url=%s", base_url)
    if is_ollama_running(base_url):
        logger.info("Ollama service already available")
        return

    hostname = urlparse(base_url).hostname
    if hostname not in LOCAL_HOSTS:
        raise ModelError(f"Ollama is not reachable at {base_url}.")

    executable = shutil.which("ollama")
    if executable is None:
        raise ModelError(
            "Ollama is not installed. Install it from https://ollama.com/download "
            "and run this command again."
        )

    process = subprocess.Popen(
        [executable, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info("started local Ollama service executable=%s", executable)
    deadline = time.monotonic() + startup_timeout

    while time.monotonic() < deadline:
        if is_ollama_running(base_url):
            logger.info("Ollama service became ready")
            return
        if process.poll() is not None:
            raise ModelError(
                f"Ollama exited before its API became ready at {base_url}."
            )
        time.sleep(0.2)

    raise ModelError(
        f"Ollama did not become ready at {base_url} within "
        f"{startup_timeout:g} seconds."
    )


def is_ollama_running(base_url: str, *, timeout: float = 1.0) -> bool:
    """Return whether the Ollama tags endpoint accepts requests."""
    request = Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout):
            return True
    except (HTTPError, URLError, OSError):
        return False
