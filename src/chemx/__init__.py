"""chemx: a small, extensible AI agent."""

import logging

from .backends import Message, ModelBackend
from .core import Agent, ChemicalsAgent, CodingAgent

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Agent", "ChemicalsAgent", "CodingAgent", "Message", "ModelBackend"]
__version__ = "0.1.0"
