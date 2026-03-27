"""
Climate Intelligence Copilot Package
"""

__version__ = "0.1.0"
__author__ = "Climate Intelligence Team"

from .src.agents import ClimateCopilot, CopilotResponse
from .src.utils import Config, Logger, VectorStoreManager

__all__ = [
    'ClimateCopilot',
    'CopilotResponse',
    'Config',
    'Logger',
    'VectorStoreManager'
]
