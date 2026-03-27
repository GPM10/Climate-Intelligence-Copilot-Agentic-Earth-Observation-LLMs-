"""
Base Agent class for the Climate Intelligence Copilot.
All agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standard result format for all agents."""
    success: bool
    agent_name: str
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Climate Intelligence Copilot.
    
    Each agent inherits from this and implements:
    - validate_input(): Data validation
    - execute(): Core logic
    - format_output(): Result formatting
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{name}")
        
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        pass
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute the agent's core logic."""
        pass
    
    @abstractmethod
    def format_output(self, result: Any) -> AgentResult:
        """Format execution result into standard format."""
        pass
    
    def run(self, input_data: Any) -> AgentResult:
        """
        Main run method that orchestrates validate -> execute -> format.
        """
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    agent_name=self.name,
                    timestamp=datetime.now(),
                    data=None,
                    metadata={},
                    error="Input validation failed"
                )
            
            result = self.execute(input_data)
            return self.format_output(result)
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                agent_name=self.name,
                timestamp=datetime.now(),
                data=None,
                metadata={},
                error=str(e)
            )
