"""
Climate Intelligence Copilot - Agent Orchestrator
Coordinates all agents (Satellite, Data, Reasoning, Policy) to answer climate questions.
"""

from typing import Dict, Any, List
import logging
from datetime import datetime
from dataclasses import dataclass

from .satellite_agent import SatelliteAgent
from .data_agent import DataAgent
from .reasoning_agent import ReasoningAgent
from .policy_agent import PolicyAgent
from .base import AgentResult

logger = logging.getLogger(__name__)


@dataclass
class CopilotResponse:
    """Unified response from the Climate Intelligence Copilot."""
    question: str
    satellite_analysis: AgentResult
    climate_data: AgentResult
    reasoning: AgentResult
    policy_recommendations: AgentResult
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'question': self.question,
            'satellite_analysis': self.satellite_analysis.data if self.satellite_analysis else None,
            'climate_data': self.climate_data.data if self.climate_data else None,
            'reasoning': self.reasoning.data if self.reasoning else None,
            'policy_recommendations': self.policy_recommendations.data if self.policy_recommendations else None,
            'timestamp': str(self.timestamp)
        }


class ClimateCopilot:
    """
    🌍 Climate Intelligence Copilot - Agentic AI System
    
    Orchestrates multiple specialized agents to provide comprehensive climate analysis:
    - Satellite Agent: Land-use and change detection
    - Data Agent: Climate datasets aggregation
    - Reasoning Agent: LLM-powered explanation
    - Policy Agent: Intervention recommendations
    
    Example:
        copilot = ClimateCopilot(config)
        response = copilot.ask(
            "Which regions in Ireland have rising land degradation and why?"
        )
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ClimateCopilot")
        
        # Initialize all agents
        self.satellite_agent = SatelliteAgent(config.get('agents', {}).get('satellite', {}))
        self.data_agent = DataAgent(config.get('agents', {}).get('data', {}))
        self.reasoning_agent = ReasoningAgent(config.get('agents', {}).get('reasoning', {}))
        self.policy_agent = PolicyAgent(config.get('agents', {}).get('policy', {}))
        
        self.logger.info("Climate Intelligence Copilot initialized successfully")
    
    def ask(self, question: str, context: Dict[str, Any] | None = None) -> CopilotResponse:
        """
        Ask the copilot a climate-related question.
        
        Args:
            question: Natural language question about climate/environment
            context: Optional context data (region, time period, etc.)
        
        Returns:
            CopilotResponse with analyzed data and recommendations
        """
        
        context = context or {}
        self.logger.info(f"Processing question: {question}")
        
        # Parse question to determine which agents to activate
        agents_to_activate = self._determine_agents(question)
        
        # Run agents in parallel (in production would use async)
        results = {}
        
        if 'satellite' in agents_to_activate:
            satellite_input = self._prepare_satellite_input(question, context)
            results['satellite'] = self.satellite_agent.run(satellite_input)
        
        if 'data' in agents_to_activate:
            data_input = self._prepare_data_input(question, context)
            results['data'] = self.data_agent.run(data_input)
        
        # Reasoning agent always runs
        reasoning_input = self._prepare_reasoning_input(question, results, context)
        results['reasoning'] = self.reasoning_agent.run(reasoning_input)
        
        # Policy agent runs if recommendation is needed
        if 'policy' in agents_to_activate:
            policy_input = self._prepare_policy_input(question, results, context)
            results['policy'] = self.policy_agent.run(policy_input)
        
        return CopilotResponse(
            question=question,
            satellite_analysis=results.get('satellite'),
            climate_data=results.get('data'),
            reasoning=results['reasoning'],
            policy_recommendations=results.get('policy'),
            timestamp=datetime.now()
        )
    
    def _determine_agents(self, question: str) -> List[str]:
        """Determine which agents should be activated for the question."""
        
        question_lower = question.lower()
        agents = ['reasoning']  # Always include reasoning
        
        # Check for satellite-related keywords
        satellite_keywords = ['satellite', 'image', 'land use', 'forest', 'deforestation', 'change', 'visual']
        if any(kw in question_lower for kw in satellite_keywords):
            agents.append('satellite')
        
        # Check for data-related keywords
        data_keywords = ['data', 'emissions', 'climate', 'temperature', 'precipitation', 'biodiversity', 'trend']
        if any(kw in question_lower for kw in data_keywords):
            agents.append('data')
        
        # Check for policy-related keywords
        policy_keywords = ['solution', 'policy', 'action', 'recommend', 'intervention', 'what should', 'how to']
        if any(kw in question_lower for kw in policy_keywords):
            agents.append('policy')
        
        return list(set(agents))  # Remove duplicates
    
    def _prepare_satellite_input(self, question: str, context: Dict) -> Dict[str, Any]:
        """Prepare input for satellite agent."""
        
        return {
            'image_path': context.get('image_path'),
            'location': context.get('region', 'Ireland'),
            'timestamp': context.get('timestamp'),
            'reference_image_path': context.get('reference_image_path')  # For change detection
        }
    
    def _prepare_data_input(self, question: str, context: Dict) -> Dict[str, Any]:
        """Prepare input for data agent."""
        
        # Infer data type from question
        data_type = context.get('data_type', 'climate')
        
        if 'emissions' in question.lower():
            data_type = 'emissions'
        elif 'biodiversity' in question.lower():
            data_type = 'biodiversity'
        elif 'land' in question.lower():
            data_type = 'land_cover'
        
        return {
            'data_type': data_type,
            'region': context.get('region', 'Ireland'),
            'temporal_range': context.get('temporal_range', [2020, 2024])
        }
    
    def _prepare_reasoning_input(self, question: str, results: Dict, context: Dict) -> Dict[str, Any]:
        """Prepare input for reasoning agent."""
        
        return {
            'question': question,
            'context': {
                'satellite_data': results.get('satellite').data if results.get('satellite') else {},
                'climate_data': results.get('data').data if results.get('data') else {},
                'region': context.get('region', 'Ireland')
            }
        }
    
    def _prepare_policy_input(self, question: str, results: Dict, context: Dict) -> Dict[str, Any]:
        """Prepare input for policy agent."""
        
        # Infer issue type from reasoning output
        issue_type = 'climate_change'  # Default
        
        if results.get('reasoning') and results['reasoning'].data:
            data = results['reasoning'].data
            if 'deforestation' in str(data).lower():
                issue_type = 'deforestation'
            elif 'emissions' in str(data).lower():
                issue_type = 'emissions'
            elif 'biodiversity' in str(data).lower():
                issue_type = 'biodiversity'
        
        return {
            'issue_type': issue_type,
            'region': context.get('region', 'Ireland'),
            'severity': context.get('severity', 'medium'),
            'context': results  # Pass all previous agent results
        }
    
    def analyze_region(self, region: str, temporal_range: List[int]) -> CopilotResponse:
        """
        Comprehensive analysis of a region over time period.
        
        Args:
            region: Region name (e.g., "Ireland")
            temporal_range: [start_year, end_year]
        
        Returns:
            Full climate intelligence analysis
        """
        
        question = f"Provide comprehensive climate analysis for {region} from {temporal_range[0]} to {temporal_range[1]}"
        context = {
            'region': region,
            'temporal_range': temporal_range,
            'analysis_type': 'comprehensive'
        }
        
        return self.ask(question, context)


__all__ = ['ClimateCopilot', 'CopilotResponse']
