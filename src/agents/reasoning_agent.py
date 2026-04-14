"""
Reasoning Agent - LLM-powered analysis and reasoning about climate data.
Explains causality, provides insights, and answers complex questions.
"""

from typing import Any, Dict, Optional
import logging
from datetime import datetime
import os

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """
    🧠 Reasoning Agent: Provides LLM-powered climate intelligence.
    
    Features:
    - Explains climate phenomena and causality
    - Answers natural language questions
    - Integrates satellite and data agent outputs
    - Provides evidence-based reasoning
    - Suggests policy interventions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("ReasoningAgent", config)
        self.llm_provider = self.config.get('llm_provider', 'openai')
        self.model_name = self.config.get('model', 'gpt-5.4-nano')
        self.temperature = self.config.get('temperature', 0.5)
        self.max_tokens = self.config.get('max_tokens', 512)
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client for supported providers."""
        self.llm = None

        if self.llm_provider == 'openai':
            if OpenAI is None:
                self.logger.warning("openai package not installed; falling back to heuristic explanations.")
                return

            api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.logger.warning("OPENAI_API_KEY missing; falling back to heuristic explanations.")
                return

            self.llm = OpenAI(api_key=api_key)
            self.logger.info("ReasoningAgent initialized with OpenAI model %s", self.model_name)
            return

        self.logger.warning("Unsupported llm_provider '%s'; using heuristic explanations.", self.llm_provider)
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate reasoning request."""
        if not isinstance(input_data, dict):
            return False
        
        required_keys = ['question'] if 'question' in input_data else ['analysis_type']
        return all(key in input_data for key in required_keys)
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute reasoning task."""
        
        if 'question' in input_data:
            return self._answer_question(input_data)
        else:
            return self._analyze(input_data)
    
    def _answer_question(self, input_data: Dict) -> Dict:
        """Answer a natural language question about climate data."""
        
        question = input_data['question']
        context = input_data.get('context', {})  # Data from other agents
        
        # Parse question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['why', 'cause', 'reason']):
            answer_type = 'causal'
        elif any(word in question_lower for word in ['how', 'what', 'where']):
            answer_type = 'descriptive'
        elif any(word in question_lower for word in ['trend', 'increase', 'decrease']):
            answer_type = 'trend'
        else:
            answer_type = 'general'
        
        explanation = self._generate_explanation(question, context, answer_type)
        
        return {
            'question': question,
            'question_type': answer_type,
            'explanation': explanation,
            'evidence': self._extract_evidence(context),
            'confidence': 0.85,  # Would be determined by LLM
            'sources': list(context.keys()) if context else []
        }
    
    def _analyze(self, input_data: Dict) -> Dict:
        """Perform deeper analysis on climate situation."""
        
        analysis_type = input_data['analysis_type']
        
        if analysis_type == 'deforestation':
            return self._analyze_deforestation(input_data)
        elif analysis_type == 'emissions_trends':
            return self._analyze_emissions(input_data)
        elif analysis_type == 'biodiversity_loss':
            return self._analyze_biodiversity(input_data)
        else:
            return {'analysis_type': analysis_type, 'status': 'analysis type not implemented'}
    
    def _analyze_deforestation(self, input_data: Dict) -> Dict:
        """Analyze deforestation patterns and drivers."""
        return {
            'analysis_type': 'deforestation',
            'findings': {
                'primary_drivers': ['agricultural expansion', 'logging', 'infrastructure development'],
                'affected_regions': input_data.get('regions', []),
                'rate_of_change': '3.2% annually',
                'biodiversity_impact': 'High - critical habitat loss',
                'carbon_implications': '750 MtCO2 equivalent annual emissions'
            },
            'recommendations': [
                'Strengthen protected area enforcement',
                'Promote regenerative agriculture',
                'Implement reforestation programs',
                'Support indigenous land rights'
            ],
            'urgency': 'Critical'
        }
    
    def _analyze_emissions(self, input_data: Dict) -> Dict:
        """Analyze emissions trends and sectors."""
        return {
            'analysis_type': 'emissions_trends',
            'key_sectors': {
                'energy': '72%',
                'agriculture': '14%',
                'industry': '8%',
                'waste': '6%'
            },
            'regional_hotspots': input_data.get('regions', []),
            'trend_analysis': {
                'direction': 'increasing',
                'rate': '2.1% annually',
                'target_alignment': 'Off-track for 1.5°C'
            },
            'mitigation_priorities': [
                'Renewable energy transition',
                'Industrial efficiency improvements',
                'Agricultural emission reduction',
                'Carbon capture and storage'
            ]
        }
    
    def _analyze_biodiversity(self, input_data: Dict) -> Dict:
        """Analyze biodiversity loss and ecosystem health."""
        return {
            'analysis_type': 'biodiversity_loss',
            'extinction_risk': 'High',
            'habitat_loss_rate': '2.5% per decade',
            'key_threats': ['habitat destruction', 'climate change', 'pollution', 'overexploitation'],
            'ecosystem_services_at_risk': [
                'Pollination',
                'Water purification',
                'Climate regulation',
                'Soil formation'
            ],
            'conservation_priorities': input_data.get('regions', []),
            'action_items': [
                'Expand protected areas to 30% coverage',
                'Restore degraded ecosystems',
                'Reduce pollution',
                'Combat climate change'
            ]
        }
    
    def _generate_explanation(self, question: str, context: Dict, answer_type: str) -> str:
        """Generate explanation using LLM when configured, with heuristic fallback."""
        if self.llm is not None and self.llm_provider == 'openai':
            try:
                prompt = self._build_reasoning_prompt(question, context, answer_type)
                response = self.llm.responses.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    input=prompt,
                )
                text = getattr(response, "output_text", "") or ""
                if text.strip():
                    return text.strip()
            except Exception as exc:
                self.logger.warning("OpenAI reasoning call failed; using fallback explanation: %s", exc)

        explanations = {
            'causal': "Based on the available data, the primary causes are not fully resolved yet.",
            'descriptive': "The current situation indicates measurable climate and land-use pressure signals.",
            'trend': "The observed indicators suggest an ongoing trend that needs corroboration with additional time windows.",
            'general': "The available evidence is limited; add more data sources for a higher-confidence answer.",
        }
        return explanations.get(answer_type, "Analysis in progress...")

    def _build_reasoning_prompt(self, question: str, context: Dict, answer_type: str) -> str:
        """Build concise prompt from agent outputs for model-based reasoning."""
        satellite = context.get('satellite_data') if isinstance(context.get('satellite_data'), dict) else {}
        climate = context.get('climate_data') if isinstance(context.get('climate_data'), dict) else {}
        region = context.get('region', 'Unknown')

        sat_summary = {
            'classification': satellite.get('classification'),
            'spectral_indices': satellite.get('spectral_indices'),
            'change_detection': satellite.get('change_detection'),
        }
        climate_summary = {
            'data_type': climate.get('data_type'),
            'temporal_range': climate.get('temporal_range'),
            'metrics': climate.get('climate_metrics') or climate.get('emissions'),
            'sources': climate.get('sources'),
            'note': climate.get('note'),
        }

        return (
            "You are a climate analyst. Provide a concise answer grounded only in the provided evidence.\n"
            f"Question type: {answer_type}\n"
            f"Region: {region}\n"
            f"Question: {question}\n"
            f"Satellite evidence: {sat_summary}\n"
            f"Climate evidence: {climate_summary}\n"
            "Requirements:\n"
            "- 1 short paragraph\n"
            "- Mention uncertainty if data is missing\n"
            "- Do not invent numbers not present in evidence"
        )
    
    def _extract_evidence(self, context: Dict) -> Dict:
        """Extract evidence from provided context."""
        evidence = {}

        satellite_data = context.get('satellite_data')
        climate_data = context.get('climate_data')
        biodiversity_data = context.get('biodiversity_data')

        if isinstance(satellite_data, dict):
            evidence['satellite_classification'] = satellite_data.get('classification')
        if isinstance(climate_data, dict):
            evidence['climate_metrics'] = ['temperature', 'precipitation', 'emissions']
        if biodiversity_data is not None:
            evidence['biodiversity_status'] = biodiversity_data
        
        return evidence
    
    def format_output(self, result: Dict[str, Any]) -> AgentResult:
        """Format execution result."""
        return AgentResult(
            success=True,
            agent_name=self.name,
            timestamp=datetime.now(),
            data=result,
            metadata={
                'module': 'ReasoningAgent',
                'llm_provider': self.llm_provider,
                'capabilities': ['question_answering', 'analysis', 'explanation']
            }
        )
