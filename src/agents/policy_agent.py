"""
Policy Agent - Suggests climate interventions and policy recommendations.
Uses rule-based logic and LLM guidance to recommend actions.
"""

from typing import Any, Dict, Optional, List
import logging
from datetime import datetime

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class PolicyAgent(BaseAgent):
    """
    🧾 Policy Agent: Generates climate policy recommendations.
    
    Features:
    - Suggests evidence-based interventions
    - Evaluates policy effectiveness
    - Prioritizes actions by impact
    - Integrates LLM reasoning with rules
    - Provides implementation guidance
    """
    
    # Policy knowledge base
    POLICY_RULES = {
        'deforestation': {
            'high': ['Protected area expansion', 'Zero-deforestation agreements', 'Reforestation'],
            'medium': ['Sustainable forestry', 'Agroforestry', 'Community programs'],
            'low': ['Awareness campaigns', 'Monitoring systems']
        },
        'emissions': {
            'high': ['Renewable energy transition', 'Industrial decarbonization'],
            'medium': ['Efficiency improvements', 'Circular economy', 'Forest protection'],
            'low': ['Public awareness', 'Carbon pricing']
        },
        'biodiversity': {
            'high': ['Protected area network', 'Habitat restoration', 'Wildlife corridors'],
            'medium': ['Sustainable agriculture', 'Pollution control', 'Invasive species management'],
            'low': ['Research', 'Education', 'Monitoring']
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("PolicyAgent", config)
        self.use_llm = self.config.get('use_llm', True)
        self.rule_based_fallback = self.config.get('rule_based_fallback', True)
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate policy request."""
        if not isinstance(input_data, dict):
            return False
        
        return 'issue_type' in input_data or 'region' in input_data
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Generate policy recommendations."""
        
        issue_type = input_data.get('issue_type')
        region = input_data.get('region')
        severity = input_data.get('severity', 'medium')
        context = input_data.get('context', {})
        
        # Get policy recommendations
        recommendations = self._get_recommendations(issue_type, severity, context)
        
        # Evaluate feasibility and impact
        evaluated_recs = self._evaluate_recommendations(recommendations, region, context)
        
        # Prioritize actions
        prioritized = self._prioritize_actions(evaluated_recs)
        
        return {
            'issue_type': issue_type,
            'region': region,
            'severity': severity,
            'recommendations': prioritized,
            'implementation_timeline': self._create_timeline(prioritized),
            'resource_requirements': self._estimate_resources(prioritized),
            'expected_impact': self._estimate_impact(prioritized)
        }
    
    def _get_recommendations(self, issue_type: str, severity: str, context: Dict) -> List[Dict]:
        """Get policy recommendations based on issue type and severity."""
        
        if not issue_type or issue_type not in self.POLICY_RULES:
            return []
        
        # Get severity-based recommendations
        severity_map = {'low': 'low', 'medium': 'medium', 'high': 'high'}
        sev_key = severity_map.get(severity.lower(), 'medium')
        
        policies = self.POLICY_RULES[issue_type].get(sev_key, [])
        
        recommendations = []
        for policy in policies:
            recommendations.append({
                'policy': policy,
                'issue_address': issue_type,
                'estimated_effectiveness': 0.65 + (len(policies) - policies.index(policy)) * 0.1,
                'co_benefits': self._get_cobenefits(policy, issue_type)
            })
        
        return recommendations
    
    def _get_cobenefits(self, policy: str, issue_type: str) -> List[str]:
        """Get co-benefits of a policy."""
        
        cobenefits_map = {
            'Protected area expansion': ['Biodiversity conservation', 'Ecosystem services', 'Tourism revenue'],
            'Renewable energy transition': ['Air quality improvement', 'Health benefits', 'Job creation'],
            'Sustainable agriculture': ['Soil health', 'Water quality', 'Farmer income'],
            'Habitat restoration': ['Carbon sequestration', 'Water regulation', 'Species recovery'],
            'Zero-deforestation agreements': ['Carbon storage', 'Indigenous rights', 'Economic stability']
        }
        
        return cobenefits_map.get(policy, [])
    
    def _evaluate_recommendations(self, recommendations: List[Dict], region: str, context: Dict) -> List[Dict]:
        """Evaluate feasibility and effectiveness of recommendations."""
        
        evaluated = []
        for rec in recommendations:
            feasibility = self._assess_feasibility(rec['policy'], region)
            impact = self._assess_impact(rec['policy'])
            cost = self._estimate_cost(rec['policy'])
            
            evaluated.append({
                **rec,
                'feasibility_score': feasibility,  # 0-1
                'impact_score': impact,             # 0-1
                'cost_category': cost,              # low, medium, high
                'roi_ratio': impact / max(0.1, {'low': 0.5, 'medium': 1.0, 'high': 2.0}.get(cost, 1.0))
            })
        
        return evaluated
    
    def _prioritize_actions(self, evaluated_recs: List[Dict]) -> List[Dict]:
        """Prioritize actions by ROI and feasibility."""
        
        # Calculate priority score: (impact * feasibility) / cost_factor
        for rec in evaluated_recs:
            rec['priority_score'] = (rec['impact_score'] * rec['feasibility_score']) / \
                                    max(0.1, {'low': 0.5, 'medium': 1.0, 'high': 2.0}.get(rec['cost_category'], 1.0))
        
        # Sort by priority
        sorted_recs = sorted(evaluated_recs, key=lambda x: x['priority_score'], reverse=True)
        
        # Add priority ranking
        for i, rec in enumerate(sorted_recs, 1):
            rec['priority_rank'] = i
        
        return sorted_recs
    
    def _assess_feasibility(self, policy: str, region: str) -> float:
        """Assess feasibility of implementing policy in region."""
        # Mock assessment - would be data-driven in production
        base_feasibility = 0.6
        region_factors = {'Ireland': 0.95, 'EU': 0.90, 'Africa': 0.65, 'Asia': 0.70}
        return min(1.0, base_feasibility * region_factors.get(region, 0.75))
    
    def _assess_impact(self, policy: str) -> float:
        """Assess potential impact of policy."""
        impact_map = {
            'Protected area expansion': 0.8,
            'Zero-deforestation agreements': 0.75,
            'Renewable energy transition': 0.85,
            'Habitat restoration': 0.7
        }
        return impact_map.get(policy, 0.6)
    
    def _estimate_cost(self, policy: str) -> str:
        """Estimate cost category of policy."""
        cost_map = {
            'Protected area expansion': 'high',
            'Renewable energy transition': 'high',
            'Sustainable agriculture': 'medium',
            'Awareness campaigns': 'low'
        }
        return cost_map.get(policy, 'medium')
    
    def _create_timeline(self, recommendations: List[Dict]) -> Dict:
        """Create implementation timeline."""
        
        return {
            'immediate_0_6_months': [
                r['policy'] for r in recommendations[:2] if r.get('priority_rank', 0) <= 2
            ],
            'short_term_6_24_months': [
                r['policy'] for r in recommendations[1:4]
            ],
            'medium_term_2_5_years': [
                r['policy'] for r in recommendations[3:]
            ]
        }
    
    def _estimate_resources(self, recommendations: List[Dict]) -> Dict:
        """Estimate resource requirements."""
        
        total_cost = sum(
            {'low': 1, 'medium': 5, 'high': 15}.get(r['cost_category'], 0)
            for r in recommendations
        )
        
        return {
            'total_estimated_cost_millions': total_cost,
            'funding_sources': ['Green Climate Fund', 'National Budget', 'Private Investment', 'Donor Support'],
            'human_resources_required': 'High',
            'capacity_building_needed': True
        }
    
    def _estimate_impact(self, recommendations: List[Dict]) -> Dict:
        """Estimate expected impact of recommendations."""
        
        avg_impact = sum(r['impact_score'] for r in recommendations) / max(1, len(recommendations))
        
        return {
            'average_effectiveness': round(avg_impact, 2),
            'expected_greenhouse_gas_reduction': f"{int(avg_impact * 100 * 50)} MtCO2 equivalent",
            'biodiversity_benefit': 'Significant' if avg_impact > 0.7 else 'Moderate',
            'timeline_to_measurable_results': '18-24 months',
            'long_term_sustainability': 'High' if avg_impact > 0.7 else 'Medium'
        }
    
    def format_output(self, result: Dict[str, Any]) -> AgentResult:
        """Format execution result."""
        return AgentResult(
            success=True,
            agent_name=self.name,
            timestamp=datetime.now(),
            data=result,
            metadata={
                'module': 'PolicyAgent',
                'use_llm': self.use_llm,
                'rule_based_fallback': self.rule_based_fallback,
                'capabilities': ['policy_recommendation', 'impact_assessment', 'prioritization']
            }
        )
