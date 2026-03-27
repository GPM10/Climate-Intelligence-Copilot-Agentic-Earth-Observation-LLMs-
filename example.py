"""
Quick start example - Using the Climate Intelligence Copilot
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agents import ClimateCopilot
from utils import Config, Logger


def main():
    """Run example climate analysis."""
    
    # Setup logging
    Logger.setup()
    
    # Load configuration
    config = Config()
    
    # Create copilot instance
    copilot = ClimateCopilot(config.to_dict())
    
    print("=" * 80)
    print("🌍 Climate Intelligence Copilot - Quick Start Example")
    print("=" * 80)
    
    # Example 1: Simple question about deforestation
    print("\n📌 Example 1: Asking about land degradation in Ireland")
    print("-" * 80)
    
    question1 = "Which regions in Ireland have rising land degradation and why?"
    response1 = copilot.ask(
        question=question1,
        context={
            'region': 'Ireland',
            'temporal_range': [2020, 2024]
        }
    )
    
    print(f"\nQuestion: {question1}")
    print(f"✓ Satellite Agent: {response1.satellite_analysis.success if response1.satellite_analysis else 'Skipped'}")
    print(f"✓ Data Agent: {response1.climate_data.success if response1.climate_data else 'Skipped'}")
    print(f"✓ Reasoning Agent: {response1.reasoning.success if response1.reasoning else 'Failed'}")
    print(f"✓ Policy Agent: {response1.policy_recommendations.success if response1.policy_recommendations else 'Skipped'}")
    
    if response1.reasoning and response1.reasoning.success:
        print("\n🧠 Analysis:")
        data = response1.reasoning.data
        print(f"  Type: {data.get('question_type', 'general')}")
        print(f"  Explanation: {data.get('explanation', 'N/A')}")
    
    if response1.policy_recommendations and response1.policy_recommendations.success:
        print("\n🧾 Policy Recommendations (Top 3):")
        data = response1.policy_recommendations.data
        for rec in data.get('recommendations', [])[:3]:
            print(f"  • {rec.get('policy', 'N/A')}")
            print(f"    Priority: {rec.get('priority_rank', 'N/A')}, Impact: {rec.get('impact_score', 0):.2f}")
    
    # Example 2: Climate data query
    print("\n\n📌 Example 2: Analyzing emissions trends")
    print("-" * 80)
    
    question2 = "What are the emissions trends in Ireland? Show data and deforestation risk."
    response2 = copilot.ask(
        question=question2,
        context={'region': 'Ireland'}
    )
    
    print(f"\nQuestion: {question2}")
    
    if response2.climate_data and response2.climate_data.success:
        print("\n📊 Climate Data Retrieved:")
        data = response2.climate_data.data
        if 'emissions' in data:
            for gas, timeseries in data['emissions'].items():
                print(f"  {gas}: Mean={timeseries.get('mean', 'N/A')}, Trend={timeseries.get('trend', 'N/A')}")
    
    # Example 3: Regional analysis
    print("\n\n📌 Example 3: Comprehensive regional analysis")
    print("-" * 80)
    
    response3 = copilot.analyze_region('Ireland', [2020, 2024])
    
    print(f"\nAnalyzing: Ireland (2020-2024)")
    print("✓ Multi-agent analysis completed")
    print("  - Satellite imagery processing")
    print("  - Climate data aggregation")
    print("  - Policy recommendations")
    
    print("\n" + "=" * 80)
    print("✅ Example complete! Check individual agent outputs above.")
    print("=" * 80)
    
    # Tips
    print("\n💡 Tips:")
    print("  • Use 'python main.py ask --question \"your question\"' for CLI")
    print("  • Check config/settings.yaml for advanced configuration")
    print("  • Configure GEE credentials in .env for satellite data")
    print("  • See notebooks/analysis.ipynb for Jupyter examples")


if __name__ == '__main__':
    main()
