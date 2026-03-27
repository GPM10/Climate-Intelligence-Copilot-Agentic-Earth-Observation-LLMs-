#!/usr/bin/env python
"""
Climate Intelligence Copilot - Main Entry Point & CLI
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agents import ClimateCopilot
from utils import Config, Logger
import logging


@click.group()
def cli():
    """🌍 Climate Intelligence Copilot - AI-powered climate decision support."""
    pass


@cli.command()
@click.option('--question', prompt='Ask about climate/environment', 
              help='Your climate question')
@click.option('--region', default='Ireland', 
              help='Geographic region')
@click.option('--config', default=None,
              help='Path to config file')
@click.option('--json-output', is_flag=True,
              help='Output as JSON')
def ask(question: str, region: str, config: Optional[str], json_output: bool):
    """
    Ask the Climate Intelligence Copilot a question.
    
    Example:
        python main.py ask --question "Which regions in Ireland have rising land degradation?"
    """
    
    # Initialize
    Logger.setup()
    cfg = Config(config)
    
    # Create copilot
    copilot = ClimateCopilot(cfg.to_dict())
    
    # Ask question
    click.echo(f"\n🤔 Processing: {question}\n")
    
    response = copilot.ask(
        question=question,
        context={'region': region}
    )
    
    # Format output
    if json_output:
        click.echo(json.dumps(response.to_dict(), indent=2))
    else:
        _print_response_pretty(response)


@cli.command()
@click.option('--region', prompt='Region to analyze', default='Ireland',
              help='Geographic region')
@click.option('--start-year', type=int, default=2020,
              help='Start year for analysis')
@click.option('--end-year', type=int, default=2024,
              help='End year for analysis')
@click.option('--config', default=None,
              help='Path to config file')
def analyze_region(region: str, start_year: int, end_year: int, config: Optional[str]):
    """
    Perform comprehensive climate analysis of a region.
    
    Example:
        python main.py analyze-region --region Ireland --start-year 2020 --end-year 2024
    """
    
    Logger.setup()
    cfg = Config(config)
    
    copilot = ClimateCopilot(cfg.to_dict())
    
    click.echo(f"\n📊 Analyzing {region} ({start_year}-{end_year})...\n")
    
    response = copilot.analyze_region(region, [start_year, end_year])
    _print_response_pretty(response)


@cli.command()
@click.argument('image_path')
@click.option('--region', default='Ireland',
              help='Geographic region')
@click.option('--config', default=None,
              help='Path to config file')
def analyze_satellite(image_path: str, region: str, config: Optional[str]):
    """
    Analyze satellite imagery.
    
    Example:
        python main.py analyze-satellite /path/to/image.tif --region Ireland
    """
    
    if not Path(image_path).exists():
        click.echo(f"❌ Image file not found: {image_path}")
        sys.exit(1)
    
    Logger.setup()
    cfg = Config(config)
    
    from agents.satellite_agent import SatelliteAgent
    
    agent = SatelliteAgent(cfg.to_dict().get('agents', {}).get('satellite', {}))
    
    click.echo(f"\n🛰️ Analyzing satellite image: {image_path}\n")
    
    result = agent.run({
        'image_path': image_path,
        'location': region,
        'timestamp': None
    })
    
    if result.success:
        click.echo("✅ Classification successful:\n")
        click.echo(json.dumps(result.data, indent=2))
    else:
        click.echo(f"❌ Error: {result.error}")


@cli.command()
@click.option('--config', default=None,
              help='Path to config file')
def status(config: Optional[str]):
    """Check system status and configuration."""
    
    Logger.setup()
    cfg = Config(config)
    
    click.echo("\n🔧 Climate Intelligence Copilot Status\n")
    click.echo("Configuration:")
    
    # Show key config items
    clickecho_item("LLM Provider", cfg.get('llm.provider', 'Not configured'))
    clickecho_item("Google Earth Engine", cfg.get('gee.project_id', 'Not configured'))
    clickecho_item("Database", cfg.get('database.type', 'Not configured'))
    
    click.echo("\nAgents:")
    click.echo("  ✓ Satellite Agent (Land-use classification)")
    click.echo("  ✓ Data Agent (Climate data aggregation)")
    click.echo("  ✓ Reasoning Agent (LLM analysis)")
    click.echo("  ✓ Policy Agent (Recommendations)")
    
    click.echo("\n✅ System ready!")


def clickecho_item(label: str, value: str):
    """Helper to echo config item."""
    click.echo(f"  {label}: {value}")


def _print_response_pretty(response):
    """Pretty print copilot response."""
    
    click.secho(f"Question: {response.question}\n", fg='cyan', bold=True)
    
    if response.satellite_analysis and response.satellite_analysis.success:
        click.secho("🛰️ Satellite Analysis:", fg='blue', bold=True)
        data = response.satellite_analysis.data
        if 'classification' in data:
            click.echo(f"  Land Use: {data['classification']['land_use_class']}")
            click.echo(f"  Confidence: {data['classification']['confidence']:.2%}")
    
    if response.climate_data and response.climate_data.success:
        click.secho("\n📊 Climate Data:", fg='blue', bold=True)
        data = response.climate_data.data
        click.echo(f"  Data Type: {data.get('data_type', 'N/A')}")
        click.echo(f"  Region: {data.get('region', 'N/A')}")
    
    if response.reasoning and response.reasoning.success:
        click.secho("\n🧠 Analysis & Reasoning:", fg='blue', bold=True)
        data = response.reasoning.data
        if 'explanation' in data:
            click.echo(f"  {data['explanation']}")
        if 'findings' in data:
            for key, value in data['findings'].items():
                click.echo(f"  {key}: {value}")
    
    if response.policy_recommendations and response.policy_recommendations.success:
        click.secho("\n🧾 Policy Recommendations:", fg='blue', bold=True)
        data = response.policy_recommendations.data
        if 'recommendations' in data:
            for i, rec in enumerate(data['recommendations'][:3], 1):
                click.echo(f"  {i}. {rec.get('policy', 'N/A')} (Priority: {rec.get('priority_rank', 'N/A')})")
    
    click.echo()


if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        sys.exit(1)
