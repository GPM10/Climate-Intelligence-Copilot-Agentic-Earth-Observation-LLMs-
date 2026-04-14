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
from training import train_satellite_model
from training.satellite import TrainConfig
from training.data_prep import (
    EuroSATPrepConfig,
    EmitPseudoLabelConfig,
    prepare_eurosat,
    prepare_emit_pseudolabels,
)
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
@click.option('--hyperspectral-cube-path', default=None,
              help='Path to a hyperspectral cube (.npy/.npz/.csv/.txt) used for satellite analysis')
@click.option('--config', default=None,
              help='Path to config file')
def analyze_region(
    region: str,
    start_year: int,
    end_year: int,
    hyperspectral_cube_path: Optional[str],
    config: Optional[str]
):
    """
    Perform comprehensive climate analysis of a region.
    
    Example:
        python main.py analyze-region --region Ireland --start-year 2020 --end-year 2024
    """
    
    Logger.setup()
    cfg = Config(config)
    
    copilot = ClimateCopilot(cfg.to_dict())
    
    click.echo(f"\n📊 Analyzing {region} ({start_year}-{end_year})...\n")
    
    if hyperspectral_cube_path:
        cube_path = Path(hyperspectral_cube_path)
        if not cube_path.exists():
            click.echo(f"Error: Hyperspectral cube file not found: {hyperspectral_cube_path}")
            sys.exit(1)

        context = {
            'region': region,
            'temporal_range': [start_year, end_year],
            'analysis_type': 'comprehensive',
            'sensor': 'hyperspectral',
            'hyperspectral_cube_path': str(cube_path),
        }
        question = (
            f"Provide comprehensive climate analysis for {region} from {start_year} to {end_year} "
            "using hyperspectral satellite data."
        )
        response = copilot.ask(question, context)
    else:
        response = copilot.analyze_region(region, [start_year, end_year])
    _print_response_pretty(response)


@cli.command()
@click.option('--data-dir', required=True,
              help='Training data root. Either <root>/<class>/* or <root>/train/<class>/* and <root>/val/<class>/*')
@click.option('--output-checkpoint', default='data/models/satellite_resnet50.pt',
              help='Path to save final checkpoint')
@click.option('--epochs', type=int, default=5,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=4,
              help='Training batch size')
@click.option('--learning-rate', type=float, default=1e-4,
              help='Optimizer learning rate')
@click.option('--val-ratio', type=float, default=0.2,
              help='Validation split ratio when no train/val folders are provided')
@click.option('--freeze-backbone', is_flag=True,
              help='Freeze ResNet backbone and train only final classifier head')
@click.option('--device', default='cpu',
              help='Training device: cpu or cuda')
@click.option('--rgb-mode', type=click.Choice(['band_selection', 'pca']), default='band_selection',
              help='How to convert hyperspectral cubes into RGB inputs')
@click.option('--hyperspectral-bands', default='10,30,50',
              help='Comma-separated band indexes for band_selection mode')
def train_satellite(
    data_dir: str,
    output_checkpoint: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    freeze_backbone: bool,
    device: str,
    rgb_mode: str,
    hyperspectral_bands: str,
):
    """Fine-tune the satellite ResNet classifier on labeled imagery/hyperspectral cubes."""
    Logger.setup()
    band_indexes = tuple(int(token.strip()) for token in hyperspectral_bands.split(",") if token.strip())
    if rgb_mode == 'band_selection' and len(band_indexes) != 3:
        click.echo("Error: --hyperspectral-bands must contain exactly 3 indexes for band_selection mode.")
        sys.exit(1)

    cfg = TrainConfig(
        data_dir=data_dir,
        output_path=output_checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        freeze_backbone=freeze_backbone,
        device=device,
        rgb_mode=rgb_mode,
        hyperspectral_bands=band_indexes,
    )
    artifacts = train_satellite_model(cfg)
    click.echo("Training complete.")
    click.echo(f"Checkpoint: {artifacts['checkpoint']}")
    click.echo(f"Best checkpoint: {artifacts['best_checkpoint']}")
    click.echo(f"Classes: {artifacts['num_classes']}")


@cli.command()
@click.option('--source-dir', required=True,
              help='Extracted EuroSAT dataset root containing class folders (AnnualCrop, Forest, etc.)')
@click.option('--output-dir', default='data/train/satellite',
              help='Output directory for train/val class folders')
@click.option('--val-ratio', type=float, default=0.2,
              help='Validation split ratio')
@click.option('--seed', type=int, default=42,
              help='Random seed')
def prepare_eurosat_data(source_dir: str, output_dir: str, val_ratio: float, seed: int):
    """Prepare EuroSAT into this repo's training folder structure."""
    cfg = EuroSATPrepConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
    )
    counts = prepare_eurosat(cfg)
    click.echo("EuroSAT preparation complete.")
    for cls, count in sorted(counts.items()):
        click.echo(f"  {cls}: {count}")


@cli.command()
@click.option('--emit-cube-path', required=True,
              help='Path to EMIT hyperspectral cube (.nc/.npy/.npz/.txt/.csv)')
@click.option('--label-raster-path', required=True,
              help='Path to aligned label raster (.tif) from WorldCover or Dynamic World')
@click.option('--output-dir', default='data/train/satellite',
              help='Output directory for generated train/val chips')
@click.option('--label-source', type=click.Choice(['worldcover', 'dynamicworld']), default='worldcover',
              help='Pseudo-label source coding scheme')
@click.option('--chip-size', type=int, default=64,
              help='Square chip size in pixels')
@click.option('--stride', type=int, default=64,
              help='Sliding window stride in pixels')
@click.option('--val-ratio', type=float, default=0.2,
              help='Validation split ratio')
@click.option('--min-valid-fraction', type=float, default=0.8,
              help='Minimum non-zero label pixel fraction required in each chip')
@click.option('--min-majority-fraction', type=float, default=0.7,
              help='Minimum majority-class fraction required in each chip')
@click.option('--max-chips-per-class', type=int, default=2000,
              help='Cap generated chips per class')
@click.option('--seed', type=int, default=42,
              help='Random seed')
def prepare_emit_pseudolabel_data(
    emit_cube_path: str,
    label_raster_path: str,
    output_dir: str,
    label_source: str,
    chip_size: int,
    stride: int,
    val_ratio: float,
    min_valid_fraction: float,
    min_majority_fraction: float,
    max_chips_per_class: int,
    seed: int,
):
    """Build labeled EMIT training chips using WorldCover/Dynamic World raster pseudo-labels."""
    cfg = EmitPseudoLabelConfig(
        emit_cube_path=emit_cube_path,
        label_raster_path=label_raster_path,
        output_dir=output_dir,
        label_source=label_source,
        chip_size=chip_size,
        stride=stride,
        val_ratio=val_ratio,
        min_valid_fraction=min_valid_fraction,
        min_majority_fraction=min_majority_fraction,
        max_chips_per_class=max_chips_per_class,
        seed=seed,
    )
    counts = prepare_emit_pseudolabels(cfg)
    click.echo("EMIT pseudo-label chip generation complete.")
    for cls, count in sorted(counts.items()):
        click.echo(f"  {cls}: {count}")


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
