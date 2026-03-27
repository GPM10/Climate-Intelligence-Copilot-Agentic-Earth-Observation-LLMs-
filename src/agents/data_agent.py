"""
Data Agent - Retrieves and processes climate datasets from multiple sources.
Handles emissions data, biodiversity indices, precipitation, temperature, etc.
"""

from typing import Any, Dict, Optional, List
import pandas as pd
import logging
from datetime import datetime

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """
    📊 Data Agent: Aggregates climate and environmental datasets.
    
    Features:
    - Retrieves emissions data (CO2, CH4, N2O)
    - Fetches biodiversity indicators
    - Collects climate observations (temperature, precipitation)
    - Caches data for efficiency
    - Handles multiple data sources
    """
    
    # Supported data sources
    DATA_SOURCES = {
        'emissions': ['EDGAR', 'CAMS'],
        'biodiversity': ['GBIF', 'BISE'],
        'climate': ['ECMWF', 'WorldClim', 'CHELSA'],
        'land_cover': ['ESA-CCI', 'Copernicus']
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("DataAgent", config)
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache = {}
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate data request parameters."""
        if not isinstance(input_data, dict):
            return False
        
        required_keys = ['data_type', 'region', 'temporal_range']
        return all(key in input_data for key in required_keys)
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute data retrieval and aggregation."""
        
        data_type = input_data['data_type']  # emissions, climate, etc.
        region = input_data['region']
        temporal_range = input_data['temporal_range']
        
        # Check cache
        cache_key = f"{data_type}_{region}_{temporal_range[0]}_{temporal_range[1]}"
        if self.cache_enabled and cache_key in self.cache:
            self.logger.info(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        # Retrieve data based on type
        if data_type == 'emissions':
            result = self._fetch_emissions_data(region, temporal_range)
        elif data_type == 'climate':
            result = self._fetch_climate_data(region, temporal_range)
        elif data_type == 'biodiversity':
            result = self._fetch_biodiversity_data(region, temporal_range)
        elif data_type == 'land_cover':
            result = self._fetch_land_cover_data(region, temporal_range)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Cache result
        if self.cache_enabled:
            self.cache[cache_key] = result
        
        return result
    
    def _fetch_emissions_data(self, region: str, temporal_range: List) -> Dict:
        """Fetch emissions data (CO2, CH4, N2O) for region."""
        # Placeholder - would integrate with EDGAR, CAMS APIs
        return {
            'data_type': 'emissions',
            'region': region,
            'temporal_range': temporal_range,
            'emissions': {
                'CO2': self._generate_mock_timeseries('CO2', temporal_range),
                'CH4': self._generate_mock_timeseries('CH4', temporal_range),
                'N2O': self._generate_mock_timeseries('N2O', temporal_range),
            },
            'units': 'Gg (Gigagrams)',
            'sources': ['EDGAR v6.0', 'CAMS Global Emission inventory']
        }
    
    def _fetch_climate_data(self, region: str, temporal_range: List) -> Dict:
        """Fetch climate observations (temperature, precipitation, humidity)."""
        return {
            'data_type': 'climate',
            'region': region,
            'temporal_range': temporal_range,
            'temperature': self._generate_mock_timeseries('Temperature', temporal_range),
            'precipitation': self._generate_mock_timeseries('Precipitation', temporal_range),
            'humidity': self._generate_mock_timeseries('Humidity', temporal_range),
            'wind_speed': self._generate_mock_timeseries('Wind Speed', temporal_range),
            'units': {'temperature': '°C', 'precipitation': 'mm', 'humidity': '%', 'wind_speed': 'm/s'},
            'sources': ['ECMWF ERA5', 'WorldClim v2.1']
        }
    
    def _fetch_biodiversity_data(self, region: str, temporal_range: List) -> Dict:
        """Fetch biodiversity indicators and species data."""
        return {
            'data_type': 'biodiversity',
            'region': region,
            'temporal_range': temporal_range,
            'biodiversity_index': 0.72,  # 0-1 scale
            'species_count': {
                'mammals': 145,
                'birds': 289,
                'amphibians': 67,
                'plants': 2341
            },
            'threatened_species': {
                'critically_endangered': 23,
                'endangered': 56,
                'vulnerable': 123
            },
            'sources': ['GBIF', 'IUCN Red List', 'BISE']
        }
    
    def _fetch_land_cover_data(self, region: str, temporal_range: List) -> Dict:
        """Fetch land cover classification data."""
        return {
            'data_type': 'land_cover',
            'region': region,
            'temporal_range': temporal_range,
            'land_cover_fractions': {
                'forest': 0.35,
                'grassland': 0.20,
                'cropland': 0.25,
                'urban': 0.12,
                'water': 0.05,
                'barren': 0.03
            },
            'sources': ['ESA CCI Land Cover', 'Copernicus Global Land Service']
        }
    
    def _generate_mock_timeseries(self, metric_name: str, temporal_range: List) -> Dict:
        """Generate mock time-series data for demonstration."""
        import numpy as np
        
        start_year, end_year = temporal_range
        years = list(range(start_year, end_year + 1))
        values = np.random.rand(len(years)).tolist() * 100  # Mock values
        trend = "increasing" if values[-1] > values[0] else "decreasing"
        
        return {
            'years': years,
            'values': [round(v, 2) for v in values],
            'trend': trend,
            'mean': round(np.mean(values), 2),
            'std_dev': round(np.std(values), 2)
        }
    
    def format_output(self, result: Dict[str, Any]) -> AgentResult:
        """Format execution result."""
        return AgentResult(
            success=True,
            agent_name=self.name,
            timestamp=datetime.now(),
            data=result,
            metadata={
                'module': 'DataAgent',
                'cache_enabled': self.cache_enabled,
                'cache_size': len(self.cache),
                'available_data_types': list(self.DATA_SOURCES.keys())
            }
        )
