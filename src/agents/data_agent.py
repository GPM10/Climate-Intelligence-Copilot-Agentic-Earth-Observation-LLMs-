"""
Data Agent - Retrieves and processes climate datasets from multiple sources.
Handles emissions data, biodiversity indices, precipitation, temperature, etc.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime

try:
    import pycountry  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pycountry = None

from data.sources import (
    EDGARDataSource,
    CAMSDataSource,
    GBIFDataSource,
    resolve_bbox_from_region,
)
from .base import BaseAgent, AgentResult

class DataAgent(BaseAgent):
    """
    📊 Data Agent: Aggregates climate and environmental datasets.

    Features:
    - Retrieves emissions data (EDGAR CSV exports)
    - Fetches climate observations via CAMS / CDS API
    - Queries biodiversity indicators from GBIF
    - Caches data for efficiency
    """

    DATA_SOURCES = {
        'emissions': ['EDGAR'],
        'biodiversity': ['GBIF'],
        'climate': ['CAMS'],
        'land_cover': ['ESA-CCI', 'Copernicus']
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("DataAgent", config)
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache: Dict[str, Dict] = {}
        self.source_config = self.config.get('data_sources', {})

        self.edgar_source = self._init_edgar_source()
        self.cams_source = self._init_cams_source()
        self.gbif_source = self._init_gbif_source()

    def _init_edgar_source(self) -> Optional[EDGARDataSource]:
        cfg = self.source_config.get('edgar', {})
        if not cfg.get('enabled', True) or not cfg.get('csv_path'):
            self.logger.warning("EDGAR data source disabled or missing csv_path; emissions queries will fail.")
            return None
        try:
            return EDGARDataSource(**{k: v for k, v in cfg.items() if k != 'enabled'})
        except Exception as exc:
            self.logger.error("Failed to initialize EDGAR data source: %s", exc)
            return None

    def _init_cams_source(self) -> Optional[CAMSDataSource]:
        cfg = self.source_config.get('cams', {})
        if not cfg.get('enabled', False):
            self.logger.warning("CAMS data source disabled; climate queries will use placeholders.")
            return None
        required_keys = ('dataset', 'api_url', 'api_key')
        if not all(cfg.get(key) for key in required_keys):
            self.logger.warning("CAMS configuration missing dataset/api_url/api_key; climate queries will fail.")
            return None
        try:
            return CAMSDataSource(
                dataset=cfg['dataset'],
                api_url=cfg['api_url'],
                api_key=cfg['api_key'],
                cache_dir=cfg.get('cache_dir', './data/cams'),
                verify=cfg.get('verify', True),
            )
        except Exception as exc:
            self.logger.error("Failed to initialize CAMS data source: %s", exc)
            return None

    def _init_gbif_source(self) -> Optional[GBIFDataSource]:
        cfg = self.source_config.get('gbif', {})
        if not cfg.get('enabled', True):
            self.logger.warning("GBIF data source disabled; biodiversity queries will fail.")
            return None
        return GBIFDataSource(user_agent=cfg.get('user_agent', 'climate-intel-copilot'), timeout=cfg.get('timeout', 30))

    def validate_input(self, input_data: Any) -> bool:
        """Validate data request parameters."""
        if not isinstance(input_data, dict):
            return False

        required_keys = ['data_type', 'region', 'temporal_range']
        return all(key in input_data for key in required_keys)

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute data retrieval and aggregation."""

        data_type = input_data['data_type']
        region = input_data['region']
        temporal_range = input_data['temporal_range']

        cache_key = f"{data_type}_{region}_{temporal_range}_{hash(str(input_data.get('bbox')))}"
        if self.cache_enabled and cache_key in self.cache:
            self.logger.info("Cache hit for %s", cache_key)
            return self.cache[cache_key]

        if data_type == 'emissions':
            result = self._fetch_emissions_data(region, temporal_range, input_data)
        elif data_type == 'climate':
            result = self._fetch_climate_data(region, temporal_range, input_data)
        elif data_type == 'biodiversity':
            result = self._fetch_biodiversity_data(region, temporal_range, input_data)
        elif data_type == 'land_cover':
            result = self._fetch_land_cover_data(region, temporal_range)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if self.cache_enabled:
            self.cache[cache_key] = result

        return result

    def _fetch_emissions_data(self, region: str, temporal_range: List, params: Dict[str, Any]) -> Dict:
        if not self.edgar_source:
            raise RuntimeError("EDGAR data source is not configured. Update config.data_sources.edgar.")

        start_year, end_year = temporal_range
        sectors = params.get('sectors')
        timeseries = self.edgar_source.fetch_country_timeseries(region, start_year, end_year, sectors=sectors)

        return {
            'data_type': 'emissions',
            'region': region,
            'temporal_range': temporal_range,
            'emissions': {
                'total_ghg': timeseries
            },
            'units': 'MtCO2e',
            'sources': [timeseries.get('source')],
        }

    def _fetch_climate_data(self, region: str, temporal_range: List, params: Dict[str, Any]) -> Dict:
        if not self.cams_source:
            raise RuntimeError("CAMS data source not configured. See config.data_sources.cams.")

        bbox = params.get('bbox')
        if not bbox:
            bbox = resolve_bbox_from_region(region)
        bbox.setdefault('name', region)

        start_year, end_year = temporal_range
        variables = params.get('variables') or self.source_config.get('cams', {}).get('variables', ['co2_surface_flux'])

        metrics = {}
        for variable in variables:
            metrics[variable] = self.cams_source.fetch_mean_timeseries(variable, bbox, start_year, end_year)

        return {
            'data_type': 'climate',
            'region': region,
            'temporal_range': temporal_range,
            'climate_metrics': metrics,
            'sources': [f"CAMS {self.source_config.get('cams', {}).get('dataset')}"],
            'bbox': bbox,
        }

    def _fetch_biodiversity_data(self, region: str, temporal_range: List, params: Dict[str, Any]) -> Dict:
        if not self.gbif_source:
            raise RuntimeError("GBIF data source is not configured.")

        start_year, end_year = temporal_range
        country_code = self._resolve_country_code(region, params.get('country_code'))
        taxon_key = params.get('taxon_key') or self.source_config.get('gbif', {}).get('taxon_key')
        summary = self.gbif_source.fetch_species_summary(
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
            taxon_key=taxon_key,
            limit=params.get('limit', 300),
        )

        return {
            'data_type': 'biodiversity',
            'region': region,
            'temporal_range': temporal_range,
            'country_code': country_code,
            'biodiversity_index': summary.get('total_records', 0) / max(1, summary.get('total_records', 1)),
            'species_count': summary.get('species_count', {}),
            'threatened_species': summary.get('threatened_species', {}),
            'sources': summary.get('sources', []),
        }

    def _fetch_land_cover_data(self, region: str, temporal_range: List) -> Dict:
        """Placeholder for ESA-CCI / Copernicus integration."""
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
            'sources': ['ESA CCI Land Cover (configure data_sources.land_cover to override)']
        }

    def _resolve_country_code(self, region: str, explicit_code: Optional[str]) -> str:
        if explicit_code:
            return explicit_code.upper()

        if len(region) == 2:
            return region.upper()

        if pycountry:
            try:
                return pycountry.countries.lookup(region).alpha_2  # type: ignore[attr-defined]
            except LookupError:
                pass

        self.logger.warning("Could not map region '%s' to ISO code; defaulting to first two letters.", region)
        return region[:2].upper()

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
