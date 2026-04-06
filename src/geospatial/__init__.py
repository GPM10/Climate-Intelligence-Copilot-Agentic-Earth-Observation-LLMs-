"""
Geospatial utilities for satellite data processing and Earth Engine integration.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
from zipfile import ZipFile
import logging
import numpy as np

from .hyperspectral import HyperspectralProcessor

try:
    from sentinelsat import SentinelAPI, geojson_to_wkt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentinelAPI = None
    geojson_to_wkt = None

try:
    import rasterio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None

try:
    import ee  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ee = None


logger = logging.getLogger(__name__)


class GeospatialProcessor:
    """Handle geospatial operations: coordinate transformation, bounding boxes, etc."""

    @staticmethod
    def create_bbox(center_lat: float, center_lon: float, side_length_km: float) -> Dict[str, float]:
        """
        Create bounding box around center point.

        Args:
            center_lat: Latitude of center
            center_lon: Longitude of center
            side_length_km: Side length in kilometers

        Returns:
            Dict with min/max lat/lon
        """

        # 1 degree ~111 km
        delta = (side_length_km / 2) / 111.0

        return {
            'min_lat': center_lat - delta,
            'max_lat': center_lat + delta,
            'min_lon': center_lon - delta,
            'max_lon': center_lon + delta
        }

    @staticmethod
    def bbox_to_geojson(bbox: Dict[str, float]) -> Dict:
        """Convert bounding box dict to GeoJSON polygon."""
        return {
            'type': 'Polygon',
            'coordinates': [[
                [bbox['min_lon'], bbox['min_lat']],
                [bbox['max_lon'], bbox['min_lat']],
                [bbox['max_lon'], bbox['max_lat']],
                [bbox['min_lon'], bbox['max_lat']],
                [bbox['min_lon'], bbox['min_lat']]
            ]]
        }

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """Validate latitude/longitude coordinates."""
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def get_geometry_bounds(geometry: Dict) -> Optional[Tuple[float, float, float, float]]:
        """Extract bounds from GeoJSON geometry."""
        if geometry['type'] == 'Point':
            coords = geometry['coordinates']
            return (coords[1], coords[0], coords[1], coords[0])
        return None


class SentinelDataHandler:
    """Handle Sentinel-2 satellite data."""

    BANDS = {
        'B1': {'name': 'Coastal aerosol', 'resolution': 60},
        'B2': {'name': 'Blue', 'resolution': 10},
        'B3': {'name': 'Green', 'resolution': 10},
        'B4': {'name': 'Red', 'resolution': 10},
        'B5': {'name': 'Vegetation Red Edge', 'resolution': 20},
        'B6': {'name': 'Vegetation Red Edge', 'resolution': 20},
        'B7': {'name': 'Vegetation Red Edge', 'resolution': 20},
        'B8': {'name': 'NIR', 'resolution': 10},
        'B8A': {'name': 'Narrow NIR', 'resolution': 20},
        'B11': {'name': 'SWIR', 'resolution': 20},
        'B12': {'name': 'SWIR', 'resolution': 20},
    }

    INDICES = {
        'NDVI': {
            'formula': '(B8 - B4) / (B8 + B4)',
            'name': 'Normalized Difference Vegetation Index',
            'range': (-1, 1),
            'interpretation': 'Vegetation health and density'
        },
        'NDBI': {
            'formula': '(B11 - B8) / (B11 + B8)',
            'name': 'Normalized Difference Built-up Index',
            'range': (-1, 1),
            'interpretation': 'Urban/built-up area detection'
        },
        'NDMI': {
            'formula': '(B8 - B11) / (B8 + B11)',
            'name': 'Normalized Difference Moisture Index',
            'range': (-1, 1),
            'interpretation': 'Moisture content'
        }
    }

    @staticmethod
    def calculate_index(band_dict: Dict[str, np.ndarray], index_name: str) -> np.ndarray:
        if index_name not in SentinelDataHandler.INDICES:
            raise ValueError(f"Unknown index: {index_name}")

        if index_name == 'NDVI':
            return (band_dict['B8'] - band_dict['B4']) / (band_dict['B8'] + band_dict['B4'] + 1e-8)
        if index_name == 'NDBI':
            return (band_dict['B11'] - band_dict['B8']) / (band_dict['B11'] + band_dict['B8'] + 1e-8)
        if index_name == 'NDMI':
            return (band_dict['B8'] - band_dict['B11']) / (band_dict['B8'] + band_dict['B11'] + 1e-8)

        return np.array([])

    @staticmethod
    def rgb_composite(band_dict: Dict[str, np.ndarray], r_band: str = 'B4', g_band: str = 'B3', b_band: str = 'B2') -> np.ndarray:
        r = SentinelDataHandler._normalize_band(band_dict[r_band])
        g = SentinelDataHandler._normalize_band(band_dict[g_band])
        b = SentinelDataHandler._normalize_band(band_dict[b_band])
        return np.stack([r, g, b], axis=2)

    @staticmethod
    def _normalize_band(band: np.ndarray) -> np.ndarray:
        band_min = band.min()
        band_max = band.max()
        return (band - band_min) / (band_max - band_min + 1e-8)

    @staticmethod
    def load_bands_from_zip(zip_path: Path, bands: List[str]) -> Dict[str, np.ndarray]:
        if rasterio is None:
            raise ImportError("rasterio is required to read Sentinel imagery. Install rasterio>=1.3.0")

        arrays = {}
        for band in bands:
            inner_path = SentinelDataHandler._find_band_path(zip_path, band)
            if not inner_path:
                raise FileNotFoundError(f"Band {band} not found inside {zip_path}")
            uri = f"zip://{zip_path}!{inner_path}"
            with rasterio.open(uri) as src:
                arrays[band] = src.read(1).astype(np.float32)
        return arrays

    @staticmethod
    def _find_band_path(zip_path: Path, band_code: str) -> Optional[str]:
        with ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith(f"_{band_code}_10m.jp2"):
                    return member
        return None

    @staticmethod
    def load_rgb_from_zip(zip_path: Path, bands: Tuple[str, str, str] = ('B4', 'B3', 'B2')) -> np.ndarray:
        band_dict = SentinelDataHandler.load_bands_from_zip(zip_path, list(bands))
        return SentinelDataHandler.rgb_composite(
            {'B4': band_dict[bands[0]], 'B3': band_dict[bands[1]], 'B2': band_dict[bands[2]]}
        )

    @staticmethod
    def compute_indices_from_zip(zip_path: Path, indices: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        indices = indices or ['NDVI']
        bands_needed = set()
        for idx in indices:
            if idx == 'NDVI':
                bands_needed.update(['B4', 'B8'])
            elif idx == 'NDBI':
                bands_needed.update(['B11', 'B8'])
            elif idx == 'NDMI':
                bands_needed.update(['B8', 'B11'])

        band_arrays = SentinelDataHandler.load_bands_from_zip(zip_path, list(bands_needed))
        stats = {}
        for idx in indices:
            array = SentinelDataHandler.calculate_index(band_arrays, idx)
            stats[idx] = {
                'mean': float(np.nanmean(array)),
                'min': float(np.nanmin(array)),
                'max': float(np.nanmax(array)),
            }
        return stats


class SentinelAPIClient:
    """Minimal Sentinel-2 downloader built on top of sentinelsat."""

    def __init__(
        self,
        username: str,
        password: str,
        api_url: str = "https://scihub.copernicus.eu/dhus",
        download_dir: str = "./data/sentinel",
    ):
        if SentinelAPI is None or geojson_to_wkt is None:
            raise ImportError("sentinelsat is required for Sentinel downloads. Install with `pip install sentinelsat`.")

        self.api = SentinelAPI(username, password, api_url)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_tile(
        self,
        bbox: Dict[str, float],
        date_range: Tuple[str, str],
        max_cloud_cover: int = 20,
        product_type: str = "S2MSI2A",
    ) -> Tuple[Path, Dict]:
        footprint = geojson_to_wkt(GeospatialProcessor.bbox_to_geojson(bbox))
        logger.info(
            "Querying Sentinel-2 %s for %s -> %s (cloud<=%s)",
            product_type,
            date_range[0],
            date_range[1],
            max_cloud_cover,
        )
        products = self.api.query(
            footprint=footprint,
            date=date_range,
            platformname="Sentinel-2",
            processinglevel="Level-2A",
            cloudcoverpercentage=(0, max_cloud_cover),
            producttype=product_type,
        )

        if not products:
            raise ValueError("No Sentinel-2 products found for the requested footprint/date.")

        product_id, product = sorted(
            products.items(),
            key=lambda item: item[1].get('cloudcoverpercentage', 100),
        )[0]
        logger.info("Downloading Sentinel product %s (%s)", product.get('title'), product_id)
        result = self.api.download(product_id, directory_path=str(self.download_dir))
        return Path(result['path']), product


class EarthEngineInterface:
    """Interface with Google Earth Engine for data retrieval."""

    _initialized = False

    @staticmethod
    def get_ee_image(collection: str, region: Dict, date_range: Tuple[str, str],
                     filters: Optional[Dict] = None) -> Dict:
        if ee is None:
            raise ImportError("earthengine-api is required for Earth Engine access. Install with `pip install earthengine-api`.")

        EarthEngineInterface._ensure_initialized()
        region_geom = ee.Geometry(region)
        image_collection = ee.ImageCollection(collection).filterBounds(region_geom).filterDate(*date_range)

        if filters:
            if 'cloud_cover' in filters:
                image_collection = image_collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', filters['cloud_cover']))
            if 'max_results' in filters:
                image_collection = image_collection.limit(filters['max_results'])

        image = image_collection.first()
        if image is None:
            raise ValueError(f"No Earth Engine images found for {collection} within {date_range}.")

        info = image.getInfo()
        download_url = image.getDownloadURL({
            'region': region_geom,
            'scale': filters.get('scale', 30) if filters else 30,
            'format': 'GeoTIFF'
        })

        return {
            'collection': collection,
            'region': region,
            'date_range': date_range,
            'properties': info.get('properties', {}),
            'bands': [band['id'] for band in info.get('bands', [])],
            'download_url': download_url,
        }

    @staticmethod
    def authenticate(service_account_email: str, service_account_path: str) -> bool:
        if ee is None:
            raise ImportError("earthengine-api is required for Earth Engine access.")
        credentials = ee.ServiceAccountCredentials(service_account_email, service_account_path)
        ee.Initialize(credentials)
        EarthEngineInterface._initialized = True
        logger.info("Authenticated Earth Engine with %s", service_account_email)
        return True

    @staticmethod
    def _ensure_initialized() -> None:
        if ee is None:
            raise ImportError("earthengine-api is required for Earth Engine access.")
        if not EarthEngineInterface._initialized:
            ee.Initialize()
            EarthEngineInterface._initialized = True


__all__ = ['GeospatialProcessor', 'SentinelDataHandler', 'SentinelAPIClient', 'EarthEngineInterface', 'HyperspectralProcessor']
