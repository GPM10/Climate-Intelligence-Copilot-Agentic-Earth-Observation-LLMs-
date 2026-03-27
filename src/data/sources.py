"""
External data source clients for Climate Intelligence Copilot.

These classes provide thin wrappers around real-world datasets so the
DataAgent can retrieve actual observations instead of mocked values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import os

import numpy as np
import pandas as pd
import requests
import xarray as xr


logger = logging.getLogger(__name__)


class EDGARDataSource:
    """
    Load greenhouse-gas inventories from an EDGAR CSV export.

    The official EDGAR portal distributes country-level totals as CSV files
    (e.g., EDGAR v7.0 GHG).  Rather than inventing yet another API, we expect
    the user to download that dataset once and point the copilot to it via
    configuration.  The loader then filters/aggregates it into the format the
    DataAgent expects.
    """

    def __init__(
        self,
        csv_path: str,
        country_column: str = "country",
        year_column: str = "year",
        value_column: str = "emissions_mtco2e",
        sector_column: Optional[str] = None,
        region_aliases: Optional[Dict[str, str]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.country_column = country_column
        self.year_column = year_column
        self.value_column = value_column
        self.sector_column = sector_column
        self.region_aliases = {k.lower(): v for k, v in (region_aliases or {}).items()}

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"EDGAR dataset not found at {self.csv_path}. "
                "Download EDGAR country totals CSV and update config.data_sources.edgar.csv_path."
            )

        self._dataframe: Optional[pd.DataFrame] = None

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = pd.read_csv(self.csv_path)
        return self._dataframe

    def _normalize_region(self, region: str) -> str:
        key = region.strip().lower()
        if key in (self.region_aliases or {}):
            return self.region_aliases[key]
        return region

    def fetch_country_timeseries(
        self, region: str, start_year: int, end_year: int, sectors: Optional[List[str]] = None
    ) -> Dict:
        df = self.dataframe.copy()
        region_key = self._normalize_region(region)
        df[self.year_column] = df[self.year_column].astype(int)
        region_mask = df[self.country_column].str.lower() == region_key.lower()
        time_mask = df[self.year_column].between(start_year, end_year)
        if self.sector_column and sectors:
            sector_mask = df[self.sector_column].isin(sectors)
            df = df[region_mask & time_mask & sector_mask]
        else:
            df = df[region_mask & time_mask]

        if df.empty:
            raise ValueError(
                f"No EDGAR data found for region '{region}' between {start_year}-{end_year}. "
                "Ensure the CSV contains that country code or add an alias."
            )

        grouped = df.groupby(self.year_column)[self.value_column]
        years = sorted(grouped.mean().index.tolist())
        values = grouped.mean().reindex(years).tolist()

        trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
        return {
            "years": years,
            "values": [round(float(v), 2) for v in values],
            "trend": trend,
            "mean": round(float(np.mean(values)), 2),
            "std_dev": round(float(np.std(values)), 2),
            "source": f"EDGAR CSV ({self.csv_path.name})",
        }


class CAMSDataSource:
    """
    Retrieve Copernicus Atmosphere Monitoring Service (CAMS) data via cdsapi.

    The downloader caches each request to NetCDF so repeated analyses reuse
    the same file.  The files can be large, so requests should be scoped to
    coarse spatial/temporal ranges.
    """

    def __init__(
        self,
        dataset: str,
        api_url: str,
        api_key: str,
        cache_dir: str = "./data/cams",
        verify: bool = True,
    ):
        try:
            import cdsapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "cdsapi is required for CAMS integration. Install with `pip install cdsapi`."
            ) from exc

        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # cdsapi expects key in KEY:SECRET format
        self.client = cdsapi.Client(url=api_url, key=api_key, verify=verify)

    def _target_path(self, variable: str, region: str, start_year: int, end_year: int) -> Path:
        safe_region = region.lower().replace(" ", "_")
        filename = f"cams_{variable}_{safe_region}_{start_year}_{end_year}.nc"
        return self.cache_dir / filename

    def fetch_mean_timeseries(
        self,
        variable: str,
        bbox: Dict[str, float],
        start_year: int,
        end_year: int,
        frequency: str = "year",
    ) -> Dict:
        target = self._target_path(variable, bbox.get("name", "region"), start_year, end_year)

        if not target.exists():
            request = {
                "variable": variable,
                "date": f"{start_year}-01-01/{end_year}-12-31",
                "area": [
                    bbox["max_lat"],
                    bbox["min_lon"],
                    bbox["min_lat"],
                    bbox["max_lon"],
                ],
                "format": "netcdf",
            }
            logger.info("Requesting CAMS dataset %s (%s) -> %s", self.dataset, variable, target)
            self.client.retrieve(self.dataset, request, str(target))

        ds = xr.open_dataset(target)
        data_array = ds[variable]

        if frequency == "month":
            grouped = data_array.groupby("time.month").mean(dim=["latitude", "longitude"])
            keys = grouped["month"].values.astype(int).tolist()
        else:
            grouped = data_array.groupby("time.year").mean(dim=["latitude", "longitude"])
            keys = grouped["year"].values.astype(int).tolist()

        values = grouped.values.tolist()
        ds.close()

        if not values:
            raise ValueError("CAMS dataset returned no values for the requested period.")

        trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
        return {
            "years" if frequency == "year" else "periods": keys,
            "values": [round(float(v), 4) for v in values],
            "trend": trend,
            "mean": round(float(np.mean(values)), 4),
            "std_dev": round(float(np.std(values)), 4),
            "source": f"CAMS {self.dataset}",
        }


class GBIFDataSource:
    """Query biodiversity metrics from the GBIF API."""

    BASE_URL = "https://api.gbif.org/v1"

    def __init__(
        self,
        user_agent: str = "climate-intelligence-copilot",
        timeout: int = 30,
    ):
        self.headers = {"User-Agent": user_agent}
        self.timeout = timeout

    def _occurrence_url(self) -> str:
        return f"{self.BASE_URL}/occurrence/search"

    def fetch_species_summary(
        self,
        country_code: str,
        start_year: int,
        end_year: int,
        taxon_key: Optional[int] = None,
        limit: int = 300,
    ) -> Dict:
        params = {
            "country": country_code.upper(),
            "year": f"{start_year},{end_year}",
            "hasCoordinate": "true",
            "limit": limit,
        }
        if taxon_key:
            params["taxonKey"] = taxon_key

        logger.info("Querying GBIF %s with %s", self._occurrence_url(), params)
        resp = requests.get(
            self._occurrence_url(),
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])

        if not results:
            return {
                "total_records": 0,
                "species_count": {},
                "threatened_species": {},
                "sources": ["GBIF API"],
            }

        species_counts: Dict[str, int] = {}
        threatened_flags = {"CR": "critically_endangered", "EN": "endangered", "VU": "vulnerable"}

        for record in results:
            species = record.get("species") or record.get("speciesKey") or "unknown"
            species_counts[species] = species_counts.get(species, 0) + 1

        threat_summary = {label: 0 for label in threatened_flags.values()}
        for record in results:
            threat = record.get("threatStatus")
            if threat in threatened_flags:
                threat_summary[threatened_flags[threat]] += 1

        return {
            "total_records": payload.get("count", len(results)),
            "species_count": species_counts,
            "threatened_species": threat_summary,
            "sources": ["GBIF API"],
        }


def resolve_bbox_from_region(region: str, fallback_bbox: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Attempt to load a bounding box from Natural Earth JSON shipped with the repo.

    This helper is tiny but useful when CAMS requests need bounding boxes even
    if the question only provided a region string.
    """

    if fallback_bbox:
        return fallback_bbox

    boundary_file = Path("config") / "regions" / "boundaries.geojson"
    if not boundary_file.exists():
        logger.warning(
            "Region boundary file %s not found; falling back to global extent.", boundary_file
        )
        return {
            "min_lon": -180.0,
            "min_lat": -90.0,
            "max_lon": 180.0,
            "max_lat": 90.0,
            "name": region,
        }

    with open(boundary_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    region_lower = region.lower()
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        names = [
            props.get("name", ""),
            props.get("NAME_EN", ""),
            props.get("ISO_A2", ""),
            props.get("ISO_A3", ""),
        ]
        if any(region_lower == str(value).lower() for value in names if value):
            bbox = feature.get("bbox")
            if bbox:
                return {
                    "min_lon": bbox[0],
                    "min_lat": bbox[1],
                    "max_lon": bbox[2],
                    "max_lat": bbox[3],
                    "name": props.get("name", region),
                }

    raise ValueError(
        f"Could not resolve bounding box for region '{region}'. Update config/regions/boundaries.geojson or pass bbox."
    )


__all__ = [
    "EDGARDataSource",
    "CAMSDataSource",
    "GBIFDataSource",
    "resolve_bbox_from_region",
]
