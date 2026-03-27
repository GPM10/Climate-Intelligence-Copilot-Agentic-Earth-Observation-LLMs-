"""
Data processing utilities for climate and environmental datasets.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ClimateDataProcessor:
    """Process climate and environmental time-series data."""
    
    @staticmethod
    def calculate_anomalies(data: np.ndarray, baseline_period: Tuple[int, int]) -> np.ndarray:
        """
        Calculate climate anomalies against baseline period.
        
        Args:
            data: Time-series data array
            baseline_period: (start_idx, end_idx) for baseline
        
        Returns:
            Anomaly data (deviations from baseline mean)
        """
        
        baseline_mean = np.mean(data[baseline_period[0]:baseline_period[1]])
        return data - baseline_mean
    
    @staticmethod
    def calculate_trends(years: List[int], values: List[float]) -> Dict:
        """
        Calculate linear trend from time-series data.
        
        Args:
            years: Year values
            values: Data values for each year
        
        Returns:
            Dict with slope, p-value, trend direction
        """
        
        if len(years) < 2:
            return {'error': 'Insufficient data points'}
        
        # Simple linear regression
        x = np.array(years, dtype=float)
        y = np.array(values, dtype=float)
        
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # R-squared calculation
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'trend_direction': trend_direction,
            'annual_change': float(slope),
            'units_per_decade': float(slope * 10) if len(years) > 0 else 0
        }
    
    @staticmethod
    def detect_extreme_events(data: np.ndarray, threshold_percentile: float = 95) -> Dict:
        """
        Detect extreme events in time-series data.
        
        Args:
            data: Time-series data
            threshold_percentile: Percentile threshold for extremes
        
        Returns:
            Dict with extreme event indices and statistics
        """
        
        threshold = np.percentile(data, threshold_percentile)
        extremes = np.where(data > threshold)[0]
        
        return {
            'threshold': float(threshold),
            'threshold_percentile': threshold_percentile,
            'extreme_indices': extremes.tolist(),
            'number_of_extremes': len(extremes),
            'frequency': len(extremes) / len(data) if len(data) > 0 else 0
        }
    
    @staticmethod
    def compare_regions(region_data: Dict[str, List[float]], years: List[int]) -> Dict:
        """
        Compare climate metrics across regions.
        
        Args:
            region_data: Dict of region_name -> values
            years: Year list
        
        Returns:
            Comparative analysis
        """
        
        comparison = {}
        
        for region, values in region_data.items():
            trends = ClimateDataProcessor.calculate_trends(years, values)
            comparison[region] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'trend': trends.get('trend_direction', 'N/A')
            }
        
        # Identify highest and lowest
        means = {r: comparison[r]['mean'] for r in region_data.keys()}
        max_region = max(means, key=means.get)
        min_region = min(means, key=means.get)
        
        return {
            'comparisons': comparison,
            'highest': max_region,
            'lowest': min_region,
            'difference': float(means[max_region] - means[min_region])
        }


class EmissionsProcessor:
    """Process greenhouse gas emissions data."""
    
    GHG_GWPS = {
        'CO2': 1.0,
        'CH4': 28,      # Over 100 years
        'N2O': 265,     # Over 100 years
        'SF6': 23500
    }
    
    @staticmethod
    def calculate_co2_equivalent(emissions_dict: Dict[str, float]) -> float:
        """
        Calculate CO2-equivalent from multiple GHGs.
        
        Args:
            emissions_dict: {gas_name: amount}
        
        Returns:
            Total CO2-equivalent
        """
        
        total_co2eq = 0
        for gas, amount in emissions_dict.items():
            gwp = EmissionsProcessor.GHG_GWPS.get(gas, 1.0)
            total_co2eq += amount * gwp
        
        return total_co2eq
    
    @staticmethod
    def calculate_sectoral_share(sectoral_emissions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate sectoral contribution to total emissions.
        
        Args:
            sectoral_emissions: {sector: amount}
        
        Returns:
            {sector: percentage}
        """
        
        total = sum(sectoral_emissions.values())
        
        if total == 0:
            return {}
        
        return {sector: (amount / total) * 100 
                for sector, amount in sectoral_emissions.items()}


class BiodiversityAnalyzer:
    """Analyze biodiversity and species data."""
    
    @staticmethod
    def calculate_indices(species_data: Dict[str, int]) -> Dict:
        """
        Calculate biodiversity indices.
        
        Args:
            species_data: {species: count}
        
        Returns:
            Various biodiversity metrics
        """
        
        # Shannon Diversity Index
        total = sum(species_data.values())
        if total == 0:
            return {'error': 'No species data'}
        
        proportions = [count / total for count in species_data.values()]
        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
        
        # Simpson Index
        simpson = 1 - sum(p ** 2 for p in proportions)
        
        return {
            'species_count': len(species_data),
            'total_individuals': total,
            'shannon_index': float(shannon),
            'simpson_index': float(simpson),
            'richness': len(species_data),
            'evenness': float(shannon / np.log(len(species_data))) if len(species_data) > 1 else 0
        }
    
    @staticmethod
    def calculate_threat_level(threatened_counts: Dict[str, int]) -> Dict:
        """
        Calculate conservation threat level.
        
        Args:
            threatened_counts: {'extinct': n, 'endangered': n, ...}
        
        Returns:
            Threat assessment
        """
        
        threat_weights = {
            'extinct': 5.0,
            'critically_endangered': 4.0,
            'endangered': 3.0,
            'vulnerable': 2.0,
            'near_threatened': 1.0,
            'least_concern': 0.5
        }
        
        total_threat = sum(
            threatened_counts.get(level, 0) * weight
            for level, weight in threat_weights.items()
        )
        
        total_species = sum(threatened_counts.values())
        
        return {
            'total_threatened': total_species,
            'threat_index': float(total_threat / max(1, total_species)),
            'threat_level': 'Critical' if total_threat > 100 else 'High' if total_threat > 50 else 'Moderate'
        }


__all__ = ['ClimateDataProcessor', 'EmissionsProcessor', 'BiodiversityAnalyzer']
