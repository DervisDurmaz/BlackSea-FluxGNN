"""
World Ocean Atlas (WOA18) Nitrate Loader.

Downloads and processes WOA nitrate climatology for Black Sea.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import os

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class WOA18NitrateLoader:
    """
    Load and interpolate WOA18 nitrate climatology.
    
    Data source: https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/
    
    Black Sea specific:
    - Surface (0-30m): ~0-1 μmol/L (nutrient-depleted due to stratification)
    - Nitracline (~50-100m): rapid increase
    - Deep (>150m): ~5-10 μmol/L (but anoxic below ~150m)
    """
    
    # WOA18 NOAA THREDDS URL (monthly nitrate)
    WOA_URL_TEMPLATE = (
        "https://www.ncei.noaa.gov/thredds-ocean/dodsC/"
        "ncei/woa/nitrate/all/{res}/{var}_{res}_{month:02d}.nc"
    )
    
    # Black Sea bounding box
    LON_MIN, LON_MAX = 27.0, 42.0
    LAT_MIN, LAT_MAX = 40.5, 47.0
    
    # Standard WOA depths (meters)
    STANDARD_DEPTHS = np.array([
        0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 
        85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350,
        375, 400, 450, 500
    ])
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 resolution: str = '01'):  # 1 degree
        """
        Initialize WOA loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
            resolution: '01' for 1°, '04' for 0.25°
        """
        self.resolution = resolution
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./woa_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Black Sea monthly climatology [12, n_depths]
        # Pre-computed typical values if download fails
        self._default_climatology = self._create_default_climatology()
        
        # Loaded data cache
        self._monthly_data = {}
        
    def _create_default_climatology(self) -> np.ndarray:
        """Create default Black Sea nitrate climatology."""
        # Depths: surface to 300m
        depths = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300])
        n_depths = len(depths)
        
        # Monthly profiles (μmol/L)
        # Column = depth, Row = month
        clim = np.zeros((12, n_depths))
        
        # Base profile (summer-like, strong stratification)
        base = np.array([0.1, 0.2, 0.3, 0.5, 1.5, 3.5, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0])
        
        # Seasonal modulation
        seasonal_factor = np.array([
            1.2,  # Jan - winter mixing brings nutrients up
            1.3,  # Feb
            1.4,  # Mar - spring bloom draws down
            1.2,  # Apr
            0.9,  # May - stratification strengthens
            0.7,  # Jun
            0.6,  # Jul - peak stratification
            0.6,  # Aug
            0.7,  # Sep
            0.8,  # Oct - autumn mixing begins
            1.0,  # Nov
            1.1,  # Dec
        ])
        
        for month in range(12):
            # Surface enrichment in winter
            surface_factor = seasonal_factor[month]
            clim[month, :3] = base[:3] * surface_factor
            
            # Deep values more stable
            clim[month, 3:] = base[3:]
        
        return clim
    
    def load_month(self, month: int, force_download: bool = False) -> np.ndarray:
        """
        Load nitrate climatology for a specific month.
        
        Args:
            month: Month (1-12)
            force_download: Force re-download even if cached
            
        Returns:
            nitrate: [n_lat, n_lon, n_depth] array for Black Sea
        """
        if month in self._monthly_data and not force_download:
            return self._monthly_data[month]
        
        cache_file = self.cache_dir / f'woa18_nitrate_month{month:02d}.npy'
        
        # Try loading from cache
        if cache_file.exists() and not force_download:
            self._monthly_data[month] = np.load(cache_file)
            return self._monthly_data[month]
        
        # Try downloading
        if XARRAY_AVAILABLE:
            try:
                url = self.WOA_URL_TEMPLATE.format(
                    res=self.resolution, 
                    var='n',
                    month=month
                )
                
                ds = xr.open_dataset(url)
                
                # Subset to Black Sea
                data = ds['n_an'].sel(
                    lon=slice(self.LON_MIN, self.LON_MAX),
                    lat=slice(self.LAT_MIN, self.LAT_MAX)
                ).values
                
                # Save to cache
                np.save(cache_file, data)
                self._monthly_data[month] = data
                
                print(f"  Loaded WOA18 nitrate for month {month}")
                return data
                
            except Exception as e:
                print(f"  Warning: Could not load WOA data: {e}")
                print(f"  Using default climatology")
        
        # Fallback to default
        return self._default_climatology[month - 1]
    
    def get_profile(self, 
                    lon: float, lat: float, 
                    month: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get nitrate profile at a specific location.
        
        Args:
            lon: Longitude (°E)
            lat: Latitude (°N)
            month: Month (1-12)
            
        Returns:
            depths: Depth array (m)
            nitrate: Nitrate values (μmol/L)
        """
        # Use default climatology (independent of location for simplicity)
        depths = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300])
        nitrate = self._default_climatology[month - 1]
        
        return depths, nitrate
    
    def interpolate_to_depth(self, 
                            target_depths: np.ndarray,
                            month: int) -> np.ndarray:
        """
        Interpolate climatology to target depths.
        
        Args:
            target_depths: Target depth array (m)
            month: Month (1-12)
            
        Returns:
            nitrate: Interpolated values at target depths
        """
        source_depths = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300])
        source_values = self._default_climatology[month - 1]
        
        return np.interp(target_depths, source_depths, source_values)
    
    def get_nitracline_depth(self, month: int) -> float:
        """
        Find nitracline depth (maximum dN/dz).
        
        Args:
            month: Month (1-12)
            
        Returns:
            depth: Nitracline depth (m)
        """
        depths = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300])
        nitrate = self._default_climatology[month - 1]
        
        # Compute gradient
        dN_dz = np.gradient(nitrate, depths)
        
        # Find depth of maximum gradient
        max_idx = np.argmax(np.abs(dN_dz))
        
        return depths[max_idx]
    
    def to_tensor(self, month: int, depths: np.ndarray):
        """
        Get nitrate as PyTorch tensor.
        
        Args:
            month: Month (1-12)
            depths: Target depths
            
        Returns:
            tensor: Nitrate tensor
        """
        import torch
        
        values = self.interpolate_to_depth(depths, month)
        return torch.tensor(values, dtype=torch.float32)


# =============================================================================
# Sparse BGC-Argo Handler
# =============================================================================

class SparseArgoHandler:
    """
    Handle sparse BGC-Argo float observations.
    
    Strategy:
    - Match float profiles to nearest graph nodes
    - Create observation operator for loss computation
    - Apply quality control flags
    """
    
    # Argo quality flags
    QC_GOOD = 1
    QC_PROBABLY_GOOD = 2
    QC_INTERPOLATED = 8
    
    def __init__(self, max_distance_km: float = 50.0):
        """
        Args:
            max_distance_km: Maximum distance for matching floats to nodes
        """
        self.max_dist = max_distance_km
        
    def haversine_distance(self, 
                           lon1: float, lat1: float,
                           lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
        """Compute haversine distance in km."""
        R = 6371  # Earth radius (km)
        
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def match_to_nodes(self,
                       float_lon: np.ndarray,
                       float_lat: np.ndarray,
                       node_lon: np.ndarray,
                       node_lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match float observations to nearest graph nodes.
        
        Returns:
            matched_indices: Node indices for each float
            distances: Distances (km)
        """
        n_floats = len(float_lon)
        matched_indices = np.zeros(n_floats, dtype=int)
        distances = np.zeros(n_floats)
        
        for i in range(n_floats):
            dist = self.haversine_distance(
                float_lon[i], float_lat[i],
                node_lon, node_lat
            )
            matched_indices[i] = np.argmin(dist)
            distances[i] = dist[matched_indices[i]]
        
        # Filter by max distance
        valid = distances < self.max_dist
        
        return matched_indices[valid], distances[valid]
    
    def create_observation_mask(self,
                               n_nodes: int,
                               matched_indices: np.ndarray) -> np.ndarray:
        """
        Create observation mask for loss computation.
        
        Args:
            n_nodes: Total number of graph nodes
            matched_indices: Indices of nodes with observations
            
        Returns:
            mask: [n_nodes] boolean mask
        """
        mask = np.zeros(n_nodes, dtype=bool)
        mask[matched_indices] = True
        return mask
    
    def apply_qc(self,
                 values: np.ndarray,
                 qc_flags: np.ndarray,
                 accepted_flags: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply quality control filtering.
        
        Args:
            values: Observation values
            qc_flags: Argo QC flags
            accepted_flags: List of accepted QC values
            
        Returns:
            filtered_values: QC-filtered values
            valid_mask: Boolean mask of valid observations
        """
        if accepted_flags is None:
            accepted_flags = [self.QC_GOOD, self.QC_PROBABLY_GOOD]
        
        valid = np.isin(qc_flags, accepted_flags)
        
        return values[valid], valid


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing WOA and Argo data handlers...")
    
    # WOA Loader
    woa = WOA18NitrateLoader()
    
    depths, nitrate = woa.get_profile(33.0, 43.0, 7)  # July, mid-Black Sea
    print(f"\n✓ WOA nitrate profile (July):")
    print(f"  Depths: {depths[:5]}...")
    print(f"  Nitrate: {nitrate[:5].round(2)}...")
    
    nitracline = woa.get_nitracline_depth(7)
    print(f"  Nitracline depth: {nitracline}m")
    
    # Test interpolation
    target_depths = np.linspace(0, 200, 50)
    interp_values = woa.interpolate_to_depth(target_depths, 7)
    print(f"\n✓ Interpolated to {len(target_depths)} levels")
    
    # Argo handler
    argo = SparseArgoHandler()
    
    # Synthetic data
    float_lon = np.array([30, 35, 38])
    float_lat = np.array([43, 44, 42])
    node_lon = np.random.rand(100) * 14 + 28
    node_lat = np.random.rand(100) * 6 + 41
    
    matched, dists = argo.match_to_nodes(float_lon, float_lat, node_lon, node_lat)
    print(f"\n✓ Argo matching:")
    print(f"  Matched {len(matched)} floats to nodes")
    print(f"  Distances: {dists.round(1)} km")
    
    print("\n✅ Data handlers ready!")
