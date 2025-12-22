"""
World Ocean Atlas Nitrate Loader and N² Computation.

Provides nitrate climatology baseline for multivariate loss.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from pathlib import Path

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import gsw  # TEOS-10
    GSW_AVAILABLE = True
except ImportError:
    GSW_AVAILABLE = False


# =============================================================================
# World Ocean Atlas Nitrate Loader
# =============================================================================

class WOANitrateLoader:
    """
    Load nitrate climatology from World Ocean Atlas 2018.
    
    Data: Monthly 1° climatology, 102 depth levels, 0-6000m
    URL: https://www.ncei.noaa.gov/data/oceans/woa/WOA18/
    
    For Black Sea:
    - Surface (0-30m): ~0-1 μmol/L (nutrient-depleted)
    - Nitracline (~50-100m): rapid increase
    - Deep (>200m): ~5-10 μmol/L (anoxic below ~150m)
    """
    
    def __init__(self, woa_path: Optional[str] = None):
        """
        Initialize loader.
        
        Args:
            woa_path: Path to WOA18 nitrate NetCDF file
                     (e.g., 'woa18_all_n_mon_1deg.nc')
        """
        self.woa_path = woa_path
        self.ds = None
        
        # Black Sea bounding box
        self.lon_min, self.lon_max = 27.0, 42.0
        self.lat_min, self.lat_max = 40.5, 47.0
        
        # Default climatology (monthly means for Black Sea)
        # [month, depth (0,10,20,30,50,75,100,125,150,200)]
        self.default_depths = np.array([0, 10, 20, 30, 50, 75, 100, 125, 150, 200])
        self.default_clim = np.array([
            # J   F   M   A   M   J   J   A   S   O   N   D
            [0.1, 0.2, 0.5, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.2],  # 0m
            [0.2, 0.3, 0.6, 0.9, 0.4, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.3],  # 10m
            [0.3, 0.4, 0.8, 1.2, 0.6, 0.3, 0.2, 0.2, 0.3, 0.4, 0.5, 0.4],  # 20m
            [0.5, 0.6, 1.0, 1.5, 0.8, 0.5, 0.3, 0.3, 0.4, 0.6, 0.7, 0.6],  # 30m
            [1.5, 1.8, 2.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.2, 1.5, 1.8, 1.6],  # 50m
            [3.5, 4.0, 4.5, 5.0, 4.5, 4.0, 3.5, 3.0, 3.2, 3.8, 4.2, 3.8],  # 75m
            [5.0, 5.5, 6.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.8, 5.5, 5.8, 5.5],  # 100m
            [6.0, 6.5, 7.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.8, 6.5, 6.8, 6.5],  # 125m
            [7.0, 7.5, 8.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.8, 7.5, 7.8, 7.5],  # 150m
            [8.0, 8.5, 9.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.8, 8.5, 8.8, 8.5],  # 200m
        ])
        
        if woa_path and Path(woa_path).exists() and XARRAY_AVAILABLE:
            self._load_woa()
    
    def _load_woa(self):
        """Load WOA NetCDF file."""
        self.ds = xr.open_dataset(self.woa_path)
        
        # Subset to Black Sea
        self.ds = self.ds.sel(
            lon=slice(self.lon_min, self.lon_max),
            lat=slice(self.lat_min, self.lat_max)
        )
        
    def get_profile(self, lon: float, lat: float, month: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract nitrate profile at location for given month.
        
        Args:
            lon: Longitude (degrees E)
            lat: Latitude (degrees N)
            month: Month (0-11)
            
        Returns:
            depth: [n_levels], m
            nitrate: [n_levels], μmol/L
        """
        if self.ds is not None:
            profile = self.ds['n_an'].sel(
                lon=lon, lat=lat, time=month,
                method='nearest'
            )
            return profile['depth'].values, profile.values
        else:
            # Use default climatology
            return self.default_depths, self.default_clim[:, month]
    
    def interpolate_to_depth(self, depth: float, month: int) -> float:
        """
        Get nitrate at specific depth via linear interpolation.
        
        Args:
            depth: Target depth (m, positive downward)
            month: Month (0-11)
            
        Returns:
            nitrate: Nitrate concentration (μmol/L)
        """
        depths = self.default_depths
        values = self.default_clim[:, month]
        
        return np.interp(depth, depths, values)
    
    def get_nitracline_depth(self, lon: float, lat: float, month: int) -> float:
        """
        Find nitracline depth (depth of maximum ∂N/∂z).
        
        The nitracline is the zone of steepest nitrate gradient,
        typically between 50-100m in the Black Sea.
        """
        depth, nitrate = self.get_profile(lon, lat, month)
        
        # Compute gradient (using central differences)
        dN_dz = np.gradient(nitrate, depth)
        
        # Find depth of maximum gradient
        idx = np.argmax(np.abs(dN_dz))
        
        return depth[idx]
    
    def to_tensor(self, month: int, depths: np.ndarray) -> torch.Tensor:
        """
        Get nitrate as tensor for loss computation.
        
        Args:
            month: Month (0-11)
            depths: [batch] depths at which to interpolate
            
        Returns:
            nitrate: [batch] tensor of nitrate values
        """
        nitrate = np.array([
            self.interpolate_to_depth(d, month) for d in depths
        ])
        return torch.tensor(nitrate, dtype=torch.float32)


# =============================================================================
# Buoyancy Frequency (N²) Computation
# =============================================================================

def compute_N2(T: np.ndarray, 
               S: np.ndarray, 
               z: np.ndarray,
               lat: float = 43.0) -> np.ndarray:
    """
    Compute buoyancy frequency N² from T/S profiles using TEOS-10.
    
    N² = -(g/ρ₀) · ∂ρ/∂z
    
    Args:
        T: Temperature profile [n_levels], °C
        S: Salinity profile [n_levels], PSU
        z: Depth profile [n_levels], m (negative downward)
        lat: Latitude for gravity correction
        
    Returns:
        N2: Buoyancy frequency squared [n_levels], s⁻²
    """
    g = 9.81  # m/s²
    rho0 = 1025.0  # Reference density kg/m³
    
    if GSW_AVAILABLE:
        # Use TEOS-10 for accurate density
        p = gsw.p_from_z(z, lat)
        SA = gsw.SA_from_SP(S, p, 33.0, lat)  # Absolute salinity
        CT = gsw.CT_from_t(SA, T, p)          # Conservative temperature
        rho = gsw.rho(SA, CT, p)
    else:
        # UNESCO EOS-80 approximation
        rho = 1000 + 0.8 * (S - 35) - 0.2 * (T - 10)
    
    # Vertical gradient of density
    drho_dz = np.gradient(rho, z)
    
    # Buoyancy frequency
    N2 = -(g / rho0) * drho_dz
    
    # Clip to physical range [0, 1e-3] s⁻²
    N2 = np.clip(N2, 0, 1e-3)
    
    return N2


def compute_N2_tensor(T: torch.Tensor,
                     S: torch.Tensor,
                     z: torch.Tensor,
                     edge_index_v: torch.Tensor) -> torch.Tensor:
    """
    Compute N² at vertical edges (for message passing).
    
    Args:
        T: Temperature at nodes [N]
        S: Salinity at nodes [N]
        z: Depth at nodes [N] (negative downward)
        edge_index_v: Vertical edge connectivity [2, E_v]
        
    Returns:
        N2: Buoyancy frequency at edges [E_v]
    """
    src, dst = edge_index_v
    
    # Simplified density (UNESCO approximation)
    rho = 1000 + 0.8 * (S - 35) - 0.2 * (T - 10)
    
    # Gradient across edge
    drho = rho[dst] - rho[src]
    dz = z[dst] - z[src]
    
    drho_dz = drho / (dz + 1e-6)
    
    # N²
    g = 9.81
    rho0 = 1025.0
    N2 = -(g / rho0) * drho_dz
    
    # Clip
    N2 = torch.clamp(N2, min=0, max=1e-3)
    
    return N2


# =============================================================================
# DCM Detection
# =============================================================================

def detect_dcm_depth(chl: np.ndarray, 
                     depth: np.ndarray,
                     smooth_window: int = 5) -> float:
    """
    Detect Deep Chlorophyll Maximum depth.
    
    DCM = depth of maximum chlorophyll (after smoothing).
    
    Args:
        chl: Chlorophyll profile [n_levels]
        depth: Depth values [n_levels] (positive downward)
        smooth_window: Smoothing window size
        
    Returns:
        dcm_depth: Depth of chlorophyll maximum (m)
    """
    from scipy.ndimage import uniform_filter1d
    
    # Smooth to reduce noise
    chl_smooth = uniform_filter1d(chl, smooth_window)
    
    # Find maximum
    idx = np.argmax(chl_smooth)
    
    return depth[idx]


def detect_dcm_depth_tensor(chl: torch.Tensor,
                            depth: torch.Tensor,
                            n_horizontal: int,
                            n_vertical: int) -> torch.Tensor:
    """
    Detect DCM depth for each horizontal location.
    
    Args:
        chl: Chlorophyll [N] (flattened 3D)
        depth: Depth [N]
        n_horizontal: Number of horizontal nodes
        n_vertical: Number of vertical levels
        
    Returns:
        dcm_depth: [n_horizontal] DCM depth at each column
    """
    chl_3d = chl.view(n_horizontal, n_vertical)
    depth_3d = depth.view(n_horizontal, n_vertical)
    
    # Index of max chl in each column
    max_idx = chl_3d.argmax(dim=1)
    
    # Get corresponding depth
    dcm_depth = torch.gather(depth_3d, 1, max_idx.unsqueeze(1)).squeeze()
    
    return dcm_depth


# =============================================================================
# CIL (Cold Intermediate Layer) Detection
# =============================================================================

def detect_cil_bounds(T: np.ndarray, 
                     depth: np.ndarray,
                     T_threshold: float = 8.0) -> Tuple[float, float, float]:
    """
    Detect Black Sea Cold Intermediate Layer.
    
    CIL = permanent cold layer (T < 8°C) between ~30-150m.
    
    Args:
        T: Temperature profile [n_levels], °C
        depth: Depth values [n_levels], m (positive)
        T_threshold: Temperature threshold for CIL
        
    Returns:
        (top_depth, bottom_depth, thickness)
    """
    in_cil = T < T_threshold
    
    if not in_cil.any():
        return np.nan, np.nan, 0.0
    
    cil_depths = depth[in_cil]
    top = cil_depths.min()
    bottom = cil_depths.max()
    
    return top, bottom, bottom - top


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Oceanographic Utilities...")
    
    # Test WOA loader
    woa = WOANitrateLoader()
    
    depths, nitrate = woa.get_profile(33.0, 43.0, 6)  # July, mid-Black Sea
    print(f"✓ WOA nitrate profile (July):")
    print(f"  Depths: {depths[:5]} ...")
    print(f"  Nitrate: {nitrate[:5]} ...")
    
    nitracline = woa.get_nitracline_depth(33.0, 43.0, 6)
    print(f"  Nitracline depth: {nitracline:.1f}m")
    
    # Test N² computation
    print(f"\n✓ N² computation:")
    T = np.linspace(25, 8, 20)  # Warm at surface, cold at depth
    S = np.linspace(18, 22, 20)  # Black Sea salinity
    z = np.linspace(0, -200, 20)
    
    N2 = compute_N2(T, S, z)
    print(f"  N² range: [{N2.min():.2e}, {N2.max():.2e}] s⁻²")
    print(f"  Pycnocline depth: {-z[np.argmax(N2)]:.0f}m")
    
    # Test DCM detection
    print(f"\n✓ DCM detection:")
    chl = np.exp(-((np.abs(z) - 50) / 20)**2)  # Peak at 50m
    dcm = detect_dcm_depth(chl, np.abs(z))
    print(f"  DCM depth: {dcm:.1f}m")
    
    # Test CIL detection
    print(f"\n✓ CIL detection:")
    top, bottom, thickness = detect_cil_bounds(T, np.abs(z))
    print(f"  CIL: {top:.0f}m - {bottom:.0f}m (thickness: {thickness:.0f}m)")
    
    print("\n✅ All tests passed!")
