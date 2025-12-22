"""
Physics-Based Feature Engineering for Black Sea Biogeochemistry.

Derives interaction terms and physics-informed features from CMEMS variables.
All features are oceanographically meaningful and can improve model performance.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

@dataclass
class OceanConstants:
    """Physical constants for Black Sea."""
    
    # General
    g: float = 9.81                    # Gravity (m/s²)
    OMEGA: float = 7.2921e-5           # Earth rotation (rad/s)
    
    # Black Sea specific (BRACKISH!)
    rho_0: float = 1012.0              # Reference density (kg/m³)
    S_0: float = 18.0                  # Reference salinity (PSU) - NOT 35!
    T_0: float = 15.0                  # Reference temperature (°C)
    
    # Coefficient of thermal expansion / haline contraction
    alpha_T: float = 0.17              # Thermal expansion (kg/m³/°C)
    beta_S: float = 0.78               # Haline contraction (kg/m³/PSU)
    
    # Biogeochemistry
    REDFIELD_N_P: float = 16.0         # Redfield N:P ratio
    REDFIELD_C_N: float = 6.625        # Redfield C:N ratio
    CHL_C_RATIO: float = 0.02          # Typical Chl:C ratio (mg Chl / mg C)
    
    # Black Sea latitude for Coriolis
    lat_mean: float = 43.5             # °N (center of Black Sea)
    
    @property
    def f(self) -> float:
        """Coriolis parameter at mean latitude."""
        return 2 * self.OMEGA * np.sin(np.radians(self.lat_mean))


CONST = OceanConstants()


# =============================================================================
# DERIVED FEATURE FUNCTIONS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. DENSITY & STRATIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_density(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Compute seawater density using simplified EOS for Black Sea.
    
    ρ = ρ₀ + β(S - S₀) - α(T - T₀)
    
    Note: For production, use TEOS-10 (gsw library)
    
    Args:
        T: Temperature (°C)
        S: Salinity (PSU)
        
    Returns:
        ρ: Density (kg/m³)
    """
    return (CONST.rho_0 + 
            CONST.beta_S * (S - CONST.S_0) - 
            CONST.alpha_T * (T - CONST.T_0))


def compute_N2(rho: np.ndarray, z: np.ndarray, 
               axis: int = 0) -> np.ndarray:
    """
    Compute buoyancy frequency squared N².
    
    N² = -(g/ρ₀) * ∂ρ/∂z
    
    Critical for stratification gating in split-kernel MP.
    
    Args:
        rho: Density profile (kg/m³)
        z: Depth array (m, positive downward)
        axis: Axis along which to compute gradient
        
    Returns:
        N²: Buoyancy frequency squared (s⁻²)
    """
    drho_dz = np.gradient(rho, z, axis=axis)
    return -(CONST.g / CONST.rho_0) * drho_dz


def compute_thermal_stratification(T_surface: np.ndarray, 
                                   T_bottom: np.ndarray) -> np.ndarray:
    """
    Thermal stratification: ΔT = T_surface - T_bottom
    
    Positive = stable stratification (warm on top)
    """
    return T_surface - T_bottom


def compute_mld_anomaly(mld: np.ndarray, 
                        mld_climatology: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mixed layer depth anomaly from climatology.
    
    Args:
        mld: Observed MLD (m)
        mld_climatology: Mean seasonal MLD (m)
        
    Returns:
        MLD anomaly (m)
    """
    if mld_climatology is None:
        return mld - np.nanmean(mld)
    return mld - mld_climatology


# ─────────────────────────────────────────────────────────────────────────────
# 2. CURRENT DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_current_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Current speed: |V| = √(u² + v²)
    """
    return np.sqrt(u**2 + v**2)


def compute_current_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Current direction in radians (oceanographic convention).
    
    0 = flow towards North
    π/2 = flow towards East
    """
    return np.arctan2(u, v)


def compute_current_direction_sincos(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode direction as sin/cos for continuity.
    
    Returns:
        sin_dir, cos_dir: Unit circle encoding of flow direction
    """
    speed = compute_current_speed(u, v) + 1e-8
    return u / speed, v / speed  # sin(θ), cos(θ)


def compute_vorticity_magnitude(u: np.ndarray, v: np.ndarray,
                                dx: float, dy: float) -> np.ndarray:
    """
    Relative vorticity: ζ = ∂v/∂x - ∂u/∂y
    
    Positive = cyclonic (counterclockwise in Northern Hemisphere)
    
    Note: For graphs, use spatial gradients from neighbors.
    """
    dvdx = np.gradient(v, dx, axis=1)  # Assuming (lat, lon) shape
    dudy = np.gradient(u, dy, axis=0)
    return dvdx - dudy


def compute_okubo_weiss(u: np.ndarray, v: np.ndarray,
                       dx: float, dy: float) -> np.ndarray:
    """
    Okubo-Weiss parameter for eddy detection.
    
    W = s_n² + s_s² - ω²
    
    W < 0 → vorticity-dominated (eddy core)
    W > 0 → strain-dominated (fronts)
    
    Args:
        u, v: Velocity components
        dx, dy: Grid spacing
        
    Returns:
        W: Okubo-Weiss parameter (s⁻²)
    """
    dudx = np.gradient(u, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    
    # Normal strain
    s_n = dudx - dvdy
    # Shear strain
    s_s = dvdx + dudy
    # Vorticity
    omega = dvdx - dudy
    
    return s_n**2 + s_s**2 - omega**2


def compute_ekman_pumping(tauuo: np.ndarray, tauvo: np.ndarray,
                         dx: float, dy: float,
                         lat: np.ndarray) -> np.ndarray:
    """
    Ekman pumping velocity: w_E = curl(τ) / (ρf)
    
    Positive = upwelling (divergence of Ekman transport)
    
    Args:
        tauuo, tauvo: Wind stress components (N/m²)
        dx, dy: Grid spacing (m)
        lat: Latitude for Coriolis parameter
        
    Returns:
        w_E: Ekman pumping velocity (m/s)
    """
    f = 2 * CONST.OMEGA * np.sin(np.radians(lat))
    
    dtauvo_dx = np.gradient(tauvo, dx, axis=1)
    dtauuo_dy = np.gradient(tauuo, dy, axis=0)
    
    curl_tau = dtauvo_dx - dtauuo_dy
    
    return curl_tau / (CONST.rho_0 * f + 1e-10)


def compute_geostrophic_velocity(zos: np.ndarray, 
                                 lat: np.ndarray,
                                 dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Geostrophic velocity from sea surface height.
    
    u_g = -(g/f) * ∂η/∂y
    v_g = (g/f) * ∂η/∂x
    
    Args:
        zos: Sea surface height (m)
        lat: Latitude for Coriolis
        dx, dy: Grid spacing (m)
        
    Returns:
        u_g, v_g: Geostrophic velocity (m/s)
    """
    f = 2 * CONST.OMEGA * np.sin(np.radians(lat))
    
    deta_dx = np.gradient(zos, dx, axis=1)
    deta_dy = np.gradient(zos, dy, axis=0)
    
    u_g = -(CONST.g / f) * deta_dy
    v_g = (CONST.g / f) * deta_dx
    
    return u_g, v_g


# ─────────────────────────────────────────────────────────────────────────────
# 3. BIOGEOCHEMISTRY INTERACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_np_ratio(no3: np.ndarray, po4: np.ndarray) -> np.ndarray:
    """
    N:P ratio (dimensionless).
    
    Redfield ratio ≈ 16:1
    - N:P < 16 → N-limited
    - N:P > 16 → P-limited
    
    Useful for identifying nutrient limitation regimes.
    """
    return no3 / (po4 + 1e-6)


def compute_np_deviation(no3: np.ndarray, po4: np.ndarray) -> np.ndarray:
    """
    Deviation from Redfield ratio: log(N:P / 16)
    
    Positive = P-limited
    Negative = N-limited
    0 = Redfield balance
    """
    ratio = compute_np_ratio(no3, po4)
    return np.log(ratio / CONST.REDFIELD_N_P + 1e-6)


def compute_chl_c_ratio(chl: np.ndarray, phyc: np.ndarray) -> np.ndarray:
    """
    Chlorophyll-to-Carbon ratio.
    
    Low Chl:C → high-light adapted cells
    High Chl:C → low-light adapted cells (shade adaptation)
    
    Useful for photoacclimation modeling.
    
    Args:
        chl: Chlorophyll (mg/m³)
        phyc: Phytoplankton carbon (mmol C/m³)
        
    Returns:
        Chl:C ratio (mg Chl / mg C)
    """
    # Convert mmol C to mg C: 1 mmol C = 12 mg C
    phyc_mg = phyc * 12.0
    return chl / (phyc_mg + 1e-6)


def compute_oxygen_saturation(o2: np.ndarray, T: np.ndarray, 
                              S: np.ndarray) -> np.ndarray:
    """
    Oxygen saturation percentage.
    
    O₂_sat = 100 * O₂_observed / O₂_equilibrium
    
    - < 100% → undersaturated (consumption > production)
    - > 100% → supersaturated (production > consumption)
    
    Uses Garcia & Gordon (1992) solubility formula.
    """
    # Simplified O2 solubility (μmol/L)
    # Fuller formula uses temperature-dependent polynomial
    Ts = np.log((298.15 - T) / (273.15 + T))
    
    # Coefficients for O2 solubility
    A0, A1, A2 = 5.80871, 3.20291, 4.17887
    B0, B1 = -0.319, -0.243
    
    o2_sat = np.exp(A0 + A1*Ts + A2*Ts**2 + 
                    S * (B0 + B1*Ts))
    
    # Convert to μmol/L (approximate)
    o2_sat = o2_sat * 44.66  
    
    return 100 * o2 / (o2_sat + 1e-6)


def compute_aou(o2: np.ndarray, o2_sat: np.ndarray) -> np.ndarray:
    """
    Apparent Oxygen Utilization.
    
    AOU = O₂_saturation - O₂_observed
    
    Positive AOU = net oxygen consumption since water left surface
    Negative AOU = supersaturation (recent photosynthesis)
    """
    return o2_sat - o2


def compute_bloom_indicator(chl: np.ndarray, 
                           chl_climatology: np.ndarray,
                           threshold: float = 2.0) -> np.ndarray:
    """
    Bloom indicator: Chl / Climatological_Chl > threshold.
    
    Args:
        chl: Observed chlorophyll
        chl_climatology: Mean seasonal chlorophyll
        threshold: Bloom threshold (default: 2x climatology)
        
    Returns:
        bloom: Boolean bloom indicator
    """
    return (chl / (chl_climatology + 1e-6)) > threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIGHT & THERMAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_par_from_shortwave(rsntds: np.ndarray, 
                               par_fraction: float = 0.45) -> np.ndarray:
    """
    Estimate PAR from net shortwave radiation.
    
    PAR ≈ 0.45 * Shortwave (400-700nm is ~45% of solar spectrum)
    
    Args:
        rsntds: Net shortwave radiation (W/m²)
        par_fraction: PAR fraction of total shortwave
        
    Returns:
        PAR (W/m² or μmol photons/m²/s with conversion)
    """
    # 1 W/m² ≈ 4.57 μmol photons/m²/s for PAR wavelengths
    par_watts = rsntds * par_fraction
    par_umol = par_watts * 4.57
    return par_umol


def compute_light_limitation(par: np.ndarray, 
                            Ik: float = 50.0) -> np.ndarray:
    """
    Light limitation factor using Smith formulation.
    
    L(I) = I / √(I² + Ik²)
    
    Ranges from 0 (dark) to 1 (saturated)
    
    Args:
        par: PAR (μmol photons/m²/s)
        Ik: Saturation irradiance (μmol/m²/s)
        
    Returns:
        Light limitation factor [0, 1]
    """
    return par / np.sqrt(par**2 + Ik**2 + 1e-6)


def compute_temperature_limitation(T: np.ndarray) -> np.ndarray:
    """
    Eppley temperature function for growth rate.
    
    f(T) = exp(0.0633 * T)
    
    Normalized to f(15°C) = 1.0
    """
    return np.exp(CONST.CHL_C_RATIO * T) / np.exp(CONST.CHL_C_RATIO * 15.0)


def compute_heat_budget(hfds: np.ndarray, hfls: np.ndarray, 
                       hfss: np.ndarray, rsntds: np.ndarray) -> np.ndarray:
    """
    Net heat flux into ocean.
    
    Q_net = SW_in + LW_net - Latent - Sensible
    
    Positive = warming
    Negative = cooling
    """
    return rsntds - hfls - hfss + hfds


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_day_of_year(dates) -> np.ndarray:
    """Extract day of year from datetime array."""
    import pandas as pd
    return pd.to_datetime(dates).dayofyear


def compute_seasonal_encoding(doy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode day-of-year as sin/cos for seasonal cycle.
    
    sin_doy: Peak at equinoxes (Mar 21, Sep 21)
    cos_doy: Peak at solstices (Jun 21, Dec 21)
    
    Returns:
        sin_doy, cos_doy: Seasonal encodings
    """
    phase = 2 * np.pi * doy / 365.25
    return np.sin(phase), np.cos(phase)


def compute_temporal_derivative(var: np.ndarray, dt: float = 1.0,
                               axis: int = -1) -> np.ndarray:
    """
    Temporal derivative: ∂var/∂t
    
    Args:
        var: Variable array with time as last dimension
        dt: Time step (days)
        axis: Time axis
        
    Returns:
        dvar/dt: Rate of change (units/day)
    """
    return np.gradient(var, dt, axis=axis)


def compute_bloom_onset(chl: np.ndarray, 
                       dchl_dt: np.ndarray,
                       dchl_threshold: float = 0.1) -> np.ndarray:
    """
    Bloom onset indicator: rapid positive change in Chl.
    
    Args:
        chl: Chlorophyll concentration
        dchl_dt: Chlorophyll rate of change
        dchl_threshold: Threshold for "rapid" change
        
    Returns:
        onset: Boolean bloom onset indicator
    """
    return (dchl_dt > dchl_threshold) & (chl > 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAPH-BASED SPATIAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_gradient_graph(values: np.ndarray,
                                   edge_index: np.ndarray,
                                   coords: np.ndarray) -> np.ndarray:
    """
    Compute spatial gradient magnitude for graph nodes.
    
    Uses finite differences along edges.
    
    Args:
        values: Node values (n_nodes,)
        edge_index: Edge connectivity (2, n_edges)
        coords: Node coordinates (n_nodes, 2) as [lon, lat]
        
    Returns:
        grad_magnitude: Gradient magnitude at each node
    """
    src, dst = edge_index
    
    # Value differences
    dval = values[dst] - values[src]
    
    # Distance (Haversine approximation)
    R = 6371  # km
    lat1, lat2 = np.radians(coords[src, 1]), np.radians(coords[dst, 1])
    dlon = np.radians(coords[dst, 0] - coords[src, 0])
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist = 2 * R * np.arcsin(np.sqrt(a))  # km
    dist = np.maximum(dist, 0.1)  # Avoid division by zero
    
    # Gradient per edge
    grad = np.abs(dval) / dist
    
    # Aggregate to nodes (max gradient)
    grad_node = np.zeros_like(values)
    np.maximum.at(grad_node, src, grad)
    np.maximum.at(grad_node, dst, grad)
    
    return grad_node


def compute_front_indicator(grad_T: np.ndarray, grad_chl: np.ndarray,
                           threshold_T: float = 0.1,
                           threshold_chl: float = 0.05) -> np.ndarray:
    """
    Front detection from temperature and chlorophyll gradients.
    
    Fronts are where both gradients are high.
    
    Returns:
        front: Front indicator [0, 1]
    """
    T_front = grad_T > threshold_T
    chl_front = grad_chl > threshold_chl
    return (T_front & chl_front).astype(float)


# =============================================================================
# FEATURE ENGINEERING CLASS
# =============================================================================

class DerivedFeatureEngine:
    """
    Compute all derived features from raw CMEMS variables.
    
    Usage:
        engine = DerivedFeatureEngine()
        features = engine.compute_all_features(raw_data)
    """
    
    # Feature categories
    STRATIFICATION_FEATURES = [
        'density', 'N2', 'thermal_stratification', 'mld_anomaly'
    ]
    
    CURRENT_FEATURES = [
        'current_speed', 'current_sin', 'current_cos',
        'geostrophic_u', 'geostrophic_v'
    ]
    
    BGC_FEATURES = [
        'np_ratio', 'np_deviation', 'chl_c_ratio', 'o2_saturation'
    ]
    
    LIGHT_FEATURES = [
        'par', 'light_limitation', 'temp_limitation', 'heat_budget'
    ]
    
    TEMPORAL_FEATURES = [
        'sin_doy', 'cos_doy', 'dchl_dt', 'dT_dt'
    ]
    
    def __init__(self, 
                 include_categories: List[str] = None,
                 protected_features: List[str] = None):
        """
        Args:
            include_categories: Which feature categories to include
                Options: 'stratification', 'current', 'bgc', 'light', 'temporal'
                Default: all
            protected_features: Features that should never be dropped
                (for PSO feature selection)
        """
        self.categories = include_categories or [
            'stratification', 'current', 'bgc', 'light', 'temporal'
        ]
        
        # Protected features for PSO (physics-critical)
        self.protected = protected_features or [
            'chl', 'thetao', 'so', 'no3', 'mlotst',
            'N2', 'current_speed', 'np_ratio', 'sin_doy', 'cos_doy'
        ]
    
    def compute_all_features(self, 
                            raw_data: Dict[str, np.ndarray],
                            doy: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute all derived features from raw CMEMS data.
        
        Args:
            raw_data: Dictionary with raw variables:
                - thetao, so, uo, vo, mlotst, zos
                - no3, po4, chl, phyc, o2
                - rsntds, hfds, hfls, hfss
            doy: Day of year (optional)
            
        Returns:
            features: Dictionary of all derived features
        """
        features = {}
        
        # Copy raw features
        for key, val in raw_data.items():
            features[key] = val
        
        # ═══ STRATIFICATION ═══
        if 'stratification' in self.categories:
            if 'thetao' in raw_data and 'so' in raw_data:
                features['density'] = compute_density(
                    raw_data['thetao'], raw_data['so']
                )
            
            if 'thetao' in raw_data and 'bottomT' in raw_data:
                features['thermal_stratification'] = compute_thermal_stratification(
                    raw_data['thetao'], raw_data['bottomT']
                )
            
            if 'mlotst' in raw_data:
                features['mld_anomaly'] = compute_mld_anomaly(raw_data['mlotst'])
        
        # ═══ CURRENTS ═══
        if 'current' in self.categories:
            if 'uo' in raw_data and 'vo' in raw_data:
                features['current_speed'] = compute_current_speed(
                    raw_data['uo'], raw_data['vo']
                )
                sin_dir, cos_dir = compute_current_direction_sincos(
                    raw_data['uo'], raw_data['vo']
                )
                features['current_sin'] = sin_dir
                features['current_cos'] = cos_dir
        
        # ═══ BGC INTERACTIONS ═══
        if 'bgc' in self.categories:
            if 'no3' in raw_data and 'po4' in raw_data:
                features['np_ratio'] = compute_np_ratio(
                    raw_data['no3'], raw_data['po4']
                )
                features['np_deviation'] = compute_np_deviation(
                    raw_data['no3'], raw_data['po4']
                )
            
            if 'chl' in raw_data and 'phyc' in raw_data:
                features['chl_c_ratio'] = compute_chl_c_ratio(
                    raw_data['chl'], raw_data['phyc']
                )
        
        # ═══ LIGHT & THERMAL ═══
        if 'light' in self.categories:
            if 'rsntds' in raw_data:
                features['par'] = compute_par_from_shortwave(raw_data['rsntds'])
                features['light_limitation'] = compute_light_limitation(features['par'])
            
            if 'thetao' in raw_data:
                features['temp_limitation'] = compute_temperature_limitation(
                    raw_data['thetao']
                )
            
            if all(k in raw_data for k in ['hfds', 'hfls', 'hfss', 'rsntds']):
                features['heat_budget'] = compute_heat_budget(
                    raw_data['hfds'], raw_data['hfls'],
                    raw_data['hfss'], raw_data['rsntds']
                )
        
        # ═══ TEMPORAL ═══
        if 'temporal' in self.categories:
            if doy is not None:
                sin_doy, cos_doy = compute_seasonal_encoding(doy)
                features['sin_doy'] = sin_doy
                features['cos_doy'] = cos_doy
            
            if 'chl' in raw_data and raw_data['chl'].ndim > 1:
                features['dchl_dt'] = compute_temporal_derivative(raw_data['chl'])
            
            if 'thetao' in raw_data and raw_data['thetao'].ndim > 1:
                features['dT_dt'] = compute_temporal_derivative(raw_data['thetao'])
        
        return features
    
    def get_feature_tensor(self, 
                          features: Dict[str, np.ndarray],
                          feature_names: List[str] = None,
                          device: str = 'cpu') -> Tuple[torch.Tensor, List[str]]:
        """
        Stack features into a tensor for model input.
        
        Args:
            features: Dictionary of features
            feature_names: Which features to include (default: all)
            device: PyTorch device
            
        Returns:
            tensor: Feature tensor (n_nodes, n_features)
            names: List of feature names
        """
        if feature_names is None:
            feature_names = list(features.keys())
        
        # Filter to 1D arrays (node features)
        valid_features = []
        valid_names = []
        
        for name in feature_names:
            if name in features:
                arr = features[name]
                if arr.ndim == 1:
                    valid_features.append(arr)
                    valid_names.append(name)
                elif arr.ndim == 2:
                    # Take first time step
                    valid_features.append(arr[:, 0])
                    valid_names.append(name)
        
        # Stack and convert to tensor
        stacked = np.stack(valid_features, axis=-1)
        tensor = torch.tensor(stacked, dtype=torch.float32, device=device)
        
        return tensor, valid_names
    
    def get_protected_indices(self, feature_names: List[str]) -> List[int]:
        """
        Get indices of protected features for PSO.
        
        These features should never be dropped during feature selection.
        """
        return [i for i, name in enumerate(feature_names) 
                if name in self.protected]


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_feature_matrix(raw_data: Dict[str, np.ndarray],
                         doy: Optional[np.ndarray] = None,
                         device: str = 'cpu') -> Tuple[torch.Tensor, List[str], List[int]]:
    """
    Create feature matrix from raw CMEMS data.
    
    Returns:
        X: Feature tensor (n_nodes, n_features)
        feature_names: List of feature names
        protected_idx: Indices of physics-critical features
    """
    engine = DerivedFeatureEngine()
    features = engine.compute_all_features(raw_data, doy)
    X, names = engine.get_feature_tensor(features, device=device)
    protected_idx = engine.get_protected_indices(names)
    
    return X, names, protected_idx


# =============================================================================
# FEATURE SUMMARY
# =============================================================================

FEATURE_CATALOG = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DERIVED FEATURE CATALOG                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STRATIFICATION (4 features)                                                 ║
║  ├── density          ρ(T,S) using simplified EOS for Black Sea              ║
║  ├── N2               Buoyancy frequency squared (stratification strength)   ║
║  ├── thermal_strat    T_surface - T_bottom                                   ║
║  └── mld_anomaly      MLD - climatological MLD                               ║
║                                                                              ║
║  CURRENTS (5 features)                                                       ║
║  ├── current_speed    √(u² + v²)                                             ║
║  ├── current_sin      sin(direction) - continuous encoding                   ║
║  ├── current_cos      cos(direction) - continuous encoding                   ║
║  ├── geostrophic_u    u_g from SSH gradient                                  ║
║  └── geostrophic_v    v_g from SSH gradient                                  ║
║                                                                              ║
║  BIOGEOCHEMISTRY (4 features)                                                ║
║  ├── np_ratio         N:P ratio (Redfield = 16)                              ║
║  ├── np_deviation     log(N:P / 16) - nutrient limitation                    ║
║  ├── chl_c_ratio      Chl:C - photoacclimation state                         ║
║  └── o2_saturation    O₂ / O₂_equilibrium × 100%                             ║
║                                                                              ║
║  LIGHT & THERMAL (4 features)                                                ║
║  ├── par              Photosynthetically Active Radiation                    ║
║  ├── light_limitation Smith light limitation I/√(I²+Ik²)                     ║
║  ├── temp_limitation  Eppley temperature function                            ║
║  └── heat_budget      Net heat flux into ocean                               ║
║                                                                              ║
║  TEMPORAL (4 features)                                                       ║
║  ├── sin_doy          sin(2π·doy/365) - seasonal cycle                       ║
║  ├── cos_doy          cos(2π·doy/365) - seasonal cycle                       ║
║  ├── dchl_dt          ∂Chl/∂t - bloom dynamics                               ║
║  └── dT_dt            ∂T/∂t - thermal evolution                              ║
║                                                                              ║
║  PROTECTED (PSO never removes):                                              ║
║  chl, thetao, so, no3, mlotst, N2, current_speed, np_ratio, sin_doy, cos_doy ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print("Feature Engineering Module for Black Sea Hybrid-FluxGNN")
    print(FEATURE_CATALOG)
    
    # Quick test with synthetic data
    print("\nTesting with synthetic data...")
    
    raw_data = {
        'thetao': np.random.rand(978) * 10 + 10,
        'so': np.random.rand(978) * 2 + 17,
        'uo': np.random.rand(978) * 0.2 - 0.1,
        'vo': np.random.rand(978) * 0.2 - 0.1,
        'no3': np.random.rand(978) * 5,
        'po4': np.random.rand(978) * 0.5,
        'chl': np.random.rand(978) * 2,
        'phyc': np.random.rand(978) * 100,
        'mlotst': np.random.rand(978) * 50 + 10,
    }
    
    engine = DerivedFeatureEngine()
    features = engine.compute_all_features(raw_data, doy=np.full(978, 180))
    
    print(f"\n✅ Generated {len(features)} features:")
    for i, name in enumerate(sorted(features.keys())):
        arr = features[name]
        print(f"   {i+1:2d}. {name:20s} - shape: {arr.shape}, "
              f"range: [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")
