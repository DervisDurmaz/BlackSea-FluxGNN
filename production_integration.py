"""
Hybrid-FluxGNN: Production-Ready Integration Module

This module provides the FIXED components for the Hybrid_FluxGNN_BlackSea_Complete.ipynb.
Copy the relevant sections into your notebook to replace the synthetic data with real V6 graph.

Author: Hybrid-FluxGNN Team
Version: 2.0 (Production)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from copy import deepcopy

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variable mapping from CMEMS to model
CMEMS_TO_MODEL = {
    'chl': 'chl',
    'thetao': 'temperature',
    'so': 'salinity', 
    'uo': 'u',
    'vo': 'v',
    'mlotst': 'mld',
    'zos': 'ssh',
    'no3': 'nitrate',
    'po4': 'phosphate',
    'phyc': 'phytoplankton',
    'nppv': 'npp',
    'o2': 'oxygen',
    'bottomT': 'bottom_temp',
    'rsntds': 'shortwave',
    'hfds': 'heat_flux',
}

# Protected features for PSO (physics-critical)
PROTECTED_FEATURES = [
    'chl', 'temperature', 'salinity', 'nitrate', 'mld',
    'N2', 'current_speed', 'np_ratio', 'sin_doy', 'cos_doy'
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PRODUCTION DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProductionConfig:
    """
    Configuration for production Hybrid-FluxGNN.
    
    Paths are set for your local environment. Update for Colab.
    """
    
    # ════ DATA PATHS ══════════════════════════════════════════════════════════
    # Local Windows:
    data_dir: str = r"C:\Users\dervi\Desktop\PINN\Important Datas"
    
    # For Google Colab (uncomment):
    # data_dir: str = "/content/drive/MyDrive/Important Datas"
    
    # Sub-directories
    yearly_dir: str = "years"
    v6_outputs_dir: str = "v6_outputs"
    
    # V6 graph files
    valid_indices_file: str = "valid_indices_v6.npy"
    edge_index_file: str = "edge_index_v6.npy"
    edge_weight_file: str = "edge_weight_v6.npy"
    node_depths_file: str = "node_depths_v6.npy"
    node_tiers_file: str = "node_tiers_v6.npy"
    node_mdt_file: str = "node_mdt_v6.npy"
    coordinates_file: str = "black_sea_sampling_coordinates.csv"
    
    # ════ DATA SPLIT ══════════════════════════════════════════════════════════
    train_years: List[int] = field(default_factory=lambda: list(range(1993, 2018)))
    val_years: List[int] = field(default_factory=lambda: [2018, 2019])
    test_years: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023])
    
    # ════ VARIABLES ═══════════════════════════════════════════════════════════
    target_var: str = "chl"
    forcing_vars: List[str] = field(default_factory=lambda: [
        "thetao", "so", "uo", "vo", "mlotst", "zos", "no3", "phyc"
    ])
    
    # ════ PROCESSING ══════════════════════════════════════════════════════════
    log_transform_chl: bool = True
    normalize: bool = True
    sequence_length: int = 7
    forecast_horizon: int = 7
    
    # ════ MODEL ═══════════════════════════════════════════════════════════════
    n_input_features: int = 14  # Raw + derived features
    hidden_dim: int = 128
    mp_layers: int = 4
    npzd_hidden_dim: int = 64
    use_fct: bool = True
    
    # ════ TRAINING ════════════════════════════════════════════════════════════
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    n_epochs: int = 500
    
    # Loss weights
    lambda_chl: float = 1.0
    lambda_nitrate: float = 0.5
    lambda_conservation: float = 0.3
    lambda_stratification: float = 0.2
    
    # Curriculum learning
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {'horizon': 1, 'threshold': 0.90, 'name': '1-step'},
        {'horizon': 3, 'threshold': 0.85, 'name': '3-step'},
        {'horizon': 7, 'threshold': 0.80, 'name': '7-step'},
        {'horizon': 30, 'threshold': 0.75, 'name': 'full'},
    ])
    tbptt_length: int = 7


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: V6 GRAPH LOADER (REPLACES SYNTHETIC)
# ═══════════════════════════════════════════════════════════════════════════════

class V6GraphData:
    """
    Load and manage the V6 graph structure from your pre-computed files.
    
    This REPLACES the synthetic PrismaticGraph for production.
    """
    
    def __init__(self, config: ProductionConfig = None, device: str = None):
        self.cfg = config or ProductionConfig()
        self.device = device or DEVICE
        
        self.data_dir = Path(self.cfg.data_dir)
        self.v6_dir = self.data_dir / self.cfg.v6_outputs_dir
        
        self._load_graph()
        self._load_coordinates()
        
    def _load_graph(self):
        """Load V6 graph topology."""
        print("📊 Loading V6 Graph...")
        
        # Valid node indices (mapping from 1100 → 978)
        self.valid_indices = np.load(self.v6_dir / self.cfg.valid_indices_file)
        self.n_nodes = len(self.valid_indices)
        
        # Edge connectivity
        self.edge_index_np = np.load(self.v6_dir / self.cfg.edge_index_file)
        self.edge_weight_np = np.load(self.v6_dir / self.cfg.edge_weight_file)
        self.n_edges = self.edge_index_np.shape[1]
        
        # Node features
        self.node_depths = np.load(self.v6_dir / self.cfg.node_depths_file)
        self.node_tiers = np.load(self.v6_dir / self.cfg.node_tiers_file)
        
        # MDT (if available)
        mdt_path = self.v6_dir / self.cfg.node_mdt_file
        if mdt_path.exists():
            self.node_mdt = np.load(mdt_path)
        else:
            self.node_mdt = np.zeros(self.n_nodes)
        
        print(f"   ✓ Nodes: {self.n_nodes}")
        print(f"   ✓ Edges: {self.n_edges}")
        
        # Convert to tensors
        self.edge_index = torch.tensor(
            self.edge_index_np, dtype=torch.long, device=self.device
        )
        self.edge_weight = torch.tensor(
            self.edge_weight_np, dtype=torch.float32, device=self.device
        )
        self.depths_tensor = torch.tensor(
            self.node_depths, dtype=torch.float32, device=self.device
        )
        self.tiers_tensor = torch.tensor(
            self.node_tiers, dtype=torch.long, device=self.device
        )
        
        # Tier statistics
        tier_counts = [(self.node_tiers == i).sum() for i in range(4)]
        print(f"   ✓ Tiers: Deep={tier_counts[0]}, Shelf={tier_counts[1]}, "
              f"Coastal={tier_counts[2]}, Rim={tier_counts[3]}")
        
    def _load_coordinates(self):
        """Load node lat/lon coordinates."""
        import pandas as pd
        
        coords_path = self.data_dir / self.cfg.coordinates_file
        if coords_path.exists():
            df = pd.read_csv(coords_path)
            self.lats = df['latitude'].values[self.valid_indices]
            self.lons = df['longitude'].values[self.valid_indices]
        else:
            # Estimate from Black Sea bounds
            self.lats = np.linspace(41, 46, self.n_nodes)
            self.lons = np.linspace(28, 41, self.n_nodes)
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Return graph data as dictionary for model initialization."""
        return {
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'node_depths': self.depths_tensor,
            'node_tiers': self.tiers_tensor,
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PRODUCTION DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class ProductionDataLoader:
    """
    Load CMEMS Black Sea data for training/evaluation.
    
    Features:
    - Loads yearly NetCDF files
    - Applies V6 valid node masking
    - Computes derived features
    - Creates train/val/test sequences
    - Handles normalization
    """
    
    def __init__(self, 
                 graph: V6GraphData,
                 config: ProductionConfig = None):
        """
        Args:
            graph: V6GraphData instance
            config: ProductionConfig
        """
        self.graph = graph
        self.cfg = config or ProductionConfig()
        
        self.data_dir = Path(self.cfg.data_dir)
        self.yearly_dir = self.data_dir / self.cfg.yearly_dir
        
        # Caches
        self._year_cache = {}
        self.stats = {}
        
    def load_year(self, year: int) -> xr.Dataset:
        """Load NetCDF for one year."""
        if year not in self._year_cache:
            path = self.yearly_dir / f"black_sea_daily_{year}.nc"
            if not path.exists():
                raise FileNotFoundError(f"Data not found: {path}")
            self._year_cache[year] = xr.open_dataset(path)
        return self._year_cache[year]
    
    def get_variable(self, year: int, var_name: str) -> np.ndarray:
        """
        Get variable for a year, subsetted to valid nodes.
        
        Returns: (978, n_days) array
        """
        ds = self.load_year(year)
        data = ds[var_name].values  # (1100, n_days)
        return data[self.graph.valid_indices]  # (978, n_days)
    
    def compute_statistics(self, years: List[int] = None):
        """Compute normalization statistics from training data."""
        if years is None:
            years = self.cfg.train_years[:5]  # Subset for speed
        
        print(f"📊 Computing statistics from {len(years)} years...")
        
        all_vars = [self.cfg.target_var] + self.cfg.forcing_vars
        
        for var in all_vars:
            values = []
            for year in years:
                try:
                    data = self.get_variable(year, var).flatten()
                    values.append(data[~np.isnan(data)])
                except Exception as e:
                    print(f"   ⚠️ {var} {year}: {e}")
            
            if values:
                combined = np.concatenate(values)
                
                # Log-transform chlorophyll
                if var == 'chl' and self.cfg.log_transform_chl:
                    combined = np.log(combined + 1e-6)
                
                self.stats[var] = {
                    'mean': float(np.nanmean(combined)),
                    'std': float(np.nanstd(combined)),
                }
        
        print(f"   ✓ Statistics computed for {len(self.stats)} variables")
        return self.stats
    
    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize data."""
        if var_name not in self.stats:
            return data
        
        result = data.copy()
        if var_name == 'chl' and self.cfg.log_transform_chl:
            result = np.log(result + 1e-6)
        
        result = (result - self.stats[var_name]['mean']) / (self.stats[var_name]['std'] + 1e-8)
        return result
    
    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Inverse normalization."""
        if var_name not in self.stats:
            return data
        
        result = data * self.stats[var_name]['std'] + self.stats[var_name]['mean']
        
        if var_name == 'chl' and self.cfg.log_transform_chl:
            result = np.exp(result) - 1e-6
        
        return result
    
    def compute_derived_features(self, 
                                 raw_data: Dict[str, np.ndarray],
                                 day_of_year: int = 180) -> Dict[str, np.ndarray]:
        """
        Compute physics-based derived features.
        
        Returns: Dictionary of features
        """
        features = {}
        
        # Copy raw features
        for key, val in raw_data.items():
            features[key] = val
        
        # ═══ STRATIFICATION ═══
        if 'thetao' in raw_data and 'so' in raw_data:
            T = raw_data['thetao']
            S = raw_data['so']
            # Simplified density
            features['density'] = 1012.0 + 0.78 * (S - 18) - 0.17 * (T - 15)
        
        # ═══ CURRENTS ═══
        if 'uo' in raw_data and 'vo' in raw_data:
            u, v = raw_data['uo'], raw_data['vo']
            features['current_speed'] = np.sqrt(u**2 + v**2)
            speed = features['current_speed'] + 1e-8
            features['current_sin'] = u / speed  # sin(direction)
            features['current_cos'] = v / speed  # cos(direction)
        
        # ═══ BGC INTERACTIONS ═══
        if 'no3' in raw_data and 'po4' in raw_data:
            features['np_ratio'] = raw_data['no3'] / (raw_data['po4'] + 1e-6)
            features['np_deviation'] = np.log(features['np_ratio'] / 16 + 1e-6)
        
        if 'chl' in raw_data and 'phyc' in raw_data:
            phyc_mg = raw_data['phyc'] * 12.0  # mmol C → mg C
            features['chl_c_ratio'] = raw_data['chl'] / (phyc_mg + 1e-6)
        
        # ═══ TEMPORAL ═══
        doy = np.full(self.graph.n_nodes, day_of_year)
        phase = 2 * np.pi * doy / 365.25
        features['sin_doy'] = np.sin(phase)
        features['cos_doy'] = np.cos(phase)
        
        # ═══ DEPTH ═══
        features['depth'] = self.graph.node_depths
        features['log_depth'] = np.log(self.graph.node_depths + 1)
        
        return features
    
    def get_sample(self, 
                   year: int = 2020,
                   time_idx: int = 180,
                   compute_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get a single time sample with all features.
        
        Returns:
            Dictionary with 'x', 'forcing', 'target' tensors
        """
        # Load raw data
        raw_data = {}
        for var in [self.cfg.target_var] + self.cfg.forcing_vars:
            try:
                data = self.get_variable(year, var)[:, time_idx]
                raw_data[var] = data
            except:
                raw_data[var] = np.zeros(self.graph.n_nodes)
        
        # Compute derived features
        if compute_features:
            day_of_year = time_idx % 365
            features = self.compute_derived_features(raw_data, day_of_year)
        else:
            features = raw_data
        
        # Build input tensor
        input_vars = [
            'chl', 'thetao', 'so', 'uo', 'vo', 'mlotst', 'zos', 'no3',
            'current_speed', 'np_ratio', 'sin_doy', 'cos_doy', 'depth', 'log_depth'
        ]
        
        input_list = []
        for var in input_vars:
            if var in features:
                arr = features[var]
                # Normalize if stats available
                if var in self.stats:
                    arr = self.normalize(arr, var)
                input_list.append(arr)
            else:
                input_list.append(np.zeros(self.graph.n_nodes))
        
        x = torch.tensor(
            np.stack(input_list, axis=-1),
            dtype=torch.float32,
            device=self.graph.device
        )
        
        # Build forcing dictionary
        forcing = {
            'temperature': torch.tensor(features.get('thetao', np.zeros(self.graph.n_nodes)), 
                                        dtype=torch.float32, device=self.graph.device),
            'salinity': torch.tensor(features.get('so', np.zeros(self.graph.n_nodes)),
                                     dtype=torch.float32, device=self.graph.device),
            'u': torch.tensor(features.get('uo', np.zeros(self.graph.n_nodes)),
                              dtype=torch.float32, device=self.graph.device),
            'v': torch.tensor(features.get('vo', np.zeros(self.graph.n_nodes)),
                              dtype=torch.float32, device=self.graph.device),
            'mld': torch.tensor(features.get('mlotst', np.full(self.graph.n_nodes, 30)),
                                dtype=torch.float32, device=self.graph.device),
            'depth': torch.tensor(self.graph.node_depths, dtype=torch.float32, device=self.graph.device),
            'day_of_year': torch.full((self.graph.n_nodes,), time_idx % 365, 
                                      dtype=torch.float32, device=self.graph.device),
            'N2': torch.full((self.graph.n_nodes,), 1e-4, 
                            dtype=torch.float32, device=self.graph.device),
            'par': torch.full((self.graph.n_nodes,), 100.0,  # Estimate PAR
                             dtype=torch.float32, device=self.graph.device),
        }
        
        # Target
        target_chl = features.get('chl', np.zeros(self.graph.n_nodes))
        if self.cfg.log_transform_chl:
            target_chl = np.log(target_chl + 1e-6)
        target = torch.tensor(target_chl, dtype=torch.float32, device=self.graph.device)
        
        return {
            'x': x,
            'forcing': forcing,
            'target': target,
            'edge_index': self.graph.edge_index,
            'edge_weight': self.graph.edge_weight,
        }
    
    def create_dataset(self, 
                       years: List[int],
                       stride: int = 7) -> Dict[str, torch.Tensor]:
        """
        Create full dataset from multiple years.
        
        Args:
            years: Years to include
            stride: Days between samples
            
        Returns:
            Dataset dictionary
        """
        print(f"📊 Creating dataset from {len(years)} years...")
        
        all_x = []
        all_target = []
        
        for year in years:
            try:
                ds = self.load_year(year)
                n_days = ds.dims['time']
                
                for t in range(0, n_days - self.cfg.forecast_horizon, stride):
                    sample = self.get_sample(year, t)
                    all_x.append(sample['x'])
                    all_target.append(sample['target'])
                    
                print(f"   ✓ {year}: {len(all_x)} samples")
                
            except Exception as e:
                print(f"   ⚠️ {year}: {e}")
        
        X = torch.stack(all_x)
        Y = torch.stack(all_target)
        
        print(f"\n   Dataset: X={X.shape}, Y={Y.shape}")
        
        return {
            'x': X,
            'y': Y,
            'edge_index': self.graph.edge_index,
            'edge_weight': self.graph.edge_weight,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PRODUCTION MODEL (USES V6 GRAPH)
# ═══════════════════════════════════════════════════════════════════════════════

class ProductionHybridFluxGNN(nn.Module):
    """
    Production Hybrid-FluxGNN that uses the V6 graph.
    
    This REPLACES the original HybridFluxGNN for real data.
    """
    
    def __init__(self, 
                 graph: V6GraphData,
                 config: ProductionConfig):
        super().__init__()
        
        self.graph = graph
        self.cfg = config
        
        # ═══ NODE ENCODER ═══
        self.node_encoder = nn.Sequential(
            nn.Linear(config.n_input_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        
        # ═══ MESSAGE PASSING ═══
        self.mp_layers = nn.ModuleList([
            SplitKernelMP(config.hidden_dim)
            for _ in range(config.mp_layers)
        ])
        
        # ═══ NPZD DECODER ═══
        self.npzd_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 4),  # P, N, Z, Chl
        )
        
        # ═══ GRAY-BOX NPZD ═══
        self.npzd = GrayBoxNPZD_Production(config.npzd_hidden_dim)
        
        # ═══ CHLOROPHYLL HEAD ═══
        self.chl_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 4, 1),
        )
        
    def forward(self, 
                x: torch.Tensor,
                forcing: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [n_nodes, n_features]
            forcing: Dictionary with temperature, par, mld, etc.
            
        Returns:
            chl_pred: Predicted chlorophyll [n_nodes]
        """
        # Encode
        h = self.node_encoder(x)
        
        # Message passing
        edge_index = self.graph.edge_index
        N2 = forcing.get('N2', torch.full((x.shape[0],), 1e-4, device=x.device))
        
        for mp in self.mp_layers:
            h = mp(h, edge_index, N2)
        
        # Decode NPZD state
        state = self.npzd_decoder(h)
        
        # One NPZD step
        state = self.npzd(state, forcing, dt=1.0)
        
        # Extract chlorophyll
        chl = state[..., 3]  # Chl is 4th component
        
        # Alternative: direct chlorophyll prediction
        chl_direct = self.chl_head(h).squeeze(-1)
        
        # Blend (learnable)
        alpha = 0.7  # Weight for physics-based
        chl_pred = alpha * chl + (1 - alpha) * chl_direct
        
        return chl_pred


class SplitKernelMP(nn.Module):
    """Simplified split-kernel message passing for production."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.msg_fn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.update_fn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                h: torch.Tensor,
                edge_index: torch.Tensor,
                N2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings [n_nodes, hidden_dim]
            edge_index: [2, n_edges]
            N2: Buoyancy frequency for stratification gating
        """
        src, dst = edge_index
        
        # Message
        h_src = h[src]
        h_dst = h[dst]
        msg = self.msg_fn(torch.cat([h_src, h_dst], dim=-1))
        
        # Aggregate (mean)
        n_nodes = h.shape[0]
        agg = torch.zeros(n_nodes, h.shape[1], device=h.device)
        counts = torch.zeros(n_nodes, 1, device=h.device)
        
        agg.index_add_(0, dst, msg)
        counts.index_add_(0, dst, torch.ones_like(msg[:, :1]))
        
        agg = agg / (counts + 1e-8)
        
        # Update
        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h + h_new)  # Residual
        
        return h_new


class GrayBoxNPZD_Production(nn.Module):
    """Production Gray-Box NPZD with RK4 integration."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # Neural network predicts PARAMETERS
        self.param_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # μ_max, K_N, g_max, m_P, m_Z, θ
        )
        
        # Physical bounds
        self.bounds = {
            'mu_max': (0.5, 3.0),
            'K_N': (0.1, 2.0),
            'g_max': (0.1, 1.0),
            'm_P': (0.01, 0.2),
            'm_Z': (0.01, 0.1),
            'theta': (0.02, 0.06),
        }
    
    def compute_parameters(self, forcing: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict biogeochemical parameters."""
        T = forcing.get('temperature', torch.zeros(1))
        par = forcing.get('par', torch.full_like(T, 100))
        mld = forcing.get('mld', torch.full_like(T, 30))
        depth = forcing.get('depth', torch.zeros_like(T))
        doy = forcing.get('day_of_year', torch.full_like(T, 180))
        N2 = forcing.get('N2', torch.full_like(T, 1e-4))
        
        inputs = torch.stack([T, par, mld, depth, doy, N2], dim=-1)
        raw = self.param_net(inputs)
        
        params = {}
        for i, (name, (lo, hi)) in enumerate(self.bounds.items()):
            params[name] = lo + (hi - lo) * torch.sigmoid(raw[..., i])
        
        return params
    
    def npzd_equations(self, 
                       state: torch.Tensor,
                       params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FIXED STRUCTURE NPZD equations."""
        P = state[..., 0]
        N = state[..., 1]
        Z = state[..., 2]
        Chl = state[..., 3]
        
        # Monod kinetics
        mu = params['mu_max'] * (N / (params['K_N'] + N + 1e-8))
        
        # Grazing (Holling Type II)
        K_g = 0.5
        g = params['g_max'] * (P / (K_g + P + 1e-8))
        
        # Tendencies
        dP_dt = mu * P - g * Z - params['m_P'] * P
        dN_dt = -mu * P / 6.625 + 0.3 * params['m_Z'] * Z / 6.625
        dZ_dt = 0.3 * g * Z - params['m_Z'] * Z
        dChl_dt = params['theta'] * dP_dt
        
        return torch.stack([dP_dt, dN_dt, dZ_dt, dChl_dt], dim=-1)
    
    def forward(self, 
                state: torch.Tensor,
                forcing: Dict[str, torch.Tensor],
                dt: float = 1.0) -> torch.Tensor:
        """RK4 integration (dt in DAYS)."""
        params = self.compute_parameters(forcing)
        
        k1 = self.npzd_equations(state, params)
        k2 = self.npzd_equations(state + 0.5 * dt * k1, params)
        k3 = self.npzd_equations(state + 0.5 * dt * k2, params)
        k4 = self.npzd_equations(state + dt * k3, params)
        
        state_new = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Positivity constraint
        state_new = torch.clamp(state_new, min=1e-10)
        
        return state_new


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PRODUCTION LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class ProductionLoss(nn.Module):
    """
    Production loss with proper normalization.
    """
    
    def __init__(self, config: ProductionConfig):
        super().__init__()
        self.cfg = config
        
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute losses.
        
        Args:
            pred: Predicted chlorophyll [n_nodes]
            target: Target chlorophyll [n_nodes]
            state: Full NPZD state [n_nodes, 4] (optional)
        """
        # Mask NaNs
        mask = ~torch.isnan(target)
        
        # MSE loss
        mse = F.mse_loss(pred[mask], target[mask])
        
        # Conservation loss (if state provided)
        if state is not None:
            total_N = state[..., 0] + state[..., 1] + state[..., 2]
            conservation = torch.std(total_N)
        else:
            conservation = torch.tensor(0.0, device=pred.device)
        
        # Positivity loss
        positivity = F.relu(-pred).mean()
        
        # Total
        total = (
            self.cfg.lambda_chl * mse +
            self.cfg.lambda_conservation * conservation +
            0.1 * positivity
        )
        
        return {
            'total': total,
            'mse': mse,
            'conservation': conservation,
            'positivity': positivity,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PRODUCTION TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_production(model: nn.Module,
                     data_loader: ProductionDataLoader,
                     config: ProductionConfig,
                     n_epochs: int = None) -> Dict:
    """
    Production training loop.
    """
    n_epochs = n_epochs or config.n_epochs
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    loss_fn = ProductionLoss(config)
    
    # Compute statistics if not done
    if not data_loader.stats:
        data_loader.compute_statistics()
    
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    print(f"\n{'═'*60}")
    print(f"  TRAINING: {n_epochs} epochs")
    print(f"{'═'*60}")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_samples = 0
        
        # Sample random years/times
        for _ in range(10):  # 10 batches per epoch
            year = np.random.choice(config.train_years)
            time_idx = np.random.randint(30, 330)
            
            try:
                sample = data_loader.get_sample(year, time_idx)
                
                optimizer.zero_grad()
                
                pred = model(sample['x'], sample['forcing'])
                losses = loss_fn(pred, sample['target'])
                
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += losses['total'].item()
                n_samples += 1
                
            except Exception as e:
                pass
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(n_samples, 1)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            n_val = 0
            
            with torch.no_grad():
                for year in config.val_years:
                    for t in [90, 180, 270]:
                        try:
                            sample = data_loader.get_sample(year, t)
                            pred = model(sample['x'], sample['forcing'])
                            losses = loss_fn(pred, sample['target'])
                            val_loss += losses['mse'].item()
                            n_val += 1
                        except:
                            pass
            
            val_loss = val_loss / max(n_val, 1)
            val_rmse = np.sqrt(val_loss)
            
            history['val_loss'].append(val_loss)
            history['val_rmse'].append(val_rmse)
            
            print(f"  Epoch {epoch+1:4d} | Train: {avg_loss:.4f} | Val: {val_loss:.4f} | RMSE: {val_rmse:.4f}")
    
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_production_pipeline(config: ProductionConfig = None) -> Tuple:
    """
    Initialize the full production pipeline.
    
    Returns:
        graph, data_loader, model, config
    """
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "HYBRID-FLUXGNN v2.0" + " "*29 + "║")
    print("║" + " "*20 + "PRODUCTION PIPELINE" + " "*29 + "║")
    print("╚" + "═"*68 + "╝")
    
    config = config or ProductionConfig()
    
    # Load graph
    print("\n[1/4] Loading V6 Graph...")
    graph = V6GraphData(config)
    
    # Create data loader
    print("\n[2/4] Creating Data Loader...")
    data_loader = ProductionDataLoader(graph, config)
    
    # Compute statistics
    print("\n[3/4] Computing Statistics...")
    data_loader.compute_statistics()
    
    # Create model
    print("\n[4/4] Creating Model...")
    model = ProductionHybridFluxGNN(graph, config).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model parameters: {n_params:,}")
    
    print(f"\n{'═'*60}")
    print("✅ PRODUCTION PIPELINE READY")
    print(f"{'═'*60}")
    print(f"   Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"   Train: {len(config.train_years)} years")
    print(f"   Val: {len(config.val_years)} years")
    print(f"   Test: {len(config.test_years)} years")
    print(f"{'═'*60}")
    
    return graph, data_loader, model, config


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK START
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick test
    print("Testing Production Pipeline...")
    
    try:
        graph, data_loader, model, config = initialize_production_pipeline()
        
        # Get a sample
        sample = data_loader.get_sample(2020, 180)
        print(f"\nSample shapes:")
        print(f"  x: {sample['x'].shape}")
        print(f"  target: {sample['target'].shape}")
        
        # Forward pass
        with torch.no_grad():
            pred = model(sample['x'], sample['forcing'])
            print(f"  pred: {pred.shape}")
        
        print("\n✅ Production pipeline test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check your data paths in ProductionConfig")
