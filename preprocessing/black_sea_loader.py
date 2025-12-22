"""
Black Sea Data Loader for Hybrid-FluxGNN.

Loads CMEMS Black Sea reanalysis data and prepares it for the GNN.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    print("Warning: xarray not available. Install with: pip install xarray netCDF4")


@dataclass
class BlackSeaDataConfig:
    """Configuration for Black Sea data loading."""
    
    # Data directories
    data_dir: str = r"C:\Users\dervi\Desktop\PINN\Important Datas"
    yearly_dir: str = "years"
    v6_outputs_dir: str = "v6_outputs"
    
    # Files
    full_nc: str = "black_sea_full_1993_2023.nc"
    coordinates_csv: str = "black_sea_sampling_coordinates.csv"
    
    # V6 graph files
    valid_indices_file: str = "valid_indices_v6.npy"
    edge_index_file: str = "edge_index_v6.npy"
    edge_weight_file: str = "edge_weight_v6.npy"
    node_tiers_file: str = "node_tiers_v6.npy"
    node_depths_file: str = "node_depths_v6.npy"
    node_mdt_file: str = "node_mdt_v6.npy"
    
    # Static files
    bathy_file: str = "cmems_mod_blk_phy_my_2.5km_static_1764749716275.nc"
    mdt_file: str = "cmems_mod_blk_phy_my_2.5km_static_1764749627687.nc"
    
    # Data split
    train_years: List[int] = field(default_factory=lambda: list(range(1993, 2018)))
    val_years: List[int] = field(default_factory=lambda: [2018, 2019])
    test_years: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023])
    
    # Variables
    target_var: str = "chl"  # Chlorophyll-a
    forcing_vars: List[str] = field(default_factory=lambda: [
        "thetao",   # Temperature
        "so",       # Salinity
        "uo",       # Eastward velocity
        "vo",       # Northward velocity
        "mlotst",   # Mixed layer depth
        "zos",      # Sea surface height
        "no3",      # Nitrate
        "phyc",     # Phytoplankton carbon
    ])
    
    # Processing
    log_transform_chl: bool = True
    normalize: bool = True
    sequence_length: int = 7  # Days for input sequence
    forecast_horizon: int = 7  # Days to predict


class BlackSeaDataLoader:
    """
    Load and preprocess Black Sea CMEMS data for Hybrid-FluxGNN.
    
    Features:
    - Load yearly NetCDF files or single merged file
    - Apply V6 graph valid indices
    - Create train/val/test splits by year (no leakage)
    - Generate sequences for time series prediction
    """
    
    def __init__(self, config: BlackSeaDataConfig = None):
        """Initialize data loader."""
        self.config = config or BlackSeaDataConfig()
        self.cfg = self.config
        
        # Paths
        self.data_dir = Path(self.cfg.data_dir)
        self.v6_dir = self.data_dir / self.cfg.v6_outputs_dir
        self.yearly_dir = self.data_dir / self.cfg.yearly_dir
        
        # Load graph structure
        self._load_graph()
        
        # Data cache
        self._data_cache = {}
        
        # Statistics for normalization
        self.stats = {}
        
    def _load_graph(self):
        """Load V6 graph structure."""
        print("📊 Loading V6 graph structure...")
        
        # Valid indices (map from original 1100 to valid 978)
        self.valid_indices = np.load(self.v6_dir / self.cfg.valid_indices_file)
        self.n_valid_nodes = len(self.valid_indices)
        print(f"   Valid nodes: {self.n_valid_nodes}")
        
        # Edge index and weights
        self.edge_index = np.load(self.v6_dir / self.cfg.edge_index_file)
        self.edge_weight = np.load(self.v6_dir / self.cfg.edge_weight_file)
        print(f"   Edges: {self.edge_index.shape[1]}")
        
        # Node features
        self.node_tiers = np.load(self.v6_dir / self.cfg.node_tiers_file)
        self.node_depths = np.load(self.v6_dir / self.cfg.node_depths_file)
        
        if (self.v6_dir / self.cfg.node_mdt_file).exists():
            self.node_mdt = np.load(self.v6_dir / self.cfg.node_mdt_file)
        else:
            self.node_mdt = np.zeros(self.n_valid_nodes)
        
        print(f"   ✓ Graph loaded")
        
    def load_year(self, year: int) -> xr.Dataset:
        """Load data for a single year."""
        if year in self._data_cache:
            return self._data_cache[year]
        
        # Try yearly file first
        yearly_file = self.yearly_dir / f"black_sea_daily_{year}.nc"
        
        if yearly_file.exists():
            ds = xr.open_dataset(yearly_file)
        else:
            # Fall back to merged file
            merged_file = self.data_dir / self.cfg.full_nc
            ds = xr.open_dataset(merged_file)
            ds = ds.sel(time=ds.time.dt.year == year)
        
        self._data_cache[year] = ds
        return ds
    
    def _subset_valid_nodes(self, data: np.ndarray) -> np.ndarray:
        """
        Subset data to valid nodes only.
        
        Args:
            data: Array with shape (n_nodes, ...) or (..., n_nodes)
            
        Returns:
            Subsetted array with valid nodes only
        """
        if data.shape[0] == 1100:
            return data[self.valid_indices]
        elif len(data.shape) > 1 and data.shape[-1] == 1100:
            return data[..., self.valid_indices]
        else:
            return data  # Already subsetted or unknown shape
    
    def get_variable(self, 
                     year: int, 
                     var_name: str,
                     subset_valid: bool = True) -> np.ndarray:
        """
        Get variable data for a year.
        
        Args:
            year: Year to load
            var_name: Variable name (e.g., 'chl', 'thetao')
            subset_valid: Whether to subset to valid nodes
            
        Returns:
            data: Array of shape (n_valid_nodes, n_time) or (n_nodes, n_time)
        """
        ds = self.load_year(year)
        
        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}")
        
        # Get values - shape is (node, time)
        data = ds[var_name].values
        
        if subset_valid:
            data = self._subset_valid_nodes(data)
        
        return data
    
    def compute_statistics(self, years: List[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute mean/std for normalization from training years.
        
        Args:
            years: Years to use (default: train_years)
            
        Returns:
            stats: {var_name: {'mean': float, 'std': float}}
        """
        if years is None:
            years = self.cfg.train_years
        
        print(f"📊 Computing statistics from {len(years)} years...")
        
        all_vars = [self.cfg.target_var] + self.cfg.forcing_vars
        
        for var in all_vars:
            all_values = []
            
            for year in years:
                try:
                    data = self.get_variable(year, var)
                    # Flatten and remove NaNs
                    valid = data[~np.isnan(data)]
                    all_values.append(valid)
                except Exception as e:
                    print(f"   Warning: Could not load {var} for {year}: {e}")
            
            if all_values:
                combined = np.concatenate(all_values)
                
                # Log transform chlorophyll
                if var == 'chl' and self.cfg.log_transform_chl:
                    combined = np.log(combined + 1e-6)
                
                self.stats[var] = {
                    'mean': float(np.nanmean(combined)),
                    'std': float(np.nanstd(combined)),
                    'min': float(np.nanmin(combined)),
                    'max': float(np.nanmax(combined)),
                }
                print(f"   {var}: mean={self.stats[var]['mean']:.4f}, std={self.stats[var]['std']:.4f}")
        
        return self.stats
    
    def normalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        Normalize data using pre-computed statistics.
        
        Args:
            data: Raw data
            var_name: Variable name
            
        Returns:
            Normalized data
        """
        if var_name not in self.stats:
            return data
        
        # Log transform chlorophyll
        if var_name == 'chl' and self.cfg.log_transform_chl:
            data = np.log(data + 1e-6)
        
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        
        return (data - mean) / (std + 1e-8)
    
    def denormalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Inverse of normalize_data."""
        if var_name not in self.stats:
            return data
        
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        
        result = data * std + mean
        
        # Inverse log transform
        if var_name == 'chl' and self.cfg.log_transform_chl:
            result = np.exp(result) - 1e-6
        
        return result
    
    def create_sequences(self,
                        years: List[int],
                        normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Create input/target sequences for training.
        
        Args:
            years: Years to use
            normalize: Whether to normalize data
            
        Returns:
            Dictionary with tensors:
            - 'x': Input features [n_samples, n_nodes, seq_len, n_features]
            - 'y': Targets [n_samples, n_nodes, horizon]
            - 'edge_index': Graph connectivity [2, n_edges]
            - 'edge_weight': Edge weights [n_edges]
        """
        print(f"📊 Creating sequences from {len(years)} years...")
        
        seq_len = self.cfg.sequence_length
        horizon = self.cfg.forecast_horizon
        
        all_x = []
        all_y = []
        
        for year in years:
            print(f"   Processing {year}...", end=" ")
            
            try:
                # Load target
                target = self.get_variable(year, self.cfg.target_var)
                if normalize:
                    target = self.normalize_data(target, self.cfg.target_var)
                
                # Load forcing
                forcing_list = []
                for var in self.cfg.forcing_vars:
                    try:
                        data = self.get_variable(year, var)
                        if normalize:
                            data = self.normalize_data(data, var)
                        forcing_list.append(data)
                    except:
                        # Missing variable - use zeros
                        forcing_list.append(np.zeros_like(target))
                
                # Stack features: [n_nodes, n_time, n_features]
                # Target + forcing
                all_features = np.stack([target] + forcing_list, axis=-1)
                n_nodes, n_time, n_features = all_features.shape
                
                # Create sliding windows
                for t in range(seq_len, n_time - horizon):
                    # Input: [n_nodes, seq_len, n_features]
                    x = all_features[:, t-seq_len:t, :]
                    
                    # Target: [n_nodes, horizon] - just chlorophyll
                    y = target[:, t:t+horizon]
                    
                    # Check for NaNs
                    if np.isnan(x).sum() == 0 and np.isnan(y).sum() == 0:
                        all_x.append(x)
                        all_y.append(y)
                
                print(f"✓ ({len(all_x)} samples so far)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # Stack and convert to tensors
        X = torch.tensor(np.stack(all_x), dtype=torch.float32)
        Y = torch.tensor(np.stack(all_y), dtype=torch.float32)
        
        print(f"\n📊 Created dataset:")
        print(f"   X shape: {X.shape} (samples, nodes, seq_len, features)")
        print(f"   Y shape: {Y.shape} (samples, nodes, horizon)")
        
        return {
            'x': X,
            'y': Y,
            'edge_index': torch.tensor(self.edge_index, dtype=torch.long),
            'edge_weight': torch.tensor(self.edge_weight, dtype=torch.float32),
            'node_depths': torch.tensor(self.node_depths, dtype=torch.float32),
            'node_tiers': torch.tensor(self.node_tiers, dtype=torch.long),
        }
    
    def get_train_val_test(self, normalize: bool = True) -> Tuple[Dict, Dict, Dict]:
        """
        Get train/val/test splits.
        
        Returns:
            train_data, val_data, test_data: Dictionaries with tensors
        """
        # Compute stats from training data
        if not self.stats:
            self.compute_statistics(self.cfg.train_years)
        
        print("\n" + "="*60)
        print("Creating Train/Val/Test Splits")
        print("="*60)
        
        train_data = self.create_sequences(self.cfg.train_years, normalize)
        print(f"\nTrain: {train_data['x'].shape[0]} samples from years {self.cfg.train_years[0]}-{self.cfg.train_years[-1]}")
        
        val_data = self.create_sequences(self.cfg.val_years, normalize)
        print(f"Val: {val_data['x'].shape[0]} samples from years {self.cfg.val_years}")
        
        test_data = self.create_sequences(self.cfg.test_years, normalize)
        print(f"Test: {test_data['x'].shape[0]} samples from years {self.cfg.test_years}")
        
        return train_data, val_data, test_data
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lat/lon coordinates for valid nodes."""
        import pandas as pd
        
        coords_file = self.data_dir / self.cfg.coordinates_csv
        df = pd.read_csv(coords_file)
        
        lats = df['latitude'].values[self.valid_indices]
        lons = df['longitude'].values[self.valid_indices]
        
        return lats, lons
    
    def to_pyg_data(self, data_dict: Dict) -> 'torch_geometric.data.Data':
        """
        Convert to PyTorch Geometric Data object.
        
        Args:
            data_dict: Output from create_sequences
            
        Returns:
            PyG Data object
        """
        from torch_geometric.data import Data
        
        # Take first sample for now (for single-graph setting)
        x = data_dict['x'][0]  # [n_nodes, seq_len, n_features]
        y = data_dict['y'][0]  # [n_nodes, horizon]
        
        # Flatten sequence dimension
        x_flat = x.reshape(x.shape[0], -1)  # [n_nodes, seq_len * n_features]
        
        return Data(
            x=x_flat,
            y=y,
            edge_index=data_dict['edge_index'],
            edge_attr=data_dict['edge_weight'].unsqueeze(-1),
            node_depths=data_dict['node_depths'],
            node_tiers=data_dict['node_tiers'],
        )


# =============================================================================
# Convenience functions
# =============================================================================

def load_black_sea_data(config: BlackSeaDataConfig = None) -> Tuple[Dict, Dict, Dict, BlackSeaDataLoader]:
    """
    Load Black Sea data with default configuration.
    
    Returns:
        train_data, val_data, test_data, loader
    """
    loader = BlackSeaDataLoader(config)
    train, val, test = loader.get_train_val_test()
    return train, val, test, loader


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing Black Sea Data Loader")
    print("="*60)
    
    # Create loader
    config = BlackSeaDataConfig()
    loader = BlackSeaDataLoader(config)
    
    # Test loading a year
    print("\nLoading 2020 data...")
    chl = loader.get_variable(2020, 'chl')
    print(f"Chlorophyll shape: {chl.shape}")
    print(f"Non-NaN values: {(~np.isnan(chl)).sum()}")
    
    # Get coordinates
    lats, lons = loader.get_coordinates()
    print(f"\nCoordinates: {len(lats)} nodes")
    print(f"Lat range: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"Lon range: [{lons.min():.2f}, {lons.max():.2f}]")
    
    print("\n✅ Data loader ready!")
