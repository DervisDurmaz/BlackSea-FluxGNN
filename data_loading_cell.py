# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING CELL FOR HYBRID-FLUXGNN
# ══════════════════════════════════════════════════════════════════════════════
#
# Copy this entire cell and add it after the imports section in the notebook.
# This loads YOUR real CMEMS Black Sea data and V6 graph structure.
#
# ══════════════════════════════════════════════════════════════════════════════

import xarray as xr
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

@dataclass
class BlackSeaDataConfig:
    """Configuration for Black Sea data loading."""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UPDATE THESE PATHS FOR YOUR ENVIRONMENT!
    # ═══════════════════════════════════════════════════════════════════════════
    
    # For Google Colab (upload to Drive first):
    # data_dir: str = "/content/drive/MyDrive/BlackSea_Data/Important Datas"
    
    # For Local Windows:
    data_dir: str = r"C:\Users\dervi\Desktop\PINN\Important Datas"
    
    yearly_dir: str = "years"
    v6_outputs_dir: str = "v6_outputs"
    
    # V6 graph files (from Black_Sea_V6_Graph_Construction.ipynb)
    valid_indices_file: str = "valid_indices_v6.npy"
    edge_index_file: str = "edge_index_v6.npy"
    edge_weight_file: str = "edge_weight_v6.npy"
    node_depths_file: str = "node_depths_v6.npy"
    node_tiers_file: str = "node_tiers_v6.npy"
    
    # Train/Val/Test split by year (prevents data leakage!)
    train_years: List[int] = field(default_factory=lambda: list(range(1993, 2018)))  # 25 years
    val_years: List[int] = field(default_factory=lambda: [2018, 2019])              # 2 years
    test_years: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023]) # 4 years
    
    # Variables (from CMEMS)
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
    sequence_length: int = 7   # Days of history
    forecast_horizon: int = 7  # Days to predict


class BlackSeaDataLoader:
    """
    Load CMEMS Black Sea data for Hybrid-FluxGNN.
    
    Features:
    - Load yearly NetCDF files from `Important Datas/years/`
    - Apply V6 valid node masking (1100 → 978 nodes)
    - Train/val/test by year (no data leakage)
    - Log-transform chlorophyll
    - Z-score normalization
    
    Usage:
        loader = BlackSeaDataLoader()
        graph = loader.get_graph_tensors()
        chl = loader.get_variable(2020, 'chl')
    """
    
    def __init__(self, config: BlackSeaDataConfig = None):
        self.cfg = config or BlackSeaDataConfig()
        self.data_dir = Path(self.cfg.data_dir)
        self.v6_dir = self.data_dir / self.cfg.v6_outputs_dir
        self.yearly_dir = self.data_dir / self.cfg.yearly_dir
        
        self._load_graph()
        self.stats = {}
        self._cache = {}
        
    def _load_graph(self):
        """Load V6 graph structure."""
        print("📊 Loading V6 Graph Structure...")
        
        # Valid indices (map from 1100 total to 978 valid nodes)
        self.valid_indices = np.load(self.v6_dir / self.cfg.valid_indices_file)
        self.n_valid_nodes = len(self.valid_indices)
        print(f"   Valid nodes: {self.n_valid_nodes}")
        
        # Graph topology
        self.edge_index = np.load(self.v6_dir / self.cfg.edge_index_file)
        self.edge_weight = np.load(self.v6_dir / self.cfg.edge_weight_file)
        print(f"   Edges: {self.edge_index.shape[1]}")
        
        # Node features
        self.node_depths = np.load(self.v6_dir / self.cfg.node_depths_file)
        self.node_tiers = np.load(self.v6_dir / self.cfg.node_tiers_file)
        
        # Tier statistics
        tier_counts = [int((self.node_tiers == i).sum()) for i in range(4)]
        print(f"   Tiers: Deep={tier_counts[0]}, Shelf={tier_counts[1]}, "
              f"Coastal={tier_counts[2]}, Rim={tier_counts[3]}")
        
        print(f"   ✅ Graph loaded successfully")
    
    def load_year(self, year: int) -> xr.Dataset:
        """Load NetCDF data for one year."""
        if year not in self._cache:
            yearly_file = self.yearly_dir / f"black_sea_daily_{year}.nc"
            if not yearly_file.exists():
                raise FileNotFoundError(f"File not found: {yearly_file}")
            self._cache[year] = xr.open_dataset(yearly_file)
        return self._cache[year]
    
    def get_variable(self, year: int, var_name: str) -> np.ndarray:
        """
        Get variable for a year, subsetted to valid nodes.
        
        Args:
            year: Year (1993-2023)
            var_name: Variable name ('chl', 'thetao', 'no3', etc.)
            
        Returns:
            data: Array of shape (978, n_days)
        """
        ds = self.load_year(year)
        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found. "
                           f"Available: {list(ds.data_vars)}")
        data = ds[var_name].values  # Shape: (1100, n_days)
        return data[self.valid_indices]  # Shape: (978, n_days)
    
    def compute_statistics(self, years: List[int] = None) -> Dict:
        """
        Compute mean/std from training data for normalization.
        
        Args:
            years: Years to use (default: first 5 train years for speed)
            
        Returns:
            stats: {var_name: {'mean': float, 'std': float}}
        """
        if years is None:
            years = self.cfg.train_years[:5]
            
        print(f"📊 Computing statistics from {len(years)} years...")
        all_vars = [self.cfg.target_var] + self.cfg.forcing_vars
        
        for var in all_vars:
            values = []
            for year in years:
                try:
                    data = self.get_variable(year, var).flatten()
                    values.append(data[~np.isnan(data)])
                except Exception as e:
                    print(f"   Warning: {var} {year} - {e}")
            
            if values:
                combined = np.concatenate(values)
                if var == 'chl' and self.cfg.log_transform_chl:
                    combined = np.log(combined + 1e-6)
                self.stats[var] = {
                    'mean': float(np.nanmean(combined)),
                    'std': float(np.nanstd(combined)),
                    'min': float(np.nanmin(combined)),
                    'max': float(np.nanmax(combined)),
                }
        
        print(f"   ✅ Statistics computed for {len(self.stats)} variables")
        return self.stats
    
    def normalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Apply log-transform and z-score normalization."""
        if var_name not in self.stats:
            return data
        
        result = data.copy()
        if var_name == 'chl' and self.cfg.log_transform_chl:
            result = np.log(result + 1e-6)
        
        result = (result - self.stats[var_name]['mean']) / (self.stats[var_name]['std'] + 1e-8)
        return result
    
    def denormalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Inverse normalization."""
        if var_name not in self.stats:
            return data
        
        result = data * self.stats[var_name]['std'] + self.stats[var_name]['mean']
        if var_name == 'chl' and self.cfg.log_transform_chl:
            result = np.exp(result) - 1e-6
        return result
    
    def get_graph_tensors(self) -> Dict[str, torch.Tensor]:
        """Return graph structure as PyTorch tensors for the model."""
        return {
            'edge_index': torch.tensor(self.edge_index, dtype=torch.long, device=DEVICE),
            'edge_weight': torch.tensor(self.edge_weight, dtype=torch.float32, device=DEVICE),
            'node_depths': torch.tensor(self.node_depths, dtype=torch.float32, device=DEVICE),
            'node_tiers': torch.tensor(self.node_tiers, dtype=torch.long, device=DEVICE),
            'n_nodes': self.n_valid_nodes,
        }
    
    def get_sample_batch(self, year: int = 2020, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get a sample batch of data for testing the model.
        
        Args:
            year: Year to sample from
            normalize: Whether to normalize
            
        Returns:
            batch: Dictionary with 'x', 'forcing', 'target' tensors
        """
        # Load target (chlorophyll)
        chl = self.get_variable(year, 'chl')
        if normalize:
            if not self.stats:
                self.compute_statistics()
            chl = self.normalize_data(chl, 'chl')
        
        # Load forcing variables
        forcing = {}
        for var in self.cfg.forcing_vars:
            try:
                data = self.get_variable(year, var)
                if normalize:
                    data = self.normalize_data(data, var)
                forcing[var] = torch.tensor(data[:, 0], dtype=torch.float32, device=DEVICE)
            except:
                forcing[var] = torch.zeros(self.n_valid_nodes, device=DEVICE)
        
        # Create input features
        x = torch.stack([
            torch.tensor(chl[:, 0], dtype=torch.float32, device=DEVICE),
            forcing['thetao'],
            forcing['so'],
            forcing['uo'],
            forcing['vo'],
            forcing['mlotst'],
            forcing['zos'],
            forcing['no3'],
            forcing.get('phyc', torch.zeros(self.n_valid_nodes, device=DEVICE)),
            torch.tensor(self.node_depths, dtype=torch.float32, device=DEVICE),
        ], dim=-1)  # Shape: (978, 10)
        
        return {
            'x': x,
            'forcing': forcing,
            'target': torch.tensor(chl, dtype=torch.float32, device=DEVICE),
            'edge_index': torch.tensor(self.edge_index, dtype=torch.long, device=DEVICE),
            'edge_weight': torch.tensor(self.edge_weight, dtype=torch.float32, device=DEVICE),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

try:
    data_cfg = BlackSeaDataConfig()
    data_loader = BlackSeaDataLoader(data_cfg)
    
    # Compute normalization statistics
    data_loader.compute_statistics()
    
    # Get graph tensors for the model
    graph_data = data_loader.get_graph_tensors()
    
    # Update main config with real values
    cfg.n_horizontal = data_loader.n_valid_nodes
    cfg.n_vertical = 1  # Surface only for now
    
    print(f"\n{'═'*60}")
    print(f"✅ REAL BLACK SEA DATA LOADED SUCCESSFULLY")
    print(f"{'═'*60}")
    print(f"   Nodes: {graph_data['n_nodes']}")
    print(f"   Edges: {graph_data['edge_index'].shape[1]}")
    print(f"   Train: {data_cfg.train_years[0]}-{data_cfg.train_years[-1]} ({len(data_cfg.train_years)} years)")
    print(f"   Val: {data_cfg.val_years}")
    print(f"   Test: {data_cfg.test_years}")
    print(f"{'═'*60}")
    
    DATA_LOADED = True
    
except Exception as e:
    print(f"\n⚠️ Could not load real data: {e}")
    print("   This is OK - the notebook will use synthetic data instead.")
    print("   To use real data, update the paths in BlackSeaDataConfig.")
    DATA_LOADED = False
    data_loader = None
    graph_data = None
