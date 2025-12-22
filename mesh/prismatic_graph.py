"""
Prismatic Graph for Black Sea oceanic modeling.

Hybrid grid: unstructured horizontal (Delaunay) + structured vertical (σ-coordinates).
Solves the 1000:1 aspect ratio singularity by separating H/V operations.
"""

import numpy as np
import torch
from scipy.spatial import Delaunay
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# σ-Coordinate Transformations (ROMS-style)
# =============================================================================

def z_to_sigma(z: float, H: float, zeta: float = 0.0) -> float:
    """
    Transform depth z to σ-coordinate.
    
    σ = (z - ζ) / (H + ζ)
    
    Args:
        z: Depth (negative downward), m
        H: Bottom depth (positive), m  
        zeta: Sea surface height anomaly, m
        
    Returns:
        σ ∈ [-1, 0]: -1 at bottom, 0 at surface
    """
    return (z - zeta) / (H + zeta)


def sigma_to_z(sigma: float, H: float, zeta: float = 0.0) -> float:
    """
    Inverse transform: σ → z.
    
    z = σ * (H + ζ) + ζ
    """
    return sigma * (H + zeta) + zeta


def sigma_stretching(sigma: np.ndarray, 
                     theta_s: float = 5.0, 
                     theta_b: float = 0.4) -> np.ndarray:
    """
    ROMS-style σ stretching for enhanced surface/bottom resolution.
    
    C(σ) = (1 - cosh(θ_s·σ)) / (cosh(θ_s) - 1)    [surface]
    C(σ) = (exp(θ_b·C) - 1) / (1 - exp(-θ_b))     [bottom]
    
    Args:
        sigma: σ values ∈ [-1, 0]
        theta_s: Surface stretching parameter (0-10, higher = more resolution)
        theta_b: Bottom stretching parameter (0-1)
        
    Returns:
        C: Stretched σ-coordinate
    """
    if theta_s > 0:
        C_surf = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
    else:
        C_surf = -sigma**2
        
    if theta_b > 0:
        C = (np.exp(theta_b * C_surf) - 1) / (1 - np.exp(-theta_b))
    else:
        C = C_surf
        
    return C


# =============================================================================
# Prismatic Graph Class
# =============================================================================

@dataclass
class GraphConfig:
    """Configuration for prismatic graph."""
    n_vertical: int = 50
    theta_s: float = 5.0      # Surface stretching
    theta_b: float = 0.4      # Bottom stretching
    min_depth: float = 10.0   # Minimum water depth (m)


class PrismaticGraph:
    """
    Prismatic (layered) graph for oceanic applications.
    
    Structure:
    - Horizontal: Unstructured Delaunay mesh (2D) fitting Black Sea coastline
    - Vertical: σ-coordinates (density-following) to minimize diapycnal error
    - Connectivity: H-neighbors via GNN edges, V-neighbors via 1D FD stencil
    
    Key Insight:
        Separate operators eliminate condition number explosion:
        - 2D Graph Convolutions → horizontal fluxes (isotropic)
        - 1D Finite Differences → vertical fluxes (structured)
    """
    
    def __init__(self,
                 horizontal_coords: np.ndarray,  # [N_h, 2] lon/lat
                 bottom_depths: np.ndarray,       # [N_h] bathymetry (positive)
                 config: Optional[GraphConfig] = None):
        """
        Initialize prismatic graph.
        
        Args:
            horizontal_coords: [N_h, 2] array of (lon, lat) coordinates
            bottom_depths: [N_h] bottom depth at each horizontal node (m, positive)
            config: Graph configuration
        """
        self.config = config or GraphConfig()
        self.horizontal_coords = horizontal_coords
        self.bottom_depths = np.maximum(bottom_depths, self.config.min_depth)
        
        self.n_horizontal = len(horizontal_coords)
        self.n_vertical = self.config.n_vertical
        self.n_total = self.n_horizontal * self.n_vertical
        
        # σ-levels (uniform in σ-space, -1 to 0)
        self.sigma_levels = np.linspace(-1, 0, self.n_vertical)
        
        # Apply stretching
        self.sigma_stretched = sigma_stretching(
            self.sigma_levels, 
            self.config.theta_s, 
            self.config.theta_b
        )
        
        # Build graph components
        self.edge_index_h = self._build_horizontal_edges()
        self.edge_index_v = self._build_vertical_edges()
        self.z_coords = self._compute_z_at_nodes()
        self.dz = self._compute_layer_thicknesses()
        
    def _build_horizontal_edges(self) -> torch.Tensor:
        """Build Delaunay triangulation edges in 2D."""
        try:
            tri = Delaunay(self.horizontal_coords)
        except Exception as e:
            raise ValueError(f"Delaunay triangulation failed: {e}. "
                           "Check for duplicate or collinear points.")
        
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                edges.add(edge)
        
        # Create bidirectional edges
        edge_list = []
        for e in edges:
            edge_list.append([e[0], e[1]])
            edge_list.append([e[1], e[0]])
            
        return torch.tensor(edge_list, dtype=torch.long).T
        
    def _build_vertical_edges(self) -> torch.Tensor:
        """Build structured vertical edges (up/down only)."""
        edges = []
        for h in range(self.n_horizontal):
            for v in range(self.n_vertical - 1):
                node_curr = h * self.n_vertical + v
                node_next = h * self.n_vertical + v + 1
                edges.append([node_curr, node_next])  # Down → up
                edges.append([node_next, node_curr])  # Up → down
        return torch.tensor(edges, dtype=torch.long).T
    
    def _compute_z_at_nodes(self) -> np.ndarray:
        """Compute actual z-coordinates at each 3D node."""
        z = np.zeros(self.n_total)
        for h in range(self.n_horizontal):
            H = self.bottom_depths[h]
            for v in range(self.n_vertical):
                node_idx = h * self.n_vertical + v
                # Use stretched σ for better resolution
                z[node_idx] = sigma_to_z(self.sigma_stretched[v], H)
        return z
    
    def _compute_layer_thicknesses(self) -> np.ndarray:
        """Compute Δz for each cell (at cell centers)."""
        dz = np.zeros((self.n_horizontal, self.n_vertical))
        for h in range(self.n_horizontal):
            H = self.bottom_depths[h]
            for v in range(self.n_vertical):
                if v == 0:
                    dz[h, v] = abs(sigma_to_z(self.sigma_stretched[1], H) - 
                                   sigma_to_z(self.sigma_stretched[0], H)) / 2
                elif v == self.n_vertical - 1:
                    dz[h, v] = abs(sigma_to_z(self.sigma_stretched[v], H) - 
                                   sigma_to_z(self.sigma_stretched[v-1], H)) / 2
                else:
                    dz[h, v] = abs(sigma_to_z(self.sigma_stretched[v+1], H) - 
                                   sigma_to_z(self.sigma_stretched[v-1], H)) / 2
        return dz.flatten()
    
    def get_node_idx(self, h_idx: int, v_idx: int) -> int:
        """Convert (horizontal, vertical) index to flat node index."""
        return h_idx * self.n_vertical + v_idx
    
    def get_hv_idx(self, node_idx: int) -> Tuple[int, int]:
        """Convert flat node index to (horizontal, vertical) index."""
        h_idx = node_idx // self.n_vertical
        v_idx = node_idx % self.n_vertical
        return h_idx, v_idx
    
    def get_horizontal_neighbors(self, h_idx: int) -> np.ndarray:
        """Get horizontal neighbor indices for a given horizontal column."""
        mask = self.edge_index_h[0] == h_idx
        return self.edge_index_h[1, mask].numpy()
    
    def compute_edge_lengths(self) -> torch.Tensor:
        """Compute Euclidean edge lengths for horizontal edges."""
        src, dst = self.edge_index_h
        coords = torch.tensor(self.horizontal_coords, dtype=torch.float32)
        
        # Convert to approximate meters (rough, for Black Sea lat ~43°)
        lon_scale = 111320 * np.cos(np.radians(43))  # m per degree lon
        lat_scale = 110540  # m per degree lat
        
        dx = (coords[dst, 0] - coords[src, 0]) * lon_scale
        dy = (coords[dst, 1] - coords[src, 1]) * lat_scale
        
        return torch.sqrt(dx**2 + dy**2)
    
    def to_tensors(self) -> dict:
        """Export graph to PyTorch tensors."""
        return {
            'edge_index_h': self.edge_index_h,
            'edge_index_v': self.edge_index_v,
            'z': torch.tensor(self.z_coords, dtype=torch.float32),
            'dz': torch.tensor(self.dz, dtype=torch.float32),
            'edge_lengths_h': self.compute_edge_lengths(),
            'sigma': torch.tensor(
                np.tile(self.sigma_stretched, self.n_horizontal), 
                dtype=torch.float32
            ),
        }
    
    def visualize_vertical_grid(self, h_idx: int = 0):
        """Visualize vertical grid at a horizontal location."""
        import matplotlib.pyplot as plt
        
        H = self.bottom_depths[h_idx]
        z_uniform = np.linspace(-H, 0, self.n_vertical)
        z_stretched = [sigma_to_z(s, H) for s in self.sigma_stretched]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        
        # Uniform σ
        axes[0].barh(range(self.n_vertical), np.ones(self.n_vertical), 
                     height=0.8, color='steelblue', alpha=0.7)
        axes[0].set_yticks(range(0, self.n_vertical, 5))
        axes[0].set_yticklabels([f'{z:.0f}m' for z in z_uniform[::5]])
        axes[0].set_title('Uniform σ-levels')
        axes[0].set_xlabel('σ-level')
        
        # Stretched σ
        layer_thickness = np.diff([sigma_to_z(0, H)] + z_stretched)
        axes[1].barh(range(self.n_vertical), abs(layer_thickness), 
                     height=0.8, color='coral', alpha=0.7)
        axes[1].set_yticks(range(0, self.n_vertical, 5))
        axes[1].set_yticklabels([f'{z:.0f}m' for z in z_stretched[::5]])
        axes[1].set_title(f'Stretched σ (θ_s={self.config.theta_s}, θ_b={self.config.theta_b})')
        axes[1].set_xlabel('Layer thickness (m)')
        
        plt.suptitle(f'Vertical Grid at H={H:.0f}m')
        plt.tight_layout()
        return fig


# =============================================================================
# Factory Functions
# =============================================================================

def create_black_sea_grid(resolution: float = 0.1) -> PrismaticGraph:
    """
    Create a prismatic graph for the Black Sea domain.
    
    Args:
        resolution: Approximate horizontal resolution in degrees
        
    Returns:
        PrismaticGraph configured for Black Sea
    """
    # Black Sea bounding box
    lon_min, lon_max = 27.0, 42.0
    lat_min, lat_max = 40.5, 47.0
    
    # Create regular grid (would use actual coastline mask in production)
    lons = np.arange(lon_min, lon_max, resolution)
    lats = np.arange(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    coords = np.column_stack([lon_grid.flatten(), lat_grid.flatten()])
    
    # Synthetic bathymetry (would load from GEBCO in production)
    # Black Sea max depth ~2200m, average ~1200m
    depths = 1000 + 500 * np.random.rand(len(coords))
    
    config = GraphConfig(
        n_vertical=50,
        theta_s=5.0,  # Enhanced surface resolution (DCM, mixed layer)
        theta_b=0.4   # Moderate bottom enhancement
    )
    
    return PrismaticGraph(coords, depths, config)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Prismatic Graph...")
    
    # Create test graph
    np.random.seed(42)
    coords = np.random.rand(100, 2) * np.array([15, 7]) + np.array([27, 40])
    depths = 500 + 1000 * np.random.rand(100)
    
    graph = PrismaticGraph(coords, depths)
    
    print(f"✓ Created graph with {graph.n_total:,} nodes")
    print(f"  - Horizontal: {graph.n_horizontal} nodes")
    print(f"  - Vertical: {graph.n_vertical} levels")
    print(f"  - H-edges: {graph.edge_index_h.shape[1]:,}")
    print(f"  - V-edges: {graph.edge_index_v.shape[1]:,}")
    
    tensors = graph.to_tensors()
    print(f"\n✓ Exported tensors:")
    for k, v in tensors.items():
        print(f"  - {k}: {v.shape}")
    
    # Test σ transformations
    z_test = -500.0
    H_test = 1000.0
    sigma = z_to_sigma(z_test, H_test)
    z_back = sigma_to_z(sigma, H_test)
    assert abs(z_test - z_back) < 1e-6, "σ transformation failed!"
    print(f"\n✓ σ-transform: z={z_test}m → σ={sigma:.3f} → z={z_back}m")
    
    print("\n✅ All tests passed!")
