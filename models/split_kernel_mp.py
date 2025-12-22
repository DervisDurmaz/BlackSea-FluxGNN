"""
Split-Kernel Anisotropic Message Passing.

Separate kernels for horizontal and vertical to respect oceanic anisotropy.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add
from typing import Optional


class SplitKernelMessagePassing(nn.Module):
    """
    Split-Kernel update that respects oceanic anisotropy.
    
    Standard GNN: h_i = h_i + Σⱼ W · m_ij  (isotropic - WRONG for ocean!)
    
    Split-Kernel:
        h_i = h_i + Σⱼ∈H(i) W_h · m_ij     [horizontal neighbors]
                  + Σₖ∈V(i) W_v · m_ik     [vertical neighbors]
    
    Key Physics:
    - Horizontal: Eddy diffusion, ~isotropic at mesoscale
    - Vertical: Stratification-controlled, strongly anisotropic
    - Ratio: κ_h / κ_v ~ 10⁶ in pycnocline!
    """
    
    def __init__(self, 
                 node_dim: int,
                 hidden_dim: int = 128,
                 use_edge_features: bool = True,
                 use_stratification_gate: bool = True):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features
        self.use_stratification_gate = use_stratification_gate
        
        # Horizontal kernel (mesoscale isotropic)
        h_input_dim = 2 * node_dim + (2 if use_edge_features else 0)  # src, dst, [dx, dy]
        self.W_h = nn.Sequential(
            nn.Linear(h_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim),
        )
        
        # Vertical kernel (stratification-aware)
        v_input_dim = 2 * node_dim + 2  # src, dst, N², dz
        self.W_v = nn.Sequential(
            nn.Linear(v_input_dim, hidden_dim),
            nn.SiLU(), 
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim),
        )
        
        # Stratification gating (strong N² → weak vertical exchange)
        if use_stratification_gate:
            self.strat_gate = nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # Residual scaling
        self.alpha_h = nn.Parameter(torch.tensor(0.1))
        self.alpha_v = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, 
                h: torch.Tensor,
                edge_index_h: torch.Tensor,
                edge_index_v: torch.Tensor,
                N2: Optional[torch.Tensor] = None,
                dz: Optional[torch.Tensor] = None,
                edge_attr_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with split H/V message passing.
        
        Args:
            h: Node features [N, D]
            edge_index_h: Horizontal edges [2, E_h]
            edge_index_v: Vertical edges [2, E_v]
            N2: Buoyancy frequency at vertical edges [E_v] or [N]
            dz: Vertical spacing at vertical edges [E_v]
            edge_attr_h: Optional horizontal edge features [E_h, ...]
            
        Returns:
            h_new: Updated node features [N, D]
        """
        n_nodes = h.size(0)
        
        # =====================================================================
        # Horizontal Messages (isotropic within layer)
        # =====================================================================
        src_h, dst_h = edge_index_h
        
        if self.use_edge_features and edge_attr_h is not None:
            m_h = torch.cat([h[src_h], h[dst_h], edge_attr_h], dim=-1)
        else:
            m_h = torch.cat([h[src_h], h[dst_h]], dim=-1)
            
        msg_h = self.W_h(m_h)
        
        # Aggregate (mean for stability)
        agg_h = scatter_mean(msg_h, dst_h, dim=0, dim_size=n_nodes)
        
        # =====================================================================
        # Vertical Messages (stratification-weighted)
        # =====================================================================
        src_v, dst_v = edge_index_v
        
        # Get N² for each edge (from source or destination node)
        if N2 is not None:
            if N2.dim() == 1 and N2.size(0) == n_nodes:
                N2_edge = N2[src_v]
            else:
                N2_edge = N2
        else:
            N2_edge = torch.zeros(edge_index_v.size(1), device=h.device)
            
        # Get dz for each edge
        if dz is not None:
            if dz.dim() == 1 and dz.size(0) == n_nodes:
                dz_edge = dz[src_v]
            else:
                dz_edge = dz
        else:
            dz_edge = torch.ones(edge_index_v.size(1), device=h.device)
        
        # Build vertical message
        m_v = torch.cat([
            h[src_v], 
            h[dst_v], 
            N2_edge.unsqueeze(-1),
            dz_edge.unsqueeze(-1)
        ], dim=-1)
        
        msg_v = self.W_v(m_v)
        
        # Apply stratification gate
        if self.use_stratification_gate:
            # Strong N² → gate close to 0 → weak vertical exchange
            # This is physics: pycnocline blocks vertical mixing
            gate = self.strat_gate(N2_edge.unsqueeze(-1))
            msg_v = msg_v * gate
        
        # Aggregate
        agg_v = scatter_mean(msg_v, dst_v, dim=0, dim_size=n_nodes)
        
        # =====================================================================
        # Combine with Residual
        # =====================================================================
        h_new = h + self.alpha_h * agg_h + self.alpha_v * agg_v
        
        return h_new


class MultiLayerSplitKernel(nn.Module):
    """
    Multiple layers of split-kernel message passing.
    """
    
    def __init__(self,
                 node_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SplitKernelMessagePassing(node_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)
        
    def forward(self, 
                h: torch.Tensor,
                edge_index_h: torch.Tensor,
                edge_index_v: torch.Tensor,
                N2: Optional[torch.Tensor] = None,
                dz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-layer forward pass.
        """
        for layer in self.layers:
            h = self.norm(h)
            h = layer(h, edge_index_h, edge_index_v, N2, dz)
            h = self.dropout(h)
            
        return h


# =============================================================================
# N² Computation Helper
# =============================================================================

def compute_N2_from_density(rho: torch.Tensor,
                            z: torch.Tensor,
                            edge_index_v: torch.Tensor) -> torch.Tensor:
    """
    Compute buoyancy frequency N² at vertical edges.
    
    N² = -(g/ρ₀) * ∂ρ/∂z
    
    Args:
        rho: Density at nodes [N]
        z: Depth at nodes [N] (negative downward)
        edge_index_v: Vertical edge connectivity [2, E_v]
        
    Returns:
        N2: Buoyancy frequency squared at edges [E_v]
    """
    g = 9.81  # m/s²
    rho0 = 1025.0  # Reference density kg/m³
    
    src, dst = edge_index_v
    
    drho = rho[dst] - rho[src]
    dz = z[dst] - z[src]
    
    drho_dz = drho / (dz + 1e-6)
    
    N2 = -(g / rho0) * drho_dz
    
    # Clip to physical range and ensure non-negative
    N2 = torch.clamp(N2, min=0, max=1e-3)
    
    return N2


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Split-Kernel Message Passing...")
    
    torch.manual_seed(42)
    
    # Test dimensions
    n_nodes = 300  # 10 horizontal × 30 vertical
    n_horizontal = 10
    n_vertical = 30
    node_dim = 64
    
    # Create test data
    h = torch.randn(n_nodes, node_dim)
    
    # Horizontal edges (simplified: connect nearby columns)
    edge_h = torch.randint(0, n_horizontal, (2, 50))
    edge_h = edge_h * n_vertical  # Connect at same depth
    
    # Vertical edges (structured)
    edges_v = []
    for col in range(n_horizontal):
        for level in range(n_vertical - 1):
            curr = col * n_vertical + level
            next_v = col * n_vertical + level + 1
            edges_v.append([curr, next_v])
            edges_v.append([next_v, curr])
    edge_v = torch.tensor(edges_v, dtype=torch.long).T
    
    # N² (stronger in pycnocline)
    N2 = torch.zeros(n_nodes)
    for col in range(n_horizontal):
        for level in range(n_vertical):
            depth = level / n_vertical  # 0 = surface, 1 = bottom
            # Peak N² in pycnocline (~0.3 = 100m in 300m water)
            N2[col * n_vertical + level] = 1e-4 * torch.exp(-((depth - 0.3) / 0.1)**2)
    
    # Test layer
    mp = SplitKernelMessagePassing(node_dim, hidden_dim=128)
    
    h_new = mp(h, edge_h, edge_v, N2=N2)
    
    print(f"✓ Input: {h.shape}")
    print(f"✓ Output: {h_new.shape}")
    print(f"✓ H-aggregation scale: {mp.alpha_h.item():.4f}")
    print(f"✓ V-aggregation scale: {mp.alpha_v.item():.4f}")
    
    # Test multi-layer
    ml = MultiLayerSplitKernel(node_dim, n_layers=4)
    h_multi = ml(h, edge_h, edge_v, N2=N2)
    print(f"\n✓ Multi-layer output: {h_multi.shape}")
    
    # Test N² computation
    rho = torch.linspace(1020, 1028, n_nodes)  # Density increasing with depth
    z = torch.linspace(0, -300, n_nodes)
    N2_computed = compute_N2_from_density(rho, z, edge_v)
    print(f"\n✓ N² range: [{N2_computed.min():.2e}, {N2_computed.max():.2e}]")
    
    print("\n✅ All tests passed!")
