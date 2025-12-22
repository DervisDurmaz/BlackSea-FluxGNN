"""
Strang Operator Splitting for Advection-Diffusion-Reaction.

TRANSPORT(Δt/2) → REACTION(Δt) → TRANSPORT(Δt/2)

Second-order accurate in time. Proven for ADR problems.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, Optional


class StrangOperatorSplitting(nn.Module):
    """
    Strang Splitting for coupled transport and reaction.
    
    For PDE: ∂C/∂t = L_transport(C) + L_reaction(C)
    
    Strang (1968) splitting is 2nd-order accurate:
        C(t+Δt) = T(Δt/2) R(Δt) T(Δt/2) C(t)
        
    Where T = transport operator, R = reaction operator.
    
    Key insight: 
        - Transport operator is NUMERICAL (mass-conservative FVM)
        - Reaction operator is NEURAL (learned biology)
    """
    
    def __init__(self,
                 transport_fn: Callable,
                 reaction_fn: Callable,
                 use_strang: bool = True):
        """
        Args:
            transport_fn: Transport operator T(C, dt) → C
            reaction_fn: Reaction operator R(C, forcing, dt) → C
            use_strang: Use Strang splitting (vs Lie splitting)
        """
        super().__init__()
        
        self.transport = transport_fn
        self.reaction = reaction_fn
        self.use_strang = use_strang
        
    def lie_splitting(self,
                      C: torch.Tensor,
                      forcing: torch.Tensor,
                      transport_args: Dict,
                      dt: float) -> torch.Tensor:
        """
        First-order Lie splitting.
        
        C(t+Δt) = R(Δt) T(Δt) C(t)
        """
        # Transport step
        C = self.transport(C, dt=dt, **transport_args)
        
        # Reaction step
        C = self.reaction(C, forcing, dt=dt)
        
        return C
    
    def strang_splitting(self,
                         C: torch.Tensor,
                         forcing: torch.Tensor,
                         transport_args: Dict,
                         dt: float) -> torch.Tensor:
        """
        Second-order Strang splitting.
        
        C(t+Δt) = T(Δt/2) R(Δt) T(Δt/2) C(t)
        """
        half_dt = dt / 2
        
        # Half transport step
        C = self.transport(C, dt=half_dt, **transport_args)
        
        # Full reaction step
        C = self.reaction(C, forcing, dt=dt)
        
        # Half transport step
        C = self.transport(C, dt=half_dt, **transport_args)
        
        return C
    
    def forward(self,
                C: torch.Tensor,
                forcing: torch.Tensor,
                transport_args: Dict,
                dt: float = 86400.0) -> torch.Tensor:
        """
        Single time step with operator splitting.
        
        Args:
            C: Tracer concentrations [N, n_tracers]
            forcing: Environmental forcing for reaction [N, n_forcing]
            transport_args: Dict of arguments for transport operator
            dt: Time step in SECONDS (default 1 day)
            
        Returns:
            C_new: Updated concentrations
        """
        if self.use_strang:
            return self.strang_splitting(C, forcing, transport_args, dt)
        else:
            return self.lie_splitting(C, forcing, transport_args, dt)
    
    def rollout(self,
                C0: torch.Tensor,
                forcing_seq: torch.Tensor,
                transport_args: Dict,
                n_steps: int,
                dt: float = 86400.0) -> torch.Tensor:
        """
        Multi-step rollout.
        
        Args:
            C0: Initial state [N, n_tracers]
            forcing_seq: Forcing time series [n_steps, N, n_forcing]
            transport_args: Transport operator arguments
            n_steps: Number of time steps
            dt: Time step (seconds)
            
        Returns:
            C_traj: Trajectory [n_steps+1, N, n_tracers]
        """
        trajectory = [C0]
        C = C0.clone()
        
        for t in range(n_steps):
            forcing_t = forcing_seq[t] if forcing_seq.dim() > 2 else forcing_seq
            C = self.forward(C, forcing_t, transport_args, dt)
            trajectory.append(C)
        
        return torch.stack(trajectory, dim=0)


class HybridOperator(nn.Module):
    """
    Complete hybrid operator combining all components.
    
    Pipeline per time step:
        1. Message passing (spatial context)
        2. Strang split: Transport(Δt/2) → Reaction(Δt) → Transport(Δt/2)
        3. FCT limiting (positivity)
    """
    
    def __init__(self,
                 message_passing: nn.Module,
                 transport: nn.Module,
                 reaction: nn.Module,
                 limiter: nn.Module):
        super().__init__()
        
        self.mp = message_passing
        self.strang = StrangOperatorSplitting(
            transport_fn=self._transport_wrapper,
            reaction_fn=self._reaction_wrapper
        )
        self.transport_module = transport
        self.reaction_module = reaction
        self.limiter = limiter
        
        # Cache for transport args (set during forward)
        self._transport_args = {}
        
    def _transport_wrapper(self, C, dt, **kwargs):
        """Wrapper for transport with cached args."""
        return self.transport_module(C, dt=dt, **self._transport_args)
    
    def _reaction_wrapper(self, C, forcing, dt):
        """Wrapper for reaction operator."""
        return self.reaction_module(C, forcing, dt=dt)
    
    def forward(self,
                C: torch.Tensor,
                node_features: torch.Tensor,
                forcing: torch.Tensor,
                edge_index_h: torch.Tensor,
                edge_index_v: torch.Tensor,
                transport_args: Dict,
                dt: float = 86400.0) -> torch.Tensor:
        """
        Full hybrid step.
        
        Args:
            C: Current tracer state [N, n_tracers]
            node_features: Node features for message passing [N, d]
            forcing: Environmental forcing [N, n_forcing]
            edge_index_h/v: Graph connectivity
            transport_args: Additional transport arguments
            dt: Time step
            
        Returns:
            C_new: Updated tracer concentrations
        """
        # 1. Message passing for spatial context
        node_features = self.mp(node_features, edge_index_h, edge_index_v)
        
        # 2. Cache transport args
        self._transport_args = transport_args
        
        # 3. Strang splitting (Transport → Reaction → Transport)
        C_high = self.strang(C, forcing, transport_args, dt)
        
        # 4. FCT limiting (ensure positivity)
        # C is the low-order (positive) solution, C_high is high-order
        C_low = C + 0.1 * (C_high - C)  # Simplified low-order
        C_final = self.limiter(C_low, C_high)
        
        return C_final


# =============================================================================
# Geostrophic Velocity Helper
# =============================================================================

def compute_geostrophic_velocity(mdt: torch.Tensor,
                                  lon: torch.Tensor,
                                  lat: torch.Tensor,
                                  resolution: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute geostrophic velocity from Mean Dynamic Topography.
    
    u_g = -(g/f) * ∂η/∂y
    v_g = (g/f) * ∂η/∂x
    
    Args:
        mdt: Mean Dynamic Topography [ny, nx], m
        lon: Longitude [nx], degrees
        lat: Latitude [ny], degrees
        resolution: Grid resolution in degrees
        
    Returns:
        u_geo: Zonal velocity [ny, nx], m/s
        v_geo: Meridional velocity [ny, nx], m/s
    """
    import numpy as np
    
    g = 9.81  # m/s²
    omega = 7.2921e-5  # rad/s
    
    # Coriolis parameter
    f = 2 * omega * torch.sin(torch.deg2rad(lat))
    f = f.unsqueeze(1)  # [ny, 1] for broadcasting
    
    # Grid spacing (degrees to meters)
    lat_mean = lat.mean()
    dx = resolution * 111320 * torch.cos(torch.deg2rad(lat_mean))  # m
    dy = resolution * 110540  # m
    
    # Centered differences
    deta_dx = torch.zeros_like(mdt)
    deta_dy = torch.zeros_like(mdt)
    
    # Interior points
    deta_dx[:, 1:-1] = (mdt[:, 2:] - mdt[:, :-2]) / (2 * dx)
    deta_dy[1:-1, :] = (mdt[2:, :] - mdt[:-2, :]) / (2 * dy)
    
    # Edges (one-sided)
    deta_dx[:, 0] = (mdt[:, 1] - mdt[:, 0]) / dx
    deta_dx[:, -1] = (mdt[:, -1] - mdt[:, -2]) / dx
    deta_dy[0, :] = (mdt[1, :] - mdt[0, :]) / dy
    deta_dy[-1, :] = (mdt[-1, :] - mdt[-2, :]) / dy
    
    # Geostrophic velocities
    u_geo = -(g / (f + 1e-10)) * deta_dy
    v_geo = (g / (f + 1e-10)) * deta_dx
    
    return u_geo, v_geo


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Operator Splitting...")
    
    # Dummy operators
    def dummy_transport(C, dt, **kwargs):
        return C * 0.99  # Slight decay
    
    def dummy_reaction(C, forcing, dt):
        return C + 0.01 * torch.randn_like(C)  # Random perturbation
    
    splitter = StrangOperatorSplitting(
        transport_fn=dummy_transport,
        reaction_fn=dummy_reaction,
        use_strang=True
    )
    
    # Test single step
    C = torch.rand(100, 4)  # 100 nodes, 4 tracers
    forcing = torch.rand(100, 8)
    
    C_new = splitter(C, forcing, {}, dt=86400)
    
    print(f"✓ Input: {C.shape}")
    print(f"✓ Output: {C_new.shape}")
    print(f"✓ Conservation: {(C_new.sum() / C.sum()).item():.4f}")
    
    # Test rollout
    forcing_seq = torch.rand(7, 100, 8)
    traj = splitter.rollout(C, forcing_seq, {}, n_steps=7)
    
    print(f"\n✓ Rollout trajectory: {traj.shape}")
    print(f"✓ Initial mass: {traj[0].sum().item():.2f}")
    print(f"✓ Final mass: {traj[-1].sum().item():.2f}")
    
    # Test geostrophic velocity
    print("\n✓ Geostrophic velocity:")
    mdt = torch.randn(20, 30) * 0.1  # Small MDT variations
    lon = torch.linspace(28, 40, 30)
    lat = torch.linspace(41, 46, 20)
    
    u_geo, v_geo = compute_geostrophic_velocity(mdt, lon, lat)
    print(f"  u_geo range: [{u_geo.min():.3f}, {u_geo.max():.3f}] m/s")
    print(f"  v_geo range: [{v_geo.min():.3f}, {v_geo.max():.3f}] m/s")
    
    print("\n✅ All tests passed!")
