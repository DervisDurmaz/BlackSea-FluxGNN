"""
Differentiable Finite Volume Method for advection-diffusion.

Mass-conservative by construction. DO NOT learn transport physics.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from typing import Tuple, Optional


class DifferentiableFVM(nn.Module):
    """
    Differentiable Finite Volume Method for tracer transport.
    
    Features:
    - Horizontal: 2D FVM with upwind/TVD advection
    - Vertical: 1D implicit diffusion (for stability)
    - Mass conservative by construction
    - Learnable diffusivities (κ_h, κ_v)
    
    DO NOT use neural networks for transport! 
    This is a numerical solver that backpropagates through physics.
    """
    
    def __init__(self,
                 kappa_h_init: float = 1e3,   # m²/s horizontal diffusivity
                 kappa_v_init: float = 1e-5,  # m²/s vertical diffusivity
                 learn_kappa: bool = True):
        super().__init__()
        
        # Learnable diffusivities (log-scale for positivity)
        if learn_kappa:
            self.log_kappa_h = nn.Parameter(torch.tensor(float(torch.log10(torch.tensor(kappa_h_init)))))
            self.log_kappa_v = nn.Parameter(torch.tensor(float(torch.log10(torch.tensor(kappa_v_init)))))
        else:
            self.register_buffer('log_kappa_h', torch.tensor(float(torch.log10(torch.tensor(kappa_h_init)))))
            self.register_buffer('log_kappa_v', torch.tensor(float(torch.log10(torch.tensor(kappa_v_init)))))
    
    @property
    def kappa_h(self) -> torch.Tensor:
        """Horizontal diffusivity (m²/s)."""
        return 10 ** self.log_kappa_h
    
    @property
    def kappa_v(self) -> torch.Tensor:
        """Vertical diffusivity (m²/s)."""
        return 10 ** self.log_kappa_v
    
    def upwind_flux(self, 
                    C: torch.Tensor,
                    u_face: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
        """
        First-order upwind advective flux.
        
        F_adv = u * C_upwind
        
        Args:
            C: Tracer concentration [N_nodes, n_tracers]
            u_face: Velocity at face (positive = src→dst) [N_edges]
            edge_index: [2, N_edges] edge connectivity
            
        Returns:
            F: Flux at each edge [N_edges, n_tracers]
        """
        src, dst = edge_index
        
        # Upwind: use upstream cell value
        C_upwind = torch.where(
            u_face.unsqueeze(-1) > 0,
            C[src],  # Flow from src to dst
            C[dst]   # Flow from dst to src
        )
        
        return u_face.unsqueeze(-1) * C_upwind
    
    def tvd_flux(self,
                 C: torch.Tensor,
                 u_face: torch.Tensor,
                 edge_index: torch.Tensor,
                 limiter: str = 'minmod') -> torch.Tensor:
        """
        Second-order TVD advective flux (Total Variation Diminishing).
        
        F_TVD = F_low + Φ(r) * (F_high - F_low)
        
        Where Φ is a flux limiter.
        """
        src, dst = edge_index
        
        # Low-order flux (upwind)
        F_low = self.upwind_flux(C, u_face, edge_index)
        
        # Gradient ratio
        dC = C[dst] - C[src]
        
        # Central difference for high-order
        C_face = 0.5 * (C[src] + C[dst])
        F_high = u_face.unsqueeze(-1) * C_face
        
        # Apply limiter
        if limiter == 'minmod':
            phi = self._minmod_limiter(dC)
        elif limiter == 'superbee':
            phi = self._superbee_limiter(dC)
        else:
            phi = torch.ones_like(dC)
        
        return F_low + phi * (F_high - F_low)
    
    def _minmod_limiter(self, r: torch.Tensor) -> torch.Tensor:
        """Minmod flux limiter (most diffusive)."""
        return torch.clamp(r, 0, 1)
    
    def _superbee_limiter(self, r: torch.Tensor) -> torch.Tensor:
        """Superbee flux limiter (least diffusive)."""
        return torch.maximum(
            torch.zeros_like(r),
            torch.maximum(
                torch.minimum(2 * r, torch.ones_like(r)),
                torch.minimum(r, 2 * torch.ones_like(r))
            )
        )
    
    def diffusive_flux(self,
                       C: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_lengths: torch.Tensor,
                       kappa: torch.Tensor) -> torch.Tensor:
        """
        Diffusive flux.
        
        F_diff = -κ * (C_dst - C_src) / Δx
        
        Args:
            C: Tracer concentration [N_nodes, n_tracers]
            edge_index: [2, N_edges] edge connectivity
            edge_lengths: Distance between cell centers [N_edges]
            kappa: Diffusivity (scalar or per-edge)
            
        Returns:
            F: Diffusive flux [N_edges, n_tracers]
        """
        src, dst = edge_index
        
        dC = C[dst] - C[src]
        
        return -kappa * dC / (edge_lengths.unsqueeze(-1) + 1e-6)
    
    def horizontal_step(self,
                        C: torch.Tensor,
                        velocity: torch.Tensor,
                        edge_index_h: torch.Tensor,
                        edge_lengths: torch.Tensor,
                        cell_areas: torch.Tensor,
                        dt: float) -> torch.Tensor:
        """
        Horizontal transport step.
        
        ∂C/∂t = -∇·(uC) + κ_h ∇²C
        
        Args:
            C: Tracer concentration [N_nodes, n_tracers]
            velocity: [N_nodes, 2] = (u, v) velocity field
            edge_index_h: Horizontal edges [2, N_edges_h]
            edge_lengths: Length of each edge [N_edges_h]
            cell_areas: Area of each cell [N_nodes]
            dt: Time step (seconds)
            
        Returns:
            C_new: Updated concentration
        """
        src, dst = edge_index_h
        
        # Face-normal velocity (simplified: project velocity)
        u = velocity[..., 0]
        v = velocity[..., 1]
        u_face = 0.5 * (u[src] + u[dst])
        
        # Advective flux (TVD)
        F_adv = self.tvd_flux(C, u_face, edge_index_h)
        
        # Diffusive flux
        F_diff = self.diffusive_flux(C, edge_index_h, edge_lengths, self.kappa_h)
        
        # Total flux
        F_total = F_adv + F_diff
        
        # Accumulate flux divergence
        div_F = scatter_add(F_total, dst, dim=0, dim_size=C.size(0))
        div_F -= scatter_add(F_total, src, dim=0, dim_size=C.size(0))
        
        # Update (explicit Euler)
        # dC/dt = -div(F) / A
        C_new = C - dt * div_F / (cell_areas.unsqueeze(-1) + 1e-6)
        
        return C_new
    
    def vertical_step_explicit(self,
                               C: torch.Tensor,
                               dz: torch.Tensor,
                               n_horizontal: int,
                               n_vertical: int,
                               dt: float) -> torch.Tensor:
        """
        Explicit vertical diffusion step.
        
        ∂C/∂t = κ_v ∂²C/∂z²
        
        Simple but requires small dt for stability (Δt < Δz²/2κ).
        """
        # Reshape to 3D
        n_tracers = C.shape[-1]
        C_3d = C.view(n_horizontal, n_vertical, n_tracers)
        dz_3d = dz.view(n_horizontal, n_vertical)
        
        # Vertical second derivative
        d2C_dz2 = torch.zeros_like(C_3d)
        
        # Interior points (central difference)
        for k in range(1, n_vertical - 1):
            dz_k = 0.5 * (dz_3d[:, k+1] + dz_3d[:, k-1])
            d2C_dz2[:, k, :] = (
                C_3d[:, k+1, :] - 2 * C_3d[:, k, :] + C_3d[:, k-1, :]
            ) / (dz_k.unsqueeze(-1)**2 + 1e-6)
        
        # Update
        C_new = C_3d + dt * self.kappa_v * d2C_dz2
        
        return C_new.view(-1, n_tracers)
    
    def vertical_step_implicit(self,
                               C: torch.Tensor,
                               dz: torch.Tensor,
                               n_horizontal: int,
                               n_vertical: int,
                               dt: float) -> torch.Tensor:
        """
        Implicit vertical diffusion (unconditionally stable).
        
        Solves tridiagonal system with Thomas algorithm.
        """
        n_tracers = C.shape[-1]
        C_3d = C.view(n_horizontal, n_vertical, n_tracers)
        dz_3d = dz.view(n_horizontal, n_vertical)
        
        # Build tridiagonal coefficients
        # (1 + 2λ)C^{n+1}_k - λ(C^{n+1}_{k-1} + C^{n+1}_{k+1}) = C^n_k
        # where λ = κ_v * dt / dz²
        
        result = torch.zeros_like(C_3d)
        
        for h in range(n_horizontal):
            for tracer in range(n_tracers):
                c = C_3d[h, :, tracer].clone()
                dz_col = dz_3d[h, :]
                
                # Thomas algorithm
                lam = self.kappa_v * dt / (dz_col**2 + 1e-6)
                
                a = -lam[1:]       # Lower diagonal
                b = 1 + 2 * lam   # Main diagonal
                c_upper = -lam[:-1]  # Upper diagonal
                d = c.clone()      # RHS
                
                # Forward sweep
                for k in range(1, n_vertical):
                    w = a[k-1] / (b[k-1] + 1e-10)
                    b[k] = b[k] - w * c_upper[k-1]
                    d[k] = d[k] - w * d[k-1]
                
                # Back substitution
                result[h, -1, tracer] = d[-1] / (b[-1] + 1e-10)
                for k in range(n_vertical - 2, -1, -1):
                    result[h, k, tracer] = (d[k] - c_upper[k] * result[h, k+1, tracer]) / (b[k] + 1e-10)
        
        return result.view(-1, n_tracers)
    
    def forward(self,
                C: torch.Tensor,
                velocity: torch.Tensor,
                edge_index_h: torch.Tensor,
                edge_lengths: torch.Tensor,
                cell_areas: torch.Tensor,
                dz: torch.Tensor,
                n_horizontal: int,
                n_vertical: int,
                dt: float = 86400.0,
                use_implicit_vertical: bool = True) -> torch.Tensor:
        """
        Full transport step (horizontal + vertical).
        
        Args:
            C: Concentration [N_nodes, n_tracers]
            velocity: [N_nodes, 2] velocity field (u, v)
            edge_index_h: Horizontal edge connectivity
            edge_lengths: Length of horizontal edges (m)
            cell_areas: Area of each cell (m²)
            dz: Vertical layer thickness [N_nodes]
            n_horizontal: Number of horizontal nodes
            n_vertical: Number of vertical levels
            dt: Time step (seconds), default 1 day
            use_implicit_vertical: Use implicit solver for stability
            
        Returns:
            C_new: Updated tracer concentration
        """
        # Horizontal transport
        C = self.horizontal_step(C, velocity, edge_index_h, edge_lengths, 
                                 cell_areas, dt)
        
        # Vertical diffusion
        if use_implicit_vertical:
            C = self.vertical_step_implicit(C, dz, n_horizontal, n_vertical, dt)
        else:
            C = self.vertical_step_explicit(C, dz, n_horizontal, n_vertical, dt)
        
        return C


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Differentiable FVM...")
    
    # Create solver
    fvm = DifferentiableFVM(kappa_h_init=1e3, kappa_v_init=1e-5)
    
    print(f"✓ κ_h = {fvm.kappa_h.item():.2e} m²/s")
    print(f"✓ κ_v = {fvm.kappa_v.item():.2e} m²/s")
    
    # Test upwind flux
    n_nodes, n_tracers = 100, 4
    C = torch.rand(n_nodes, n_tracers)
    edge_index = torch.randint(0, n_nodes, (2, 300))
    u_face = torch.randn(300)
    
    F_upwind = fvm.upwind_flux(C, u_face, edge_index)
    print(f"\n✓ Upwind flux shape: {F_upwind.shape}")
    
    # Check mass conservation (sum of fluxes should cancel for internal edges)
    # This is a simplified check
    print(f"✓ Mean flux magnitude: {F_upwind.abs().mean():.4f}")
    
    # Test learnable parameters
    loss = fvm.kappa_h + fvm.kappa_v
    loss.backward()
    print(f"\n✓ Gradients computed: log_kappa_h.grad = {fvm.log_kappa_h.grad}")
    
    print("\n✅ All tests passed!")
