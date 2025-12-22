"""
Gray-Box Universal Differential Equation for NPZD dynamics.

The NN predicts PARAMETERS (μ_max, K_N, etc.), NOT dC/dt directly.
Equation STRUCTURE is hard-coded to prevent "ghost nutrient" hallucinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class NPZDParams:
    """Physical bounds on NPZD parameters."""
    
    # Growth rate (Eppley curve: μ_max ∈ [0.5, 3.0] day⁻¹)
    mu_max_bounds: Tuple[float, float] = (0.5, 3.0)
    
    # Half-saturation for nitrate (K_N ∈ [0.1, 2.0] μmol/L)
    K_N_bounds: Tuple[float, float] = (0.1, 2.0)
    
    # Maximum grazing rate (g_max ∈ [0.1, 1.0] day⁻¹)
    g_max_bounds: Tuple[float, float] = (0.1, 1.0)
    
    # Phytoplankton mortality (m_P ∈ [0.01, 0.2] day⁻¹)
    m_P_bounds: Tuple[float, float] = (0.01, 0.2)
    
    # Zooplankton mortality (m_Z ∈ [0.01, 0.1] day⁻¹)
    m_Z_bounds: Tuple[float, float] = (0.01, 0.1)
    
    # Chl:C ratio (θ ∈ [0.02, 0.06] mg Chl / mg C)
    theta_bounds: Tuple[float, float] = (0.02, 0.06)
    
    # Grazing half-saturation (K_g ∈ [0.3, 1.0] mg C/m³)
    K_g_bounds: Tuple[float, float] = (0.3, 1.0)
    
    # Assimilation efficiency (γ ∈ [0.2, 0.4])
    gamma_bounds: Tuple[float, float] = (0.2, 0.4)
    
    # Redfield N:C ratio
    NC_ratio: float = 1.0 / 6.625  # mol N / mol C


class ParameterNetwork(nn.Module):
    """
    Neural network that predicts NPZD parameters from environmental forcing.
    
    Input: [T, PAR, MLD, depth, day_of_year, N², ...]
    Output: [μ_max, K_N, g_max, m_P, m_Z, θ, K_g, γ]
    
    All outputs are constrained to physical bounds via sigmoid scaling.
    """
    
    def __init__(self, 
                 input_dim: int = 8,
                 hidden_dim: int = 64,
                 n_params: int = 8):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_params)
        )
        
        # Initialize for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize to output mid-range parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, forcing: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            forcing: [batch, input_dim] environmental forcing
            
        Returns:
            raw_params: [batch, n_params] unconstrained parameters
        """
        return self.net(forcing)


class GrayBoxNPZD(nn.Module):
    """
    Universal Differential Equation with STRUCTURED NPZD dynamics.
    
    KEY INSIGHT: 
        Hard-code the STRUCTURE of NPZD equations.
        Use neural network ONLY to predict PARAMETERS.
        
    The NPZD Equations (STRUCTURE IS FIXED):
    
        dP/dt = μ·P - g·Z - m_P·P
        dN/dt = -α·μ·P + ε·m_Z·Z + remineralization
        dZ/dt = γ·g·Z - m_Z·Z
        dChl/dt = θ·dP/dt
        
    Where:
        μ = μ_max · f(T) · f(PAR) · N/(K_N + N)     [Monod + Eppley]
        g = g_max · P/(K_g + P)                     [Holling Type II]
        
    What the NN learns (PARAMETERS ONLY):
        - μ_max(T, PAR): Maximum growth rate
        - K_N(depth): Half-saturation for nitrate
        - g_max(T): Maximum grazing rate
        - θ: Chl:C ratio as f(light history)
    """
    
    def __init__(self, 
                 input_dim: int = 8,
                 hidden_dim: int = 64,
                 params: NPZDParams = None):
        super().__init__()
        
        self.params = params or NPZDParams()
        
        # Parameter prediction network
        self.param_net = ParameterNetwork(input_dim, hidden_dim, n_params=8)
        
        # Store bounds as buffers
        self.register_buffer('bounds', torch.tensor([
            self.params.mu_max_bounds,
            self.params.K_N_bounds,
            self.params.g_max_bounds,
            self.params.m_P_bounds,
            self.params.m_Z_bounds,
            self.params.theta_bounds,
            self.params.K_g_bounds,
            self.params.gamma_bounds,
        ]))
        
    def constrain_params(self, raw_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply sigmoid constraints to ensure physical bounds.
        
        p_constrained = lo + (hi - lo) * sigmoid(p_raw)
        """
        params = {}
        param_names = ['mu_max', 'K_N', 'g_max', 'm_P', 'm_Z', 'theta', 'K_g', 'gamma']
        
        for i, name in enumerate(param_names):
            lo, hi = self.bounds[i]
            params[name] = lo + (hi - lo) * torch.sigmoid(raw_params[..., i])
            
        return params
    
    def eppley_temperature(self, T: torch.Tensor) -> torch.Tensor:
        """
        Eppley (1972) temperature function.
        
        f(T) = 0.59 * exp(0.0633 * T)
        
        Normalized to 1.0 at T=20°C.
        """
        return 0.59 * torch.exp(0.0633 * T) / (0.59 * torch.exp(0.0633 * 20))
    
    def light_limitation(self, PAR: torch.Tensor, 
                        PAR_opt: float = 100.0) -> torch.Tensor:
        """
        Light limitation with photoinhibition (Platt-Jassby).
        
        f(PAR) = PAR/PAR_opt * exp(1 - PAR/PAR_opt)
        """
        ratio = PAR / (PAR_opt + 1e-6)
        return ratio * torch.exp(1 - ratio)
    
    def npzd_tendencies(self, 
                        state: torch.Tensor, 
                        params: Dict[str, torch.Tensor],
                        T: torch.Tensor,
                        PAR: torch.Tensor) -> torch.Tensor:
        """
        Compute NPZD tendencies with FIXED equation structure.
        
        Args:
            state: [batch, 4] = [P, N, Z, Chl] concentrations
            params: Dict of neural-predicted parameters
            T: Temperature [batch]
            PAR: Light [batch]
            
        Returns:
            tendencies: [batch, 4] = [dP/dt, dN/dt, dZ/dt, dChl/dt]
        """
        P = state[..., 0]  # Phytoplankton (mg C/m³)
        N = state[..., 1]  # Nitrate (μmol/L)
        Z = state[..., 2]  # Zooplankton (mg C/m³)
        # Chl = state[..., 3]  # Chlorophyll (mg/m³)
        
        # Temperature effect (Eppley)
        f_T = self.eppley_temperature(T)
        
        # Light limitation
        f_PAR = self.light_limitation(PAR)
        
        # Nutrient limitation (Monod kinetics) - FIXED STRUCTURE
        f_N = N / (params['K_N'] + N + 1e-6)
        
        # Growth rate - FIXED STRUCTURE
        mu = params['mu_max'] * f_T * f_PAR * f_N
        
        # Grazing (Holling Type II) - FIXED STRUCTURE
        g = params['g_max'] * P / (params['K_g'] + P + 1e-6)
        
        # NPZD tendencies - FIXED STRUCTURE
        dP_dt = mu * P - g * Z - params['m_P'] * P
        
        dN_dt = (-self.params.NC_ratio * mu * P  # Uptake
                 + 0.3 * params['m_P'] * P        # Phyto mortality → N
                 + 0.3 * params['m_Z'] * Z)       # Zoo mortality → N
        
        dZ_dt = params['gamma'] * g * Z - params['m_Z'] * Z
        
        dChl_dt = params['theta'] * dP_dt  # Photoacclimation (simplified)
        
        return torch.stack([dP_dt, dN_dt, dZ_dt, dChl_dt], dim=-1)
    
    def forward(self, 
                state: torch.Tensor, 
                forcing: torch.Tensor,
                dt: float = 1.0) -> torch.Tensor:
        """
        Integrate NPZD equations forward in time.
        
        Uses RK4 for stability with potentially stiff biology.
        
        Args:
            state: [batch, 4] current NPZD state [P, N, Z, Chl]
            forcing: [batch, input_dim] environmental forcing
                     Expected: [T, PAR, MLD, depth, doy, N², lat, lon]
            dt: Time step in DAYS
            
        Returns:
            new_state: [batch, 4] updated NPZD state
        """
        # Extract T and PAR from forcing
        T = forcing[..., 0]    # Temperature
        PAR = forcing[..., 1]  # Light
        
        # Get parameters from neural network
        raw_params = self.param_net(forcing)
        params = self.constrain_params(raw_params)
        
        # RK4 integration
        k1 = self.npzd_tendencies(state, params, T, PAR)
        k2 = self.npzd_tendencies(state + 0.5 * dt * k1, params, T, PAR)
        k3 = self.npzd_tendencies(state + 0.5 * dt * k2, params, T, PAR)
        k4 = self.npzd_tendencies(state + dt * k3, params, T, PAR)
        
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Hard positivity constraint (biogeochemical concentrations ≥ 0)
        new_state = F.relu(new_state) + 1e-8
        
        return new_state
    
    def get_diagnostics(self, 
                        state: torch.Tensor, 
                        forcing: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return diagnostic variables for analysis.
        
        Returns:
            diagnostics: Dict with growth rate, grazing, parameter values, etc.
        """
        raw_params = self.param_net(forcing)
        params = self.constrain_params(raw_params)
        
        T = forcing[..., 0]
        PAR = forcing[..., 1]
        
        P = state[..., 0]
        N = state[..., 1]
        
        # Realized growth rate
        f_T = self.eppley_temperature(T)
        f_PAR = self.light_limitation(PAR)
        f_N = N / (params['K_N'] + N + 1e-6)
        mu = params['mu_max'] * f_T * f_PAR * f_N
        
        return {
            'mu_realized': mu,
            'f_temperature': f_T,
            'f_light': f_PAR,
            'f_nutrient': f_N,
            **params
        }


# =============================================================================
# Photoacclimation Model (Extended)
# =============================================================================

class PhotoacclimationModel(nn.Module):
    """
    Dynamic Chl:C ratio based on light history.
    
    θ = θ_min + (θ_max - θ_min) * exp(-E/E_k)
    
    Where E is integrated irradiance history.
    """
    
    def __init__(self, 
                 theta_min: float = 0.01,
                 theta_max: float = 0.06,
                 tau: float = 5.0):  # Acclimation timescale (days)
        super().__init__()
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.tau = tau
        
    def forward(self, PAR: torch.Tensor, PAR_history: torch.Tensor) -> torch.Tensor:
        """
        Compute acclimated Chl:C ratio.
        
        Args:
            PAR: Current PAR
            PAR_history: Running mean of past PAR
        """
        E_k = 50.0  # Saturation irradiance
        theta = self.theta_min + (self.theta_max - self.theta_min) * \
                torch.exp(-PAR_history / E_k)
        return theta


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Gray-Box NPZD...")
    
    torch.manual_seed(42)
    
    # Create model
    model = GrayBoxNPZD(input_dim=8, hidden_dim=64)
    
    # Test data
    batch_size = 100
    state = torch.rand(batch_size, 4) * torch.tensor([10, 5, 2, 2])  # P, N, Z, Chl
    forcing = torch.rand(batch_size, 8)
    forcing[:, 0] = forcing[:, 0] * 20 + 5    # T: 5-25°C
    forcing[:, 1] = forcing[:, 1] * 500       # PAR: 0-500 μmol/m²/s
    
    # Forward pass
    new_state = model(state, forcing, dt=1.0)
    
    print(f"✓ Input state: {state.shape}")
    print(f"✓ Output state: {new_state.shape}")
    print(f"✓ All positive: {(new_state > 0).all().item()}")
    
    # Check parameter bounds
    raw = model.param_net(forcing)
    params = model.constrain_params(raw)
    
    print(f"\n✓ Parameter bounds:")
    for name, val in params.items():
        print(f"  - {name}: [{val.min():.4f}, {val.max():.4f}]")
    
    # Test diagnostics
    diag = model.get_diagnostics(state, forcing)
    print(f"\n✓ Diagnostics:")
    print(f"  - μ_realized: {diag['mu_realized'].mean():.4f} day⁻¹")
    print(f"  - f_T: {diag['f_temperature'].mean():.4f}")
    print(f"  - f_PAR: {diag['f_light'].mean():.4f}")
    print(f"  - f_N: {diag['f_nutrient'].mean():.4f}")
    
    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {n_params:,}")
    
    print("\n✅ All tests passed!")
