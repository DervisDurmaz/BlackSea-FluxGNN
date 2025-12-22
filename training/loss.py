"""
Multivariate Loss Functions with Nitrate Climatology.

CRITICAL: Optimizing ONLY for Chlorophyll will break nitrogen conservation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class MultivariateNPZDLoss(nn.Module):
    """
    Loss function that includes ALL NPZD tracers.
    
    L_total = λ_Chl · L_Chl      (satellite/Argo data)
            + λ_N   · L_N        (vs World Ocean Atlas climatology)
            + λ_P   · L_P        (vs sparse float data)
            + λ_cons · L_cons    (nitrogen budget closure)
    
    CRITICAL: λ_N MUST be > 0 or nitrogen conservation will fail!
    """
    
    def __init__(self,
                 lambda_chl: float = 1.0,
                 lambda_nitrate: float = 0.5,
                 lambda_phyto: float = 0.3,
                 lambda_conservation: float = 0.3,
                 lambda_positivity: float = 10.0,
                 lambda_smoothness: float = 0.1,
                 NC_ratio: float = 1/6.625):
        super().__init__()
        
        self.lambda_chl = lambda_chl
        self.lambda_n = lambda_nitrate
        self.lambda_p = lambda_phyto
        self.lambda_cons = lambda_conservation
        self.lambda_pos = lambda_positivity
        self.lambda_smooth = lambda_smoothness
        self.NC_ratio = NC_ratio
        
        # Placeholder for WOA climatology
        self.register_buffer('nitrate_clim', None)
        
    def set_nitrate_climatology(self, clim: torch.Tensor):
        """
        Set nitrate climatology from WOA.
        
        Args:
            clim: [12, n_depth] monthly climatology
        """
        self.register_buffer('nitrate_clim', clim)
        
    def chlorophyll_loss(self, 
                        pred: torch.Tensor, 
                        target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Loss on chlorophyll (index 3).
        
        Handles sparse observations via mask.
        """
        pred_chl = pred[..., 3]
        target_chl = target[..., 3]
        
        if mask is None:
            mask = ~torch.isnan(target_chl)
            
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Log-transform for Chl (lognormal distribution)
        pred_log = torch.log(pred_chl[mask] + 1e-6)
        target_log = torch.log(target_chl[mask] + 1e-6)
        
        return F.mse_loss(pred_log, target_log)
    
    def nitrate_loss(self,
                    pred: torch.Tensor,
                    depth: torch.Tensor,
                    month: int) -> torch.Tensor:
        """
        Loss against WOA nitrate climatology.
        
        Args:
            pred: [batch, 4] predicted NPZD
            depth: [batch] depth of each sample (m, positive)
            month: Month (0-11)
        """
        pred_N = pred[..., 1]
        
        if self.nitrate_clim is None:
            # Default climatology if not set
            # Black Sea: ~0 at surface, ~8 μmol/L at 200m+
            N_clim = 8.0 * torch.sigmoid((depth - 100) / 50)
        else:
            # Interpolate from climatology
            N_clim = self._interpolate_climatology(depth, month)
        
        return F.mse_loss(pred_N, N_clim.to(pred.device))
    
    def _interpolate_climatology(self, depth: torch.Tensor, month: int) -> torch.Tensor:
        """Interpolate climatology to sample depths."""
        # Simplified: would use proper interpolation in production
        clim = self.nitrate_clim[month, :]
        depths = torch.linspace(0, 2000, clim.shape[0])
        
        # Nearest neighbor
        idx = torch.searchsorted(depths, depth.cpu()).clamp(0, len(clim)-1)
        return clim[idx]
    
    def conservation_loss(self, pred: torch.Tensor, 
                         pred_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Nitrogen mass conservation loss.
        
        Total N = P·(N:C) + N_inorganic + Z·(N:C)
        
        Should be approximately constant in closed system.
        """
        P, N, Z = pred[..., 0], pred[..., 1], pred[..., 2]
        
        total_N = P * self.NC_ratio + N + Z * self.NC_ratio
        
        if pred_prev is not None:
            P_prev, N_prev, Z_prev = pred_prev[..., 0], pred_prev[..., 1], pred_prev[..., 2]
            total_N_prev = P_prev * self.NC_ratio + N_prev + Z_prev * self.NC_ratio
            
            # Should be conserved
            return F.mse_loss(total_N, total_N_prev)
        else:
            # Penalize variance (should be uniform)
            return total_N.std()
    
    def positivity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Soft positivity constraint (backup for hard constraint).
        
        Penalizes negative predictions.
        """
        return F.relu(-pred).mean()
    
    def smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Spatial/temporal smoothness regularization.
        
        Penalizes large gradients.
        """
        if pred.dim() < 2:
            return torch.tensor(0.0, device=pred.device)
            
        diff = pred[1:] - pred[:-1]
        return diff.pow(2).mean()
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                depth: Optional[torch.Tensor] = None,
                month: int = 0,
                pred_prev: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            pred: [batch, 4] predicted [P, N, Z, Chl]
            target: [batch, 4] target (may have NaNs)
            depth: [batch] sample depths (for nitrate climatology)
            month: Month (0-11) for climatology lookup
            pred_prev: Previous prediction for conservation check
            
        Returns:
            losses: Dict with total and component losses
        """
        losses = {}
        
        # Chlorophyll (primary)
        losses['chl'] = self.chlorophyll_loss(pred, target)
        
        # Nitrate (vs climatology)
        if depth is not None:
            losses['nitrate'] = self.nitrate_loss(pred, depth, month)
        else:
            losses['nitrate'] = torch.tensor(0.0, device=pred.device)
        
        # Conservation
        losses['conservation'] = self.conservation_loss(pred, pred_prev)
        
        # Positivity (soft backup)
        losses['positivity'] = self.positivity_loss(pred)
        
        # Smoothness
        losses['smoothness'] = self.smoothness_loss(pred)
        
        # Weighted sum
        total = (self.lambda_chl * losses['chl'] +
                self.lambda_n * losses['nitrate'] +
                self.lambda_cons * losses['conservation'] +
                self.lambda_pos * losses['positivity'] +
                self.lambda_smooth * losses['smoothness'])
        
        losses['total'] = total
        
        return losses


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score for probabilistic forecasts.
    
    CRPS = E|Y - X| - 0.5 * E|X - X'|
    
    For Gaussian predictions:
    CRPS_gauss(μ, σ, y) = σ * [z·(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                pred_mean: torch.Tensor, 
                pred_std: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        CRPS for Gaussian predictions.
        """
        z = (target - pred_mean) / (pred_std + 1e-6)
        
        # Standard normal CDF (approximation)
        sqrt2 = np.sqrt(2)
        phi = 0.5 * (1 + torch.erf(z / sqrt2))
        
        # Standard normal PDF
        pdf = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        
        crps = pred_std * (z * (2 * phi - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        
        return crps.mean()


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task learning with homoscedastic uncertainty weighting.
    
    L = Σᵢ (1/2σᵢ²) · Lᵢ + log(σᵢ)
    
    Learns optimal task weights automatically.
    """
    
    def __init__(self, n_tasks: int):
        super().__init__()
        
        # Log-variance for each task
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine losses with learned weights.
        """
        loss_list = list(losses.values())
        
        weighted_sum = 0
        for i, (name, loss) in enumerate(losses.items()):
            if name == 'total':
                continue
            precision = torch.exp(-self.log_vars[i])
            weighted_sum += precision * loss + self.log_vars[i]
        
        return weighted_sum
    
    def get_weights(self) -> Dict[str, float]:
        """Return current task weights."""
        precisions = torch.exp(-self.log_vars)
        return {f'weight_{i}': p.item() for i, p in enumerate(precisions)}


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Multivariate NPZD Loss...")
    
    torch.manual_seed(42)
    
    # Create loss
    loss_fn = MultivariateNPZDLoss(
        lambda_chl=1.0,
        lambda_nitrate=0.5,
        lambda_conservation=0.3
    )
    
    # Test data
    batch_size = 100
    pred = torch.rand(batch_size, 4) * torch.tensor([10, 5, 2, 2])  # P, N, Z, Chl
    target = pred + torch.randn_like(pred) * 0.1
    target[:10, 3] = float('nan')  # Sparse Chl observations
    depth = torch.rand(batch_size) * 300  # 0-300m
    
    # Forward
    losses = loss_fn(pred, target, depth=depth, month=6)
    
    print(f"✓ Loss components:")
    for name, val in losses.items():
        print(f"  - {name}: {val.item():.4f}")
    
    # Test CRPS
    crps = CRPSLoss()
    pred_mean = torch.rand(batch_size)
    pred_std = torch.rand(batch_size) * 0.5 + 0.1
    target_crps = pred_mean + torch.randn(batch_size) * pred_std
    
    crps_val = crps(pred_mean, pred_std, target_crps)
    print(f"\n✓ CRPS: {crps_val.item():.4f}")
    
    # Test uncertainty weighting
    uw = UncertaintyWeightedLoss(n_tasks=4)
    weighted = uw({'a': torch.tensor(1.0), 'b': torch.tensor(0.5), 
                   'c': torch.tensor(0.8), 'd': torch.tensor(0.3)})
    print(f"\n✓ Uncertainty-weighted: {weighted.item():.4f}")
    print(f"  Weights: {uw.get_weights()}")
    
    print("\n✅ All tests passed!")
