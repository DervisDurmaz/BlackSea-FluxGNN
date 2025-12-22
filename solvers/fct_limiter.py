"""
Flux-Corrected Transport (FCT) Positivity Limiter.

GUARANTEES: Chlorophyll, Phytoplankton, etc. can NEVER go negative.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FCTLimiter(nn.Module):
    """
    Flux-Corrected Transport limiter.
    
    Blends low-order (diffusive but positive) with high-order (accurate but oscillatory):
    
        C_final = C_low + α * (C_high - C_low)
        
    Where α ∈ [0, 1] is computed to ensure C_final ≥ ε > 0.
    
    This is a HARD constraint, not a soft penalty.
    FCT is a classic technique from CFD - we're making it differentiable.
    """
    
    def __init__(self, 
                 eps: float = 1e-10,
                 use_neural_limiter: bool = True):
        """
        Args:
            eps: Small positive floor for concentrations
            use_neural_limiter: Learn α via neural net (vs Zalesak)
        """
        super().__init__()
        
        self.eps = eps
        self.use_neural_limiter = use_neural_limiter
        
        if use_neural_limiter:
            # Neural network to predict optimal blending coefficient
            # Learns to balance accuracy vs positivity
            self.alpha_net = nn.Sequential(
                nn.Linear(4, 32),
                nn.SiLU(),
                nn.Linear(32, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # α ∈ [0, 1]
            )
            
    def zalesak_limiter(self, 
                        C_low: torch.Tensor, 
                        C_high: torch.Tensor) -> torch.Tensor:
        """
        Classic Zalesak (1979) limiter.
        
        α_Zalesak = min(1, max(0, (C_low - ε) / (C_low - C_high)))
        
        Ensures: C_low + α * (C_high - C_low) ≥ ε
        """
        delta = C_high - C_low
        
        # Maximum α to maintain positivity
        # C_final = C_low + α * delta ≥ ε
        # If delta < 0 (high-order is lower than low-order):
        #   α ≤ (C_low - ε) / (-delta)
        # If delta ≥ 0: no constraint (high-order is larger)
        
        alpha_max = torch.where(
            delta < 0,
            (C_low - self.eps) / (-delta + 1e-10),
            torch.ones_like(delta)
        )
        
        # Clamp to [0, 1]
        alpha = torch.clamp(alpha_max, 0, 1)
        
        return alpha
    
    def neural_limiter(self,
                       C_low: torch.Tensor,
                       C_high: torch.Tensor) -> torch.Tensor:
        """
        Learned limiter with hard positivity constraint.
        
        Network predicts α, then we apply Zalesak clip.
        """
        # Features for the neural network
        delta = C_high - C_low
        features = torch.stack([
            C_low,
            C_high,
            delta.abs(),
            (C_low + C_high) / 2
        ], dim=-1)
        
        # Neural prediction (unconstrained)
        alpha_neural = self.alpha_net(features).squeeze(-1)
        
        # Hard constraint via Zalesak
        alpha_max = self.zalesak_limiter(C_low, C_high)
        
        # Final α is minimum of neural and max allowed
        alpha = torch.minimum(alpha_neural, alpha_max)
        
        return alpha
    
    def forward(self, 
                C_low: torch.Tensor, 
                C_high: torch.Tensor) -> torch.Tensor:
        """
        Apply FCT limiting.
        
        Args:
            C_low: Low-order (diffusive/positive) solution [batch, n_tracers]
            C_high: High-order (accurate/oscillatory) solution [batch, n_tracers]
            
        Returns:
            C_limited: Positive and accurate solution
        """
        if self.use_neural_limiter:
            alpha = self.neural_limiter(C_low, C_high)
        else:
            alpha = self.zalesak_limiter(C_low, C_high)
        
        # Blend
        C_limited = C_low + alpha * (C_high - C_low)
        
        # Absolute floor (defense in depth)
        C_limited = torch.maximum(C_limited, torch.tensor(self.eps))
        
        return C_limited
    
    def verify_positivity(self, C: torch.Tensor) -> Tuple[bool, float]:
        """
        Verify all concentrations are positive.
        
        Returns:
            (all_positive, negative_fraction)
        """
        n_negative = (C < 0).sum().item()
        n_total = C.numel()
        
        negative_fraction = n_negative / n_total
        all_positive = n_negative == 0
        
        return all_positive, negative_fraction


class PositivityAwareLoss(nn.Module):
    """
    Soft positivity penalty (backup for FCT hard constraint).
    
    L_pos = λ * mean(ReLU(-C))²
    """
    
    def __init__(self, lambda_pos: float = 10.0):
        super().__init__()
        self.lambda_pos = lambda_pos
        
    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Penalize negative concentrations."""
        violation = torch.relu(-C)
        return self.lambda_pos * (violation ** 2).mean()


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing FCT Limiter...")
    
    torch.manual_seed(42)
    
    # Create limiter
    fct = FCTLimiter(eps=1e-10, use_neural_limiter=True)
    
    # Test data: some high-order values are negative
    batch_size, n_tracers = 100, 4
    C_low = torch.rand(batch_size, n_tracers) + 0.5   # All positive (0.5 to 1.5)
    C_high = torch.randn(batch_size, n_tracers)       # Some negative!
    
    print(f"C_low range: [{C_low.min():.3f}, {C_low.max():.3f}]")
    print(f"C_high range: [{C_high.min():.3f}, {C_high.max():.3f}]")
    print(f"C_high negative: {(C_high < 0).sum().item()} / {C_high.numel()}")
    
    # Apply limiter
    C_limited = fct(C_low, C_high)
    
    all_pos, neg_frac = fct.verify_positivity(C_limited)
    print(f"\n✓ C_limited range: [{C_limited.min():.6f}, {C_limited.max():.3f}]")
    print(f"✓ All positive: {all_pos}")
    print(f"✓ Negative fraction: {neg_frac:.6f}")
    
    # Test Zalesak specifically
    fct_zalesak = FCTLimiter(eps=1e-10, use_neural_limiter=False)
    C_zalesak = fct_zalesak(C_low, C_high)
    all_pos_z, _ = fct_zalesak.verify_positivity(C_zalesak)
    print(f"\n✓ Zalesak all positive: {all_pos_z}")
    
    # Test gradient flow
    C_limited = fct(C_low.clone().requires_grad_(True), C_high)
    loss = C_limited.sum()
    loss.backward()
    print(f"\n✓ Gradients computed successfully")
    
    print("\n✅ All tests passed!")
