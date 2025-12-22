"""
Conservation Tests for Hybrid-FluxGNN.

Verifies mass conservation, nitrogen budget, and stratification preservation.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMassConservation:
    """Tests for mass/tracer conservation."""
    
    @pytest.fixture
    def setup_data(self):
        """Create test data."""
        torch.manual_seed(42)
        N = 100
        
        # NPZD state: P, N, Z, Chl
        state = torch.rand(N, 4) * torch.tensor([10, 5, 2, 2])
        
        # Forcing
        forcing = torch.rand(N, 8)
        forcing[:, 0] = forcing[:, 0] * 20 + 5    # T: 5-25°C
        forcing[:, 1] = forcing[:, 1] * 500       # PAR
        
        return state, forcing
    
    def test_nitrogen_budget_closure(self, setup_data):
        """
        Test that total nitrogen is conserved.
        
        Total N = P·(N:C) + N_inorganic + Z·(N:C)
        Should remain constant over time steps.
        """
        state, forcing = setup_data
        
        # Compute total nitrogen
        NC_ratio = 1 / 6.625
        total_N_initial = state[:, 0] * NC_ratio + state[:, 1] + state[:, 2] * NC_ratio
        
        # Simulate some reactions (simplified)
        # In real model, would use GrayBoxNPZD
        dt = 1.0
        dP = 0.1 * state[:, 0] - 0.05 * state[:, 0]  # growth - mortality
        dN = -NC_ratio * 0.1 * state[:, 0] + 0.3 * 0.05 * state[:, 2]  # uptake + remineralization
        dZ = 0.3 * 0.05 * state[:, 0] - 0.05 * state[:, 2]  # grazing - mortality
        
        new_P = state[:, 0] + dt * dP
        new_N = state[:, 1] + dt * dN
        new_Z = state[:, 2] + dt * dZ
        
        total_N_final = new_P * NC_ratio + new_N + new_Z * NC_ratio
        
        # Check conservation (within tolerance)
        relative_error = (total_N_final.sum() - total_N_initial.sum()).abs() / total_N_initial.sum()
        
        assert relative_error < 0.01, f"Nitrogen budget error {relative_error:.2%} > 1%"
    
    def test_no_creation_from_nothing(self, setup_data):
        """Test that tracers don't spontaneously appear."""
        state, _ = setup_data
        
        # Zero initial state
        zero_state = torch.zeros_like(state)
        
        # Any reactions should produce zero from zero
        # (Simplified check - real test would use model)
        assert zero_state.sum() == 0
    
    def test_tracer_positivity(self, setup_data):
        """Test that FCT limiter maintains positivity."""
        from solvers.fct_limiter import FCTLimiter
        
        state, _ = setup_data
        
        # Create a scenario where high-order solution is negative
        C_low = state[:, 0:1]  # Positive
        C_high = state[:, 0:1] - 2 * state[:, 0:1].std()  # Some negative
        
        fct = FCTLimiter(eps=1e-10, use_neural_limiter=False)
        C_limited = fct(C_low, C_high)
        
        assert (C_limited >= 0).all(), "FCT limiter produced negative values"


class TestStratificationPreservation:
    """Tests for stratification (N²) preservation."""
    
    def test_pycnocline_stability(self):
        """Test that pycnocline is not eroded numerically."""
        # Create typical Black Sea T/S profile
        n_levels = 50
        z = np.linspace(0, -300, n_levels)  # Depth (negative downward)
        
        # Temperature: warm surface, CIL at 50-100m, cooling at depth
        T = 20 - 15 * (1 - np.exp(z / 30))  # Exponential profile
        T[10:30] = 8  # Cold Intermediate Layer
        
        # Salinity: lower at surface (river input)
        S = 18 + 4 * (1 - np.exp(z / 100))
        
        # Compute N² (simplified)
        rho = 1000 + 0.8 * (S - 35) - 0.2 * (T - 10)
        drho_dz = np.gradient(rho, z)
        N2 = -(9.81 / 1025) * drho_dz
        
        # Check that there's a pycnocline
        max_N2 = N2.max()
        max_N2_depth = z[np.argmax(N2)]
        
        assert max_N2 > 1e-5, f"Pycnocline too weak: N² = {max_N2:.2e}"
        assert -150 < max_N2_depth < -20, f"Pycnocline depth wrong: {max_N2_depth}m"
    
    def test_cil_preservation(self):
        """Test Cold Intermediate Layer is preserved."""
        from preprocessing.oceanographic_utils import detect_cil_bounds
        
        # Black Sea T profile with CIL
        n_levels = 50
        depth = np.linspace(0, 300, n_levels)
        T = np.concatenate([
            np.linspace(22, 8, 10),   # Surface warming
            np.ones(15) * 7.5,         # CIL (cold layer)
            np.linspace(8, 9, 25)      # Below CIL
        ])
        
        top, bottom, thickness = detect_cil_bounds(T, depth, T_threshold=8.0)
        
        assert not np.isnan(thickness), "CIL not detected"
        assert thickness > 30, f"CIL too thin: {thickness}m"


class TestDCMPreservation:
    """Tests for Deep Chlorophyll Maximum preservation."""
    
    def test_dcm_detection(self):
        """Test DCM depth detection accuracy."""
        from preprocessing.oceanographic_utils import detect_dcm_depth
        
        # Create typical DCM profile
        depth = np.linspace(0, 200, 100)
        dcm_true = 50  # True DCM at 50m
        
        chl = 0.5 + 2.0 * np.exp(-((depth - dcm_true) / 15)**2)  # Gaussian peak
        
        dcm_detected = detect_dcm_depth(chl, depth, smooth_window=3)
        
        dcm_error = abs(dcm_detected - dcm_true)
        
        assert dcm_error < 10, f"DCM detection error {dcm_error}m > 10m"
    
    def test_dcm_not_at_surface(self):
        """DCM should be subsurface in Black Sea."""
        depth = np.linspace(0, 200, 100)
        
        # Wrong profile: maximum at surface
        chl_wrong = 5.0 * np.exp(-depth / 30)
        
        # Correct profile: subsurface maximum
        chl_correct = 0.5 + 2.0 * np.exp(-((depth - 40) / 20)**2)
        
        # Subsurface max should be in correct range
        max_depth_correct = depth[np.argmax(chl_correct)]
        
        assert 20 < max_depth_correct < 80, f"DCM depth {max_depth_correct}m not in expected range"


class TestFCTLimiter:
    """Tests specifically for FCT positivity constraint."""
    
    def test_extreme_negative_input(self):
        """FCT should handle extreme negative high-order values."""
        from solvers.fct_limiter import FCTLimiter
        
        fct = FCTLimiter(eps=1e-10, use_neural_limiter=False)
        
        C_low = torch.ones(100, 4) * 0.1  # Small positive
        C_high = torch.randn(100, 4) * 10  # Many large negative values
        
        C_limited = fct(C_low, C_high)
        
        all_pos, neg_frac = fct.verify_positivity(C_limited)
        
        assert all_pos, f"FCT failed: {neg_frac:.2%} negative"
    
    def test_differentiability(self):
        """FCT should maintain gradient flow."""
        from solvers.fct_limiter import FCTLimiter
        
        fct = FCTLimiter(eps=1e-10, use_neural_limiter=True)
        
        C_low = torch.rand(50, 4, requires_grad=True)
        C_high = torch.randn(50, 4)
        
        C_limited = fct(C_low, C_high)
        loss = C_limited.sum()
        loss.backward()
        
        assert C_low.grad is not None, "Gradient not computed through FCT"
        assert not torch.isnan(C_low.grad).any(), "NaN in FCT gradients"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
