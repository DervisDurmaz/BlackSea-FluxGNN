"""
Hybrid-FluxGNN: Black Sea Biogeochemical Forecasting.

A physics-informed Graph Neural Network for Black Sea chlorophyll prediction.

Architecture:
    - Neural-Reaction / Numerical-Transport hybrid
    - Gray-Box UDE for NPZD (learns parameters, not dC/dt)
    - Differentiable FVM for conservative transport
    - Split-Kernel message passing for anisotropic ocean physics
    - FCT limiter for hard positivity constraints
"""

__version__ = "1.0.0"
__author__ = "N/A"
__license__ = "MIT"

# Package imports
from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent

# Submodule availability flags
MESH_AVAILABLE = True
MODELS_AVAILABLE = True
SOLVERS_AVAILABLE = True
TRAINING_AVAILABLE = True
PREPROCESSING_AVAILABLE = True
VISUALIZATION_AVAILABLE = True

# Lazy imports
def get_model():
    """Get the main HybridFluxGNN model."""
    from models.hybrid_fluxgnn import HybridFluxGNN
    return HybridFluxGNN

def get_graph():
    """Get the PrismaticGraph class."""
    from mesh.prismatic_graph import PrismaticGraph
    return PrismaticGraph

def get_trainer():
    """Get the CurriculumTrainer class."""
    from training.curriculum import CurriculumTrainer
    return CurriculumTrainer

def get_loss():
    """Get the MultivariateNPZDLoss class."""
    from training.loss import MultivariateNPZDLoss
    return MultivariateNPZDLoss

# Version info
def version_info():
    """Print version and component information."""
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     HYBRID-FLUXGNN v{__version__}                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Mesh:           prismatic_graph.py (σ-coordinates, Delaunay)        ║
║  Models:         gray_box_ude.py, split_kernel_mp.py, hybrid_fluxgnn ║
║  Solvers:        differentiable_fvm.py, fct_limiter.py, operator_split║
║  Training:       curriculum.py, loss.py, hpo.py, ensemble.py         ║
║  Preprocessing:  oceanographic_utils.py, woa_loader.py               ║
║  Visualization:  nature_figures.py (15 templates)                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

# Print on import
print(f"Hybrid-FluxGNN v{__version__} loaded")
