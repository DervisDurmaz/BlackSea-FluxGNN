"""
Colab Setup Cell - Copy this to the first cell of your notebook

This sets up the environment for Google Colab, including:
- Cloning the repository
- Installing dependencies  
- Mounting Google Drive for data
- Configuring paths
"""

# ═══════════════════════════════════════════════════════════════════════════════
# COLAB SETUP - Run this cell first!
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🌊 Setting up Hybrid-FluxGNN in Google Colab...")
    
    # ════════════════════════════════════════════════════════════════════════
    # OPTION 1: Clone from GitHub (uncomment if using GitHub)
    # ════════════════════════════════════════════════════════════════════════
    
    # REPO_URL = "https://github.com/YOUR_USERNAME/BlackSea-FluxGNN.git"
    # 
    # if not os.path.exists('/content/BlackSea-FluxGNN'):
    #     !git clone {REPO_URL}
    # 
    # %cd /content/BlackSea-FluxGNN
    # sys.path.insert(0, '/content/BlackSea-FluxGNN')
    
    # ════════════════════════════════════════════════════════════════════════
    # OPTION 2: Mount Google Drive (if you uploaded files to Drive)
    # ════════════════════════════════════════════════════════════════════════
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Path to your project in Google Drive
    PROJECT_PATH = '/content/drive/MyDrive/PINN'
    DATA_PATH = '/content/drive/MyDrive/PINN'
    V6_PATH = '/content/drive/MyDrive/PINN/v6'
    
    # Add to Python path
    sys.path.insert(0, PROJECT_PATH)
    
    print(f"   ✓ Project path: {PROJECT_PATH}")
    print(f"   ✓ Data path: {DATA_PATH}")
    print(f"   ✓ V6 path: {V6_PATH}")
    
    # ════════════════════════════════════════════════════════════════════════
    # Install dependencies
    # ════════════════════════════════════════════════════════════════════════
    
    print("\n📦 Installing dependencies...")
    
    # Core packages (most are pre-installed in Colab)
    !pip install -q xarray netCDF4 optuna gsw
    
    # PyTorch Geometric (Colab-specific)
    import torch
    TORCH_VERSION = torch.__version__.split('+')[0]
    CUDA_VERSION = torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
    
    !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
    !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
    !pip install -q torch-geometric
    
    print("   ✓ Dependencies installed")
    
    # ════════════════════════════════════════════════════════════════════════
    # Configure environment
    # ════════════════════════════════════════════════════════════════════════
    
    # Set environment variables for data paths
    os.environ['BLACKSEA_DATA_DIR'] = DATA_PATH
    os.environ['BLACKSEA_V6_DIR'] = V6_PATH
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️ No GPU detected - using CPU")
    
    print("\n" + "═"*60)
    print("✅ COLAB SETUP COMPLETE!")
    print("═"*60)
    
else:
    print("Running locally - no Colab setup needed")
    PROJECT_PATH = os.getcwd()
    DATA_PATH = PROJECT_PATH
    V6_PATH = os.path.join(PROJECT_PATH, 'data', 'v6_outputs')


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def verify_setup():
    """Verify that everything is set up correctly."""
    print("\n🔍 Verifying setup...")
    
    errors = []
    
    # Check imports
    try:
        import torch
        import numpy as np
        print(f"   ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
    
    try:
        import xarray as xr
        print(f"   ✓ xarray available")
    except ImportError as e:
        errors.append(f"xarray: {e}")
    
    try:
        from torch_geometric.data import Data
        print(f"   ✓ PyTorch Geometric available")
    except ImportError as e:
        errors.append(f"PyG: {e}")
    
    # Check data path
    if os.path.exists(DATA_PATH):
        print(f"   ✓ Data directory found")
        
        # Check V6 outputs
        v6_path = os.path.join(DATA_PATH, 'v6_outputs')
        if os.path.exists(v6_path):
            files = os.listdir(v6_path)
            print(f"   ✓ V6 outputs: {len(files)} files")
        else:
            errors.append("V6 outputs directory not found")
    else:
        errors.append(f"Data directory not found: {DATA_PATH}")
    
    # Check project modules
    try:
        from production_integration import ProductionConfig
        print(f"   ✓ production_integration module available")
    except ImportError as e:
        print(f"   ⚠️ production_integration not found (OK if notebook is self-contained)")
    
    if errors:
        print("\n⚠️ Issues found:")
        for err in errors:
            print(f"   - {err}")
    else:
        print("\n✅ All checks passed!")
    
    return len(errors) == 0

# Run verification
verify_setup()
