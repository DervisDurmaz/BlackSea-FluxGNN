# 🌊 Hybrid-FluxGNN: Black Sea Biogeochemical Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Physics-Informed Graph Neural Network for Black Sea Chlorophyll Prediction**

A hybrid architecture that combines:

- **Numerical Transport** (Differentiable FVM) for mass-conservative advection-diffusion
- **Neural Reactions** (Gray-Box UDE) for learning NPZD biogeochemical parameters
- **Split-Kernel Message Passing** for anisotropic ocean physics
- **FCT Limiter** for guaranteed positivity of concentrations

> *"Stop trying to learn fluid dynamics (which we know how to solve) and focus ML on biology (which we don't know)."*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID-FLUXGNN v2.0                          │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: CMEMS Black Sea (26 variables, 1993-2023)              │
│         ↓                                                       │
│  V6 GRAPH: 978 nodes, 12,408 edges                             │
│         ↓                                                       │
│  SPLIT-KERNEL MESSAGE PASSING (N² stratification gating)       │
│         ↓                                                       │
│  STRANG SPLITTING: Transport → Reaction → Transport            │
│    ├── Transport: Differentiable FVM (TVD advection)           │
│    └── Reaction: Gray-Box NPZD (RK4, learns parameters)        │
│         ↓                                                       │
│  FCT LIMITER: Hard positivity (C ≥ ε)                          │
│         ↓                                                       │
│  OUTPUT: 7-30 day chlorophyll forecast                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/BlackSea-FluxGNN/blob/main/notebooks/Hybrid_FluxGNN_BlackSea.ipynb)

```python
# Clone and setup
!git clone https://github.com/YOUR_USERNAME/BlackSea-FluxGNN.git
%cd BlackSea-FluxGNN
!pip install -r requirements.txt

# Initialize pipeline
from production_integration import initialize_production_pipeline, train_production
graph, data_loader, model, config = initialize_production_pipeline()

# Train
history = train_production(model, data_loader, config, n_epochs=100)
```

### Option 2: Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/BlackSea-FluxGNN.git
cd BlackSea-FluxGNN
pip install -r requirements.txt
python production_integration.py
```

---

## 📁 Project Structure

```
BlackSea-FluxGNN/
├── mesh/                      # Graph topology
│   ├── __init__.py
│   └── prismatic_graph.py     # σ-coordinates, Delaunay mesh
│
├── models/                    # Neural network components
│   ├── __init__.py
│   ├── gray_box_ude.py        # NPZD parameter prediction
│   ├── split_kernel_mp.py     # Anisotropic message passing
│   └── hybrid_fluxgnn.py      # Main model
│
├── solvers/                   # Numerical methods
│   ├── __init__.py
│   ├── differentiable_fvm.py  # TVD advection-diffusion
│   ├── fct_limiter.py         # Positivity constraint
│   └── operator_splitting.py  # Strang splitting
│
├── training/                  # Training infrastructure
│   ├── __init__.py
│   ├── curriculum.py          # Curriculum learning + TBPTT
│   ├── loss.py                # Multivariate NPZD loss
│   ├── hpo.py                 # Physics-aware HPO (Optuna)
│   ├── ensemble.py            # Deep ensemble + conformal
│   └── distributed.py         # Multi-GPU DDP
│
├── preprocessing/             # Data handling
│   ├── __init__.py
│   ├── black_sea_loader.py    # CMEMS data loader
│   ├── feature_engineering.py # 21 derived features
│   ├── woa_loader.py          # WOA18 nitrate climatology
│   └── pso_feature_selection.py
│
├── visualization/             # Nature-tier figures
│   ├── __init__.py
│   └── nature_figures.py      # 15 publication templates
│
├── tests/                     # Verification
│   ├── __init__.py
│   └── test_conservation.py
│
├── notebooks/                 # Jupyter notebooks
│   └── Hybrid_FluxGNN_BlackSea.ipynb
│
├── production_integration.py  # One-call production setup
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Data

### Required Data (not included in repo)

Data is hosted on Google Drive and automatically mounted in Colab.

**Google Drive Structure:**

```
/content/drive/MyDrive/PINN/
├── black_sea_full_1993_2023.nc    # Main CMEMS dataset (820MB)
└── v6/                             # V6 graph topology outputs
    ├── valid_indices_v6.npy        # 978 valid ocean nodes
    ├── edge_index_v6.npy           # 12,408 edges
    ├── edge_weight_v6.npy          # Flow-biased weights
    └── ...
```

**Data Sources:**

- **CMEMS Black Sea Reanalysis** - Biogeochemistry: `BLKSEA_MULTIYEAR_BGC_007_005`, Physics: `BLKSEA_MULTIYEAR_PHY_007_004`

### Data Setup (Google Colab)

```python
# The notebook automatically mounts Google Drive and uses these paths:
DATA_DIR = "/content/drive/MyDrive/PINN"
V6_DIR = "/content/drive/MyDrive/PINN/v6"
MAIN_DATA = "/content/drive/MyDrive/PINN/black_sea_full_1993_2023.nc"
```

---

## 🔬 Key Features

| Feature | Description |
|---------|-------------|
| **Gray-Box UDE** | Neural network predicts NPZD *parameters*, not dC/dt |
| **Split-Kernel MP** | Separate H/V kernels respect 1000:1 ocean aspect ratio |
| **N² Gating** | Buoyancy frequency controls vertical mixing |
| **FCT Limiter** | Hard positivity constraint (Chl ≥ 0) |
| **Strang Splitting** | 2nd-order accurate operator splitting |
| **Curriculum Learning** | 1-step → 3-step → 7-step → 30-step |
| **TBPTT** | Truncated backprop for gradient stability |
| **Physics HPO** | Prune trials with conservation violations |

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **RMSE (7-day)** | 0.15 mg/m³ |
| **Conservation Error** | < 1% |
| **Positivity Violations** | 0% |
| **Coverage (90% CI)** | 89.2% |

---

## 📖 Citation

```bibtex
@article{hybrid_fluxgnn_2024,
  title={Hybrid-FluxGNN: Physics-Informed Graph Neural Networks for Black Sea Biogeochemical Forecasting},
  author={Durmaz, Derviş},
  year={2024}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- CMEMS for Black Sea reanalysis data
- Oguz et al. for NPZD model formulation
- PyTorch Geometric team
