"""
Nature-Tier Visualizations for Hybrid-FluxGNN.

Publication-quality figures (300 DPI, Wong colorblind-safe palette).
Designed for Nature/Science journals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# =============================================================================
# WONG COLORBLIND-SAFE PALETTE
# =============================================================================

WONG = {
    'blue': '#0072B2',
    'orange': '#E69F00', 
    'green': '#009E73',
    'yellow': '#F0E442',
    'sky': '#56B4E9',
    'vermilion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}

# Sequential colormaps for ocean data
OCEAN_CMAP = plt.cm.viridis
CHL_CMAP = plt.cm.YlGn
TEMP_CMAP = plt.cm.RdYlBu_r
ERROR_CMAP = plt.cm.RdBu_r

# Figure settings
FIGSIZE_SINGLE = (6, 5)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_FULL = (12, 10)
DPI = 300
FONTSIZE = 10
LABELSIZE = 12


def set_nature_style():
    """Apply Nature journal style to matplotlib."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': FONTSIZE,
        'axes.labelsize': LABELSIZE,
        'axes.titlesize': LABELSIZE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
    })

set_nature_style()


# =============================================================================
# FIGURE 1: TAYLOR DIAGRAM
# =============================================================================

def plot_taylor_diagram(stats: Dict[str, Dict[str, float]], 
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Taylor Diagram for model skill assessment.
    
    Args:
        stats: Dict of {model_name: {'std': float, 'corr': float, 'rmse': float}}
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=FIGSIZE_SINGLE)
    
    # Create polar axes
    ax = fig.add_subplot(111, polar=True)
    
    # Limit to first quadrant
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    
    # Reference point (observations)
    ref_std = 1.0
    ax.plot(0, ref_std, 'ko', markersize=10, label='Reference', zorder=5)
    
    # Plot models
    colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['vermilion'], WONG['purple']]
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (name, stat) in enumerate(stats.items()):
        theta = np.arccos(stat['corr'])
        r = stat['std']
        ax.plot(theta, r, markers[i % len(markers)], 
                color=colors[i % len(colors)], 
                markersize=8, label=name, zorder=4)
    
    # RMSE contours
    for rmse in [0.25, 0.5, 0.75, 1.0]:
        angles = np.linspace(0, np.pi/2, 50)
        rs = np.sqrt(1 + rmse**2 - 2*rmse*np.cos(angles))
        ax.plot(angles, rs, 'k--', alpha=0.3, linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Standard Deviation (normalized)', labelpad=10)
    ax.set_ylabel('Correlation', labelpad=30)
    ax.set_title('Taylor Diagram: Model Skill Assessment', pad=15)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 2: SPATIAL ERROR MAP
# =============================================================================

def plot_spatial_error_map(lon: np.ndarray, lat: np.ndarray, 
                           error: np.ndarray, 
                           title: str = "Spatial Distribution of Prediction Error",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Spatial error map with Black Sea coastline.
    
    Args:
        lon: Longitude array
        lat: Latitude array
        error: Error values (RMSE or bias)
        title: Figure title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Create scatter plot
    vmax = np.percentile(np.abs(error), 95)
    scatter = ax.scatter(lon, lat, c=error, cmap=ERROR_CMAP, 
                        s=20, alpha=0.8, vmin=-vmax, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Prediction Error (mg Chl/m³)')
    
    # Black Sea bounds
    ax.set_xlim(27, 42)
    ax.set_ylim(40.5, 47)
    
    # Labels
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(title)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 3: BENCHMARK COMPARISON HEATMAP
# =============================================================================

def plot_benchmark_heatmap(metrics: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap comparing multiple models across metrics.
    
    Args:
        metrics: {model_name: {metric_name: value}}
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    models = list(metrics.keys())
    metric_names = list(metrics[models[0]].keys())
    
    # Create matrix
    data = np.array([[metrics[m][met] for met in metric_names] for m in models])
    
    # Normalize each column
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    
    # Heatmap
    im = ax.imshow(data_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Annotate with actual values
    for i in range(len(models)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title('Model Benchmark Comparison')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Score (lower = better)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 4: PARETO FRONT (HPO)
# =============================================================================

def plot_pareto_front(trials: List[Dict], 
                      objectives: List[str] = ['RMSE', 'Physics', 'Stratification'],
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Multi-objective Pareto front visualization.
    
    Args:
        trials: List of {obj1: val, obj2: val, obj3: val, 'pareto': bool}
        objectives: Names of objectives
        save_path: Path to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for ax, (i, j) in zip(axes, pairs):
        # All trials
        all_x = [t[objectives[i]] for t in trials]
        all_y = [t[objectives[j]] for t in trials]
        ax.scatter(all_x, all_y, c='lightgray', s=30, alpha=0.5, label='All trials')
        
        # Pareto front
        pareto = [t for t in trials if t.get('pareto', False)]
        pareto_x = [t[objectives[i]] for t in pareto]
        pareto_y = [t[objectives[j]] for t in pareto]
        ax.scatter(pareto_x, pareto_y, c=WONG['vermilion'], s=80, 
                  edgecolors='black', linewidths=0.5, label='Pareto optimal', zorder=5)
        
        # Connect Pareto points
        if pareto:
            sorted_idx = np.argsort(pareto_x)
            ax.plot(np.array(pareto_x)[sorted_idx], np.array(pareto_y)[sorted_idx],
                   color=WONG['vermilion'], alpha=0.5, linestyle='--')
        
        ax.set_xlabel(objectives[i])
        ax.set_ylabel(objectives[j])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle('Multi-Objective HPO: Pareto Front', fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 5: UNCERTAINTY CALIBRATION
# =============================================================================

def plot_uncertainty_calibration(expected: np.ndarray, observed: np.ndarray,
                                 intervals: List[float] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Calibration plot for uncertainty quantification.
    
    Args:
        expected: Expected coverage [0-1]
        observed: Observed coverage [0-1]
        intervals: Confidence levels used
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    
    # Fill between
    ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
    
    # Observed calibration
    ax.plot(expected, observed, 'o-', color=WONG['blue'], 
            markersize=8, linewidth=2, label='Model calibration')
    
    # Shade over/under-confident regions
    over_mask = observed < expected
    under_mask = observed > expected
    
    ax.scatter(expected[over_mask], observed[over_mask], 
              c=WONG['vermilion'], s=100, marker='v', zorder=5, label='Over-confident')
    ax.scatter(expected[under_mask], observed[under_mask], 
              c=WONG['green'], s=100, marker='^', zorder=5, label='Under-confident')
    
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Uncertainty Calibration Plot')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 6: SKILL DEGRADATION WITH HORIZON
# =============================================================================

def plot_skill_degradation(horizons: np.ndarray, 
                           metrics: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Forecast skill vs prediction horizon.
    
    Args:
        horizons: Forecast horizons (days)
        metrics: {variable: skill_array}
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['purple']]
    markers = ['o', 's', '^', 'D']
    
    for i, (var, skill) in enumerate(metrics.items()):
        ax.plot(horizons, skill, f'{markers[i]}-', color=colors[i],
                markersize=6, linewidth=2, label=var)
        
        # Fill uncertainty
        if hasattr(skill, 'std'):
            ax.fill_between(horizons, skill - skill.std, skill + skill.std,
                           color=colors[i], alpha=0.2)
    
    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Skill=0.5')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Skill Score')
    ax.set_title('Variable-wise Skill Degradation')
    ax.set_xlim(0, max(horizons))
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 7: SEASONAL SKILL BREAKDOWN
# =============================================================================

def plot_seasonal_skill(months: np.ndarray, 
                        skill: Dict[str, np.ndarray],
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Seasonal breakdown of forecast skill.
    
    Args:
        months: Month indices (1-12)
        skill: {variable: skill_per_month}
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Heatmap
    data = np.array(list(skill.values()))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(np.arange(len(skill)))
    ax.set_yticklabels(list(skill.keys()))
    
    # Annotate
    for i in range(len(skill)):
        for j in range(12):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if data[i, j] < 0.5 else 'black')
    
    ax.set_xlabel('Month')
    ax.set_title('Seasonal Skill Breakdown')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Skill Score')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 8: VERTICAL PROFILE COMPARISON
# =============================================================================

def plot_vertical_profile(depth: np.ndarray,
                          observed: np.ndarray,
                          predicted: np.ndarray,
                          uncertainty: Optional[np.ndarray] = None,
                          variable: str = 'Chlorophyll',
                          units: str = 'mg/m³',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Vertical profile comparison with uncertainty.
    
    Args:
        depth: Depth array (positive downward)
        observed: Observed values
        predicted: Predicted values
        uncertainty: Prediction uncertainty (std)
        variable: Variable name
        units: Units for x-axis
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # Observations
    ax.plot(observed, depth, 'ko-', markersize=6, linewidth=0,
            label='Observations', zorder=5)
    
    # Prediction
    ax.plot(predicted, depth, '-', color=WONG['blue'], linewidth=2,
            label='Hybrid-FluxGNN')
    
    # Uncertainty
    if uncertainty is not None:
        ax.fill_betweenx(depth, predicted - 2*uncertainty, predicted + 2*uncertainty,
                        color=WONG['blue'], alpha=0.2, label='95% CI')
    
    # DCM marker
    dcm_idx = np.argmax(predicted)
    ax.axhline(y=depth[dcm_idx], color=WONG['green'], linestyle='--', 
               alpha=0.7, label=f'DCM: {depth[dcm_idx]:.0f}m')
    
    ax.invert_yaxis()
    ax.set_xlabel(f'{variable} ({units})')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Vertical Profile: {variable}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 9: CONSERVATION ANALYSIS
# =============================================================================

def plot_conservation_analysis(time: np.ndarray,
                               total_N: np.ndarray,
                               components: Dict[str, np.ndarray],
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Nitrogen budget conservation analysis.
    
    Args:
        time: Time array
        total_N: Total nitrogen over time
        components: {component_name: time_series}
        save_path: Path to save
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Panel A: Components
    ax1 = axes[0]
    colors = [WONG['green'], WONG['blue'], WONG['orange'], WONG['purple']]
    
    ax1.stackplot(time, *components.values(), labels=list(components.keys()),
                  colors=colors, alpha=0.7)
    ax1.set_ylabel('Nitrogen (μmol/L)')
    ax1.set_title('A. Nitrogen Budget Components')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Conservation error
    ax2 = axes[1]
    relative_error = (total_N - total_N[0]) / total_N[0] * 100
    
    ax2.plot(time, relative_error, color=WONG['vermilion'], linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
    
    ax2.fill_between(time, -10, 10, color=WONG['green'], alpha=0.1, label='Acceptable')
    
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Conservation Error (%)')
    ax2.set_title('B. Nitrogen Conservation Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-15, 15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 10: TRAINING DYNAMICS
# =============================================================================

def plot_training_dynamics(history: Dict[str, List],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Training dynamics with curriculum stages.
    
    Args:
        history: {train_loss, val_loss, val_rmse, stage}
        save_path: Path to save
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_FULL)
    
    epochs = np.arange(len(history['train_loss']))
    
    # A. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], color=WONG['blue'], label='Train', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], color=WONG['orange'], label='Validation', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('A. Training & Validation Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # B. RMSE
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_rmse'], color=WONG['green'], linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE (mg Chl/m³)')
    ax2.set_title('B. Validation RMSE')
    ax2.grid(True, alpha=0.3)
    
    # C. Curriculum stages
    ax3 = axes[1, 0]
    stages = np.array(history['stage'])
    stage_names = ['1-step', '3-step', '7-step', 'Full']
    
    for i, name in enumerate(stage_names):
        mask = stages == i
        if mask.any():
            ax3.fill_between(epochs[mask], 0, 1, alpha=0.3, label=name)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Stage Progress')
    ax3.set_title('C. Curriculum Learning Stages')
    ax3.legend(loc='lower right')
    ax3.set_ylim(0, 1)
    
    # D. Learning rate (if available)
    ax4 = axes[1, 1]
    if 'lr' in history:
        ax4.plot(epochs, history['lr'], color=WONG['purple'], linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('D. Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Learning rate\nnot tracked', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('D. Learning Rate Schedule')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 11: PREDICTION VS OBSERVATION SCATTER
# =============================================================================

def plot_prediction_scatter(observed: np.ndarray, 
                            predicted: np.ndarray,
                            variable: str = 'Chlorophyll',
                            units: str = 'mg/m³',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Prediction vs observation scatter with density.
    
    Args:
        observed: Observed values
        predicted: Predicted values
        variable: Variable name
        units: Units
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # 2D histogram for density
    from scipy.stats import gaussian_kde
    
    xy = np.vstack([observed, predicted])
    try:
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones_like(observed)
    
    # Sort by density
    idx = z.argsort()
    x, y, z = observed[idx], predicted[idx], z[idx]
    
    scatter = ax.scatter(x, y, c=z, s=10, cmap='viridis', alpha=0.6)
    
    # 1:1 line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='1:1 line')
    
    # Statistics
    from scipy.stats import pearsonr
    r, _ = pearsonr(observed, predicted)
    rmse = np.sqrt(np.mean((observed - predicted)**2))
    bias = np.mean(predicted - observed)
    
    stats_text = f'R² = {r**2:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(f'Observed {variable} ({units})')
    ax.set_ylabel(f'Predicted {variable} ({units})')
    ax.set_title(f'Prediction vs Observation: {variable}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Density')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 12: ERROR DISTRIBUTION
# =============================================================================

def plot_error_distribution(errors: np.ndarray,
                            variable: str = 'Chlorophyll',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Error distribution with QQ plot.
    
    Args:
        errors: Prediction errors
        variable: Variable name
        save_path: Path to save
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    
    # A. Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=50, density=True, color=WONG['blue'], 
             alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit normal
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    x = np.linspace(errors.min(), errors.max(), 100)
    ax1.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, 
             label=f'Normal fit (μ={mu:.3f}, σ={std:.3f})')
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Density')
    ax1.set_title(f'A. {variable} Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # B. QQ plot
    ax2 = axes[1]
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(WONG['blue'])
    ax2.get_lines()[0].set_markeredgecolor('black')
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[1].set_color(WONG['vermilion'])
    ax2.set_title('B. Q-Q Plot (Normal)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 13: TIME SERIES EXAMPLE
# =============================================================================

def plot_time_series(time: np.ndarray,
                     observed: np.ndarray,
                     predicted: np.ndarray,
                     uncertainty: Optional[np.ndarray] = None,
                     variable: str = 'Chlorophyll',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Time series comparison with uncertainty bands.
    
    Args:
        time: Time array
        observed: Observed values
        predicted: Predicted values
        uncertainty: Prediction std
        variable: Variable name
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    # Observations
    ax.plot(time, observed, 'ko', markersize=4, label='Observations', alpha=0.7)
    
    # Predictions
    ax.plot(time, predicted, '-', color=WONG['blue'], linewidth=2, 
            label='Hybrid-FluxGNN')
    
    # Uncertainty
    if uncertainty is not None:
        ax.fill_between(time, predicted - 2*uncertainty, predicted + 2*uncertainty,
                       color=WONG['blue'], alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{variable}')
    ax.set_title(f'{variable} Time Series at Selected Location')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 14: ARCHITECTURE DIAGRAM
# =============================================================================

def plot_architecture_diagram(save_path: Optional[str] = None) -> plt.Figure:
    """
    Schematic of Hybrid-FluxGNN architecture.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Box style
    box_kwargs = dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='black', linewidth=1.5)
    
    # Boxes
    boxes = [
        (1, 6, 'Input\n(Forcing + State)', WONG['sky']),
        (4, 6, 'Split-Kernel\nMessage Passing', WONG['blue']),
        (7, 6, 'Gray-Box\nNPZD (UDE)', WONG['green']),
        (10, 6, 'Output\n(NPZD state)', WONG['orange']),
        (4, 3, 'Differentiable\nFVM Transport', WONG['purple']),
        (7, 3, 'FCT\nLimiter', WONG['vermilion']),
        (5.5, 1, 'Strang Splitting\nR(Δt/2) ∘ T(Δt) ∘ R(Δt/2)', 'white'),
    ]
    
    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                              boxstyle='round,pad=0.05',
                              facecolor=color, edgecolor='black',
                              linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow_kwargs = dict(arrowstyle='->', color='black', lw=1.5)
    
    arrows = [
        ((1.8, 6), (3.2, 6)),
        ((4.8, 6), (6.2, 6)),
        ((7.8, 6), (9.2, 6)),
        ((4, 5.6), (4, 3.4)),
        ((4.8, 3), (6.2, 3)),
        ((7.8, 3), (7.8, 5.6)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_title('Hybrid-FluxGNN Architecture', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 15: ENSEMBLE SPREAD
# =============================================================================

def plot_ensemble_spread(time: np.ndarray,
                         ensemble_members: np.ndarray,
                         observed: Optional[np.ndarray] = None,
                         variable: str = 'Chlorophyll',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Ensemble spread visualization.
    
    Args:
        time: Time array
        ensemble_members: [n_members, n_time] array
        observed: Optional observations
        variable: Variable name
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    n_members = ensemble_members.shape[0]
    
    # Individual members
    for i in range(n_members):
        ax.plot(time, ensemble_members[i], '-', color=WONG['blue'], 
                alpha=0.3, linewidth=1)
    
    # Ensemble mean
    mean = ensemble_members.mean(axis=0)
    ax.plot(time, mean, '-', color=WONG['blue'], linewidth=2.5, 
            label='Ensemble mean')
    
    # Ensemble spread
    std = ensemble_members.std(axis=0)
    ax.fill_between(time, mean - 2*std, mean + 2*std,
                   color=WONG['blue'], alpha=0.2, label='±2σ spread')
    
    # Observations
    if observed is not None:
        ax.plot(time, observed, 'ko', markersize=4, label='Observations')
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{variable}')
    ax.set_title(f'{variable}: {n_members}-Member Ensemble')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# CONVENIENCE: GENERATE ALL DEMO FIGURES
# =============================================================================

def generate_all_demo_figures(output_dir: str = './figures'):
    """Generate all 15 figures with synthetic data for demonstration."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Nature-tier demo figures...")
    
    # 1. Taylor Diagram
    stats = {
        'Hybrid-FluxGNN': {'std': 0.95, 'corr': 0.92, 'rmse': 0.3},
        'Persistence': {'std': 1.2, 'corr': 0.65, 'rmse': 0.7},
        'Climatology': {'std': 0.8, 'corr': 0.55, 'rmse': 0.8},
        'LSTM': {'std': 1.0, 'corr': 0.85, 'rmse': 0.45},
    }
    plot_taylor_diagram(stats, f'{output_dir}/01_taylor_diagram.png')
    print("  ✓ Taylor diagram")
    
    # 2. Spatial error map
    lon = np.random.rand(500) * 14 + 28
    lat = np.random.rand(500) * 6 + 41
    error = np.random.randn(500) * 0.3
    plot_spatial_error_map(lon, lat, error, save_path=f'{output_dir}/02_spatial_error.png')
    print("  ✓ Spatial error map")
    
    # 3. Benchmark heatmap
    metrics = {
        'Hybrid-FluxGNN': {'RMSE': 0.15, 'MAE': 0.10, 'R²': 0.92, 'Skill': 0.88},
        'LSTM': {'RMSE': 0.25, 'MAE': 0.18, 'R²': 0.85, 'Skill': 0.78},
        'GNN': {'RMSE': 0.22, 'MAE': 0.15, 'R²': 0.87, 'Skill': 0.82},
        'Persistence': {'RMSE': 0.45, 'MAE': 0.35, 'R²': 0.65, 'Skill': 0.45},
    }
    plot_benchmark_heatmap(metrics, save_path=f'{output_dir}/03_benchmark_heatmap.png')
    print("  ✓ Benchmark heatmap")
    
    # 4. Pareto front
    trials = []
    for _ in range(100):
        t = {'RMSE': np.random.rand()*0.5, 'Physics': np.random.rand()*0.3, 
             'Stratification': np.random.rand()*20, 'pareto': np.random.rand() > 0.9}
        trials.append(t)
    plot_pareto_front(trials, save_path=f'{output_dir}/04_pareto_front.png')
    print("  ✓ Pareto front")
    
    # 5. Uncertainty calibration
    expected = np.linspace(0, 1, 10)
    observed = expected + np.random.randn(10) * 0.05
    observed = np.clip(observed, 0, 1)
    plot_uncertainty_calibration(expected, observed, save_path=f'{output_dir}/05_calibration.png')
    print("  ✓ Uncertainty calibration")
    
    # 6. Skill degradation
    horizons = np.arange(1, 31)
    skill = {
        'Chlorophyll': 0.95 * np.exp(-horizons/15),
        'Nitrate': 0.9 * np.exp(-horizons/20),
        'SST': 0.98 * np.exp(-horizons/25),
    }
    plot_skill_degradation(horizons, skill, save_path=f'{output_dir}/06_skill_degradation.png')
    print("  ✓ Skill degradation")
    
    # 7. Seasonal skill
    months = np.arange(12)
    skill = {
        'Chlorophyll': np.random.rand(12) * 0.3 + 0.6,
        'Nitrate': np.random.rand(12) * 0.3 + 0.5,
        'SST': np.random.rand(12) * 0.2 + 0.75,
    }
    plot_seasonal_skill(months, skill, save_path=f'{output_dir}/07_seasonal_skill.png')
    print("  ✓ Seasonal skill")
    
    # 8. Vertical profile
    depth = np.linspace(0, 200, 50)
    chl_obs = 0.5 + 2 * np.exp(-((depth - 50)/20)**2) + np.random.randn(50) * 0.1
    chl_pred = 0.5 + 2 * np.exp(-((depth - 48)/22)**2)
    uncertainty = np.ones(50) * 0.2
    plot_vertical_profile(depth, chl_obs, chl_pred, uncertainty, 
                         save_path=f'{output_dir}/08_vertical_profile.png')
    print("  ✓ Vertical profile")
    
    # 9. Conservation
    time = np.arange(100)
    total_N = 10 + np.cumsum(np.random.randn(100)) * 0.1
    components = {
        'Phyto-N': np.ones(100) * 3 + np.random.randn(100) * 0.2,
        'Nitrate': np.ones(100) * 5 + np.random.randn(100) * 0.3,
        'Zoo-N': np.ones(100) * 2 + np.random.randn(100) * 0.1,
    }
    plot_conservation_analysis(time, total_N, components, 
                              save_path=f'{output_dir}/09_conservation.png')
    print("  ✓ Conservation analysis")
    
    # 10. Training dynamics
    history = {
        'train_loss': [1/(i+1) + np.random.rand()*0.1 for i in range(200)],
        'val_loss': [1.2/(i+1) + np.random.rand()*0.1 for i in range(200)],
        'val_rmse': [0.5/(i+1)**0.3 + np.random.rand()*0.02 for i in range(200)],
        'stage': [0]*50 + [1]*50 + [2]*50 + [3]*50,
    }
    plot_training_dynamics(history, save_path=f'{output_dir}/10_training_dynamics.png')
    print("  ✓ Training dynamics")
    
    # 11. Scatter
    observed = np.random.rand(1000) * 5
    predicted = observed + np.random.randn(1000) * 0.3
    plot_prediction_scatter(observed, predicted, save_path=f'{output_dir}/11_scatter.png')
    print("  ✓ Prediction scatter")
    
    # 12. Error distribution
    errors = np.random.randn(1000) * 0.3
    plot_error_distribution(errors, save_path=f'{output_dir}/12_error_dist.png')
    print("  ✓ Error distribution")
    
    # 13. Time series
    time = np.arange(365)
    observed = 1 + 0.5*np.sin(2*np.pi*time/365) + np.random.randn(365)*0.1
    predicted = 1 + 0.5*np.sin(2*np.pi*time/365)
    uncertainty = np.ones(365) * 0.15
    plot_time_series(time, observed, predicted, uncertainty, 
                    save_path=f'{output_dir}/13_time_series.png')
    print("  ✓ Time series")
    
    # 14. Architecture
    plot_architecture_diagram(save_path=f'{output_dir}/14_architecture.png')
    print("  ✓ Architecture diagram")
    
    # 15. Ensemble
    time = np.arange(100)
    ensemble = np.sin(time/10)[:, None] + np.random.randn(100, 5) * 0.2
    plot_ensemble_spread(time, ensemble.T, save_path=f'{output_dir}/15_ensemble.png')
    print("  ✓ Ensemble spread")
    
    print(f"\n✅ All 15 figures saved to {output_dir}/")


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Nature-tier visualization module...")
    
    # Generate demo figures
    generate_all_demo_figures('./test_figures')
    
    print("\n✅ Visualization module ready!")
