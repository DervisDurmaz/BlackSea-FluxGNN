"""
PSO-based Feature Selection with Physics Constraints.

Binary Particle Swarm Optimization for selecting optimal feature subsets
while respecting oceanographic physics constraints.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


class PSOFeatureSelector:
    """
    Binary Particle Swarm Optimization for feature selection.
    
    Key Features:
    - Binary encoding: particle[i] = 1 means feature i is selected
    - Protected features: ALWAYS selected (lat, lon, depth, T, S, etc.)
    - Multi-objective fitness: accuracy + parsimony + physics compliance
    
    Search Space:
        25 candidate features → 2^25 = 33 million possible subsets
        PSO explores efficiently without exhaustive search
        
    Usage:
        >>> selector = PSOFeatureSelector(n_features=25)
        >>> best_mask, best_fitness = selector.optimize(X, y)
        >>> X_selected = X[:, best_mask > 0.5]
    """
    
    def __init__(self,
                 n_features: int = 25,
                 n_particles: int = 30,
                 n_iterations: int = 100,
                 w: float = 0.7,      # Inertia weight
                 c1: float = 1.5,     # Cognitive (personal best)
                 c2: float = 1.5,     # Social (global best)
                 protected_indices: Optional[List[int]] = None,
                 feature_names: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        Initialize PSO Feature Selector.
        
        Args:
            n_features: Total number of candidate features
            n_particles: Swarm size (30-50 typical)
            n_iterations: Max iterations (100-200 typical)
            w: Inertia weight (0.4-0.9, controls exploration/exploitation)
            c1: Cognitive coefficient (personal best influence)
            c2: Social coefficient (global best influence)
            protected_indices: Indices of physics-critical features (always selected)
            feature_names: Optional list of feature names for reporting
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.feature_names = feature_names
        
        np.random.seed(random_state)
        
        # Protected features (physics-critical, never removed)
        # Default: [lat, lon, depth, time, T, S, deptho, mdt]
        self.protected = protected_indices or [0, 1, 2, 3, 4, 5, 6, 7]
        
        # Physics-important features with weights
        self.physics_weights = {
            'n2': 1.0,           # Buoyancy frequency (stratification)
            'mld': 0.9,          # Mixed layer depth
            'par': 0.8,          # Photosynthetically active radiation
            'dcm_depth': 0.7,    # Deep chlorophyll maximum
            'nitracline': 0.6,   # Nitracline depth
            'f_coriolis': 0.5,   # Coriolis parameter
        }
        
        # Initialize swarm
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((n_particles, n_features))
        
        # Personal and global bests
        self.p_best = self.particles.copy()
        self.p_best_fitness = np.full(n_particles, -np.inf)
        self.g_best = None
        self.g_best_fitness = -np.inf
        
        # History for convergence analysis
        self.history = {
            'fitness': [],
            'n_features': [],
            'best_particle': []
        }
        
    def _initialize_particles(self) -> np.ndarray:
        """Initialize with random feature subsets, protected always ON."""
        particles = np.random.rand(self.n_particles, self.n_features) > 0.5
        # Force protected features to 1
        particles[:, self.protected] = 1
        return particles.astype(float)
        
    def _sigmoid(self, v: np.ndarray) -> np.ndarray:
        """Transfer function for binary PSO."""
        # Clip to avoid overflow
        v = np.clip(v, -500, 500)
        return 1 / (1 + np.exp(-v))
        
    def _quick_cv_score(self, X: np.ndarray, y: np.ndarray, cv: int = 3) -> float:
        """Quick cross-validation score for fitness evaluation."""
        if X.shape[1] == 0:
            return 0.0
            
        model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            return max(scores.mean(), 0.0)  # R² can be negative
        except Exception:
            return 0.0
            
    def _physics_compliance(self, particle: np.ndarray, 
                           feature_name_to_idx: Optional[dict] = None) -> float:
        """
        Score based on physics-important features being selected.
        
        High-value features for oceanography:
        - N² (buoyancy frequency): Critical for stratification
        - MLD (mixed layer depth): Critical for vertical mixing
        - PAR (light): Critical for photosynthesis
        """
        if feature_name_to_idx is None:
            # Use index-based weights if no mapping provided
            physics_features = {
                8: 1.0,   # Assumed: N²
                9: 0.9,   # Assumed: MLD
                10: 0.8,  # Assumed: PAR
                11: 0.7,  # Assumed: DCM depth
                12: 0.5,  # Assumed: f
            }
        else:
            physics_features = {}
            for name, weight in self.physics_weights.items():
                if name in feature_name_to_idx:
                    physics_features[feature_name_to_idx[name]] = weight
        
        score = 0.0
        for idx, weight in physics_features.items():
            if idx < len(particle) and particle[idx] > 0.5:
                score += weight
                
        max_score = sum(physics_features.values()) if physics_features else 1.0
        return score / max_score if max_score > 0 else 0.0
        
    def fitness(self, particle: np.ndarray, X: np.ndarray, y: np.ndarray,
                alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2) -> float:
        """
        Multi-objective fitness function.
        
        Fitness = α·Accuracy + β·Parsimony + γ·Physics_Score
        
        Args:
            particle: Binary feature mask
            X: Full feature matrix
            y: Target variable
            alpha: Weight for accuracy (0.6 default)
            beta: Weight for parsimony (0.2 default)
            gamma: Weight for physics compliance (0.2 default)
            
        Returns:
            fitness: Weighted combination of objectives
        """
        selected = particle > 0.5
        n_selected = selected.sum()
        
        if n_selected == 0:
            return -np.inf
            
        # Evaluate model with selected features
        X_subset = X[:, selected]
        accuracy = self._quick_cv_score(X_subset, y)
        
        # Parsimony penalty (prefer fewer features)
        parsimony = 1 - (n_selected / self.n_features)
        
        # Physics bonus (reward physics-critical features)
        physics_score = self._physics_compliance(particle)
        
        # Weighted combination
        fitness = alpha * accuracy + beta * parsimony + gamma * physics_score
        
        return fitness
        
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target variable [n_samples]
            verbose: Print progress
            
        Returns:
            best_mask: Binary mask of selected features
            best_fitness: Fitness of best solution
        """
        for iteration in range(self.n_iterations):
            # Evaluate all particles
            for i in range(self.n_particles):
                fit = self.fitness(self.particles[i], X, y)
                
                # Update personal best
                if fit > self.p_best_fitness[i]:
                    self.p_best_fitness[i] = fit
                    self.p_best[i] = self.particles[i].copy()
                    
                # Update global best
                if fit > self.g_best_fitness:
                    self.g_best_fitness = fit
                    self.g_best = self.particles[i].copy()
                    
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                
                # Velocity update (PSO equations)
                cognitive = self.c1 * r1 * (self.p_best[i] - self.particles[i])
                social = self.c2 * r2 * (self.g_best - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # Binary position update (sigmoid transfer)
                prob = self._sigmoid(self.velocities[i])
                self.particles[i] = (np.random.rand(self.n_features) < prob).astype(float)
                
                # Enforce protected features
                self.particles[i, self.protected] = 1
                
            # Store history
            self.history['fitness'].append(self.g_best_fitness)
            self.history['n_features'].append(int(self.g_best.sum()))
            self.history['best_particle'].append(self.g_best.copy())
                
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1}/{self.n_iterations}: "
                      f"Fitness = {self.g_best_fitness:.4f}, "
                      f"Features = {int(self.g_best.sum())}/{self.n_features}")
                  
        return self.g_best, self.g_best_fitness
        
    def get_selected_features(self) -> List[int]:
        """Get indices of selected features."""
        if self.g_best is None:
            raise ValueError("Must run optimize() first")
        return np.where(self.g_best > 0.5)[0].tolist()
        
    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features (if feature_names provided)."""
        if self.feature_names is None:
            raise ValueError("feature_names not provided during initialization")
        indices = self.get_selected_features()
        return [self.feature_names[i] for i in indices]
        
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot convergence history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Fitness convergence
        axes[0].plot(self.history['fitness'], 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('PSO Convergence')
        axes[0].grid(True, alpha=0.3)
        
        # Number of features
        axes[1].plot(self.history['n_features'], 'r-', linewidth=2)
        axes[1].axhline(y=len(self.protected), color='g', linestyle='--', 
                       label=f'Protected ({len(self.protected)})')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title('Feature Count Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples, n_features = 500, 25
    
    # True relevant features: indices 0-7 (protected) + 8, 10, 15 (important)
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + 2*X[:, 8] + 0.5*X[:, 10] + X[:, 15] + 
         0.1 * np.random.randn(n_samples))
    
    # Feature names for demonstration
    feature_names = [
        'lat', 'lon', 'depth', 'time', 'T', 'S', 'deptho', 'mdt',  # Protected
        'n2', 'mld', 'par', 'dcm_depth', 'nitracline', 'f_coriolis',  # Physics
        'sin_doy', 'cos_doy', 'Ro', 'Ri', 'Ek', 'dist_coast', 'dist_river',
        'lunar', 'cil_thick', 'pycno_depth', 'zeu'
    ]
    
    # Run PSO feature selection
    selector = PSOFeatureSelector(
        n_features=n_features,
        n_particles=20,
        n_iterations=50,
        feature_names=feature_names
    )
    
    best_mask, best_fitness = selector.optimize(X, y, verbose=True)
    
    print("\n" + "="*50)
    print("PSO Feature Selection Results")
    print("="*50)
    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Selected Features ({int(best_mask.sum())}/{n_features}):")
    for name in selector.get_selected_feature_names():
        print(f"  - {name}")
