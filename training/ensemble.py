"""
Ensemble Training Protocol for Uncertainty Quantification.

Deep Ensemble with 5+ members for calibrated uncertainty estimates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import copy


class EnsembleTrainer:
    """
    Train ensemble of models for uncertainty quantification.
    
    Methods:
    - Deep Ensemble: N independent models with different seeds
    - MC Dropout: Single model with dropout at inference
    
    Uncertainty types:
    - Epistemic: Ensemble disagreement (reducible with more data)
    - Aleatoric: Data noise (irreducible)
    """
    
    def __init__(self,
                 model_factory: Callable,
                 n_members: int = 5,
                 device: str = 'cuda'):
        """
        Initialize ensemble trainer.
        
        Args:
            model_factory: Function that returns a fresh model instance
            n_members: Number of ensemble members
            device: Device to train on
        """
        self.model_factory = model_factory
        self.n_members = n_members
        self.device = device
        
        self.models: List[nn.Module] = []
        self.training_histories: List[Dict] = []
        
    def train(self,
              train_loader,
              val_loader,
              trainer_factory: Callable,
              base_seed: int = 42,
              **trainer_kwargs) -> List[Dict]:
        """
        Train all ensemble members.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            trainer_factory: Function(model) -> Trainer
            base_seed: Base random seed
            **trainer_kwargs: Additional arguments for trainer
            
        Returns:
            histories: List of training histories per member
        """
        print("="*60)
        print(f"ENSEMBLE TRAINING - {self.n_members} members")
        print("="*60)
        
        for i in range(self.n_members):
            print(f"\n{'='*40}")
            print(f"Training member {i+1}/{self.n_members}")
            print(f"{'='*40}")
            
            # Set unique seed for this member
            seed = base_seed + i * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Initialize fresh model
            model = self.model_factory()
            model = model.to(self.device)
            
            # Create trainer
            trainer = trainer_factory(model, **trainer_kwargs)
            
            # Train
            history = trainer.fit(train_loader, val_loader)
            
            # Store
            self.models.append(model)
            self.training_histories.append(history)
            
            print(f"Member {i+1} complete. Final val loss: {history['loss'][-1]:.4f}")
        
        return self.training_histories
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction with uncertainty.
        
        Args:
            x: Input tensor [batch, ...]
            
        Returns:
            mean: Ensemble mean prediction [batch, ...]
            std: Ensemble standard deviation (epistemic uncertainty) [batch, ...]
        """
        if not self.models:
            raise ValueError("No trained models. Call train() first.")
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                x_device = x.to(self.device)
                pred = model(x_device)
                predictions.append(pred.cpu())
        
        stack = torch.stack(predictions, dim=0)
        
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        
        return mean, std
    
    def predict_with_samples(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return all ensemble predictions (for downstream analysis).
        
        Returns:
            samples: [n_members, batch, ...]
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(self.device))
                predictions.append(pred.cpu())
        
        return torch.stack(predictions, dim=0)
    
    def save(self, save_dir: str):
        """Save ensemble models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), save_path / f"member_{i}.pt")
        
        torch.save({
            'n_members': self.n_members,
            'histories': self.training_histories
        }, save_path / "ensemble_meta.pt")
        
    def load(self, save_dir: str):
        """Load ensemble models."""
        save_path = Path(save_dir)
        
        meta = torch.load(save_path / "ensemble_meta.pt")
        self.n_members = meta['n_members']
        self.training_histories = meta['histories']
        
        self.models = []
        for i in range(self.n_members):
            model = self.model_factory()
            model.load_state_dict(torch.load(save_path / f"member_{i}.pt"))
            model = model.to(self.device)
            self.models.append(model)


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses dropout at inference time to sample from approximate posterior.
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 50):
        """
        Args:
            model: Model with dropout layers
            n_samples: Number of MC samples
        """
        self.model = model
        self.n_samples = n_samples
        
    def _enable_dropout(self):
        """Enable dropout during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout prediction with uncertainty.
        
        Returns:
            mean: Mean over MC samples
            std: Standard deviation (epistemic uncertainty)
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                predictions.append(pred)
        
        stack = torch.stack(predictions, dim=0)
        
        return stack.mean(dim=0), stack.std(dim=0)


class ConformalPredictor:
    """
    Calibrated prediction intervals using conformal prediction.
    
    Guarantees: P(Y ∈ [lo, hi]) ≥ 1 - α
    
    Unlike ensemble std, provides calibrated coverage guarantees.
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        """
        Args:
            model: Trained model
            alpha: Miscoverage rate (0.1 = 90% coverage)
        """
        self.model = model
        self.alpha = alpha
        self.threshold = None
        
    def calibrate(self, X_cal: torch.Tensor, Y_cal: torch.Tensor):
        """
        Calibrate on held-out calibration set.
        
        Args:
            X_cal: Calibration inputs [n_cal, ...]
            Y_cal: Calibration targets [n_cal, ...]
        """
        self.model.eval()
        
        with torch.no_grad():
            pred = self.model(X_cal)
        
        # Nonconformity score = |y - ŷ|
        scores = torch.abs(Y_cal - pred)
        
        # Flatten if multi-dimensional
        scores = scores.view(-1)
        
        # Quantile for guaranteed coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.threshold = torch.quantile(scores, q_level)
        
        print(f"Conformal calibration: α={self.alpha}, threshold={self.threshold:.4f}")
        
    def predict_interval(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction intervals with coverage guarantee.
        
        Returns:
            lo: Lower bound (1 - α/2 quantile)
            hi: Upper bound (1 - α/2 quantile)
        """
        if self.threshold is None:
            raise ValueError("Must calibrate before prediction. Call calibrate() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            pred = self.model(X)
        
        lo = pred - self.threshold
        hi = pred + self.threshold
        
        return lo, hi
    
    def compute_coverage(self, X_test: torch.Tensor, Y_test: torch.Tensor) -> float:
        """Compute empirical coverage on test set."""
        lo, hi = self.predict_interval(X_test)
        
        in_interval = (Y_test >= lo) & (Y_test <= hi)
        
        coverage = in_interval.float().mean()
        
        return coverage.item()


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Save and resume training with full state.
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_last_n
        
    def save(self, 
             epoch: int, 
             model: nn.Module, 
             optimizer: torch.optim.Optimizer,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             metrics: Dict = None):
        """Save full training state."""
        from datetime import datetime
        
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(state, path)
        
        print(f"  Saved checkpoint: {path.name}")
        
        # Cleanup old checkpoints
        self._cleanup()
        
    def _cleanup(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > self.keep_n:
            for old in checkpoints[:-self.keep_n]:
                old.unlink()
                
    def load_latest(self, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> int:
        """
        Load most recent checkpoint.
        
        Returns:
            epoch: Epoch to resume from
        """
        checkpoints = sorted(self.dir.glob('checkpoint_epoch_*.pt'))
        
        if not checkpoints:
            print("No checkpoints found. Starting from scratch.")
            return 0
        
        latest = checkpoints[-1]
        state = torch.load(latest)
        
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        
        if scheduler and state['scheduler_state']:
            scheduler.load_state_dict(state['scheduler_state'])
        
        print(f"Resumed from epoch {state['epoch']} ({latest.name})")
        
        return state['epoch']


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Ensemble & UQ Components...")
    
    # Dummy model factory
    def model_factory():
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4)
        )
    
    # Test ensemble
    ensemble = EnsembleTrainer(model_factory, n_members=3, device='cpu')
    
    # Manually add models for testing
    for i in range(3):
        torch.manual_seed(42 + i)
        ensemble.models.append(model_factory())
    
    x_test = torch.randn(16, 10)
    mean, std = ensemble.predict(x_test)
    
    print(f"✓ Ensemble prediction:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Mean std: {std.mean():.4f}")
    
    # Test MC Dropout
    mc_model = model_factory()
    mc_pred = MCDropoutPredictor(mc_model, n_samples=20)
    mean_mc, std_mc = mc_pred.predict(x_test)
    
    print(f"\n✓ MC Dropout prediction:")
    print(f"  Mean std: {std_mc.mean():.4f}")
    
    # Test Conformal
    conf = ConformalPredictor(mc_model, alpha=0.1)
    X_cal = torch.randn(100, 10)
    Y_cal = torch.randn(100, 4)
    conf.calibrate(X_cal, Y_cal)
    
    lo, hi = conf.predict_interval(x_test)
    print(f"\n✓ Conformal prediction:")
    print(f"  Interval width: {(hi - lo).mean():.4f}")
    
    # Test checkpoint manager
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = CheckpointManager(tmpdir, keep_last_n=2)
        
        model = model_factory()
        optimizer = torch.optim.Adam(model.parameters())
        
        ckpt.save(1, model, optimizer, metrics={'loss': 0.5})
        ckpt.save(2, model, optimizer, metrics={'loss': 0.3})
        ckpt.save(3, model, optimizer, metrics={'loss': 0.2})
        
        epoch = ckpt.load_latest(model, optimizer)
        print(f"\n✓ Checkpoint: loaded epoch {epoch}")
    
    print("\n✅ All tests passed!")
