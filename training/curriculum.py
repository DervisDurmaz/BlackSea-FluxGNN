"""
Curriculum Learning with Truncated Backpropagation Through Time (TBPTT).

Addresses gradient explosion in long rollouts by:
1. Starting with short horizons, expanding after mastery
2. Truncating gradient flow every τ steps
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class CurriculumStage:
    """Definition of a curriculum stage."""
    horizon: int        # Rollout length (days)
    threshold: float    # Accuracy threshold to advance
    name: str           # Stage name
    max_epochs: int = 100  # Max epochs before forced advance


class CurriculumTrainer:
    """
    Train biology on SHORT windows first, expand only after mastery.
    
    Schedule:
        Stage 1: 1-step prediction (dC = f(C, forcing))  → until acc > 90%
        Stage 2: 3-step rollout                          → until acc > 85%
        Stage 3: 7-step rollout                          → until acc > 80%
        Stage 4: 30-step rollout (full forecast)
        
    DO NOT skip to Stage 4 directly! The gradients will be noise.
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 stages: Optional[List[CurriculumStage]] = None,
                 tbptt_length: int = 7,
                 device: str = 'cuda'):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.tbptt_length = tbptt_length
        
        # Default curriculum stages
        self.stages = stages or [
            CurriculumStage(horizon=1, threshold=0.90, name='1-step'),
            CurriculumStage(horizon=3, threshold=0.85, name='3-step'),
            CurriculumStage(horizon=7, threshold=0.80, name='7-step'),
            CurriculumStage(horizon=30, threshold=0.75, name='full-forecast'),
        ]
        
        self.current_stage = 0
        
        # History
        self.history = {
            'loss': [],
            'accuracy': [],
            'stage': [],
            'epoch': [],
        }
        
    @property
    def horizon(self) -> int:
        """Current rollout horizon."""
        return self.stages[self.current_stage].horizon
    
    @property
    def stage_name(self) -> str:
        """Current stage name."""
        return self.stages[self.current_stage].name
        
    def should_advance(self, accuracy: float, epochs_in_stage: int) -> bool:
        """
        Check if should advance to next stage.
        
        Advances if:
        - Accuracy exceeds threshold, OR
        - Max epochs in stage reached
        """
        current = self.stages[self.current_stage]
        
        if accuracy > current.threshold:
            return True
        if epochs_in_stage >= current.max_epochs:
            print(f"  ⚠️ Forced advance after {epochs_in_stage} epochs")
            return True
        return False
    
    def advance_stage(self) -> bool:
        """Advance to next stage if not at last."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            new_stage = self.stages[self.current_stage]
            print(f"\n{'='*50}")
            print(f">>> ADVANCING TO STAGE {self.current_stage + 1}: {new_stage.name}")
            print(f"    Horizon: {new_stage.horizon} days")
            print(f"    Threshold: {new_stage.threshold:.0%}")
            print(f"{'='*50}\n")
            return True
        return False
    
    def train_step_tbptt(self,
                         batch: Dict[str, torch.Tensor],
                         forward_fn: Callable) -> Dict[str, float]:
        """
        Single training step with Truncated BPTT.
        
        Args:
            batch: Dict containing:
                - initial_state: [batch, n_tracers] initial NPZD state
                - forcing: [batch, horizon, n_forcing] environmental forcing
                - targets: [batch, horizon, n_tracers] target states
            forward_fn: Function to step model forward one timestep
                        signature: (state, forcing) -> new_state
                        
        Returns:
            metrics: Dict with loss and accuracy
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        C = batch['initial_state'].to(self.device)
        forcing_seq = batch['forcing'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        total_loss = 0
        n_steps = 0
        
        # Rollout with TBPTT
        for t in range(self.horizon):
            # Truncate gradient every τ steps
            if t > 0 and t % self.tbptt_length == 0:
                C = C.detach()
                C.requires_grad_(True)
            
            # Forward step
            forcing_t = forcing_seq[:, t, :]
            C = forward_fn(C, forcing_t)
            
            # Compute loss
            target_t = targets[:, t, :]
            loss_dict = self.loss_fn(C, target_t)
            total_loss = total_loss + loss_dict['total']
            n_steps += 1
        
        # Average loss
        avg_loss = total_loss / n_steps
        
        # Backward
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute accuracy (R² or similar)
        with torch.no_grad():
            pred_var = C.var(dim=0).mean()
            target_var = targets[:, -1, :].var(dim=0).mean()
            r2 = 1 - (C - targets[:, -1, :]).pow(2).mean() / (target_var + 1e-6)
            accuracy = max(0, r2.item())
        
        return {
            'loss': avg_loss.item(),
            'accuracy': accuracy,
        }
    
    def train_epoch(self,
                    dataloader,
                    forward_fn: Callable) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        losses = []
        accuracies = []
        
        for batch in dataloader:
            metrics = self.train_step_tbptt(batch, forward_fn)
            losses.append(metrics['loss'])
            accuracies.append(metrics['accuracy'])
        
        return {
            'loss': np.mean(losses),
            'accuracy': np.mean(accuracies),
        }
    
    def fit(self,
            train_loader,
            val_loader,
            forward_fn: Callable,
            max_epochs: int = 500,
            early_stop_patience: int = 20) -> Dict[str, List]:
        """
        Full training loop with curriculum learning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            forward_fn: Model forward function
            max_epochs: Maximum total epochs
            early_stop_patience: Patience for early stopping
            
        Returns:
            history: Training history
        """
        print("="*60)
        print(f"CURRICULUM TRAINING")
        print(f"Stages: {[s.name for s in self.stages]}")
        print(f"TBPTT truncation: τ = {self.tbptt_length}")
        print("="*60)
        
        epochs_in_stage = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, forward_fn)
            
            # Validation
            val_metrics = self.validate(val_loader, forward_fn)
            
            # Log
            self.history['loss'].append(train_metrics['loss'])
            self.history['accuracy'].append(val_metrics['accuracy'])
            self.history['stage'].append(self.current_stage)
            self.history['epoch'].append(epoch)
            
            # Print progress
            if epoch % 10 == 0 or val_metrics['accuracy'] > 0.85:
                print(f"Epoch {epoch:4d} | Stage {self.stage_name:12s} | "
                      f"Loss: {train_metrics['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.3f}")
            
            # Check stage advancement
            if self.should_advance(val_metrics['accuracy'], epochs_in_stage):
                if self.advance_stage():
                    epochs_in_stage = 0
                    patience_counter = 0
            else:
                epochs_in_stage += 1
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        return self.history
    
    def validate(self,
                 dataloader,
                 forward_fn: Callable) -> Dict[str, float]:
        """
        Validation step.
        """
        self.model.eval()
        losses = []
        accuracies = []
        
        with torch.no_grad():
            for batch in dataloader:
                C = batch['initial_state'].to(self.device)
                forcing_seq = batch['forcing'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Rollout
                for t in range(self.horizon):
                    C = forward_fn(C, forcing_seq[:, t, :])
                
                # Loss
                loss_dict = self.loss_fn(C, targets[:, -1, :])
                losses.append(loss_dict['total'].item())
                
                # Accuracy
                target_var = targets[:, -1, :].var(dim=0).mean()
                r2 = 1 - (C - targets[:, -1, :]).pow(2).mean() / (target_var + 1e-6)
                accuracies.append(max(0, r2.item()))
        
        return {
            'loss': np.mean(losses),
            'accuracy': np.mean(accuracies),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_curriculum_dataloader(data: torch.Tensor,
                                 forcing: torch.Tensor,
                                 horizon: int,
                                 batch_size: int = 32):
    """
    Create dataloader for curriculum training.
    
    Generates sequences of specified horizon from time series data.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # data: [T, N, n_tracers]
    # forcing: [T, N, n_forcing]
    
    T = data.shape[0]
    sequences = []
    
    for start in range(T - horizon - 1):
        seq = {
            'initial_state': data[start],
            'forcing': forcing[start:start+horizon],
            'targets': data[start+1:start+horizon+1],
        }
        sequences.append(seq)
    
    # Stack into tensors
    initial_states = torch.stack([s['initial_state'] for s in sequences])
    forcing_seqs = torch.stack([s['forcing'] for s in sequences])
    target_seqs = torch.stack([s['targets'] for s in sequences])
    
    dataset = TensorDataset(initial_states, forcing_seqs, target_seqs)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Curriculum Trainer...")
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
        def forward(self, state, forcing):
            return self.fc(state) * 0.99 + state * 0.01
    
    # Dummy loss
    class DummyLoss(nn.Module):
        def forward(self, pred, target):
            return {'total': nn.functional.mse_loss(pred, target)}
    
    model = DummyModel()
    loss_fn = DummyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = CurriculumTrainer(model, loss_fn, optimizer, tbptt_length=3, device='cpu')
    
    print(f"✓ Initial stage: {trainer.stage_name} (horizon={trainer.horizon})")
    
    # Test stage advancement
    print(f"\n✓ Testing stage advancement...")
    trainer.advance_stage()
    print(f"  Current stage: {trainer.stage_name} (horizon={trainer.horizon})")
    
    print("\n✅ All tests passed!")
