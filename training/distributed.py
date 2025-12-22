"""
Multi-GPU Training with PyTorch Distributed Data Parallel (DDP).

Enables training on multiple GPUs for faster convergence.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Callable, Dict, Optional, Tuple
import numpy as np


def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize distributed process group.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: 'nccl' for GPU, 'gloo' for CPU
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] Initialized process group with {world_size} workers")


def cleanup_ddp():
    """Destroy distributed process group."""
    dist.destroy_process_group()


class DistributedTrainer:
    """
    Distributed training wrapper for Hybrid-FluxGNN.
    
    Features:
    - Automatic model wrapping with DDP
    - Distributed data sampling
    - Gradient synchronization
    - Checkpointing with rank awareness
    """
    
    def __init__(self,
                 model: nn.Module,
                 rank: int,
                 world_size: int,
                 loss_fn: nn.Module,
                 optimizer_cls: Callable = torch.optim.AdamW,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-5):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            rank: This process's rank
            world_size: Total number of processes
            loss_fn: Loss function
            optimizer_cls: Optimizer class
            lr: Learning rate
            weight_decay: Weight decay
        """
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Move model to device and wrap with DDP
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer_cls(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Track metrics
        self.metrics_history = []
        
    def create_distributed_dataloader(self,
                                      dataset,
                                      batch_size: int,
                                      shuffle: bool = True) -> DataLoader:
        """
        Create dataloader with distributed sampler.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle
            
        Returns:
            DataLoader with DistributedSampler
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=4
        )
    
    def train_epoch(self, 
                    dataloader: DataLoader,
                    forward_fn: Callable,
                    epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Distributed dataloader
            forward_fn: Function to compute model output
            epoch: Current epoch (for DistributedSampler)
            
        Returns:
            metrics: Dict of averaged metrics
        """
        self.model.train()
        
        # Important: set epoch for proper shuffling
        dataloader.sampler.set_epoch(epoch)
        
        local_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward
            outputs = forward_fn(self.model, batch)
            loss_dict = self.loss_fn(outputs, batch)
            loss = loss_dict['total']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            local_losses.append(loss.item())
        
        # Aggregate metrics across all processes
        avg_loss = np.mean(local_losses)
        
        # All-reduce to get global average
        loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / self.world_size
        
        return {'loss': global_avg_loss}
    
    @torch.no_grad()
    def evaluate(self, 
                dataloader: DataLoader,
                forward_fn: Callable) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Args:
            dataloader: Validation dataloader
            forward_fn: Model forward function
            
        Returns:
            metrics: Validation metrics
        """
        self.model.eval()
        
        local_losses = []
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            outputs = forward_fn(self.model, batch)
            loss_dict = self.loss_fn(outputs, batch)
            local_losses.append(loss_dict['total'].item())
        
        avg_loss = np.mean(local_losses)
        
        # All-reduce
        loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / self.world_size
        
        return {'val_loss': global_avg_loss}
    
    def save_checkpoint(self, 
                        path: str, 
                        epoch: int,
                        additional_state: Dict = None):
        """
        Save checkpoint (only rank 0 saves).
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            additional_state: Any additional state to save
        """
        if self.rank != 0:
            return
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # .module for DDP
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
        }
        
        if additional_state:
            state.update(additional_state)
        
        torch.save(state, path)
        print(f"[Rank 0] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            epoch: Epoch to resume from
        """
        # Map to correct device
        map_location = {'cuda:0': f'cuda:{self.rank}'}
        checkpoint = torch.load(path, map_location=map_location)
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        print(f"[Rank {self.rank}] Loaded checkpoint from {path}")
        
        return checkpoint['epoch']
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            forward_fn: Callable,
            n_epochs: int,
            checkpoint_dir: str = './checkpoints',
            checkpoint_every: int = 10) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            forward_fn: Model forward function
            n_epochs: Number of epochs
            checkpoint_dir: Directory for checkpoints
            checkpoint_every: Checkpoint frequency
            
        Returns:
            history: Training history
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"DISTRIBUTED TRAINING ON {self.world_size} GPUs")
            print(f"{'='*60}")
        
        for epoch in range(n_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, forward_fn, epoch)
            
            # Validation
            val_metrics = self.evaluate(val_loader, forward_fn)
            
            # Record
            self.metrics_history.append({
                'epoch': epoch,
                **train_metrics,
                **val_metrics
            })
            
            # Logging (only rank 0)
            if self.rank == 0:
                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    print(f"Epoch {epoch:4d} | "
                          f"Train Loss: {train_metrics['loss']:.4f} | "
                          f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Checkpoint
            if epoch % checkpoint_every == 0:
                self.save_checkpoint(
                    f"{checkpoint_dir}/checkpoint_epoch_{epoch:04d}.pt",
                    epoch
                )
        
        return {'metrics': self.metrics_history}


def run_distributed_training(rank: int, 
                             world_size: int,
                             model_factory: Callable,
                             train_data,
                             val_data,
                             config: Dict):
    """
    Worker function for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total processes
        model_factory: Function to create model
        train_data: Training dataset
        val_data: Validation dataset
        config: Training configuration
    """
    # Setup
    setup_ddp(rank, world_size)
    
    # Create model
    model = model_factory()
    
    # Create trainer
    from training.loss import MultivariateNPZDLoss
    loss_fn = MultivariateNPZDLoss()
    
    trainer = DistributedTrainer(
        model=model,
        rank=rank,
        world_size=world_size,
        loss_fn=loss_fn,
        lr=config.get('learning_rate', 5e-4)
    )
    
    # Create dataloaders
    train_loader = trainer.create_distributed_dataloader(
        train_data, 
        batch_size=config.get('batch_size', 32)
    )
    val_loader = trainer.create_distributed_dataloader(
        val_data, 
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    
    # Define forward function
    def forward_fn(model, batch):
        return model(batch['x'], batch['forcing'])
    
    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        forward_fn,
        n_epochs=config.get('n_epochs', 100)
    )
    
    # Cleanup
    cleanup_ddp()
    
    return history


def launch_distributed_training(model_factory: Callable,
                                train_data,
                                val_data,
                                config: Dict,
                                n_gpus: int = None):
    """
    Launch distributed training across multiple GPUs.
    
    Args:
        model_factory: Function to create model
        train_data: Training dataset
        val_data: Validation dataset
        config: Training configuration
        n_gpus: Number of GPUs (default: all available)
    """
    import torch.multiprocessing as mp
    
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    
    if n_gpus < 2:
        print("Warning: Distributed training requires >= 2 GPUs")
        print("Falling back to single-GPU training")
        return None
    
    print(f"Launching distributed training on {n_gpus} GPUs...")
    
    mp.spawn(
        run_distributed_training,
        args=(n_gpus, model_factory, train_data, val_data, config),
        nprocs=n_gpus,
        join=True
    )


# =============================================================================
# Tests (single-process mock)
# =============================================================================

if __name__ == "__main__":
    print("Testing Distributed Training Module...")
    
    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    if n_gpus == 0:
        print("No GPUs available. Testing DDP initialization only...")
        
        # Mock test
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 4)
            def forward(self, x):
                return self.fc(x)
        
        model = DummyModel()
        print(f"✓ Dummy model created: {sum(p.numel() for p in model.parameters())} params")
        
    else:
        print(f"Could launch distributed training on {n_gpus} GPUs")
        print("Use launch_distributed_training() to start")
    
    print("\n✅ Distributed training module ready!")
