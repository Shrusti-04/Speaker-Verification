"""
Training Script for Speaker Verification Models
Supports both ECAPA-TDNN and TiTANet architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import project modules
from src.dataset import SpeakerVerificationDataset, collate_fn
from src.augmentation import AudioAugmentation
from src.models.ecapa_tdnn import ECAPA_TDNN_Wrapper, AAMSoftmax
from src.models.titanet import TiTANet_Wrapper
from src.evaluation import VerificationMetrics, compute_cosine_similarity
from src.visualization import plot_training_history


class Trainer:
    """
    Trainer class for speaker verification models
    """
    
    def __init__(self, config_path: str, model_type: str = "ecapa"):
        """
        Args:
            config_path: Path to configuration YAML file
            model_type: Type of model ("ecapa" or "titanet")
        """
        self.model_type = model_type
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            self.config['hardware']['device']
            if torch.cuda.is_available()
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create directories
        self._create_directories()
        
        # Initialize model
        self._initialize_model()
        
        # Initialize loss function
        self._initialize_loss()
        
        # Initialize optimizer and scheduler
        self._initialize_optimizer()
        
        # Initialize data loaders
        self._initialize_dataloaders()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_eer': []
        }
        
        self.best_eer = float('inf')
        self.current_epoch = 0
    
    def _create_directories(self):
        """Create necessary directories"""
        checkpoint_dir = self.config['checkpoint']['save_dir']
        log_dir = self.config['logging']['log_dir']
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Log directory: {log_dir}")
    
    def _initialize_model(self):
        """Initialize the model"""
        if self.model_type == "ecapa":
            self.model = ECAPA_TDNN_Wrapper(
                embedding_dim=self.config['model']['embedding_dim'],
                num_speakers=self.config['dataset']['total_speakers'],
                pretrained_path=self.config['model']['pretrained_path'],
                freeze_encoder=(
                    self.config['training']['freeze_encoder_epochs'] > 0
                )
            )
            print("Initialized ECAPA-TDNN model")
            
            # Load pretrained weights
            self.model.load_pretrained(device=str(self.device))
        
        elif self.model_type == "titanet":
            self.model = TiTANet_Wrapper(
                embedding_dim=self.config['model']['embedding_dim'],
                num_speakers=self.config['dataset']['total_speakers'],
                pretrained_path=self.config['model']['pretrained_path'],
                freeze_encoder=(
                    self.config['training']['freeze_encoder_epochs'] > 0
                )
            )
            print("Initialized TiTANet model")
            
            # Load pretrained weights
            self.model.load_pretrained(device=str(self.device))
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
    
    def _initialize_loss(self):
        """Initialize loss function"""
        loss_type = self.config['training']['loss_function']
        
        if loss_type == "aam_softmax":
            self.criterion = AAMSoftmax(
                embedding_dim=self.config['model']['embedding_dim'],
                num_classes=self.config['dataset']['total_speakers'],
                margin=self.config['training']['margin'],
                scale=self.config['training']['scale']
            ).to(self.device)
            print("Using AAM-Softmax loss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using Cross-Entropy loss")
    
    def _initialize_optimizer(self):
        """Initialize optimizer and scheduler"""
        # Collect parameters
        if isinstance(self.criterion, AAMSoftmax):
            params = list(self.model.parameters()) + list(self.criterion.parameters())
        else:
            params = self.model.parameters()
        
        # Create optimizer
        optimizer_name = self.config['training']['optimizer'].lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        print(f"Using {optimizer_name} optimizer")
        
        # Create scheduler
        scheduler_name = self.config['training']['scheduler'].lower()
        if scheduler_name == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['training']['patience']
            )
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        else:
            self.scheduler = None
        
        print(f"Using {scheduler_name} scheduler")
    
    def _initialize_dataloaders(self):
        """Initialize data loaders"""
        # Create augmentation
        if self.config['augmentation']['apply']:
            augmentation = AudioAugmentation(
                sample_rate=self.config['dataset']['sample_rate'],
                apply_noise=self.config['augmentation']['noise']['apply'],
                apply_reverb=self.config['augmentation']['reverb']['apply'],
                apply_speed_perturb=self.config['augmentation']['speed_perturb']['apply'],
                prob_augment=self.config['augmentation']['prob_augment']
            )
        else:
            augmentation = None
        
        # Get combined data parameters from config (with defaults for backward compatibility)
        use_combined_data = self.config['dataset'].get('use_combined_data', False)
        train_split_ratio = self.config['dataset'].get('train_split_ratio', 0.8)
        random_seed = self.config['dataset'].get('random_seed', 42)
        
        # Create full training dataset
        full_dataset = SpeakerVerificationDataset(
            data_root=self.config['dataset']['data_root'],
            split=self.config['dataset']['train_dir'],
            sample_rate=self.config['dataset']['sample_rate'],
            min_duration=self.config['dataset']['min_duration'],
            max_duration=self.config['dataset']['max_duration'],
            transform=augmentation,
            use_combined_data=use_combined_data,
            train_split_ratio=train_split_ratio,
            random_seed=random_seed
        )
        
        # Split into train and validation
        val_split = self.config['validation']['val_split']
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=self.config['hardware']['pin_memory'],
            drop_last=True  # Drop incomplete batches to avoid BatchNorm issues
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=self.config['hardware']['pin_memory'],
            drop_last=True  # Drop incomplete batches to avoid BatchNorm issues
        )
    
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            waveform = batch['waveform'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.criterion, AAMSoftmax):
                # Extract embeddings and compute AAM-Softmax loss
                embeddings = self.model(waveform, return_embedding=True)
                loss = self.criterion(embeddings, labels)
            else:
                # Get logits and compute cross-entropy loss
                logits = self.model(waveform, return_embedding=False)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                logits = self.model(waveform, return_embedding=False)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For EER calculation
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                waveform = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                embeddings = self.model(waveform, return_embedding=True)
                logits = self.model.classifier(embeddings)
                
                # Compute loss
                loss = self.criterion(embeddings, labels) if isinstance(
                    self.criterion, AAMSoftmax
                ) else self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store embeddings for EER calculation
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        # Compute EER (simplified - compare same class vs different class)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        eer = self._compute_eer(all_embeddings, all_labels)
        
        return avg_loss, accuracy, eer
    
    def _compute_eer(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute EER from embeddings"""
        from src.evaluation import compute_eer
        
        # Sample pairs for EER calculation (to save time)
        n_samples = min(1000, len(embeddings))
        indices = torch.randperm(len(embeddings))[:n_samples]
        
        scores = []
        targets = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                idx1, idx2 = indices[i], indices[j]
                score = compute_cosine_similarity(
                    embeddings[idx1:idx1+1],
                    embeddings[idx2:idx2+1]
                ).item()
                
                target = 1 if labels[idx1] == labels[idx2] else 0
                
                scores.append(score)
                targets.append(target)
        
        scores = np.array(scores)
        targets = np.array(targets)
        
        eer, _ = compute_eer(scores, targets)
        
        return eer
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['num_epochs']} epochs...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Unfreeze encoder after specified epochs
            if epoch == self.config['training']['freeze_encoder_epochs']:
                self.model.unfreeze_encoder()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_eer = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_eer'].append(val_eer)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val EER: {val_eer*100:.2f}%")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_eer)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if val_eer < self.best_eer:
                self.best_eer = val_eer
                self._save_checkpoint(is_best=True)
                print(f"✓ New best EER: {val_eer*100:.2f}%")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(is_best=False)
        
        print(f"\nTraining completed! Best EER: {self.best_eer*100:.2f}%")
        
        # Save training history plot
        self._save_history_plot()
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.current_epoch+1}.pt'
        
        self.model.save_checkpoint(
            str(checkpoint_path),
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict()
        )
    
    def _save_history_plot(self):
        """Save training history plot"""
        log_dir = Path(self.config['logging']['log_dir'])
        plot_path = log_dir / 'training_history.png'
        
        plot_training_history(
            self.history,
            metrics=['loss', 'acc', 'eer'],
            save_path=str(plot_path)
        )


def main():
    parser = argparse.ArgumentParser(description='Train Speaker Verification Model')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model', type=str, default='ecapa',
        choices=['ecapa', 'titanet'],
        help='Model type (ecapa or titanet)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(args.config, model_type=args.model)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n{'='*60}")
        print(f"Resuming training from checkpoint: {args.resume}")
        print(f"{'='*60}\n")
        trainer.model.load_checkpoint(args.resume, device=str(trainer.device))
        trainer.model.to(trainer.device)
        # Ensure encoder is unfrozen for continued training
        if hasattr(trainer.model, 'embedding_model') and trainer.model.embedding_model is not None:
            for param in trainer.model.embedding_model.parameters():
                param.requires_grad = True
            print("Encoder weights unfrozen for continued training")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
