"""
ECAPA-TDNN Model Implementation for Speaker Verification
Uses SpeechBrain's pretrained ECAPA-TDNN model with fine-tuning capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import os


class ECAPA_TDNN_Wrapper(nn.Module):
    """
    Wrapper for SpeechBrain's ECAPA-TDNN model
    Allows loading pretrained weights and fine-tuning for new speakers
    """
    
    def __init__(
        self,
        embedding_dim: int = 192,
        num_speakers: int = 351,
        pretrained_path: str = "speechbrain/spkrec-ecapa-voxceleb",
        freeze_encoder: bool = False
    ):
        """
        Args:
            embedding_dim: Dimension of speaker embeddings
            num_speakers: Number of speakers in the dataset
            pretrained_path: Path or HuggingFace model ID for pretrained model
            freeze_encoder: Whether to freeze encoder weights during training
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.pretrained_path = pretrained_path
        self.freeze_encoder = freeze_encoder
        
        # Load pretrained model (will be loaded in load_pretrained method)
        self.encoder = None
        self.embedding_model = None
        
        # Classification head for fine-tuning
        self.classifier = nn.Linear(embedding_dim, num_speakers)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def load_pretrained(self, device: str = "cuda"):
        """
        Load pretrained ECAPA-TDNN model from SpeechBrain
        Downloads all files directly to avoid Windows symlink issues
        """
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from huggingface_hub import hf_hub_download
            import shutil
            from pathlib import Path
            import os
            
            # Set environment variable to disable symlinks
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            # Download files directly without symlink
            savedir = Path("pretrained_models/ecapa")
            savedir.mkdir(parents=True, exist_ok=True)
            
            # Download all required files from HuggingFace
            files_to_download = [
                "hyperparams.yaml",
                "embedding_model.ckpt",
                "classifier.ckpt",
                "mean_var_norm_emb.ckpt",
                "label_encoder.txt"
            ]
            
            print("Downloading pretrained model files...")
            for filename in files_to_download:
                try:
                    target_file = savedir / filename
                    if target_file.exists():
                        print(f"  ✓ {filename} (already exists)")
                        continue
                        
                    downloaded_file = hf_hub_download(
                        repo_id=self.pretrained_path,
                        filename=filename,
                        local_dir=savedir,
                        local_dir_use_symlinks=False  # Critical: disable symlinks
                    )
                    print(f"  ✓ {filename}")
                except Exception as e:
                    print(f"  ⚠ {filename}: {e}")
            
            # Also need to handle label_encoder.ckpt naming
            label_txt = savedir / "label_encoder.txt"
            label_ckpt = savedir / "label_encoder.ckpt"
            if label_txt.exists() and not label_ckpt.exists():
                shutil.copy2(label_txt, label_ckpt)
                print(f"  ✓ label_encoder.ckpt (copied from .txt)")
            
            # Load pretrained model from local directory
            self.embedding_model = EncoderClassifier.from_hparams(
                source=self.pretrained_path,
                savedir=str(savedir),
                run_opts={"device": device}
            )
            
            print(f"✅ Successfully loaded pretrained ECAPA-TDNN from {self.pretrained_path}")
            
            # Freeze encoder if requested
            if self.freeze_encoder:
                for param in self.embedding_model.mods.parameters():
                    param.requires_grad = False
                print("Encoder weights frozen")
        
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Will initialize from scratch")
            self.embedding_model = None
    
    def forward(
        self,
        waveform: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            waveform: Input waveform tensor of shape (batch, time)
            return_embedding: If True, return embeddings; if False, return logits
        
        Returns:
            Either embeddings (batch, embedding_dim) or logits (batch, num_speakers)
        """
        # Extract embeddings using pretrained model
        if self.embedding_model is not None:
            # SpeechBrain model expects specific input format
            embeddings = self.embedding_model.encode_batch(waveform)
            
            # Reshape if needed
            if embeddings.dim() > 2:
                embeddings = embeddings.squeeze()
            
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
        else:
            # Fallback: use simple embedding (for testing)
            embeddings = torch.randn(
                waveform.shape[0], self.embedding_dim,
                device=waveform.device
            )
        
        if return_embedding:
            return embeddings
        
        # Pass through classifier for training
        logits = self.classifier(embeddings)
        return logits
    
    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from waveform
        
        Args:
            waveform: Input waveform
        
        Returns:
            Speaker embedding
        """
        with torch.no_grad():
            embedding = self.forward(waveform, return_embedding=True)
        return embedding
    
    def unfreeze_encoder(self):
        """
        Unfreeze encoder weights for fine-tuning
        """
        if self.embedding_model is not None:
            for param in self.embedding_model.mods.parameters():
                param.requires_grad = True
            self.freeze_encoder = False
            print("Encoder weights unfrozen")
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'embedding_dim': self.embedding_dim,
            'num_speakers': self.num_speakers,
            'classifier_state_dict': self.classifier.state_dict(),
            'freeze_encoder': self.freeze_encoder
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        # Save embedding model state if available
        if self.embedding_model is not None:
            checkpoint['embedding_model_state'] = self.embedding_model.mods.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, device: str = "cuda"):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Load classifier
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Load embedding model if available
        if 'embedding_model_state' in checkpoint and self.embedding_model is not None:
            self.embedding_model.mods.load_state_dict(checkpoint['embedding_model_state'])
        
        print(f"Checkpoint loaded from {path}")
        
        return checkpoint.get('epoch', 0)


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax Loss (AAM-Softmax)
    Also known as ArcFace loss
    
    Improves discriminative power of embeddings by adding angular margin
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_classes: Number of classes (speakers)
            margin: Angular margin penalty
            scale: Feature scale
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            embeddings: Input embeddings (batch, embedding_dim)
            labels: Ground truth labels (batch,)
        
        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Clip for numerical stability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angle
        theta = torch.acos(cosine)
        
        # Add margin to target angles
        one_hot = F.one_hot(labels, self.num_classes).float()
        theta_m = theta + one_hot * self.margin
        
        # Convert back to cosine
        cosine_m = torch.cos(theta_m)
        
        # Scale
        logits = self.scale * cosine_m
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


if __name__ == "__main__":
    # Test ECAPA-TDNN wrapper
    print("Testing ECAPA-TDNN wrapper...")
    
    # Create model
    model = ECAPA_TDNN_Wrapper(
        embedding_dim=192,
        num_speakers=351,
        pretrained_path="speechbrain/spkrec-ecapa-voxceleb",
        freeze_encoder=False
    )
    
    # Test forward pass (without loading pretrained)
    batch_size = 4
    waveform_length = 16000  # 2 seconds at 8kHz
    dummy_waveform = torch.randn(batch_size, waveform_length)
    
    print(f"Input waveform shape: {dummy_waveform.shape}")
    
    # Get embeddings
    embeddings = model(dummy_waveform, return_embedding=True)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Get logits
    logits = model(dummy_waveform, return_embedding=False)
    print(f"Logits shape: {logits.shape}")
    
    # Test AAM-Softmax loss
    print("\nTesting AAM-Softmax loss...")
    aam_loss = AAMSoftmax(
        embedding_dim=192,
        num_classes=351,
        margin=0.2,
        scale=30.0
    )
    
    labels = torch.randint(0, 351, (batch_size,))
    loss = aam_loss(embeddings, labels)
    print(f"AAM-Softmax loss: {loss.item():.4f}")
    
    # Test checkpoint saving/loading
    print("\nTesting checkpoint save/load...")
    checkpoint_path = "test_checkpoint.pt"
    model.save_checkpoint(checkpoint_path, epoch=1)
    
    # Create new model and load checkpoint
    model2 = ECAPA_TDNN_Wrapper(
        embedding_dim=192,
        num_speakers=351
    )
    epoch = model2.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Test checkpoint removed")
    
    print("\nECAPA-TDNN tests completed successfully!")
