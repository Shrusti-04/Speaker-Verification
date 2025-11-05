"""
Evaluation Script for Speaker Verification Models
Comprehensive evaluation including EER, minDCF, and visualizations
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import project modules
from src.dataset import SpeakerVerificationDataset, PairwiseVerificationDataset
from src.models.ecapa_tdnn import ECAPA_TDNN_Wrapper
from src.models.titanet import TiTANet_Wrapper
from src.evaluation import (
    VerificationMetrics, compute_eer, compute_minDCF,
    compute_cosine_similarity, print_metrics
)
from src.verification import CosineScorer, PLDAScorer
from src.visualization import plot_tsne, plot_roc_curve, plot_score_distribution
from sklearn.metrics import roc_curve


class Evaluator:
    """
    Evaluator for speaker verification models
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        model_type: str = "ecapa",
        scorer_type: str = "cosine"
    ):
        """
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            model_type: Type of model ("ecapa" or "titanet")
            scorer_type: Type of scorer ("cosine" or "plda")
        """
        self.model_type = model_type
        self.scorer_type = scorer_type
        
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
        
        # Initialize model
        self._initialize_model(checkpoint_path)
        
        # Initialize scorer
        self._initialize_scorer()
        
        # Load test dataset
        self._load_test_dataset()
    
    def _initialize_model(self, checkpoint_path: str):
        """Initialize and load model"""
        if self.model_type == "ecapa":
            self.model = ECAPA_TDNN_Wrapper(
                embedding_dim=self.config['model']['embedding_dim'],
                num_speakers=self.config['dataset']['total_speakers'],
                pretrained_path=self.config['model']['pretrained_path']
            )
        elif self.model_type == "titanet":
            self.model = TiTANet_Wrapper(
                embedding_dim=self.config['model']['embedding_dim'],
                num_speakers=self.config['dataset']['total_speakers'],
                pretrained_path=self.config['model']['pretrained_path']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # CRITICAL: Load pretrained base model FIRST
        self.model.load_pretrained(device=str(self.device))
        
        # THEN load fine-tuned checkpoint (overwrites encoder with fine-tuned weights)
        self.model.load_checkpoint(checkpoint_path, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {self.model_type} model from {checkpoint_path}")
    
    def _initialize_scorer(self):
        """Initialize scoring function"""
        if self.scorer_type == "cosine":
            self.scorer = CosineScorer(normalize=True)
            print("Using cosine similarity scorer")
        elif self.scorer_type == "plda":
            self.scorer = PLDAScorer(
                embedding_dim=self.config['model']['embedding_dim']
            )
            print("Using PLDA scorer (will be fitted on training data)")
        else:
            raise ValueError(f"Unknown scorer type: {self.scorer_type}")
    
    def _load_test_dataset(self):
        """Load test dataset"""
        self.test_dataset = SpeakerVerificationDataset(
            data_root=self.config['dataset']['data_root'],
            split=self.config['dataset']['test_dir'],
            sample_rate=self.config['dataset']['sample_rate'],
            min_duration=self.config['dataset']['min_duration'],
            max_duration=self.config['dataset']['max_duration'],
            use_combined_data=self.config['dataset'].get('use_combined_data', False),
            train_split_ratio=self.config['dataset'].get('train_split_ratio', 0.8),
            random_seed=self.config['dataset'].get('random_seed', 42)
        )
        
        print(f"Loaded test dataset: {len(self.test_dataset)} samples")
    
    def extract_all_embeddings(self, dataset):
        """Extract embeddings for all samples in dataset"""
        embeddings = []
        labels = []
        speaker_ids = []
        
        print("Extracting embeddings...")
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                sample = dataset[i]
                waveform = sample['waveform'].unsqueeze(0).to(self.device)
                
                embedding = self.model.extract_embedding(waveform)
                
                embeddings.append(embedding.cpu().numpy())
                labels.append(sample['label'].item())
                speaker_ids.append(sample['speaker_id'])
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        
        return embeddings, labels, speaker_ids
    
    def fit_plda(self):
        """Fit PLDA scorer on training data"""
        if self.scorer_type != "plda":
            return
        
        print("\nFitting PLDA on training data...")
        
        # Load training dataset
        train_dataset = SpeakerVerificationDataset(
            data_root=self.config['dataset']['data_root'],
            split=self.config['dataset']['train_dir'],
            sample_rate=self.config['dataset']['sample_rate'],
            use_combined_data=self.config['dataset'].get('use_combined_data', False),
            train_split_ratio=self.config['dataset'].get('train_split_ratio', 0.8),
            random_seed=self.config['dataset'].get('random_seed', 42)
        )
        
        # Extract embeddings
        train_embeddings, train_labels, _ = self.extract_all_embeddings(train_dataset)
        
        # Fit PLDA
        self.scorer.fit(train_embeddings, train_labels)
    
    def evaluate_verification(self, num_pairs: int = 10000):
        """
        Evaluate speaker verification performance
        
        Args:
            num_pairs: Number of verification pairs to generate (will be split 50/50 positive/negative)
        """
        print(f"\nEvaluating verification with {num_pairs} pairs...")
        
        # Extract test embeddings
        test_embeddings, test_labels, test_speaker_ids = self.extract_all_embeddings(
            self.test_dataset
        )
        
        # Generate verification pairs (balanced)
        scores = []
        targets = []
        
        n_samples = len(test_embeddings)
        n_positive = num_pairs // 2
        n_negative = num_pairs - n_positive
        
        # Group samples by speaker for efficient pair generation
        speaker_to_indices = {}
        for idx, label in enumerate(test_labels):
            if label not in speaker_to_indices:
                speaker_to_indices[label] = []
            speaker_to_indices[label].append(idx)
        
        # Generate positive pairs (same speaker) - much faster
        print(f"Generating {n_positive} positive pairs...")
        for _ in tqdm(range(n_positive), desc="Positive pairs"):
            # Pick a random speaker with at least 2 samples
            valid_speakers = [spk for spk, indices in speaker_to_indices.items() if len(indices) >= 2]
            if not valid_speakers:
                break
            speaker = np.random.choice(valid_speakers)
            
            # Pick two random samples from this speaker
            idx1, idx2 = np.random.choice(speaker_to_indices[speaker], size=2, replace=False)
            
            # Compute score
            if isinstance(self.scorer, CosineScorer):
                score = self.scorer.score(
                    torch.tensor(test_embeddings[idx1]),
                    torch.tensor(test_embeddings[idx2])
                ).item()
            else:  # PLDA
                score = self.scorer.score(
                    test_embeddings[idx1],
                    test_embeddings[idx2]
                )
            
            scores.append(score)
            targets.append(1)
        
        # Generate negative pairs (different speakers)
        print(f"Generating {n_negative} negative pairs...")
        speakers = list(speaker_to_indices.keys())
        for _ in tqdm(range(n_negative), desc="Negative pairs"):
            # Pick two different speakers
            spk1, spk2 = np.random.choice(speakers, size=2, replace=False)
            
            # Pick one sample from each speaker
            idx1 = np.random.choice(speaker_to_indices[spk1])
            idx2 = np.random.choice(speaker_to_indices[spk2])
            
            # Compute score
            if isinstance(self.scorer, CosineScorer):
                score = self.scorer.score(
                    torch.tensor(test_embeddings[idx1]),
                    torch.tensor(test_embeddings[idx2])
                ).item()
            else:  # PLDA
                score = self.scorer.score(
                    test_embeddings[idx1],
                    test_embeddings[idx2]
                )
            
            scores.append(score)
            targets.append(0)
        
        scores = np.array(scores)
        targets = np.array(targets)
        
        # Compute metrics
        metrics = VerificationMetrics()
        metrics.add_batch(scores, targets)
        results = metrics.compute_all_metrics()
        
        # Print results
        print_metrics(results, f"{self.model_type.upper()} Verification Results")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(targets, scores)
        plot_roc_curve(
            fpr, tpr, results['eer'],
            title=f"{self.model_type.upper()} ROC Curve",
            save_path=f"results/{self.model_type}_roc_curve.png"
        )
        
        # Plot score distribution
        genuine_scores = scores[targets == 1]
        impostor_scores = scores[targets == 0]
        plot_score_distribution(
            genuine_scores, impostor_scores,
            results['eer_threshold'],
            title=f"{self.model_type.upper()} Score Distribution",
            save_path=f"results/{self.model_type}_score_distribution.png"
        )
        
        return results
    
    def visualize_embeddings(self):
        """Create t-SNE visualization of embeddings"""
        print("\nCreating t-SNE visualization...")
        
        # Extract embeddings
        embeddings, labels, speaker_ids = self.extract_all_embeddings(
            self.test_dataset
        )
        
        # Sample speakers if too many
        unique_labels = np.unique(labels)
        if len(unique_labels) > 50:
            print(f"Sampling 50 speakers from {len(unique_labels)} for visualization")
            selected_speakers = np.random.choice(unique_labels, size=50, replace=False)
            mask = np.isin(labels, selected_speakers)
            embeddings = embeddings[mask]
            labels = labels[mask]
        
        # Create t-SNE plot
        plot_tsne(
            embeddings, labels,
            title=f"{self.model_type.upper()} Speaker Embeddings (t-SNE)",
            save_path=f"results/{self.model_type}_tsne.png",
            perplexity=min(30, len(embeddings) // 5)
        )
    
    def evaluate_all(self):
        """Run complete evaluation"""
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Fit PLDA if needed
        if self.scorer_type == "plda":
            self.fit_plda()
        
        # Evaluate verification
        results = self.evaluate_verification()
        
        # Create visualizations
        self.visualize_embeddings()
        
        # Save results
        results_path = f"results/{self.model_type}_results.txt"
        with open(results_path, 'w') as f:
            f.write(f"{self.model_type.upper()} Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\nResults saved to {results_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Speaker Verification Model')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model', type=str, default='ecapa',
        choices=['ecapa', 'titanet'],
        help='Model type'
    )
    parser.add_argument(
        '--scorer', type=str, default='cosine',
        choices=['cosine', 'plda'],
        help='Scoring method'
    )
    parser.add_argument(
        '--num-pairs', type=int, default=10000,
        help='Number of verification pairs to evaluate'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(
        args.config,
        args.checkpoint,
        model_type=args.model,
        scorer_type=args.scorer
    )
    
    # Run evaluation
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
