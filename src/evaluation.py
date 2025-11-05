"""
Evaluation Metrics for Speaker Verification
Implements EER, minDCF, and other verification metrics
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Tuple, Dict, List


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and corresponding threshold
    
    Args:
        scores: Similarity scores (higher means more similar)
        labels: Binary labels (1 for same speaker, 0 for different)
    
    Returns:
        Tuple of (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FPR = FNR (EER point)
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    
    # Alternative calculation using interpolation for more accuracy
    try:
        eer_interp = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = eer_interp
    except:
        pass  # Use the simpler method if interpolation fails
    
    return float(eer), float(eer_threshold)


def compute_minDCF(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0
) -> Tuple[float, float]:
    """
    Compute minimum Detection Cost Function (minDCF)
    
    Args:
        scores: Similarity scores
        labels: Binary labels
        p_target: Prior probability of target speaker
        c_miss: Cost of missing a target
        c_fa: Cost of false alarm
    
    Returns:
        Tuple of (minDCF, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Compute DCF for all thresholds
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
    # Find minimum
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]
    
    # Normalize by minimum possible cost
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf_normalized = min_dcf / c_default
    
    return float(min_dcf_normalized), float(min_dcf_threshold)


def compute_accuracy_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> float:
    """
    Compute accuracy at a given threshold
    
    Args:
        scores: Similarity scores
        labels: Binary labels
        threshold: Decision threshold
    
    Returns:
        Accuracy
    """
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    return float(accuracy)


def compute_far_frr(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    
    Args:
        scores: Similarity scores
        labels: Binary labels
        threshold: Decision threshold
    
    Returns:
        Tuple of (FAR, FRR)
    """
    predictions = scores >= threshold
    
    # FAR: false positives among negative samples
    negatives = labels == 0
    if negatives.sum() > 0:
        far = (predictions[negatives]).sum() / negatives.sum()
    else:
        far = 0.0
    
    # FRR: false negatives among positive samples
    positives = labels == 1
    if positives.sum() > 0:
        frr = (~predictions[positives]).sum() / positives.sum()
    else:
        frr = 0.0
    
    return float(far), float(frr)


def compute_cosine_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding (can be batched)
        embedding2: Second embedding (can be batched)
    
    Returns:
        Cosine similarity scores
    """
    # Normalize embeddings
    embedding1_norm = torch.nn.functional.normalize(embedding1, p=2, dim=-1)
    embedding2_norm = torch.nn.functional.normalize(embedding2, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = (embedding1_norm * embedding2_norm).sum(dim=-1)
    
    return similarity


def compute_euclidean_distance(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Euclidean distance between two embeddings
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
    
    Returns:
        Euclidean distances
    """
    distance = torch.norm(embedding1 - embedding2, p=2, dim=-1)
    return distance


class VerificationMetrics:
    """
    Class to accumulate and compute verification metrics
    """
    
    def __init__(self):
        self.scores = []
        self.labels = []
    
    def add_batch(self, scores: np.ndarray, labels: np.ndarray):
        """
        Add a batch of scores and labels
        
        Args:
            scores: Similarity scores
            labels: Binary labels
        """
        self.scores.extend(scores.flatten().tolist())
        self.labels.extend(labels.flatten().tolist())
    
    def compute_all_metrics(
        self,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute all verification metrics
        
        Returns:
            Dictionary of metrics
        """
        if len(self.scores) == 0:
            return {}
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Compute EER
        eer, eer_threshold = compute_eer(scores, labels)
        
        # Compute minDCF
        min_dcf, dcf_threshold = compute_minDCF(
            scores, labels, p_target, c_miss, c_fa
        )
        
        # Compute accuracy at EER threshold
        acc_at_eer = compute_accuracy_at_threshold(scores, labels, eer_threshold)
        
        # Compute FAR and FRR at EER threshold
        far, frr = compute_far_frr(scores, labels, eer_threshold)
        
        metrics = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'minDCF': min_dcf,
            'dcf_threshold': dcf_threshold,
            'accuracy_at_eer': acc_at_eer,
            'far_at_eer': far,
            'frr_at_eer': frr,
            'num_trials': len(scores),
            'num_positive': int(labels.sum()),
            'num_negative': int((1 - labels).sum())
        }
        
        return metrics
    
    def reset(self):
        """
        Reset accumulated scores and labels
        """
        self.scores = []
        self.labels = []
    
    def __len__(self):
        return len(self.scores)


def evaluate_verification_pairs(
    model: torch.nn.Module,
    pairs: List[Tuple[torch.Tensor, torch.Tensor, int]],
    device: str = "cuda",
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate verification performance on a list of audio pairs
    
    Args:
        model: Speaker verification model
        pairs: List of (audio1, audio2, label) tuples
        device: Device to run evaluation on
        batch_size: Batch size for processing
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics_calculator = VerificationMetrics()
    
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            
            audio1_batch = torch.stack([p[0] for p in batch_pairs]).to(device)
            audio2_batch = torch.stack([p[1] for p in batch_pairs]).to(device)
            labels_batch = np.array([p[2] for p in batch_pairs])
            
            # Extract embeddings
            emb1 = model.extract_embedding(audio1_batch)
            emb2 = model.extract_embedding(audio2_batch)
            
            # Compute similarity scores
            scores = compute_cosine_similarity(emb1, emb2)
            scores = scores.cpu().numpy()
            
            # Add to metrics
            metrics_calculator.add_batch(scores, labels_batch)
    
    # Compute all metrics
    metrics = metrics_calculator.compute_all_metrics()
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Verification Metrics"):
    """
    Pretty print verification metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    if 'eer' in metrics:
        print(f"Equal Error Rate (EER):        {metrics['eer']*100:.2f}%")
        print(f"EER Threshold:                 {metrics['eer_threshold']:.4f}")
    
    if 'minDCF' in metrics:
        print(f"Minimum Detection Cost (minDCF): {metrics['minDCF']:.4f}")
        print(f"DCF Threshold:                 {metrics['dcf_threshold']:.4f}")
    
    if 'accuracy_at_eer' in metrics:
        print(f"Accuracy at EER:               {metrics['accuracy_at_eer']*100:.2f}%")
    
    if 'far_at_eer' in metrics:
        print(f"FAR at EER:                    {metrics['far_at_eer']*100:.2f}%")
        print(f"FRR at EER:                    {metrics['frr_at_eer']*100:.2f}%")
    
    if 'num_trials' in metrics:
        print(f"\nNumber of trials:              {metrics['num_trials']}")
        print(f"  Positive pairs:              {metrics['num_positive']}")
        print(f"  Negative pairs:              {metrics['num_negative']}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test metrics computation
    print("Testing verification metrics...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate scores: positive pairs have higher scores
    positive_scores = np.random.beta(8, 2, n_samples // 2)  # Skewed towards 1
    negative_scores = np.random.beta(2, 8, n_samples // 2)  # Skewed towards 0
    
    scores = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    scores = scores[indices]
    labels = labels[indices]
    
    print(f"Generated {n_samples} samples")
    print(f"  Positive: {labels.sum()}")
    print(f"  Negative: {(1-labels).sum()}")
    
    # Compute EER
    eer, eer_threshold = compute_eer(scores, labels)
    print(f"\nEER: {eer*100:.2f}%")
    print(f"EER Threshold: {eer_threshold:.4f}")
    
    # Compute minDCF
    min_dcf, dcf_threshold = compute_minDCF(scores, labels)
    print(f"\nminDCF: {min_dcf:.4f}")
    print(f"DCF Threshold: {dcf_threshold:.4f}")
    
    # Test VerificationMetrics class
    print("\nTesting VerificationMetrics class...")
    metrics_calc = VerificationMetrics()
    metrics_calc.add_batch(scores, labels)
    metrics = metrics_calc.compute_all_metrics()
    
    print_metrics(metrics, "Test Metrics")
    
    # Test cosine similarity
    print("Testing cosine similarity...")
    emb1 = torch.randn(10, 192)
    emb2 = torch.randn(10, 192)
    similarity = compute_cosine_similarity(emb1, emb2)
    print(f"Similarity shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    
    print("\nMetrics tests completed successfully!")
