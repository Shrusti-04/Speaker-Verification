"""
Visualization Module for Speaker Verification
Implements t-SNE visualization and other plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Optional, List, Tuple
import torch


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization of Speaker Embeddings",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create t-SNE visualization of speaker embeddings
    
    Args:
        embeddings: Speaker embeddings (n_samples, embedding_dim)
        labels: Speaker labels (n_samples,)
        title: Plot title
        save_path: Path to save figure (optional)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        random_state: Random seed
        figsize: Figure size
    """
    print(f"Computing t-SNE projection (perplexity={perplexity})...")
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,  # Changed from n_iter to max_iter
        random_state=random_state,
        verbose=1
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    n_speakers = len(unique_labels)
    
    # Use a colormap
    if n_speakers <= 20:
        # Use distinct colors for few speakers
        colors = plt.cm.tab20(np.linspace(0, 1, n_speakers))
    else:
        # Use continuous colormap for many speakers
        colors = plt.cm.viridis(np.linspace(0, 1, n_speakers))
    
    # Plot each speaker
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Speaker {label}' if n_speakers <= 20 else None,
            alpha=0.6,
            s=50
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Add legend only if not too many speakers
    if n_speakers <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return embeddings_2d


def plot_confusion_matrix(
    similarity_matrix: np.ndarray,
    speaker_ids: List[str],
    title: str = "Speaker Similarity Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot similarity/confusion matrix for speakers
    
    Args:
        similarity_matrix: Similarity matrix (n_speakers, n_speakers)
        speaker_ids: List of speaker IDs
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        xticklabels=speaker_ids if len(speaker_ids) <= 50 else False,
        yticklabels=speaker_ids if len(speaker_ids) <= 50 else False,
        cmap='viridis',
        cbar=True,
        square=True,
        vmin=0,
        vmax=1
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Speaker ID', fontsize=12)
    plt.ylabel('Speaker ID', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    eer: float,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot ROC curve with EER point
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        eer: Equal error rate
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    # Mark EER point
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    plt.plot(
        fpr[eer_idx], tpr[eer_idx], 'go',
        markersize=10, label=f'EER = {eer*100:.2f}%'
    )
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_det_curve(
    fpr: np.ndarray,
    fnr: np.ndarray,
    eer: float,
    title: str = "DET Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot Detection Error Tradeoff (DET) curve
    
    Args:
        fpr: False positive rates
        fnr: False negative rates
        eer: Equal error rate
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    from scipy.stats import norm
    
    plt.figure(figsize=figsize)
    
    # Convert to normal deviate scale
    fpr_nd = norm.ppf(np.clip(fpr, 1e-10, 1-1e-10))
    fnr_nd = norm.ppf(np.clip(fnr, 1e-10, 1-1e-10))
    
    # Plot DET curve
    plt.plot(fpr_nd, fnr_nd, 'b-', linewidth=2, label='DET Curve')
    
    # Mark EER point
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    plt.plot(
        fpr_nd[eer_idx], fnr_nd[eer_idx], 'ro',
        markersize=10, label=f'EER = {eer*100:.2f}%'
    )
    
    # Set tick labels as percentages
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    tick_labels = [f'{t*100:.1f}%' for t in ticks]
    tick_locs = norm.ppf(ticks)
    
    plt.xticks(tick_locs, tick_labels)
    plt.yticks(tick_locs, tick_labels)
    
    plt.xlabel('False Positive Rate (%)', fontsize=12)
    plt.ylabel('False Negative Rate (%)', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            axes[i].plot(
                history[train_key],
                label=f'Train {metric}',
                linewidth=2
            )
        
        if val_key in history:
            axes[i].plot(
                history[val_key],
                label=f'Val {metric}',
                linewidth=2
            )
        
        axes[i].set_xlabel('Epoch', fontsize=11)
        axes[i].set_ylabel(metric.capitalize(), fontsize=11)
        axes[i].set_title(f'{metric.capitalize()} vs Epoch', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_score_distribution(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
    title: str = "Score Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot distribution of genuine and impostor scores
    
    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        threshold: Decision threshold
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot histograms
    plt.hist(
        impostor_scores, bins=50, alpha=0.5,
        label='Impostor', color='red', density=True
    )
    plt.hist(
        genuine_scores, bins=50, alpha=0.5,
        label='Genuine', color='green', density=True
    )
    
    # Plot threshold line
    plt.axvline(
        threshold, color='black', linestyle='--',
        linewidth=2, label=f'Threshold = {threshold:.3f}'
    )
    
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_embedding_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int = 3,
    title: str = "Embedding Distribution (First 3 dimensions)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot distribution of first few embedding dimensions
    
    Args:
        embeddings: Speaker embeddings
        labels: Speaker labels
        n_components: Number of dimensions to plot
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d' if n_components == 3 else None)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    if n_components == 3:
        # 3D scatter plot
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                embeddings[mask, 2],
                c=[colors[i]],
                label=f'Speaker {label}' if len(unique_labels) <= 20 else None,
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_zlabel('Dimension 3', fontsize=11)
    else:
        # 2D scatter plot
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[i]],
                label=f'Speaker {label}' if len(unique_labels) <= 20 else None,
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
    
    plt.title(title, fontsize=16, fontweight='bold')
    
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Generate dummy embeddings
    np.random.seed(42)
    n_speakers = 10
    n_samples_per_speaker = 20
    embedding_dim = 192
    
    embeddings = []
    labels = []
    
    for speaker in range(n_speakers):
        # Generate embeddings for this speaker (clustered)
        speaker_mean = np.random.randn(embedding_dim) * 5
        for _ in range(n_samples_per_speaker):
            emb = speaker_mean + np.random.randn(embedding_dim)
            embeddings.append(emb)
            labels.append(speaker)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Generated {embeddings.shape[0]} embeddings with {n_speakers} speakers")
    
    # Test t-SNE visualization
    print("\nPlotting t-SNE...")
    embeddings_2d = plot_tsne(
        embeddings, labels,
        title="Test t-SNE Visualization",
        perplexity=15,
        n_iter=500
    )
    
    # Test score distribution
    print("\nPlotting score distribution...")
    genuine_scores = np.random.beta(8, 2, 1000)
    impostor_scores = np.random.beta(2, 8, 1000)
    
    plot_score_distribution(
        genuine_scores,
        impostor_scores,
        threshold=0.5,
        title="Test Score Distribution"
    )
    
    # Test training history
    print("\nPlotting training history...")
    history = {
        'train_loss': [1.5, 1.2, 0.9, 0.7, 0.5],
        'val_loss': [1.6, 1.3, 1.0, 0.8, 0.7],
        'train_accuracy': [0.5, 0.6, 0.7, 0.8, 0.85],
        'val_accuracy': [0.48, 0.58, 0.68, 0.75, 0.80]
    }
    
    plot_training_history(history, metrics=['loss', 'accuracy'])
    
    print("\nVisualization tests completed!")
