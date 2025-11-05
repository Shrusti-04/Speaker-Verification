"""
Speaker Verification Module
Implements cosine similarity and PLDA scoring for speaker verification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CosineScorer:
    """
    Cosine similarity scorer for speaker verification
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize embeddings before computing similarity
        """
        self.normalize = normalize
    
    def score(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding (batch, dim) or (dim,)
            embedding2: Second embedding (batch, dim) or (dim,)
        
        Returns:
            Cosine similarity scores
        """
        if self.normalize:
            embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=-1)
            embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=-1)
        
        # Compute cosine similarity
        if embedding1.dim() == 1:
            similarity = (embedding1 * embedding2).sum()
        else:
            similarity = (embedding1 * embedding2).sum(dim=-1)
        
        return similarity
    
    def score_trials(
        self,
        enrollment_embeddings: torch.Tensor,
        test_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Score multiple enrollment-test pairs
        
        Args:
            enrollment_embeddings: Enrollment embeddings (n_enrollment, dim)
            test_embeddings: Test embeddings (n_test, dim)
        
        Returns:
            Score matrix (n_enrollment, n_test)
        """
        if self.normalize:
            enrollment_embeddings = torch.nn.functional.normalize(
                enrollment_embeddings, p=2, dim=-1
            )
            test_embeddings = torch.nn.functional.normalize(
                test_embeddings, p=2, dim=-1
            )
        
        # Compute all pairwise similarities
        scores = torch.mm(enrollment_embeddings, test_embeddings.t())
        
        return scores


class PLDAScorer:
    """
    Probabilistic Linear Discriminant Analysis (PLDA) scorer
    for speaker verification
    """
    
    def __init__(self, embedding_dim: int = 192):
        """
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.is_fitted = False
        
        # PLDA parameters (will be learned during training)
        self.mean = None
        self.transform_matrix = None
        self.within_class_precision = None
        self.between_class_precision = None
    
    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """
        Fit PLDA model on training data
        
        Args:
            embeddings: Training embeddings (n_samples, embedding_dim)
            labels: Speaker labels (n_samples,)
        """
        print("Fitting PLDA model...")
        
        # Center the data
        self.mean = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - self.mean
        
        # Use LDA for dimensionality reduction and class separation
        lda = LinearDiscriminantAnalysis()
        
        try:
            # Fit LDA
            lda.fit(centered_embeddings, labels)
            self.transform_matrix = lda.scalings_
            
            # Transform embeddings
            transformed = lda.transform(centered_embeddings)
            
            # Compute within-class and between-class covariance
            self._compute_covariance_matrices(transformed, labels)
            
            self.is_fitted = True
            print("PLDA model fitted successfully")
        
        except Exception as e:
            print(f"Error fitting PLDA: {e}")
            print("Falling back to simple LDA")
            self.is_fitted = False
    
    def _compute_covariance_matrices(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """
        Compute within-class and between-class covariance matrices
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        n_features = embeddings.shape[1]
        
        # Within-class covariance
        within_class_cov = np.zeros((n_features, n_features))
        class_means = []
        
        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            class_mean = np.mean(class_embeddings, axis=0)
            class_means.append(class_mean)
            
            centered = class_embeddings - class_mean
            within_class_cov += np.dot(centered.T, centered)
        
        within_class_cov /= len(embeddings)
        
        # Between-class covariance
        overall_mean = np.mean(embeddings, axis=0)
        between_class_cov = np.zeros((n_features, n_features))
        
        for i, label in enumerate(unique_labels):
            n_samples = np.sum(labels == label)
            diff = class_means[i] - overall_mean
            between_class_cov += n_samples * np.outer(diff, diff)
        
        between_class_cov /= len(embeddings)
        
        # Compute precision matrices (inverse covariances)
        try:
            self.within_class_precision = np.linalg.inv(
                within_class_cov + 1e-6 * np.eye(n_features)
            )
            self.between_class_precision = np.linalg.inv(
                between_class_cov + 1e-6 * np.eye(n_features)
            )
        except:
            # Use pseudo-inverse if singular
            self.within_class_precision = np.linalg.pinv(within_class_cov)
            self.between_class_precision = np.linalg.pinv(between_class_cov)
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using PLDA
        
        Args:
            embeddings: Input embeddings
        
        Returns:
            Transformed embeddings
        """
        if not self.is_fitted:
            return embeddings
        
        # Center
        centered = embeddings - self.mean
        
        # Apply transformation
        if self.transform_matrix is not None:
            transformed = np.dot(centered, self.transform_matrix)
            return transformed
        
        return centered
    
    def score(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute PLDA score between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            PLDA score (higher means more similar)
        """
        if not self.is_fitted:
            # Fall back to cosine similarity
            return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        
        # Transform embeddings
        emb1_transformed = self.transform(embedding1.reshape(1, -1)).flatten()
        emb2_transformed = self.transform(embedding2.reshape(1, -1)).flatten()
        
        # Compute PLDA score using Mahalanobis distance
        diff = emb1_transformed - emb2_transformed
        
        # Score based on within-class precision
        score = -np.dot(diff, np.dot(self.within_class_precision, diff))
        
        return float(score)
    
    def score_trials(
        self,
        enrollment_embeddings: np.ndarray,
        test_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Score multiple enrollment-test pairs
        
        Args:
            enrollment_embeddings: Enrollment embeddings (n_enrollment, dim)
            test_embeddings: Test embeddings (n_test, dim)
        
        Returns:
            Score matrix (n_enrollment, n_test)
        """
        n_enrollment = enrollment_embeddings.shape[0]
        n_test = test_embeddings.shape[0]
        scores = np.zeros((n_enrollment, n_test))
        
        for i in range(n_enrollment):
            for j in range(n_test):
                scores[i, j] = self.score(
                    enrollment_embeddings[i],
                    test_embeddings[j]
                )
        
        return scores


class SpeakerVerifier:
    """
    High-level speaker verification system
    """
    
    def __init__(
        self,
        model: nn.Module,
        scorer_type: str = "cosine",
        device: str = "cuda"
    ):
        """
        Args:
            model: Speaker embedding model
            scorer_type: Type of scorer ("cosine" or "plda")
            device: Device to run model on
        """
        self.model = model
        self.scorer_type = scorer_type
        self.device = device
        
        # Initialize scorer
        if scorer_type == "cosine":
            self.scorer = CosineScorer(normalize=True)
        elif scorer_type == "plda":
            embedding_dim = getattr(model, 'embedding_dim', 192)
            self.scorer = PLDAScorer(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown scorer type: {scorer_type}")
        
        self.model.to(device)
        self.model.eval()
    
    def enroll_speaker(
        self,
        audio_files: List[str]
    ) -> torch.Tensor:
        """
        Enroll a speaker using multiple audio files
        
        Args:
            audio_files: List of audio file paths
        
        Returns:
            Speaker embedding (averaged across files)
        """
        embeddings = []
        
        with torch.no_grad():
            for audio_file in audio_files:
                # Load audio
                import torchaudio
                waveform, sr = torchaudio.load(audio_file)
                
                # Resample if needed
                if sr != 8000:
                    resampler = torchaudio.transforms.Resample(sr, 8000)
                    waveform = resampler(waveform)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                waveform = waveform.to(self.device)
                
                # Extract embedding
                embedding = self.model.extract_embedding(waveform)
                embeddings.append(embedding)
        
        # Average embeddings
        enrollment_embedding = torch.stack(embeddings).mean(dim=0)
        
        return enrollment_embedding
    
    def verify(
        self,
        enrollment_embedding: torch.Tensor,
        test_audio: str,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Verify if test audio matches enrolled speaker
        
        Args:
            enrollment_embedding: Enrolled speaker embedding
            test_audio: Path to test audio file
            threshold: Decision threshold
        
        Returns:
            Tuple of (is_same_speaker, score)
        """
        # Load test audio
        import torchaudio
        waveform, sr = torchaudio.load(test_audio)
        
        # Resample if needed
        if sr != 8000:
            resampler = torchaudio.transforms.Resample(sr, 8000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.to(self.device)
        
        # Extract test embedding
        with torch.no_grad():
            test_embedding = self.model.extract_embedding(waveform)
        
        # Compute score
        if isinstance(self.scorer, CosineScorer):
            score = self.scorer.score(enrollment_embedding, test_embedding)
            score = score.item()
        else:  # PLDA
            score = self.scorer.score(
                enrollment_embedding.cpu().numpy(),
                test_embedding.cpu().numpy()
            )
        
        # Make decision
        is_same_speaker = score >= threshold
        
        return is_same_speaker, score


if __name__ == "__main__":
    # Test cosine scorer
    print("Testing CosineScorer...")
    scorer = CosineScorer(normalize=True)
    
    emb1 = torch.randn(192)
    emb2 = emb1 + 0.1 * torch.randn(192)  # Similar embedding
    emb3 = torch.randn(192)  # Different embedding
    
    score_similar = scorer.score(emb1, emb2)
    score_different = scorer.score(emb1, emb3)
    
    print(f"Score (similar embeddings): {score_similar:.4f}")
    print(f"Score (different embeddings): {score_different:.4f}")
    
    # Test batch scoring
    print("\nTesting batch scoring...")
    enrollment_embs = torch.randn(5, 192)
    test_embs = torch.randn(10, 192)
    
    scores = scorer.score_trials(enrollment_embs, test_embs)
    print(f"Score matrix shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Test PLDA scorer
    print("\nTesting PLDAScorer...")
    plda_scorer = PLDAScorer(embedding_dim=192)
    
    # Generate dummy training data
    n_speakers = 50
    n_samples_per_speaker = 10
    embeddings = []
    labels = []
    
    for speaker in range(n_speakers):
        # Generate embeddings for this speaker
        speaker_mean = np.random.randn(192)
        for _ in range(n_samples_per_speaker):
            emb = speaker_mean + 0.3 * np.random.randn(192)
            embeddings.append(emb)
            labels.append(speaker)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Training data: {embeddings.shape}")
    
    # Fit PLDA
    plda_scorer.fit(embeddings, labels)
    
    # Test scoring
    if plda_scorer.is_fitted:
        test_emb1 = embeddings[0]
        test_emb2 = embeddings[1]  # Same speaker
        test_emb3 = embeddings[n_samples_per_speaker]  # Different speaker
        
        score_same = plda_scorer.score(test_emb1, test_emb2)
        score_diff = plda_scorer.score(test_emb1, test_emb3)
        
        print(f"PLDA score (same speaker): {score_same:.4f}")
        print(f"PLDA score (different speaker): {score_diff:.4f}")
    
    print("\nVerification module tests completed successfully!")
