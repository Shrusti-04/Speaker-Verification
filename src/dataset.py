"""
Dataset Module for Speaker Verification
Handles loading and preprocessing of audio files for training and testing
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import random
from pathlib import Path


class SpeakerVerificationDataset(Dataset):
    """
    Dataset class for speaker verification task
    Handles Train and Test folders with speaker-specific subdirectories
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "Train",
        sample_rate: int = 8000,
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        transform=None,
        use_combined_data: bool = False,
        train_split_ratio: float = 0.8,
        random_seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing Train and Test folders
            split: Either "Train" or "Test"
            sample_rate: Target sample rate (8000 Hz for telephone quality)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            transform: Optional transform to apply (augmentation)
            use_combined_data: If True, combine Train and Test folders and split by ratio
            train_split_ratio: Ratio for train/test split when use_combined_data=True
            random_seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.transform = transform
        self.use_combined_data = use_combined_data
        self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed
        
        # Load file paths and speaker labels
        self.audio_files, self.labels, self.speaker_ids = self._load_dataset()
        self.num_speakers = len(set(self.speaker_ids))
        
        # Create speaker to index mapping
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted(set(self.speaker_ids)))}
        
        print(f"Loaded {split} dataset:")
        print(f"  Total files: {len(self.audio_files)}")
        print(f"  Total speakers: {self.num_speakers}")
        print(f"  Files per speaker: ~{len(self.audio_files) / self.num_speakers:.1f}")
    
    def _load_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load all audio files from the dataset directory
        Expected structure: data_root/split/speaker_id/audio_files.wav
        
        If use_combined_data=True, combines Train and Test folders and splits
        by train_split_ratio per speaker
        """
        if self.use_combined_data:
            return self._load_combined_dataset()
        else:
            return self._load_single_split_dataset()
    
    def _load_single_split_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load dataset from a single split directory (original behavior)
        """
        audio_files = []
        labels = []
        speaker_ids = []
        
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Iterate through speaker directories
        speaker_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for speaker_idx, speaker_dir in enumerate(speaker_dirs):
            speaker_id = speaker_dir.name
            
            # Get all .wav files in the speaker directory
            wav_files = sorted(speaker_dir.glob("*.wav"))
            
            for wav_file in wav_files:
                audio_files.append(str(wav_file))
                labels.append(speaker_idx)
                speaker_ids.append(speaker_id)
        
        return audio_files, labels, speaker_ids
    
    def _load_combined_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Load dataset from both Train and Test folders, then split per speaker
        """
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        audio_files = []
        labels = []
        speaker_ids = []
        
        # Collect all speakers from both Train and Test directories
        train_dir = self.data_root / "Train"
        test_dir = self.data_root / "Test"
        
        all_speakers = set()
        if train_dir.exists():
            all_speakers.update([d.name for d in train_dir.iterdir() if d.is_dir()])
        if test_dir.exists():
            all_speakers.update([d.name for d in test_dir.iterdir() if d.is_dir()])
        
        all_speakers = sorted(all_speakers)
        
        # For each speaker, collect all files from both directories
        for speaker_idx, speaker_id in enumerate(all_speakers):
            speaker_files = []
            
            # Collect from Train directory
            train_speaker_dir = train_dir / speaker_id
            if train_speaker_dir.exists():
                speaker_files.extend(sorted(train_speaker_dir.glob("*.wav")))
            
            # Collect from Test directory
            test_speaker_dir = test_dir / speaker_id
            if test_speaker_dir.exists():
                speaker_files.extend(sorted(test_speaker_dir.glob("*.wav")))
            
            # Shuffle files for this speaker
            random.shuffle(speaker_files)
            
            # Split into train/test based on ratio
            num_files = len(speaker_files)
            num_train = int(num_files * self.train_split_ratio)
            
            if self.split == "Train":
                selected_files = speaker_files[:num_train]
            else:  # Test
                selected_files = speaker_files[num_train:]
            
            # Add to dataset
            for wav_file in selected_files:
                audio_files.append(str(wav_file))
                labels.append(speaker_idx)
                speaker_ids.append(speaker_id)
        
        return audio_files, labels, speaker_ids
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess audio file
        
        Returns:
            Dictionary containing:
                - waveform: Preprocessed audio tensor
                - label: Speaker label
                - speaker_id: Original speaker ID
                - file_path: Path to audio file
        """
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        speaker_id = self.speaker_ids[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure waveform is within duration limits
        waveform = self._adjust_duration(waveform)
        
        # Apply augmentation if provided
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        return {
            'waveform': waveform.squeeze(0),  # Remove channel dimension
            'label': torch.tensor(label, dtype=torch.long),
            'speaker_id': speaker_id,
            'file_path': audio_path
        }
    
    def _adjust_duration(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Adjust audio duration to be within min and max limits
        """
        num_samples = waveform.shape[1]
        min_samples = int(self.min_duration * self.sample_rate)
        max_samples = int(self.max_duration * self.sample_rate)
        
        # If too short, repeat the waveform
        if num_samples < min_samples:
            repeat_factor = int(np.ceil(min_samples / num_samples))
            waveform = waveform.repeat(1, repeat_factor)
            waveform = waveform[:, :min_samples]
        
        # If too long, randomly crop
        elif num_samples > max_samples:
            start_idx = random.randint(0, num_samples - max_samples)
            waveform = waveform[:, start_idx:start_idx + max_samples]
        
        return waveform
    
    def get_speaker_files(self, speaker_id: str) -> List[str]:
        """
        Get all audio files for a specific speaker
        """
        return [self.audio_files[i] for i in range(len(self.audio_files)) 
                if self.speaker_ids[i] == speaker_id]


class PairwiseVerificationDataset(Dataset):
    """
    Dataset for pairwise speaker verification task
    Generates positive and negative pairs for verification
    """
    
    def __init__(
        self,
        base_dataset: SpeakerVerificationDataset,
        num_pairs_per_speaker: int = 10,
        positive_ratio: float = 0.5
    ):
        """
        Args:
            base_dataset: Base speaker verification dataset
            num_pairs_per_speaker: Number of pairs to generate per speaker
            positive_ratio: Ratio of positive pairs (same speaker)
        """
        self.base_dataset = base_dataset
        self.num_pairs_per_speaker = num_pairs_per_speaker
        self.positive_ratio = positive_ratio
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        print(f"Generated {len(self.pairs)} verification pairs")
        print(f"  Positive pairs: {sum(p[2] for p in self.pairs)}")
        print(f"  Negative pairs: {sum(1-p[2] for p in self.pairs)}")
    
    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate positive and negative pairs
        Returns list of tuples: (file1_idx, file2_idx, label)
        where label is 1 for same speaker, 0 for different speakers
        """
        pairs = []
        
        # Group files by speaker
        speaker_to_files = {}
        for idx, speaker_id in enumerate(self.base_dataset.speaker_ids):
            if speaker_id not in speaker_to_files:
                speaker_to_files[speaker_id] = []
            speaker_to_files[speaker_id].append(idx)
        
        speakers = list(speaker_to_files.keys())
        
        for speaker in speakers:
            speaker_files = speaker_to_files[speaker]
            
            # Generate positive pairs (same speaker)
            num_positive = int(self.num_pairs_per_speaker * self.positive_ratio)
            for _ in range(num_positive):
                if len(speaker_files) >= 2:
                    idx1, idx2 = random.sample(speaker_files, 2)
                    pairs.append((idx1, idx2, 1))
            
            # Generate negative pairs (different speakers)
            num_negative = self.num_pairs_per_speaker - num_positive
            for _ in range(num_negative):
                other_speaker = random.choice([s for s in speakers if s != speaker])
                idx1 = random.choice(speaker_files)
                idx2 = random.choice(speaker_to_files[other_speaker])
                pairs.append((idx1, idx2, 0))
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pair of audio samples
        """
        idx1, idx2, label = self.pairs[idx]
        
        sample1 = self.base_dataset[idx1]
        sample2 = self.base_dataset[idx2]
        
        return {
            'waveform1': sample1['waveform'],
            'waveform2': sample2['waveform'],
            'label': torch.tensor(label, dtype=torch.float32),
            'speaker_id1': sample1['speaker_id'],
            'speaker_id2': sample2['speaker_id']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable length audio
    Pads all waveforms to the same length
    """
    # Find maximum length in batch
    max_len = max(item['waveform'].shape[0] for item in batch)
    
    # Pad waveforms
    waveforms = []
    labels = []
    speaker_ids = []
    file_paths = []
    
    for item in batch:
        waveform = item['waveform']
        # Pad to max length
        if waveform.shape[0] < max_len:
            padding = torch.zeros(max_len - waveform.shape[0])
            waveform = torch.cat([waveform, padding])
        
        waveforms.append(waveform)
        labels.append(item['label'])
        speaker_ids.append(item['speaker_id'])
        file_paths.append(item['file_path'])
    
    return {
        'waveform': torch.stack(waveforms),
        'label': torch.stack(labels),
        'speaker_id': speaker_ids,
        'file_path': file_paths
    }


def get_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    transform=None,
    use_combined_data: bool = False,
    train_split_ratio: float = 0.8,
    random_seed: int = 42,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for speaker verification
    
    Args:
        data_root: Root directory of dataset
        split: "Train" or "Test"
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        transform: Optional transforms/augmentations
        use_combined_data: If True, combine Train and Test folders and split by ratio
        train_split_ratio: Ratio for train/test split when use_combined_data=True
        random_seed: Random seed for reproducible splits
        **kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader instance
    """
    dataset = SpeakerVerificationDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        use_combined_data=use_combined_data,
        train_split_ratio=train_split_ratio,
        random_seed=random_seed,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    data_root = "data"
    
    # Test train dataset
    train_dataset = SpeakerVerificationDataset(
        data_root=data_root,
        split="Train",
        sample_rate=8000
    )
    
    print(f"\nSample from training set:")
    sample = train_dataset[0]
    print(f"  Waveform shape: {sample['waveform'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Speaker ID: {sample['speaker_id']}")
    print(f"  File path: {sample['file_path']}")
    
    # Test test dataset
    test_dataset = SpeakerVerificationDataset(
        data_root=data_root,
        split="Test",
        sample_rate=8000
    )
    
    # Test dataloader
    train_loader = get_dataloader(
        data_root=data_root,
        split="Train",
        batch_size=8,
        num_workers=0  # Set to 0 for testing
    )
    
    print(f"\nTest DataLoader:")
    batch = next(iter(train_loader))
    print(f"  Batch waveform shape: {batch['waveform'].shape}")
    print(f"  Batch labels shape: {batch['label'].shape}")
    print(f"  Number of speaker IDs: {len(batch['speaker_id'])}")
