"""
Data Augmentation Module for Speaker Verification
Implements various augmentation techniques to improve model robustness
"""

import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
from typing import Optional, List, Tuple


class AudioAugmentation(nn.Module):
    """
    Composite audio augmentation module
    Applies random augmentations to improve model robustness
    """
    
    def __init__(
        self,
        sample_rate: int = 8000,
        apply_noise: bool = True,
        apply_reverb: bool = True,
        apply_speed_perturb: bool = True,
        apply_time_masking: bool = False,
        apply_freq_masking: bool = False,
        prob_augment: float = 0.5,
        noise_snr_low: int = 0,
        noise_snr_high: int = 15,
        reverb_room_sizes: List[float] = [0.1, 0.5, 1.0],
        speed_factors: List[float] = [0.95, 1.0, 1.05],
        time_mask_param: int = 10,
        freq_mask_param: int = 10,
        num_time_masks: int = 2,
        num_freq_masks: int = 2
    ):
        """
        Args:
            sample_rate: Audio sample rate
            apply_noise: Whether to apply additive noise
            apply_reverb: Whether to apply reverberation
            apply_speed_perturb: Whether to apply speed perturbation
            apply_time_masking: Whether to apply time masking (SpecAugment)
            apply_freq_masking: Whether to apply frequency masking (SpecAugment)
            prob_augment: Probability of applying augmentation
            noise_snr_low: Minimum SNR for noise addition
            noise_snr_high: Maximum SNR for noise addition
            reverb_room_sizes: Room sizes for reverberation
            speed_factors: Speed perturbation factors
            time_mask_param: Maximum time mask size
            freq_mask_param: Maximum frequency mask size
            num_time_masks: Number of time masks
            num_freq_masks: Number of frequency masks
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.apply_noise = apply_noise
        self.apply_reverb = apply_reverb
        self.apply_speed_perturb = apply_speed_perturb
        self.apply_time_masking = apply_time_masking
        self.apply_freq_masking = apply_freq_masking
        self.prob_augment = prob_augment
        
        # Noise augmentation parameters
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high
        
        # Reverb parameters
        self.reverb_room_sizes = reverb_room_sizes
        
        # Speed perturbation parameters
        self.speed_factors = speed_factors
        
        # Masking parameters
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation to waveform
        
        Args:
            waveform: Input waveform (1D or 2D tensor)
        
        Returns:
            Augmented waveform
        """
        # Skip augmentation with probability (1 - prob_augment)
        if random.random() > self.prob_augment:
            return waveform
        
        # Apply speed perturbation first (changes length)
        if self.apply_speed_perturb and random.random() > 0.5:
            waveform = self.speed_perturbation(waveform)
        
        # Apply additive noise
        if self.apply_noise and random.random() > 0.5:
            waveform = self.add_noise(waveform)
        
        # Apply reverberation
        if self.apply_reverb and random.random() > 0.5:
            waveform = self.add_reverb(waveform)
        
        return waveform
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add white noise to waveform with random SNR
        
        Args:
            waveform: Input waveform
        
        Returns:
            Noisy waveform
        """
        # Generate random SNR
        snr_db = random.uniform(self.noise_snr_low, self.noise_snr_high)
        
        # Generate white noise
        noise = torch.randn_like(waveform)
        
        # Calculate signal and noise power
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        # Add scaled noise
        noisy_waveform = waveform + scale * noise
        
        return noisy_waveform
    
    def add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add reverberation to waveform
        Simple implementation using convolution with exponential decay
        
        Args:
            waveform: Input waveform
        
        Returns:
            Reverberant waveform
        """
        # Select random room size
        room_size = random.choice(self.reverb_room_sizes)
        
        # Create impulse response (exponential decay)
        ir_length = int(room_size * self.sample_rate)
        if ir_length < 10:
            ir_length = 10
        
        decay_time = room_size * 0.5  # seconds
        time = torch.linspace(0, ir_length / self.sample_rate, ir_length)
        impulse_response = torch.exp(-time / decay_time)
        
        # Normalize impulse response
        impulse_response = impulse_response / impulse_response.sum()
        
        # Handle batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            squeeze_dims = True
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            squeeze_dims = False
        else:
            squeeze_dims = False
        
        # Prepare impulse response for convolution
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution (reverberation)
        reverb_waveform = torch.nn.functional.conv1d(
            waveform,
            impulse_response,
            padding=ir_length // 2
        )
        
        # Trim to original length
        reverb_waveform = reverb_waveform[:, :, :waveform.shape[-1]]
        
        # Remove added dimensions
        if squeeze_dims:
            reverb_waveform = reverb_waveform.squeeze(0).squeeze(0)
        else:
            reverb_waveform = reverb_waveform.squeeze(1)
        
        # Mix original and reverberant signal
        mix_ratio = random.uniform(0.3, 0.7)
        mixed = mix_ratio * waveform.squeeze() + (1 - mix_ratio) * reverb_waveform
        
        return mixed
    
    def speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply speed perturbation to waveform
        
        Args:
            waveform: Input waveform
        
        Returns:
            Speed-perturbed waveform
        """
        # Select random speed factor
        speed_factor = random.choice(self.speed_factors)
        
        if speed_factor == 1.0:
            return waveform
        
        # Handle batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_dim = True
        else:
            squeeze_dim = False
        
        # Calculate new sample rate
        new_sample_rate = int(self.sample_rate * speed_factor)
        
        # Resample to change speed
        resampler = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=new_sample_rate
        )
        perturbed = resampler(waveform)
        
        # Resample back to original rate (changes duration)
        resampler_back = torchaudio.transforms.Resample(
            orig_freq=new_sample_rate,
            new_freq=self.sample_rate
        )
        perturbed = resampler_back(perturbed)
        
        if squeeze_dim:
            perturbed = perturbed.squeeze(0)
        
        return perturbed


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    Applied to spectrograms (time-frequency domain)
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 10,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        prob_augment: float = 0.5
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            prob_augment: Probability of applying augmentation
        """
        super().__init__()
        
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.prob_augment = prob_augment
        
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram
        
        Args:
            spectrogram: Input spectrogram of shape (..., freq, time)
        
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.prob_augment:
            return spectrogram
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_masking(spectrogram)
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            spectrogram = self.time_masking(spectrogram)
        
        return spectrogram


class BackgroundNoise:
    """
    Add real background noise from noise samples
    Useful when you have a noise dataset
    """
    
    def __init__(
        self,
        noise_dir: Optional[str] = None,
        sample_rate: int = 8000,
        snr_low: int = 0,
        snr_high: int = 15
    ):
        """
        Args:
            noise_dir: Directory containing noise audio files
            sample_rate: Audio sample rate
            snr_low: Minimum SNR in dB
            snr_high: Maximum SNR in dB
        """
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.snr_low = snr_low
        self.snr_high = snr_high
        
        self.noise_files = []
        if noise_dir is not None:
            self._load_noise_files()
    
    def _load_noise_files(self):
        """
        Load list of noise files from directory
        """
        import glob
        self.noise_files = glob.glob(f"{self.noise_dir}/*.wav")
        if len(self.noise_files) == 0:
            print(f"Warning: No noise files found in {self.noise_dir}")
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add background noise to waveform
        
        Args:
            waveform: Clean waveform
        
        Returns:
            Noisy waveform
        """
        if len(self.noise_files) == 0:
            # Fall back to white noise
            return self._add_white_noise(waveform)
        
        # Select random noise file
        noise_file = random.choice(self.noise_files)
        
        try:
            # Load noise
            noise, sr = torchaudio.load(noise_file)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                noise = resampler(noise)
            
            # Convert to mono
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            
            noise = noise.squeeze(0)
            
            # Match length
            if noise.shape[0] < waveform.shape[0]:
                # Repeat noise if too short
                repeat_factor = int(np.ceil(waveform.shape[0] / noise.shape[0]))
                noise = noise.repeat(repeat_factor)
            
            # Random crop
            if noise.shape[0] > waveform.shape[0]:
                start_idx = random.randint(0, noise.shape[0] - waveform.shape[0])
                noise = noise[start_idx:start_idx + waveform.shape[0]]
            
            # Add noise with random SNR
            snr_db = random.uniform(self.snr_low, self.snr_high)
            signal_power = waveform.pow(2).mean()
            noise_power = noise.pow(2).mean()
            
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt(signal_power / (snr_linear * noise_power))
            
            noisy_waveform = waveform + scale * noise
            
            return noisy_waveform
        
        except Exception as e:
            print(f"Error loading noise file {noise_file}: {e}")
            return self._add_white_noise(waveform)
    
    def _add_white_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add white noise as fallback
        """
        snr_db = random.uniform(self.snr_low, self.snr_high)
        noise = torch.randn_like(waveform)
        
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        return waveform + scale * noise


if __name__ == "__main__":
    # Test augmentation
    print("Testing audio augmentation...")
    
    # Create dummy audio (1 second at 8kHz)
    sample_rate = 8000
    duration = 1.0
    waveform = torch.randn(int(sample_rate * duration))
    
    print(f"Original waveform shape: {waveform.shape}")
    print(f"Original waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    
    # Test noise addition
    augmenter = AudioAugmentation(
        sample_rate=sample_rate,
        apply_noise=True,
        apply_reverb=False,
        apply_speed_perturb=False,
        prob_augment=1.0  # Always apply for testing
    )
    
    noisy = augmenter.add_noise(waveform)
    print(f"\nNoisy waveform range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    # Test reverb
    augmenter_reverb = AudioAugmentation(
        sample_rate=sample_rate,
        apply_noise=False,
        apply_reverb=True,
        apply_speed_perturb=False,
        prob_augment=1.0
    )
    
    reverb = augmenter_reverb.add_reverb(waveform)
    print(f"Reverberant waveform shape: {reverb.shape}")
    
    # Test speed perturbation
    augmenter_speed = AudioAugmentation(
        sample_rate=sample_rate,
        apply_noise=False,
        apply_reverb=False,
        apply_speed_perturb=True,
        prob_augment=1.0,
        speed_factors=[0.9, 1.1]
    )
    
    perturbed = augmenter_speed.speed_perturbation(waveform)
    print(f"Speed-perturbed waveform shape: {perturbed.shape}")
    
    # Test full augmentation pipeline
    print("\nTesting full augmentation pipeline...")
    full_augmenter = AudioAugmentation(
        sample_rate=sample_rate,
        apply_noise=True,
        apply_reverb=True,
        apply_speed_perturb=True,
        prob_augment=1.0
    )
    
    augmented = full_augmenter(waveform)
    print(f"Fully augmented waveform shape: {augmented.shape}")
    
    # Test batch augmentation
    print("\nTesting batch augmentation...")
    batch_waveform = torch.randn(4, int(sample_rate * duration))
    batch_augmented = full_augmenter(batch_waveform)
    print(f"Batch augmented shape: {batch_augmented.shape}")
    
    # Test SpecAugment
    print("\nTesting SpecAugment...")
    spec_augmenter = SpecAugment(
        freq_mask_param=10,
        time_mask_param=10,
        num_freq_masks=2,
        num_time_masks=2,
        prob_augment=1.0
    )
    
    # Create dummy spectrogram
    spectrogram = torch.randn(1, 80, 100)  # (batch, freq, time)
    augmented_spec = spec_augmenter(spectrogram)
    print(f"Augmented spectrogram shape: {augmented_spec.shape}")
    print(f"Number of masked values: {(augmented_spec == 0).sum().item()}")
    
    print("\nAugmentation tests completed successfully!")
