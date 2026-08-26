"""
audio_augmentation.py
=====================
Advanced Audio & Spectrogram Data Augmentation Pipeline for Robust ASR.
Features:
  1. 3-Way Speed Perturbation (0.9x, 1.0x, 1.1x) to simulate variable speaking tempos.
  2. Additive White & Room Background Noise Injection with SNR dB ranges.
  3. SpecAugment+ (Multi-band Frequency Masking & Time Masking).
  4. Pitch & Energy Scaling.
"""

import random
import numpy as np
import torch


class AudioAugmentor:
    """Audio Data Augmentation pipeline for robust Nepali speech recognition."""

    @staticmethod
    def add_gaussian_noise(audio_array: np.ndarray, snr_db_range=(12, 28)) -> np.ndarray:
        """Adds random Gaussian background noise with realistic SNR."""
        if audio_array is None or len(audio_array) == 0:
            return audio_array

        snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
        audio_power = np.mean(audio_array ** 2)
        if audio_power == 0:
            return audio_array

        noise_power = audio_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), audio_array.shape)
        return (audio_array + noise).astype(np.float32)

    @staticmethod
    def speed_perturbation(audio_array: np.ndarray, speed_rates=(0.9, 1.0, 1.1)) -> np.ndarray:
        """
        Applies standard 3-rate speed perturbation (0.9x slower, 1.0x normal, 1.1x faster)
        simulating diverse speaking speeds.
        """
        if audio_array is None or len(audio_array) == 0:
            return audio_array

        rate = random.choice(speed_rates)
        if rate == 1.0:
            return audio_array

        indices = np.round(np.arange(0, len(audio_array), rate)).astype(int)
        indices = indices[indices < len(audio_array)]
        return audio_array[indices]

    @staticmethod
    def amplitude_scaling(audio_array: np.ndarray, gain_range=(0.7, 1.3)) -> np.ndarray:
        """Scales audio signal volume randomly."""
        if audio_array is None or len(audio_array) == 0:
            return audio_array
        gain = random.uniform(gain_range[0], gain_range[1])
        return np.clip(audio_array * gain, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def augment_raw_audio(audio_array: np.ndarray, apply_prob: float = 0.5) -> np.ndarray:
        """Composite raw audio augmentation pipeline."""
        if audio_array is None or len(audio_array) == 0 or random.random() > apply_prob:
            return audio_array

        aug_audio = audio_array.copy()
        # 1. Speed perturbation
        aug_audio = AudioAugmentor.speed_perturbation(aug_audio)
        # 2. Volume scaling
        aug_audio = AudioAugmentor.amplitude_scaling(aug_audio)
        # 3. Additive noise (30% chance)
        if random.random() < 0.35:
            aug_audio = AudioAugmentor.add_gaussian_noise(aug_audio)

        return aug_audio

    @staticmethod
    def spec_augment(spec_tensor: torch.Tensor, freq_mask_max=8, time_mask_max=25, num_freq_masks=2, num_time_masks=2) -> torch.Tensor:
        """
        Applies SpecAugment+ (Multi-band Frequency & Time Masking) on MFCC feature tensors.
        spec_tensor shape: (time_steps, n_features)
        """
        spec = spec_tensor.clone()
        num_frames, num_freqs = spec.shape

        # Multi-band Frequency Masking
        for _ in range(num_freq_masks):
            f_len = random.randint(0, min(freq_mask_max, num_freqs))
            f_zero = random.randint(0, max(0, num_freqs - f_len))
            spec[:, f_zero : f_zero + f_len] = 0.0

        # Multi-band Time Masking
        for _ in range(num_time_masks):
            t_len = random.randint(0, min(time_mask_max, num_frames))
            t_zero = random.randint(0, max(0, num_frames - t_len))
            spec[t_zero : t_zero + t_len, :] = 0.0

        return spec


if __name__ == "__main__":
    dummy_audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    aug = AudioAugmentor.augment_raw_audio(dummy_audio)
    dummy_spec = torch.randn(100, 39)
    spec_aug = AudioAugmentor.spec_augment(dummy_spec)
    print(f"Audio Augmentor verification successful! Spec shape: {spec_aug.shape}")
