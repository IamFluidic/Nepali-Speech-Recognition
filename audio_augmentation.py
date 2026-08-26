import numpy as np
import torch
import torch.nn as nn
import random

class AudioAugmentor:
    """Audio Data Augmentation pipeline for robust speech recognition."""

    @staticmethod
    def add_gaussian_noise(audio_array, snr_db_range=(10, 25)):
        """Adds random Gaussian noise to audio signal with target SNR in dB."""
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
    def speed_perturbation(audio_array, speed_factors=(0.9, 1.1)):
        """Resamples audio to simulate different speaking speeds."""
        if audio_array is None or len(audio_array) == 0:
            return audio_array
            
        factor = random.uniform(speed_factors[0], speed_factors[1])
        indices = np.round(np.arange(0, len(audio_array), factor)).astype(int)
        indices = indices[indices < len(audio_array)]
        return audio_array[indices]

    @staticmethod
    def spec_augment(spec_tensor, freq_mask_max=8, time_mask_max=20):
        """
        Applies SpecAugment (Frequency and Time Masking) on feature spectrogram tensors.
        spec_tensor shape: (time_steps, n_features)
        """
        spec = spec_tensor.clone()
        num_frames, num_freqs = spec.shape

        # 1. Frequency Masking
        f_len = random.randint(0, min(freq_mask_max, num_freqs))
        f_zero = random.randint(0, max(0, num_freqs - f_len))
        spec[:, f_zero : f_zero + f_len] = 0.0

        # 2. Time Masking
        t_len = random.randint(0, min(time_mask_max, num_frames))
        t_zero = random.randint(0, max(0, num_frames - t_len))
        spec[t_zero : t_zero + t_len, :] = 0.0

        return spec

if __name__ == "__main__":
    dummy_spec = torch.randn(100, 39)
    augmented = AudioAugmentor.spec_augment(dummy_spec)
    print(f"SpecAugment test successful! Tensor shape: {augmented.shape}")
