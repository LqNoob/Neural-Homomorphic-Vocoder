# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from functools import lru_cache
from numpy import pi
from typing import Union, Optional, Tuple, Iterable


def repeat_interpolate(x: Tensor, frame_size: int) -> Tensor:
    """Upsample by repeating in each frame.
    """
    # x: [..., n_frame], y: [..., n_sample = n_frame * frame_size]
    return torch.repeat_interleave(x, frame_size, -1)


def linear_interpolate(x: Tensor, frame_size: int) -> Tensor:
    """Wrapper of 1D linear interpolation.
    """
    # x: [n_batch, n_channel, n_frame], y: [n_batch, n_channel, n_frame * frame_size]
    return torch.nn.functional.interpolate(x, scale_factor=frame_size, mode="linear", align_corners=False)


@lru_cache(2)
def freq_multiplier(n_harmonic: int, device: torch.device) -> Tensor:
    """Generate the frequency multiplier [[[1], [2], ..., [n_harmonic]]]
    This function is LRU cached.
    Returns:
        multiplier: [1, n_harmonic, 1]
    """
    a = torch.as_tensor([[1.0 * k for k in range(1, n_harmonic + 1)]]).reshape(1, n_harmonic, 1).to(device)
    return a


def freq_antialias_mask(sampling_rate: Union[int, float], freq_tensor: Tensor, hard_boundary: Optional[bool] = True) -> Tensor:
    """Return a harmonic amplitude mask that silence any harmonics above sampling_rate / 2.
    Args:
        freq_tensor (Tensor): Tensor of any shape (...), values in Hertz.
    Returns:
        mask[freq_tensor > fs / 2] are zeroed.
    """
    if hard_boundary:
        return (freq_tensor < sampling_rate / 2.0).float()
    else:
        return torch.sigmoid(-(freq_tensor - sampling_rate / 2.0))


def harmonic_amplitudes_to_signal(f0_t: Tensor, harmonic_amplitudes_t: Tensor,
                                  sampling_rate: int, min_f0: float) -> Tensor:
    """Generate harmonic signal from given frequency and harmonic amplitudes.
    The phase of sinusoids are assumed to be all zero. The periodic function
    used is SINE.

    Args:
        f0_t: [n_batch, 1, n_sample]. Fundamental frequency per sampling point in Hertz.
        harmonic_amplitudes_t: [n_batch, n_harmonic, n_sample]. Harmonic amplitudes per sampling point.
        sampling_rate: Sampling rate in Hertz.
        min_f0: Minimum f0 to accept. All f0_t below min_f0 are ignored.

    Returns:
        signal: [n_batch, 1, n_sample]. Sum of sinusoids with given harmonic amplitudes and fundamental frequencies.
    """
    _, n_harmonic, _ = harmonic_amplitudes_t.shape
    f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t

    # [n_batch, n_harmonic, n_sample]
    weight_map = freq_antialias_mask(sampling_rate, f0_map) * harmonic_amplitudes_t
    f0_map_cum = f0_t.cumsum(dim=-1) * freq_multiplier(n_harmonic, f0_t.device)
    w0_map_cum = f0_map_cum * 2.0 * pi / sampling_rate

    # [n_batch, 1, n_sample]
    source = torch.sum(torch.sin(w0_map_cum) * weight_map, dim=-2, keepdim=True)
    source = (~(f0_t < min_f0)).float() * source
    return source * 0.01


def generate_impulse_train(f0_t: Tensor, n_harmonic: int,
                           sampling_rate: Union[int, float],
                           source_amp: Optional[float] = 0.01,
                           min_f0: Optional[float] = 1.0) -> Tensor:
    """Generate impulse train with sinusoidal synthesis.
    Args:
        f0_t: [n_batch, 1, n_sample]
        n_harmonic: Maximum number of harmonics in sinusoidal synthesis.
        sampling_rate: Sampling rate in Hertz.
    Returns:
        signal: [n_batch, 1, n_sample]
    """

    # [n_batch, n_harmonic, n_sample]
    f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
    weight_map = freq_antialias_mask(sampling_rate, f0_map)  

    w0_map_cum = (f0_t.cumsum(dim=-1) * 2.0 * pi / sampling_rate * freq_multiplier(n_harmonic, f0_t.device))

    # [n_batch, 1, n_sample]
    source = torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)
    source = (~(f0_t < min_f0)).float() * source

    return source * source_amp


def generate_random_Noise(f0_sample: Tensor,
                          min_f0_sample: Optional[float] = 1.,
                          noise_std: Optional[float] = 0.001) -> Tensor:
   
    vuv = (f0_sample > min_f0_sample).float()
    noise_amp = vuv * noise_std + (1 - vuv) * noise_std * 10
    noise_source = noise_amp * torch.randn_like(f0_sample, device=f0_sample.device)

    return noise_source


if __name__ == "__main__":

    import glob
    import os
    import librosa
    import numpy as np

    f0_path = 'BZNSYP_22050/5ms/f0'
    for fpath in glob.glob(os.path.join(f0_path,'*.f0'))[-1:]:
        f0 = np.fromfile(fpath, dtype=np.float32)
        f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(-1)
        print(f0.shape)
        
        up_sampled_data = repeat_interpolate(f0.permute(0, 2, 1), 128)
        sine_waves = generate_impulse_train(up_sampled_data, 7, 22050)
        librosa.output.write_wav('sin_zjl.wav', sine_waves.squeeze().numpy().astype(np.float32), sr=22050)
        #librosa.output.write_wav('noi_zjl.wav', torch.randn_like(sine_waves).squeeze().numpy().astype(np.float32)*0.01, sr=22050)
