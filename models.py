# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F

from cepstrum import complex_cepstrum_to_imp, complex_cepstrum_scale, complex_cepstrum_lowpass_mask
from conv_filter import ltv_fir, hann_ltv_fir
from source_generator_zjlww import generate_impulse_train, generate_random_Noise, repeat_interpolate

from trainable_filter import ExpDecayReverb
from reverb import TrainableFIRReverb


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class nn_filter_Estimator(torch.nn.Module):
    def __init__(self, in_channels=80, channels=256, out_channels=222, \
                 kernel_size=3, padding=1, \
                 ccep_size=222, fft_size=1024, \
                 max_quefrency=math.ceil(22.05*6), \
                 device='cuda'):
        super(nn_filter_Estimator, self).__init__()

        self.convs_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels, channels, kernel_size, padding=padding),
            torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=8),
            torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=8),
        ])
        self.final_layer = torch.nn.Conv1d(channels, out_channels, kernel_size, padding=padding)

        self.fft_size = fft_size
        self.ccep_size = ccep_size
        self.max_quefrency = max_quefrency

        self.cepstrum_scale = complex_cepstrum_scale(self.ccep_size, device=device).unsqueeze(-1)
        self.cepstrum_lowpass_mask = complex_cepstrum_lowpass_mask(self.ccep_size, self.max_quefrency, device=device).unsqueeze(-1)

        self.convs_layers.apply(init_weights)
        self.final_layer.apply(init_weights)


    def forward(self, feat):
        # [B, n_mels, n_frame]
        x = feat

        # inference complex cepstrum
        for l in self.convs_layers:
            x = l(x)
            x = F.relu(x)
        
        # generate impulse response from complex cepstrum
        #ccep = self.final_layer(x) * complex_cepstrum_scale(self.ccep_size).unsqueeze(-1)  # [B, ccep_size, n_frame]
        #ccep = ccep * complex_cepstrum_lowpass_mask(self.ccep_size, self.max_quefrency).unsqueeze(-1).to(ccep)  # liftering method in quefrency domain
        ccep = self.final_layer(x) * self.cepstrum_scale
        ccep = ccep * self.cepstrum_lowpass_mask
        ccep_impluse = complex_cepstrum_to_imp(ccep, self.fft_size, dim=1)   # [B, fft_size, n_frame]

        return ccep_impluse


def ccep_ltvfilter(input_signal, impluse_filters, frame_size, window_type=None):
    
    # input_signal: [B, 1, n_samples], impluse_filters: [B, t_frame, fft_size]
    if window_type == None:
        filtered_signal = ltv_fir(input_signal, impluse_filters, frame_size=frame_size)  # [B, 1, t_frame * frame_size]
    else:
        filtered_signal = hann_ltv_fir(input_signal, impluse_filters, frame_size=frame_size)
    
    return filtered_signal


class Generator(torch.nn.Module):
    """ Model definition
    """
    def __init__(self, sampling_rate=22050, \
                       frame_length=128, fft_size=2048, n_mels=80, \
                       ccep_size=222, reverb_length=1000, \
                       envelop_max_quefrency=math.ceil(22050 / 1000 * 6), \
                       noise_max_quefrency=math.ceil(22050 / 1000 * 6), \
                       harmonic_num=200, use_weight_norm=True, \
                       device='cuda'):
        super(Generator, self).__init__()

        self.envelop_max_quefrency = envelop_max_quefrency
        self.noise_max_quefrency = noise_max_quefrency
        self.frame_length = frame_length
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.ccep_size = ccep_size
        self.reverb_length = reverb_length
        
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = 0.1
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = 0.003

        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        
        # nn filter modules: generate complex cepstrum impluse response
        self.source_filter = nn_filter_Estimator(in_channels=self.n_mels, ccep_size=self.ccep_size, fft_size=self.fft_size, max_quefrency=self.envelop_max_quefrency, \
                                                device=device)
        self.noise_filter = nn_filter_Estimator(in_channels=self.n_mels, ccep_size=self.ccep_size, fft_size=self.fft_size, max_quefrency=self.noise_max_quefrency, \
                                                device=device)
        
        # final trainable FIR filter
        self.reverb = TrainableFIRReverb(self.reverb_length, device=device)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, feat, pitch):
        # input: (B, 80, t_frame), (B, 1, t_frame)
        #print('feature: ', feat.shape, f0.shape)
        
        # [B, fft_size, t_frame]
        source_impulse = self.source_filter(feat)
        noise_impulse = self.noise_filter(feat)
        
        # [B, 1, n_samples]
        interpolate_pitch = repeat_interpolate(pitch, self.frame_length)
        source_train = generate_impulse_train(interpolate_pitch, n_harmonic=self.harmonic_num, sampling_rate=self.sampling_rate, source_amp=self.sine_amp)
        source_noise = generate_random_Noise(interpolate_pitch, noise_std=self.noise_std)
        
        # [B, 1, n_samples]
        filtered_source = ccep_ltvfilter(source_train, source_impulse.transpose(1, 2), frame_size=self.frame_length)
        filtered_noise = ccep_ltvfilter(source_noise, noise_impulse.transpose(1, 2), frame_size=self.frame_length)
        
        # [B, 1, n_samples]
        out = self.reverb(filtered_source + filtered_noise)
        
        return out
        
        
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
