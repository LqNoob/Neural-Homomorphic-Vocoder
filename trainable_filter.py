# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from conv_filter import fir


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    x = x.to(torch.float32)
    return max_value * torch.sigmoid(x)**torch.log(torch.tensor(exponent).to(x.device)) + threshold


class ExpDecayReverb(torch.nn.Module):
    """ FIR: trainable finite impulse response filter  """
    def __init__(self, reverb_length=1000, trainable=True, add_dry=False, scale_fn=exp_sigmoid, fft_size=2048):
        super(ExpDecayReverb, self).__init__()

        self._reverb_length = reverb_length
        self._scale_fn = scale_fn
        self.trainable = trainable
        self.fft_size = fft_size

        self._gain  = torch.nn.Parameter(torch.Tensor(1,).zero_())
        torch.nn.init.constant_(self._gain, 2.0)

        #self._decay = torch.nn.Parameter(torch.Tensor(1,).zero_())
        #torch.nn.init.constant_(self._decay, 4.0)
        self._decay = torch.nn.Parameter(torch.FloatTensor([0.995]), requires_grad=False)

    def _match_dimensions(self, audio, ir):
        """Tile the impulse response variable to match the batch size."""
        # Add batch dimension.
        if len(ir.shape) == 1:
          ir = torch.unsqueeze(ir, dim=0)

        # Match batch dimension.
        batch_size = int(audio.shape[0])
        return ir.repeat((batch_size, 1))

    def _get_ir(self, gain, decay):
        """Simple exponential decay of white noise."""
        gain = self._scale_fn(gain)
        #decay_exponent = 2.0 + torch.exp(decay)

        time = torch.linspace(0.0, 1.0, self._reverb_length).unsqueeze(dim=0).to(gain.device)
        noise = torch.Tensor(1, self._reverb_length).uniform_(-1.0, 1.0).to(gain.device)
        
        #ir = gain * torch.exp(-decay_exponent * time) * noise
        ir = gain * torch.exp(-decay * time) * noise
        return ir

    def _mask_dry_ir(self, ir):
        """Set first impulse response to zero to mask the dry signal."""
        # Make IR 2-D [batch, ir_size].
        if len(ir.shape) == 1:
          ir = ir.unsqueeze(dim=0)  # Add a batch dimension
        if len(ir.shape) == 3:
          ir = ir[:, :, 0]  # Remove unnessary channel dimension.
        
        # Mask the dry signal.
        dry_mask = torch.zeros([int(ir.shape[0]), 1], dtype=torch.float32).to(ir.device)
        return torch.cat([dry_mask, ir[:, 1:]], dim=1)

    def add_reverb(self, audio, gain=None, decay=None):
        if self.trainable:
            gain, decay = torch.unsqueeze(self._gain, dim=0), torch.unsqueeze(self._decay, dim=0)
        elif gain is None or decay is None:
            raise ValueError('Must provide "gain" and "decay" tensors if ExpDecayReverb trainable=False.')

        ir = self._get_ir(gain, decay)
        if self.trainable:
            ir = self._match_dimensions(audio, ir)

        #print(audio.dtype, ir.dtype)
        audio, ir = audio.to(torch.float32), ir.to(torch.float32)
        
        ir = self._mask_dry_ir(ir)
        # audio: [B, 1, n_samples], ir: [B, 1, filter_size]
        wet = fir(audio, ir, is_causal=self.trainable)

        return wet


