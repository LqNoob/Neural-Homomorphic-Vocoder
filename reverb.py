# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from conv_filter import fir


class TrainableFIRReverb(nn.Module):
    def __init__(self, reverb_length=1000, device="cuda"):
        super(TrainableFIRReverb, self).__init__()

        # default reverb length is set to 3sec.
        # thus this model can max out t60 to 3sec, which corresponds to rich chamber characters.
        self.reverb_length = reverb_length
        self.device = device

        # impulse response of reverb.
        self.fir = nn.Parameter(torch.rand(1, self.reverb_length, dtype=torch.float32).to(self.device) * 2 - 1, requires_grad=True)

        # Initialized drywet to around 26%.
        # but equal-loudness crossfade between identity impulse and fir reverb impulse is not implemented yet.
        self.drywet = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32).to(self.device), requires_grad=True)

        # Initialized decay to 5, to make t60 = 1sec.
        #self.decay = nn.Parameter(torch.tensor([3.0], dtype=torch.float32).to(self.device), requires_grad=True)
        self.decay_rate = nn.Parameter(torch.tensor([0.995], dtype=torch.float32).to(self.device), requires_grad=False)

    def forward(self, input_signal):
        """
        Compute FIR Reverb
        Output:
            output_signal : batch of reverberated signals
        """

        # Build decaying impulse response and send it to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        # Dry-wet mixing is done by mixing impulse response, rather than mixing at the final stage.

        """ TODO 
        Not numerically stable decay method?
        """
        #decay_envelope = torch.exp(-(torch.exp(self.decay) + 2) * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device))
        decay_envelope = torch.exp(-self.decay_rate * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device))
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(self.device)
        ir_identity[:, 0] = 1

        """ TODO
        Equal-loudness(intensity) crossfade between to ir.
        """
        final_fir = (torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity)
        final_fir = final_fir.unsqueeze(0).repeat((input_signal.shape[0], 1, 1))
        
        filter_signal = fir(input_signal, final_fir, is_causal=True)

        # TODO batched ??
        return filter_signal
