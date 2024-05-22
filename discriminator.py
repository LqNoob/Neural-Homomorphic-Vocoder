#!/usr/bin/env python
# _*_coding:utf-8_*_

""" reference from wavenet
    https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/models/parallel_wavegan.py#L110
"""

import math
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from layers.residual_block import Conv1d, Conv1d1x1, ResidualBlock
from layers.upsample import ConvInUpsampleNetwork
from layers import upsample

LRELU_SLOPE = 0.1

# We applied Weight Normalization to all network weights
class Discriminator(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=14,
                 max_dilation=64,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 aux_context_window=0,
                 dropout=0.0,
                 bias=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 upsample_conditional_features=True,
                 upsample_net="ConvInUpsampleNetwork",
                 upsample_params={"upsample_scales": [4, 4, 4, 2]},
                 ):
        """ Discriminator module. """
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.max_dilation = max_dilation
        self.kernel_size = kernel_size

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define conv + upsampling network
        if upsample_conditional_features and upsample_net=="ConvInUpsampleNetwork":
            upsample_params.update({
                "use_causal_conv": use_causal_conv,
                "aux_channels": aux_channels,
                "aux_context_window": aux_context_window,
            })
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            self.upsample_factor = np.prod(upsample_params["upsample_scales"])
        else:
            self.upsample_net = None
            self.upsample_factor = 1

        # define residual blocks
        loop_factor = math.floor(math.log2(max_dilation)) + 1
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer%loop_factor)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(skip_channels, skip_channels, bias=True),
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(skip_channels, out_channels, bias=True),
        ])

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        #x, c = x.permute(0, 2, 1), c.permute(0, 2, 1)
        # perform upsampling
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)

        # encode to hidden representation
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        # (B, T, 1)
        return x.permute(0, 2, 1)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

