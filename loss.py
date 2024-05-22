# -*- coding: utf-8 -*-

import torch


class STFTLoss(torch.nn.Module):
    def __init__(self, \
                 fft_lengths=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 3072, 4096, 6144, 8192], \
                 window_lengths=[128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 3072, 4096], \
                 hop_lengths=[32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024], \
                 loss_scale_type="log_linear"):
        """
        STFT Loss
        fft_lengths: list of int    = window_lengths*2
        window_lengths: list of int = [128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 3072, 4096]
        hop_lengths: list of int    = window_lengths*0.25
        loss_scale_type: str defining the scale of loss
        """
        super(STFTLoss, self).__init__()
        self.window_lengths = window_lengths
        self.fft_lengths = fft_lengths
        self.hop_lengths = hop_lengths
        self.loss_scale_type = loss_scale_type

    def forward(self, x, y):
        """
        x: FloatTensor [Batch, 1, T]
        y: FloatTensor [Batch, 1, T]
        returns: FloatTensor [] as total loss
        """
        x, y = x.squeeze(1), y.squeeze(1)
        loss = 0.0
        batch_size = x.size(0)
        z = torch.cat([x, y], dim=0) # [2 x Batch, T]
        for fft_length, window_length, hop_length in zip(self.fft_lengths, self.window_lengths, self.hop_lengths):
            window = torch.hann_window(window_length).to(x.device)
            
            Z = torch.stft(z, fft_length, hop_length, window_length, window, return_complex=False) # [2 x Batch, Frame, 2]
            SquareZ = Z.pow(2).sum(dim=-1) + 1e-10 # [2 x Batch, Frame]
            SquareX, SquareY = SquareZ.split(batch_size, dim=0)
            MagZ = SquareZ.sqrt()
            MagX, MagY = MagZ.split(batch_size, dim=0)
            if self.loss_scale_type == "log_linear":
                loss += (MagX - MagY).abs().mean() + 0.5 * (SquareX.log() - SquareY.log()).abs().mean()
            elif self.loss_scale_type == "linear":
                loss += (MagX - MagY).abs().mean()
            else:
                raise RuntimeError(f"Unrecognized loss scale type {self.loss_scale_type}")

        return loss


def adversarial_loss(disc_real_outputs, disc_generated_outputs):
    # hinge loss
    dis_loss = torch.mean(torch.maximum(1 - disc_real_outputs, 0)) + torch.mean(torch.maximum(1 + disc_generated_outputs, 0))
    gen_loss = - torch.mean(disc_generated_outputs)

    return dis_loss, gen_loss

