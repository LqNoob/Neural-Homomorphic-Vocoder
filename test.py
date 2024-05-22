# -*- coding: utf-8 -*-

import librosa
import torch
from conv_filter import ltv_fir, hann_ltv_fir
from features_extract import mel_spectrogram

torch.manual_seed(3407) 


def test_ltvfilter():

    input_signal = torch.randn(2, 1, 32*128)
    input_filters = torch.randn(2, 32, 1024)
    
    output_result = ltv_fir(input_signal, input_filters, frame_size=128, method='fft')  # n_fft = 2048
    print(output_result.shape, output_result)
    
    output_result = hann_ltv_fir(input_signal, input_filters, frame_size=128, method='fft')
    print(output_result.shape, output_result)


def test_nframe():

    n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax = 1024, 80, 22050, 128, 512, 40, 7600
    
    wav_path = 'BZNSYP_22050/wav_22050/009997.wav'
    data, sample_rate = librosa.load(wav_path)
    #print(data.shape)
    
    torch_data = torch.from_numpy(data).unsqueeze(0)[:, :32*hop_size]
    melspec = mel_spectrogram(torch_data, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
    print(melspec.shape)
    
    frame_signal = torch.nn.functional.unfold(torch_data.unsqueeze(1).unsqueeze(-1), kernel_size=(hop_size, 1), stride=(hop_size, 1))
    print(frame_signal.shape)


