# -*- coding: utf-8 -*-

import numpy as np
import parselmouth
import pyworld
from pyworld import dio, stonemask
from pyreaper import reaper


def log_f0(f0):
    #对非零f0取对数
    nonzero_idxs = np.where(f0 != 0)[0]
    f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
    return f0

def cmvn_f0(f0, mean, std):
    new_f0 = f0.copy()
    #均值方差归一化
    zero_idxs = np.where(new_f0 == 0.0)[0]
    new_f0 -= mean
    new_f0 /= std
    new_f0[zero_idxs] = 0.0
    return new_f0


def calculate_frame_f0_dio(audio_norm, fs, f0_min, f0_max, hop_length, mel_len):
    #计算每帧的pitch
    f0, timeaxis = pyworld.dio(
            audio_norm,
            fs,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            frame_period=1000 * hop_length / fs,
    )
    f0 = pyworld.stonemask(audio_norm, f0, timeaxis, fs)
    f0 = f0.reshape(-1)
    assert np.abs(mel_len - f0.shape[0]) <= 1.0
    f0_error = False
    if (f0 == 0).all():
        f0_error = True
    return f0, f0_error

def calculate_frame_f0_praat(audio_path, mel_len):
    #计算每帧的pitch
    snd = parselmouth.Sound(audio_path)
    f0 = snd.to_pitch(time_step=snd.duration / (mel_len + 6)).selected_array['frequency']
    assert np.abs(mel_len - f0.shape[0]) <= 1.0
    f0_error = False
    if (f0 == 0).all():
        f0_error = True
    return f0, f0_error

def repaer_stonemask(double_x, frame_length, sampling_rate):
    """
    double_x: numpy double array [Samples]
    frame_length: int, # of samples in a single frame
    sampling_rate: int
    returns: numpy double array [Frames]
    """
    n_frames = len(double_x) // frame_length
    n_samples = n_frames * frame_length
    double_x = double_x[:n_samples]
    int_x = np.clip(double_x * (65536 // 2), -32768, 32767).astype(np.int16)
    times = np.linspace(0, n_frames - 1, n_frames) * frame_length / sampling_rate + frame_length / 2 / sampling_rate
    _, _, f0_times, f0, _ = reaper(int_x, sampling_rate, minf0=40.0, maxf0=600.0)
    coarse_f0 = np.interp(times, f0_times, f0)
    fine_f0 = pyworld.stonemask(double_x, coarse_f0, times, sampling_rate)

    return fine_f0


if __name__ == '__main__':

    from features_extract import mel_spectrogram
    import librosa
    import torch
    
    def test_f0_version():
    
        n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax = 1024, 80, 22050, 256, 1024, 0, 8000  # mel_len + 3
        wav_path = 'BZNSYP_22050/wav_22050/009991.wav'
        
        audio_data, sample_rate = librosa.load(wav_path, sr=None)
        assert sample_rate == sampling_rate, sample_rate

        melspec = mel_spectrogram(torch.from_numpy(audio_data).unsqueeze(0), n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        print(melspec.shape, audio_data.shape, melspec.dtype, melspec)
        
        audio_data, sample_rate = librosa.load(wav_path, sr=None, dtype=np.float64)
        f0_version1 = repaer_stonemask(audio_data, hop_size, sample_rate)
        print(f0_version1.shape, f0_version1.dtype, f0_version1.shape[-1] == melspec.shape[-1])
        
        #f0_version2 = calculate_frame_f0_dio(audio_data, sample_rate, fmin, fmax, hop_size, melspec.shape[-1])
        #print(f0_version2.shape)
        
        f0_version3 = calculate_frame_f0_praat(wav_path, melspec.shape[-1])
        print(f0_version3[0].shape, f0_version3[0].dtype, f0_version3[-1], f0_version3[0].shape[-1] == melspec.shape[-1])
        print(f0_version3, f0_version3[0].astype(np.float32))
    
    test_f0_version()
    exit(0)
    
    import glob, os
    import torch
    from librosa.util import normalize

    root_dir = 'BZNSYP_22050/feat'
    wav_dir = 'BZNSYP_22050/wav_22050'
    
    pitch_dir = os.path.join(root_dir, 'f0')
    melspec_dir = os.path.join(root_dir, 'melspec')
    
    if not os.path.exists(pitch_dir) or not os.path.exists(melspec_dir):
        os.makedirs(pitch_dir, exist_ok=True)
        os.makedirs(melspec_dir, exist_ok=True)
    
    n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax = 2048, 80, 22050, 128, 512, 40, 7600
    for wav_path in glob.glob(os.path.join(wav_dir, '*.wav')):
        print(wav_path)
        
        audio_data, sample_rate = librosa.load(wav_path, sr=None)
        assert sample_rate == sampling_rate, sample_rate
        audio_data = normalize(audio_data) * 0.95
        
        # [1, 80, t_frame]
        melspec = mel_spectrogram(torch.from_numpy(audio_data).unsqueeze(0), n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        #print(melspec.shape)
        
        # [t_frame]
        pitch, pitch_error = calculate_frame_f0_praat(wav_path, melspec.shape[-1])
        if pitch_error:
            print('<------------->', wav_path)
            continue
        pitch = pitch.astype(np.float32)
        
        wav_name = os.path.basename(wav_path)[:-4]
        pitch_path = os.path.join(pitch_dir, wav_name + '.npy')
        np.save(pitch_path, pitch)
        melspec_path = os.path.join(melspec_dir, wav_name + '.npy')
        np.save(melspec_path, melspec)
        
        #break
    
    
