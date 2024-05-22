from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import time
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from features_extract import mel_spectrogram, MAX_WAV_VALUE, load_wav
from compute_pitch import calculate_frame_f0_praat
from models import Generator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(device=device).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = glob.glob(os.path.join(a.input_wavs_dir, '*.wav'))
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    total_rtf = 0.0
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            start_time = time.time()
            wav, sr = load_wav(filname)
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            
            x = get_mel(wav.unsqueeze(0)).squeeze().unsqueeze(0)
            f0, _ = calculate_frame_f0_praat(filname, x.shape[-1])
            f0 = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            
            y_g_hat = generator(x, f0)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.split(filname)[-1].split('.')[0] + '_generated_nhv.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)
            total_rtf += (time.time() - start_time) / (len(audio) / h.sampling_rate)

        total_rtf = total_rtf / len(filelist)
        print("\nFinished generation of utterances (RTF = %.3f)." % total_rtf)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

