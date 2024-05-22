import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from env import AttrDict, build_env
from features_extract import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator
from loss import STFTLoss
from discriminator import Discriminator
from utils_common import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, NoamLR


torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(rank, a, h, pretrained=False):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(device=device).to(device)
    discriminator = Discriminator().to(device)
    stft_loss = STFTLoss().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    # --> resume or adaptation from checkpoint
    if pretrained:
        if os.path.isdir(a.adaptation_path):
            cp_g = scan_checkpoint(a.adaptation_path, 'g_')
            cp_do = scan_checkpoint(a.adaptation_path, 'do_')

        steps = 0
        last_epoch = -1
        if cp_g is None or cp_do is None:
            state_dict_do = None
            last_epoch = -1
        else:
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            generator.load_state_dict(state_dict_g['generator'])
            discriminator.load_state_dict(state_dict_do['discriminator'])
    else:
        if os.path.isdir(a.checkpoint_path):
            cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
            cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

        steps = 0
        if cp_g is None or cp_do is None:
            state_dict_do = None
            last_epoch = -1
        else:
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            generator.load_state_dict(state_dict_g['generator'])
            discriminator.load_state_dict(state_dict_do['discriminator'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    # TODO
    optim_g = torch.optim.AdamW(generator.parameters(), 0.0006, betas=[0.9, 0.999], eps=1e-08)
    optim_d = torch.optim.AdamW(discriminator.parameters(), 0.0002, betas=[0.9, 0.999], eps=1e-08)

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    # TODO
    scheduler_g = NoamLR(optim_g, last_epoch, warmup_steps=4000, init_lr=0.0006, min_lr=0.00001)
    scheduler_d = NoamLR(optim_d, last_epoch, warmup_steps=20000, init_lr=0.0002, min_lr=0.00001)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=False,
                              drop_last=True)

    if rank == 0:
        if validation_filelist != None:
            validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                                  h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                                  fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                                  base_mels_path=a.input_mels_dir)
            validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                           sampler=None,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, pitch, _ = batch
            
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            pitch = torch.autograd.Variable(pitch.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x, pitch.unsqueeze(1))

            # Discriminator
            optim_d.zero_grad()
            disc_generated_outputs = discriminator(y_g_hat.detach(), x)
            disc_real_outputs = discriminator(y, x)
            
            loss_disc_all = torch.mean(torch.maximum(1 - disc_real_outputs, torch.zeros(1, device=device))) + \
                            torch.mean(torch.maximum(1 + disc_generated_outputs, torch.zeros(1, device=device)))
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            disc_generated_outputs = discriminator(y_g_hat, x)

            loss_G = - torch.mean(disc_generated_outputs)
            loss_R = stft_loss(y_g_hat, y)
            #loss_gen_all = 4.0*loss_G + loss_R
            loss_gen_all = loss_G + loss_R

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = stft_loss(y_g_hat, y).item()

                    print('Epoch/Steps : {:d}/{:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(epoch, steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/loss_disc_all", loss_disc_all, steps)

                    sw.add_scalar("learnrate/learning_rate_d", scheduler_d.get_last_lr()[0], steps)
                    sw.add_scalar("learnrate/learning_rate_g", scheduler_g.get_last_lr()[0], steps)


                # Validation
                #if steps % a.validation_interval == 0 and validation_filelist != None:  # and steps != 0:
                #    generator.eval()
                #    torch.cuda.empty_cache()
                #    val_err_tot = 0
                #    with torch.no_grad():
                #        for j, batch in enumerate(validation_loader):
                #            x, y, _, y_mel = batch
                #            y_g_hat = generator(x.to(device))
                #            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                #            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                #                                          h.hop_size, h.win_size,
                #                                          h.fmin, h.fmax_for_loss)
                #            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                #            if j <= 4:
                #                if steps == 0:
                #                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                #                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                #                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                #                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                #                                             h.sampling_rate, h.hop_size, h.win_size,
                #                                             h.fmin, h.fmax)
                #                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                #                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                #        val_err = val_err_tot / (j+1)
                #        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                #    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    #parser.add_argument('--input_wavs_dir', default='/data/sdd/beibeihu/dataset/allspeakers_20201223150014')
    parser.add_argument('--input_wavs_dir', default='/data/sdh/liqiang/projects/code/vocoder/neural-homomorphic-vocoder/vocoder_netural_homomorphic')

    parser.add_argument('--input_mels_dir', default='ft_dataset')

    parser.add_argument('--input_training_file', default='test_text/annotations_biaobei_20240426.txt')

    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')

    parser.add_argument('--checkpoint_path', default='checkpoints/cp_nhv_BZNSYP_gt_default')
    parser.add_argument('--adaptation_path', default='pretrained_model')

    parser.add_argument('--config', default='config_fmax_v1.json')
    parser.add_argument('--training_epochs', default=6200, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
