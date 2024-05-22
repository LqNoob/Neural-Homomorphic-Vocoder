import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler
from typing import Union


class NoamLR(_LRScheduler):
    """The LR scheduler proposed by Noam
    """
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1, \
                 warmup_steps: Union[int, float] = 20000, init_lr: float = 0.0002, \
                 min_lr: float = 0.00001):

        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.init_lr = init_lr

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def get_lr(self, power=0.35):
        step_num = self.last_epoch + 1
        return [np.maximum(self.init_lr * self.warmup_steps ** power * \
                          np.minimum(step_num * self.warmup_steps ** (-1 - power), step_num ** -power), \
                          self.min_lr)]


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

