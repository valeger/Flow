import os
import argparse
from typing import Dict

import numpy as np
import torch
import torch.utils.data
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from .data.load import load_data
from .model import train_and_evaluate

def parse() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='Name of a model to train',
                        default=1)
    parser.add_argument('--lr', type=float, help='Learning rate',
                        default=5e-4)
    parser.add_argument('--name', type=str, help='Name of a model to save',
                        default='flow_model')
    parser.add_argument('--gpu', type=bool, help='Enable CUDA',
                        default='True')
    return parser.parse_args().__dict__

def show_plot(train_losses: np.ndarray, 
              test_losses: np.ndarray, 
              title: str) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    try:
        os.makedirs('images')
    except:
        pass

    fname = os.path.join('images', 'train_plot.png')
    plt.savefig(fname, format='png')

def show_samples(samples: torch.Tensor, title: str, nrow: int = 10) -> None:
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    try:
        os.makedirs('images')
    except:
        pass

    fname = os.path.join('images', f'{title}.png')
    plt.savefig(fname, format='png')

if __name__ == '__main__':
    params = parse()
    train_data, test_data = load_data()

    train_losses, test_losses, samples, interpolations = train_and_evaluate(train_data, test_data, params)
    samples = samples.astype('float')
    interpolations = interpolations.astype('float')

    show_plot(train_losses, test_losses, 'Train Plot')
    show_samples(samples * 255.0, 'Samples')
    show_samples(interpolations * 255.0, 'Interpolations', nrow=6)