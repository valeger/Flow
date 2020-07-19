from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

def preprocess(x: torch.Tensor, reverse: bool = False, dequantize: bool = True) -> Tuple[torch.Tensor]:
    if reverse: 
        x = 1.0 / (1 + torch.exp(-x))
        x -= 0.05
        x /= 0.9
        return x
    else:
        
        if dequantize:
            x += torch.distributions.Uniform(0.0, 1.0).sample(x.shape).cuda()
        x /= 4.0

        # logit operation
        x *= 0.9
        x += 0.05
        logit = torch.log(x) - torch.log(1.0 - x)
        log_det = F.softplus(logit) + F.softplus(-logit) \
            + torch.log(torch.tensor(0.9)) - torch.log(torch.tensor(4.0))
        return logit, torch.sum(log_det, dim=(1, 2, 3))

def interpolate(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    model.eval()
    good = [5, 13, 16, 19, 22]
    indices = []
    for index in good:
        indices.append(index*2)
        indices.append(index*2+1)
    with torch.no_grad():
        actual_images = next(iter(test_loader))[indices].to('cpu')
        assert actual_images.shape[0] % 2 == 0
        logit_actual_images, _ = preprocess(actual_images.float(), dequantize=False)
        latent_images, _ = model.f(logit_actual_images)
        latents = []
        for i in range(0, actual_images.shape[0], 2):
            a = latent_images[i:i+1]
            b = latent_images[i + 1:i+2]
            diff = (b - a)/5.0
            latents.append(a)
            for j in range(1, 5):
                latents.append(a + diff * float(j))
            latents.append(b)
        latents = torch.cat(latents, dim=0)
        logit_results = model.g(latents)
        results = preprocess(logit_results, reverse=True)
        return results.cpu().numpy()

def train(model: nn.Module, 
          train_loader: DataLoader, 
          optimizer: Optimizer,
          device: torch.device) -> torch.Tensor:
    losses = []
    for images in train_loader:
        images = images.to(device)
        logit_x, log_det = preprocess(images.float())
        log_prob = model.log_prob(logit_x)
        log_prob += log_det

        batch_loss = -torch.mean(log_prob) / (3.0 * 32.0 * 32.0)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_loss = float(batch_loss.data)
        losses.append(batch_loss)
    return losses

def train_epochs(model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 params: Dict) -> Tuple[List]:
    epochs, lr, fname = params['epochs'], params['lr'], params['name']
    device = params['device']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        losses_per_epoch = train(model, train_loader, optimizer, device)
        test_loss = eval_loss(model, test_loader, device)
        train_losses.extend(losses_per_epoch)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch} Train loss: {np.mean(losses_per_epoch):.4f}\n')
        print(f'Epoch {epoch} Test loss: {test_loss:.4f}\n')

    model.save_model(fname)
    return train_losses, test_losses

def eval_loss(model: nn.Module, test_loader: DataLoader, device: torch.device) -> np.float64:
    model.eval()
    losses = []
    for images in test_loader:
        with torch.no_grad():
            images = images.to(device)
            logit_x, log_det = preprocess(images.float())
            log_prob = model.log_prob(logit_x)
            log_prob += log_det

            loss = -torch.mean(log_prob) / (3.0 * 32.0 * 32.0)
            loss = loss.item()
            losses.append(loss)
    return np.mean(losses)