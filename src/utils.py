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

