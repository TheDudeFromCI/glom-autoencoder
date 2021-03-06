import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam


def generate_layer_to_rgb(layer_dim):
    print('Generating layer-to-rgb convolution...')
    start_time = time.time()

    to_rgb = nn.Sequential(
        nn.Conv2d(layer_dim, 3, 1, 1, 0),
        nn.Tanh()
    )

    to_layer = nn.Conv2d(3, layer_dim, 1, 1, 0)
    ae = nn.Sequential(to_rgb, to_layer)
    optim = Adam(ae.parameters(), lr=1e-3)

    for _ in range(256):
        noise = torch.randn((128, layer_dim, 1, 1))
        loss = F.mse_loss(ae(noise), noise)

        optim.zero_grad()
        loss.backward()
        optim.step()

    elapsed_time = (time.time() - start_time) * 1000
    print(f"Finished in {elapsed_time:.1f} ms.")

    return nn.Sequential(
        to_rgb,
        nn.UpsamplingNearest2d(scale_factor=16)
    )
