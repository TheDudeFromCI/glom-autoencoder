from models import GLOM, GlomReconstructor
from util import generate_layer_to_rgb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

IMAGE_DIM = 3
IMAGE_SIZE = 128
LAYER_DIM = 128
N_LAYERS = 5

print(
    f"Training model for {IMAGE_DIM}x{IMAGE_SIZE}x{IMAGE_SIZE} images. ({N_LAYERS} layers, {LAYER_DIM} channels each.)")

to_rgb = generate_layer_to_rgb(LAYER_DIM)

glom = GLOM(IMAGE_DIM, LAYER_DIM, N_LAYERS)
reconstructor = GlomReconstructor(N_LAYERS, LAYER_DIM)

pipeline = nn.Sequential(glom, reconstructor)
optim = Adam(pipeline.parameters(), lr=1e-4)

glom.cuda()
reconstructor.cuda()
to_rgb.cuda()

print('Loading dataset at ./data...')
ds = ImageDataset('./data')
train_loader = DataLoader(ds, batch_size=32, num_workers=3, shuffle=True, pin_memory=True, drop_last=True)

print('Starting training...')
for epoch in range(100):
    batch_index = 0
    for batch in tqdm(train_loader, smoothing=0):
        batch = batch.cuda()
        x = glom(batch)
        for i in range(10):
            x = glom.timestep(x)

        y = reconstructor(x)
        loss = F.mse_loss(y, batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_index += 1
        print(f"Ep: {epoch}, Bat: {batch_index}, Loss: {loss.item():.4f}")

    l = torch.stack([x[i][0] for i in range(N_LAYERS)])
    l = to_rgb(l)

    batch = torch.unsqueeze(batch[0], 0)
    y = torch.unsqueeze(y[0], 0)
    batch = torch.cat([batch, y, l])
    save_image(batch, 'example.png', nrow=1)
