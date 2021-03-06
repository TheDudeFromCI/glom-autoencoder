import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()

        self.pre = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, 1, kernel // 2),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(channels, channels, kernel, 1, kernel // 2)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.conv(x) + x
        return x


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pre = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.skip = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.conv(x) + self.skip(x)
        return x


class GLOM(nn.Module):
    def __init__(self, img_channels, layer_dim, n_layers):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(img_channels, layer_dim, 16, 16, 0),
                ResidualBlock(layer_dim, 1),
                ResidualBlock(layer_dim, 1),
            ))

        layer_up = []
        for _ in range(n_layers - 1):
            layer_up.append(nn.Sequential(
                nn.Conv2d(layer_dim, layer_dim, 1, 1, 0),
                ResidualBlock(layer_dim, 1),
                ResidualBlock(layer_dim, 1),
            ))

        layer_down = []
        for _ in range(n_layers - 1):
            layer_down.append(nn.Sequential(
                nn.Conv2d(layer_dim, layer_dim, 1, 1, 0),
                ResidualBlock(layer_dim, 1),
                ResidualBlock(layer_dim, 1),
            ))

        self.n_layers = n_layers
        self.to_layer = nn.ModuleList(layers)
        self.layer_up = nn.ModuleList(layer_up)
        self.layer_down = nn.ModuleList(layer_down)

    def forward(self, x):
        layers = [self.to_layer[i](x) for i in range(self.n_layers)]
        return layers

    def timestep(self, layers):
        layers_next = [0] * self.n_layers
        for i in range(self.n_layers):
            from_below = 0 if i == 0 else self.layer_down[i - 1](layers[i])
            from_above = 0 if i == self.n_layers - 1 else self.layer_up[i](layers[i])
            attention = self.attention(layers, i)
            layers_next[i] = from_below + from_above + attention

        return layers_next

    def attention(self, layers, layer_index):
        batch_size, channels, width, height = layers[0].shape
        L = 0.5

        x = 0
        for i in range(layer_index + 1):
            weight = L ** (layer_index - i)

            val = layers[i].view(batch_size, -1, width * height)
            att = torch.bmm(val.permute(0, 2, 1), val)

            x += weight * att

        x = x.softmax(dim=-1)

        val = layers[layer_index].view(batch_size, -1, width * height)
        out = torch.bmm(val, x.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)

        return out


class GlomReconstructor(nn.Module):
    def __init__(self, n_layers, layer_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(n_layers * layer_dim, 256, 3, 1, 1),
            ResidualBlock(256, 3),
            ResidualBlockUp(256, 128),

            ResidualBlock(128, 3),
            ResidualBlockUp(128, 64),

            ResidualBlock(64, 3),
            ResidualBlockUp(64, 32),

            ResidualBlock(32, 3),
            ResidualBlockUp(32, 32),

            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, layers):
        x = torch.cat(layers, dim=1)
        x = self.model(x)
        return x
