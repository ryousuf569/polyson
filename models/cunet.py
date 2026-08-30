# code reference https://github.com/gabolsgabs/cunet
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion import timestep_embedding


class FiLM(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)
        nn.init.normal_(self.gamma.weight, std=0.02)
        nn.init.ones_(self.gamma.bias)
        nn.init.normal_(self.beta.weight, std=0.02)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.film1 = FiLM(out_channels, cond_dim)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.film2 = FiLM(out_channels, cond_dim)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, cond):
        h = self.conv1(F.leaky_relu(self.norm1(x), 0.2))
        h = F.leaky_relu(self.film1(h, cond), 0.2)
        h = self.conv2(self.norm2(h))
        h = self.film2(h, cond)
        return F.leaky_relu(h + self.skip(x), 0.2)


class SelfAttention(nn.Module):
    def __init__(self, channels, heads=4, groups=8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = F.scaled_dot_product_attention(q.transpose(-2, -1),
                                             k.transpose(-2, -1),
                                             v.transpose(-2, -1))
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, attention=False):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, cond_dim)
        self.attn = SelfAttention(out_channels) if attention else None
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.res(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, cond_dim,
                 attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2,
                                     padding=1)
        self.res = ResBlock(in_channels + skip_channels, out_channels, cond_dim)
        self.attn = SelfAttention(out_channels) if attention else None

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.res(torch.cat([x, skip], dim=1), cond)
        if self.attn is not None:
            x = self.attn(x)
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, n_classes, in_channels=1, base_channels=64,
                 channel_mults=(1, 2, 4, 4), cond_dim=128, attention_levels=(2, 3)):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, cond_dim)
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cond_dim, cond_dim),
        )
        self.control = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cond_dim, cond_dim),
            nn.LeakyReLU(0.2),
        )
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.downs = nn.ModuleList()
        current = base_channels
        skips = []
        for i, out_channels in enumerate(channels):
            self.downs.append(DownBlock(current, out_channels, cond_dim,
                                        attention=i in attention_levels))
            skips.append(out_channels)
            current = out_channels

        self.mid1 = ResBlock(current, current, cond_dim)
        self.mid_attn = SelfAttention(current)
        self.mid2 = ResBlock(current, current, cond_dim)

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channels))):
            out_channels = channels[i - 1] if i > 0 else base_channels
            self.ups.append(UpBlock(current, skips[i], out_channels, cond_dim,
                                    attention=i in attention_levels))
            current = out_channels

        self.out_norm = nn.GroupNorm(min(8, current), current)
        self.out_conv = nn.Conv2d(current, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, labels):
        cond = self.embedding(labels) + self.time_mlp(timestep_embedding(t, self.cond_dim))
        cond = self.control(cond)
        h = self.stem(x)
        skips = []
        for block in self.downs:
            h, skip = block(h, cond)
            skips.append(skip)
        h = self.mid2(self.mid_attn(self.mid1(h, cond)), cond)
        for block, skip in zip(self.ups, reversed(skips)):
            h = block(h, skip, cond)
        return self.out_conv(F.leaky_relu(self.out_norm(h), 0.2))
