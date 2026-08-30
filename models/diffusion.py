import math as m
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
SCHEDULE = "cosine"

DDIM_STEPS = 50
DDIM_ETA = 0.0
CLIP_X0 = 3.0

class NoiseSchedule(nn.Module):
    def __init__(self, num_timesteps=1000, schedule="cosine", beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.num_timesteps = num_timesteps

        if schedule == "cosine":
            beta = self.get_cosine_beta_schedule(num_timesteps)
        else:
            beta = torch.linspace(beta_start, beta_end, num_timesteps)
            
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0) # alpha bar
        alpha_bar_prev = F.pad(alpha_bar[:-1], pad=(1, 0), value=1.)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_min_alpha_bar = torch.sqrt(1 - alpha_bar)
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        self.register_buffer('sqrt_one_min_alpha_bar', sqrt_one_min_alpha_bar)
        self.register_buffer('posterior_variance', posterior_variance)

    def extract(self, buffer, t, shape):
        out = buffer.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def get_cosine_beta_schedule(self, num_timesteps, s=0.008, max_beta=0.999):
        """cosine beta schedule as proposed by Nichol and Dhariwal (2021)."""
        timesteps = torch.linspace(0, num_timesteps, num_timesteps + 1)

        # Compute f(t) using the cosine formula
        alphas_bar = (torch.cos(((timesteps / num_timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2)

        # Normalize by f(0)
        alphas_bar = alphas_bar / alphas_bar[0]

        # Calculate beta_t = 1 - (alpha_cumprod_t / alpha_cumprod_{t-1})
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])

        # Cap betas to prevent singularities near the end of the schedule
        return torch.clamp(betas, max=max_beta)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, schedule):
        super().__init__()
        self.model = model
        self.schedule = schedule

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        s = self.schedule
        x_t = (s.extract(s.sqrt_alpha_bar, t, x_0.shape) * x_0
               + s.extract(s.sqrt_one_min_alpha_bar, t, x_0.shape) * noise)
        return x_t

    def predict_x0_from_noise(self, x_t, t, noise, clip=CLIP_X0):
        s = self.schedule
        x_0 = ((x_t - s.extract(s.sqrt_one_min_alpha_bar, t, x_t.shape) * noise)
               / s.extract(s.sqrt_alpha_bar, t, x_t.shape))
        if clip is not None:
            x_0 = x_0.clamp(-clip, clip)
        return x_0

    def p_losses(self, x_0, labels, t=None, noise=None):
        if t is None:
            t = torch.randint(0, self.schedule.num_timesteps, (x_0.shape[0],),
                              device=x_0.device)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        return F.mse_loss(self.model(x_t, t, labels), noise)

    def forward(self, x_0, labels):
        return self.p_losses(x_0, labels)

class DDPMSampler():
    def __init__(self, diffusion):
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(self, labels, shape, device):
        d = self.diffusion
        s = d.schedule
        x = torch.randn(shape, device=device)

        for step in reversed(range(s.num_timesteps)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            noise = d.model(x, t, labels)
            x_0 = d.predict_x0_from_noise(x, t, noise)

            beta = s.extract(s.beta, t, x.shape)
            alpha = s.extract(s.alpha, t, x.shape)
            alpha_bar = s.extract(s.alpha_bar, t, x.shape)
            alpha_bar_prev = s.extract(s.alpha_bar_prev, t, x.shape)

            mean = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar) * x_0
                    + torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar) * x)

            # The last step lands on the clean estimate, so adding noise there
            # would leave grain in every sample
            if step > 0:
                variance = s.extract(s.posterior_variance, t, x.shape)
                x = mean + torch.sqrt(variance) * torch.randn_like(x)
            else:
                x = mean
        return x

class DDIMSampler():
    def __init__(self, diffusion):
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(self, labels, shape, device, num_steps=DDIM_STEPS, eta=DDIM_ETA):
        d = self.diffusion
        s = d.schedule
        step_ratio = max(s.num_timesteps // num_steps, 1)
        timesteps = list(range(0, s.num_timesteps, step_ratio))[::-1]
        x = torch.randn(shape, device=device)

        for i, step in enumerate(timesteps):
            prev_step = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            noise = d.model(x, t, labels)
            x_0 = d.predict_x0_from_noise(x, t, noise)

            alpha_bar = s.extract(s.alpha_bar, t, x.shape)
            # A negative index would wrap around to the last alpha bar, which is
            # near zero, so the final step is pinned to one by hand
            if prev_step >= 0:
                prev = torch.full((shape[0],), prev_step, dtype=torch.long,
                                  device=device)
                alpha_bar_prev = s.extract(s.alpha_bar, prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar)

            sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                     * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
            direction = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0)) * noise
            x = torch.sqrt(alpha_bar_prev) * x_0 + direction
            if eta > 0 and prev_step >= 0:
                x = x + sigma * torch.randn_like(x)
        return x

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-m.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def normalize_mel(mel, mean, std):
    return (mel - mean) / std

def denormalize_mel(mel, mean, std):
    return mel * std + mean
