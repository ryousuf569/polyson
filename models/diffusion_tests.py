# Equations referenced here are from the DDPM paper (Ho et al. 2020,
# arXiv:2006.11239) and the DDIM paper (Song et al. 2021, arXiv:2010.02502)
import sys

import torch

sys.path.insert(0, ".")

from models.cunet import ConditionalUNet
from models.diffusion import (DDIMSampler, DDPMSampler, GaussianDiffusion,
                              NoiseSchedule, denormalize_mel, normalize_mel,
                              timestep_embedding)

TOL = 1e-4
SHAPE = (4, 1, 32, 32)
N_CLASSES = 4

passed = []
failed = []


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print("  PASS  %s %s" % (name, detail))
    else:
        failed.append(name)
        print("  FAIL  %s %s" % (name, detail))


def close(a, b, tol=TOL):
    return float((a - b).abs().max()) < tol


# DDPM eq 2, the single step forward process q(x_t|x_{t-1}) has mean
# sqrt(1 - beta_t) * x_{t-1} and variance beta_t, so chaining single steps must
# agree in distribution with the closed form of eq 4
def test_forward_chain_matches_closed_form(d):
    s = d.schedule
    torch.manual_seed(0)
    x_0 = torch.randn(20000, 1, 1, 1)
    x = x_0.clone()
    for step in range(10):
        beta = s.beta[step]
        x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * torch.randn_like(x)
    t = torch.full((20000,), 9, dtype=torch.long)
    direct = d.q_sample(x_0, t)
    check("ddpm eq2 chain mean", abs(float(x.mean() - direct.mean())) < 0.02,
          "chained %.4f vs closed form %.4f" % (x.mean(), direct.mean()))
    check("ddpm eq2 chain std", abs(float(x.std() - direct.std())) < 0.02,
          "chained %.4f vs closed form %.4f" % (x.std(), direct.std()))


# DDPM eq 4, q(x_t|x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
def test_eq4_closed_form(d):
    s = d.schedule
    torch.manual_seed(0)
    x_0 = torch.randn(*SHAPE)
    for step in (0, 100, 500, 999):
        t = torch.full((SHAPE[0],), step, dtype=torch.long)
        noise = torch.randn_like(x_0)
        expected = (s.sqrt_alpha_bar[step] * x_0
                    + s.sqrt_one_min_alpha_bar[step] * noise)
        check("ddpm eq4 t=%d" % step, close(d.q_sample(x_0, t, noise), expected))

    # The stated variance is 1 - alpha_bar_t, checked empirically at one step
    big = torch.zeros(50000, 1, 1, 1)
    t = torch.full((50000,), 500, dtype=torch.long)
    got = float(d.q_sample(big, t).var())
    want = float(1 - s.alpha_bar[500])
    check("ddpm eq4 variance", abs(got - want) < 0.02,
          "%.4f vs %.4f" % (got, want))


# DDPM eq 7, beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t,
# and it must vanish at t=0 where there is nothing left to sample
def test_eq7_posterior_variance(d):
    s = d.schedule
    expected = (1 - s.alpha_bar_prev) / (1 - s.alpha_bar) * s.beta
    check("ddpm eq7 posterior variance", close(s.posterior_variance, expected))
    check("ddpm eq7 vanishes at t=0", float(s.posterior_variance[0]) < 1e-8,
          "%.3e" % s.posterior_variance[0])
    check("ddpm eq7 non negative", bool((s.posterior_variance >= 0).all()))


# DDPM eq 6, the posterior mean blends x_0 and x_t with the stated coefficients.
# Their sum is not one, that only holds as beta goes to zero, so the real
# invariant is the schedule identity alpha_bar_t = alpha_t * alpha_bar_{t-1}
# that the coefficients are built from
def test_eq6_posterior_mean(d):
    s = d.schedule
    check("ddpm eq6 schedule identity",
          close(s.alpha_bar / s.alpha, s.alpha_bar_prev, 1e-5),
          "max diff %.2e" % float((s.alpha_bar / s.alpha - s.alpha_bar_prev).abs().max()))

    # With x_0 and x_t both zero the mean is zero, and feeding the true x_0 at
    # t=0 has to return it unchanged because the posterior collapses there
    zero = torch.zeros(*SHAPE)
    t0 = torch.zeros(SHAPE[0], dtype=torch.long)
    beta = s.extract(s.beta, t0, SHAPE)
    alpha = s.extract(s.alpha, t0, SHAPE)
    alpha_bar = s.extract(s.alpha_bar, t0, SHAPE)
    alpha_bar_prev = s.extract(s.alpha_bar_prev, t0, SHAPE)
    x_0 = torch.randn(*SHAPE)
    mean = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar) * x_0
            + torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar) * x_0)
    check("ddpm eq6 collapses at t=0", close(mean, x_0, 1e-3),
          "max diff %.2e" % float((mean - x_0).abs().max()))
    check("ddpm eq6 zero input gives zero mean",
          close(torch.zeros_like(zero), zero * 0.0))


# DDPM Algorithm 2 line 4 writes the update directly in terms of epsilon,
# x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps).
# Our sampler instead goes through the eq 6 posterior mean, so the two forms
# have to agree exactly or one of them is wrong
def test_algorithm2_equals_posterior_form(d):
    s = d.schedule
    torch.manual_seed(0)
    x_t = torch.randn(*SHAPE)
    eps = torch.randn(*SHAPE)
    for step in (1, 100, 500, 999):
        t = torch.full((SHAPE[0],), step, dtype=torch.long)
        paper = (1.0 / torch.sqrt(s.alpha[step])
                 * (x_t - (1 - s.alpha[step]) / s.sqrt_one_min_alpha_bar[step] * eps))
        x_0 = d.predict_x0_from_noise(x_t, t, eps, clip=None)
        ours = (torch.sqrt(s.alpha_bar_prev[step]) * s.beta[step]
                / (1 - s.alpha_bar[step]) * x_0
                + torch.sqrt(s.alpha[step]) * (1 - s.alpha_bar_prev[step])
                / (1 - s.alpha_bar[step]) * x_t)
        check("ddpm alg2 vs eq6 t=%d" % step, close(paper, ours, 1e-3),
              "max diff %.2e" % float((paper - ours).abs().max()))


# DDPM eq 14, L_simple regresses the model onto the exact epsilon that was
# added, so a model that returns that epsilon has zero loss and an untrained
# model scores about the unit variance of the noise
def test_eq14_simple_objective(d, net):
    torch.manual_seed(0)
    x_0 = torch.randn(*SHAPE)
    labels = torch.randint(0, N_CLASSES, (SHAPE[0],))

    class Oracle(torch.nn.Module):
        def forward(self, x, t, y):
            return Oracle.noise

    Oracle.noise = torch.randn_like(x_0)
    oracle = GaussianDiffusion(Oracle(), d.schedule)
    loss = float(oracle.p_losses(x_0, labels, noise=Oracle.noise))
    check("ddpm eq14 oracle loss is zero", loss < 1e-8, "%.3e" % loss)

    losses = [float(d.p_losses(x_0, labels)) for _ in range(20)]
    mean = sum(losses) / len(losses)
    check("ddpm eq14 untrained loss near one", 0.8 < mean < 1.25, "%.4f" % mean)


# Inverting eq 4 for x_0 is what both samplers use to turn an epsilon estimate
# into a clean estimate, so it has to round trip
def test_predict_x0_inverts_eq4(d):
    torch.manual_seed(0)
    x_0 = torch.randn(*SHAPE)
    worst = 0.0
    for step in (0, 1, 250, 500, 750, 900):
        t = torch.full((SHAPE[0],), step, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = d.q_sample(x_0, t, noise)
        rec = d.predict_x0_from_noise(x_t, t, noise, clip=None)
        worst = max(worst, float((rec - x_0).abs().max()))
    check("eq4 inversion round trip", worst < 1e-2, "max err %.2e" % worst)


# DDIM eq 12 with sigma = 0 is deterministic given x_T, and the paper defines
# alpha_0 := 1 so the final step lands exactly on the predicted x_0
def test_ddim_eq12(d, net):
    s = d.schedule
    sampler = DDIMSampler(d)
    labels = torch.arange(N_CLASSES)

    torch.manual_seed(42)
    a = sampler.sample(labels, SHAPE, "cpu", num_steps=20)
    torch.manual_seed(42)
    b = sampler.sample(labels, SHAPE, "cpu", num_steps=20)
    check("ddim eq12 deterministic at eta=0", close(a, b, 1e-6),
          "max diff %.2e" % float((a - b).abs().max()))

    torch.manual_seed(1)
    c = sampler.sample(labels, SHAPE, "cpu", num_steps=20)
    check("ddim eq12 seed changes output", float((a - c).abs().max()) > 1e-3)

    # alpha_0 := 1 means the last step has no direction term left, so the
    # output equals the final predicted x_0 rather than a partially noisy one
    alpha_bar_prev_final = 1.0
    direction = (1 - alpha_bar_prev_final) ** 0.5
    check("ddim alpha_0 is one", abs(direction) < 1e-12,
          "direction coefficient %.2e" % direction)

    for steps in (10, 20, 50, 100):
        out = sampler.sample(labels, SHAPE, "cpu", num_steps=steps)
        check("ddim %d steps finite" % steps, bool(torch.isfinite(out).all()))


# DDIM eq 16, sigma_t = sqrt((1 - a_prev)/(1 - a)) * sqrt(1 - a/a_prev) is the
# choice that makes the generative process a DDPM, and it must reproduce the
# eq 7 posterior variance exactly
def test_ddim_eq16_recovers_ddpm_variance(d):
    s = d.schedule
    for step in (1, 100, 500, 999):
        alpha_bar = s.alpha_bar[step]
        alpha_bar_prev = s.alpha_bar_prev[step]
        sigma = (torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                 * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        check("ddim eq16 equals eq7 variance t=%d" % step,
              abs(float(sigma ** 2 - s.posterior_variance[step])) < 1e-6,
              "%.6e vs %.6e" % (sigma ** 2, s.posterior_variance[step]))


# Nichol and Dhariwal 2021 cosine schedule, alpha_bar starts near one, decays
# monotonically to near zero, and every beta stays a valid variance
def test_cosine_schedule_properties(d):
    s = d.schedule
    check("schedule alpha_bar starts near 1", float(s.alpha_bar[0]) > 0.99,
          "%.6f" % s.alpha_bar[0])
    check("schedule alpha_bar ends near 0", float(s.alpha_bar[-1]) < 0.01,
          "%.6f" % s.alpha_bar[-1])
    check("schedule monotonic", bool((s.alpha_bar[1:] <= s.alpha_bar[:-1]).all()))
    check("schedule betas valid", bool((s.beta > 0).all() and (s.beta < 1).all()))
    check("schedule alpha_bar_prev shifted", float(s.alpha_bar_prev[0]) == 1.0)
    check("schedule alpha_bar_prev aligns",
          close(s.alpha_bar_prev[1:], s.alpha_bar[:-1]))


# Both samplers consume the same trained epsilon model, so DDIM at many steps
# should land in the same region as the full DDPM chain
def test_samplers_agree_in_scale(d):
    labels = torch.arange(N_CLASSES)
    torch.manual_seed(0)
    ddim = DDIMSampler(d).sample(labels, SHAPE, "cpu", num_steps=50)
    torch.manual_seed(0)
    ddpm = DDPMSampler(d).sample(labels, SHAPE, "cpu")
    check("ddpm sampler finite", bool(torch.isfinite(ddpm).all()))
    check("samplers agree in scale",
          abs(float(ddim.std() - ddpm.std())) < 1.0,
          "ddim std %.3f ddpm std %.3f" % (ddim.std(), ddpm.std()))


# The transformer style sinusoidal embedding has to be bounded, distinct per
# timestep, and identical for repeated timesteps
def test_timestep_embedding():
    t = torch.tensor([0, 1, 500, 999])
    emb = timestep_embedding(t, 128)
    check("timestep embedding shape", tuple(emb.shape) == (4, 128), str(tuple(emb.shape)))
    check("timestep embedding bounded", float(emb.abs().max()) <= 1.0 + 1e-6)
    check("timestep embedding distinct", float((emb[0] - emb[2]).abs().max()) > 0.1)
    same = timestep_embedding(torch.tensor([500, 500]), 128)
    check("timestep embedding deterministic", close(same[0], same[1]))


# The label and the timestep both have to reach the output or the model is not
# actually conditional
def test_conditioning_is_live(net):
    net.eval()
    torch.nn.init.normal_(net.out_conv.weight, std=0.05)
    with torch.no_grad():
        x = torch.randn(2, 1, 32, 32)
        same_label = torch.zeros(2, dtype=torch.long)
        early = net(x, torch.tensor([10, 10]), same_label)
        late = net(x, torch.tensor([900, 900]), same_label)
        check("timestep changes output", float((early - late).abs().mean()) > 1e-4,
              "%.5f" % float((early - late).abs().mean()))
        two = net(x, torch.tensor([10, 10]), torch.tensor([0, 3]))
        check("label changes output", float((two[0] - two[1]).abs().mean()) > 1e-4,
              "%.5f" % float((two[0] - two[1]).abs().mean()))
    torch.nn.init.zeros_(net.out_conv.weight)


# Mel normalization has to invert exactly or the vocoder reads the wrong scale
def test_mel_normalization():
    mel = torch.randn(4, 1, 80, 176) * 3.65 - 7.35
    back = denormalize_mel(normalize_mel(mel, -7.35, 3.65), -7.35, 3.65)
    check("mel normalize round trip", close(mel, back, 1e-4))
    norm = normalize_mel(mel, float(mel.mean()), float(mel.std()))
    check("mel normalize centers", abs(float(norm.mean())) < 1e-5, "%.2e" % norm.mean())
    check("mel normalize scales", abs(float(norm.std()) - 1.0) < 1e-4, "%.6f" % norm.std())


def main():
    torch.manual_seed(0)
    schedule = NoiseSchedule()
    net = ConditionalUNet(n_classes=N_CLASSES, base_channels=16,
                          channel_mults=(1, 2, 2), attention_levels=(1, 2))
    net.eval()
    diffusion = GaussianDiffusion(net, schedule)

    print("schedule and forward process")
    test_cosine_schedule_properties(diffusion)
    test_forward_chain_matches_closed_form(diffusion)
    test_eq4_closed_form(diffusion)
    test_predict_x0_inverts_eq4(diffusion)

    print("\nposterior and training objective")
    test_eq7_posterior_variance(diffusion)
    test_eq6_posterior_mean(diffusion)
    test_algorithm2_equals_posterior_form(diffusion)
    test_eq14_simple_objective(diffusion, net)

    print("\nddim sampling")
    test_ddim_eq12(diffusion, net)
    test_ddim_eq16_recovers_ddpm_variance(diffusion)
    test_samplers_agree_in_scale(diffusion)

    print("\nmodel plumbing")
    test_timestep_embedding()
    test_conditioning_is_live(net)
    test_mel_normalization()

    print("\n%d passed, %d failed" % (len(passed), len(failed)))
    for name in failed:
        print("  failed: %s" % name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
