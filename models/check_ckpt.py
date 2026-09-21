# python models/check_ckpt.py [CHECKPOINT ...] [--n N] [--classify N] [--seed N]
# Numeric health check of trained checkpoints, no sampling or vocoder. Every
# number uses the same clips, timesteps and noise, so checkpoints compare 1:1.
# There is no held-out split, so these measure fit to the training corpus
import argparse
import glob
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.cunet import ConditionalUNet
from models.diffusion import GaussianDiffusion, NoiseSchedule
from models.train import INDEX_PATH, STATS_PATH, MelDataset

CHECKPOINT_DIR = os.path.join("models", "checkpoints")
T_BUCKETS = [(0, 100), (100, 300), (300, 600), (600, 900), (900, 1000)]
CLASSIFY_DRAWS = 16


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = ConditionalUNet(n_classes=ckpt["n_classes"])
    net.load_state_dict(ckpt["model"])
    net.eval()
    return GaussianDiffusion(net, NoiseSchedule()).to(device), ckpt


# NaN/Inf anywhere means the checkpoint is dead, the global norm shows drift
def weight_health(state):
    bad = sum(int((~torch.isfinite(v)).sum()) for v in state.values() if v.is_floating_point())
    norm = sum(float(v.float().pow(2).sum()) for v in state.values() if v.is_floating_point()) ** 0.5
    return bad, norm


# Per-sample eps MSE for fixed (x_0, t, noise), under the given labels
def eps_mse(diffusion, x_0, t, noise, labels, batch=16):
    out = []
    for i in range(0, x_0.shape[0], batch):
        sl = slice(i, i + batch)
        x_t = diffusion.q_sample(x_0[sl], t[sl], noise[sl])
        pred = diffusion.model(x_t, t[sl], labels[sl])
        out.append(F.mse_loss(pred, noise[sl], reduction="none").flatten(1).mean(1))
    return torch.cat(out)


# Diffusion classifier: the label whose conditional model best denoises a clip
# is the model's guess for it. Chance is 1 / n_classes
def classify(diffusion, x_0, labels, n_classes, gen, device):
    correct = 0
    for i in range(x_0.shape[0]):
        t = torch.randint(0, 1000, (CLASSIFY_DRAWS,), generator=gen).to(device)
        noise = torch.randn((CLASSIFY_DRAWS,) + x_0.shape[1:], generator=gen).to(device)
        # every class sees the same draws, so only the label differs
        t_all = t.repeat(n_classes)
        noise_all = noise.repeat(n_classes, 1, 1, 1)
        x_all = x_0[i:i + 1].expand(n_classes * CLASSIFY_DRAWS, *x_0.shape[1:])
        cls = torch.arange(n_classes, device=device).repeat_interleave(CLASSIFY_DRAWS)
        err = eps_mse(diffusion, x_all, t_all, noise_all, cls, batch=16)
        correct += int(err.view(n_classes, CLASSIFY_DRAWS).mean(1).argmin() == labels[i])
    return correct / x_0.shape[0]


def epoch_of(path):
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Numeric checkpoint health check.")
    parser.add_argument("checkpoints", nargs="*",
                        help="checkpoint paths, default every one in %s" % CHECKPOINT_DIR)
    parser.add_argument("--n", type=int, default=512, help="clips for the loss metrics")
    parser.add_argument("--classify", type=int, default=128,
                        help="clips for the diffusion-classifier accuracy, 0 to skip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = args.checkpoints or glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    paths = sorted(paths, key=epoch_of)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One fixed eval set shared by every checkpoint
    data = MelDataset(INDEX_PATH, STATS_PATH)
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(data), generator=gen)[:args.n].tolist()
    x_0 = torch.stack([data[i][0] for i in idx]).to(device)
    labels = torch.tensor([data[i][1] for i in idx], device=device)
    t = torch.randint(0, 1000, (len(idx),), generator=gen).to(device)
    noise = torch.randn(x_0.shape, generator=gen).to(device)
    # a wrong label per clip: shift by a random nonzero offset
    n_classes_data = int(labels.max()) + 1
    shift = torch.randint(1, max(n_classes_data, 2), (len(idx),), generator=gen).to(device)
    classify_seed = int(torch.randint(0, 2**31, (1,), generator=gen))

    print("eval clips %d, device %s" % (len(idx), device))
    header = ("epoch", "bad", "|w|", "mse", *["t%d-%d" % b for b in T_BUCKETS],
              "null-true", "wrong-true", "cls_acc")
    print("  ".join("%10s" % h for h in header))

    for path in paths:
        diffusion, ckpt = load(path, device)
        n_classes = ckpt["n_classes"]
        bad, norm = weight_health(ckpt["model"])

        true = eps_mse(diffusion, x_0, t, noise, labels)
        buckets = [float(true[(t >= lo) & (t < hi)].mean()) for lo, hi in T_BUCKETS]

        # Positive gaps mean the labels carry information the model uses
        null_gap = wrong_gap = float("nan")
        if diffusion.null_label is not None:
            null = torch.full_like(labels, diffusion.null_label)
            null_gap = float(eps_mse(diffusion, x_0, t, noise, null).mean() - true.mean())
        wrong = (labels + shift) % n_classes
        wrong_gap = float(eps_mse(diffusion, x_0, t, noise, wrong).mean() - true.mean())

        acc = float("nan")
        if args.classify > 0:
            k = min(args.classify, len(idx))
            acc = classify(diffusion, x_0[:k], labels[:k], n_classes,
                           torch.Generator().manual_seed(classify_seed), device)

        row = [str(ckpt.get("epoch", -1) + 1), str(bad), "%.1f" % norm,
               "%.5f" % float(true.mean()), *["%.5f" % b for b in buckets],
               "%+.5f" % null_gap, "%+.5f" % wrong_gap, "%.3f" % acc]
        print("  ".join("%10s" % r for r in row))

    print("chance cls_acc %.3f" % (1.0 / n_classes))


if __name__ == "__main__":
    sys.exit(main())
