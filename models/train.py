import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")

from models.cunet import ConditionalUNet
from models.diffusion import (COND_DROPOUT, GaussianDiffusion, NoiseSchedule,
                              normalize_mel)

INDEX_PATH = os.path.join("data", "processed", "index.csv")
STATS_PATH = os.path.join("data", "processed", "mel_stats.json")
LABELS_PATH = os.path.join("data", "processed", "labels.json")
CHECKPOINT_DIR = os.path.join("models", "checkpoints")

BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 200
LOG_EVERY = 50
SAVE_EVERY = 10


# One clean mel per clip, clamped to the dynamic range preprocess.py measured
# and min-max scaled onto [-1, 1], plus the subclass label for FiLM
class MelDataset(Dataset):
    def __init__(self, index_path, stats_path):
        self.rows = list(csv.DictReader(open(index_path, newline="", encoding="utf-8")))
        stats = json.load(open(stats_path, encoding="utf-8"))
        if "mel_ref" not in stats:
            raise SystemExit("%s predates the min-max normalization, rebuild it "
                             "with: python data/preprocess.py --stats-only" % stats_path)
        self.ref = stats["mel_ref"]
        self.floor = stats["mel_floor"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mel = torch.from_numpy(np.load(row["mel_path"])).float().unsqueeze(0)
        mel = normalize_mel(mel, self.ref, self.floor)
        return mel, int(row["class_idx"])


# Standard DDPM training loop: diffusion.p_losses samples t and noise, builds
# x_t in closed form, and returns MSE(eps_pred, eps) for one batch
def train(diffusion, loader, optimizer, device, epochs, start_epoch=0, n_classes=None):
    net = diffusion.model
    step = 0

    def save(epoch):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "cunet_epoch%d.pt" % (epoch + 1))
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "n_classes": n_classes,
        }, ckpt_path)
        print("saved %s" % ckpt_path)

    for epoch in range(start_epoch, epochs):
        net.train()
        running = 0.0
        for x_0, class_idx in loader:
            x_0 = x_0.to(device)
            class_idx = class_idx.to(device)

            loss = diffusion.p_losses(x_0, class_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss)
            step += 1
            if step % LOG_EVERY == 0:
                print("epoch %d step %d loss %.4f" % (epoch, step, float(loss)))

        print("epoch %d done, avg loss %.4f" % (epoch, running / len(loader)))

        if (epoch + 1) % SAVE_EVERY == 0 or epoch + 1 == epochs:
            save(epoch)


def main():
    parser = argparse.ArgumentParser(description="Train the CU-Net DDPM on preprocessed mels.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--cond-dropout", type=float, default=COND_DROPOUT,
                        dest="cond_dropout",
                        help="fraction of training labels replaced by the null "
                             "token, for classifier-free guidance (default: %.2f)"
                             % COND_DROPOUT)
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    labels = json.load(open(LABELS_PATH, encoding="utf-8"))
    dataset = MelDataset(INDEX_PATH, STATS_PATH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device == "cuda"))
    print("%d clips, %d subclasses" % (len(dataset), labels["n_subclasses"]))
    print("mel range [%.3f, %.3f] scaled to [-1, 1], label dropout %.2f"
          % (dataset.floor, dataset.ref, args.cond_dropout))

    net = ConditionalUNet(n_classes=labels["n_subclasses"])
    # Label dropout is what buys classifier-free guidance at sampling time
    diffusion = GaussianDiffusion(net, NoiseSchedule(),
                                  cond_dropout=args.cond_dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print("resumed from %s at epoch %d" % (args.resume, start_epoch))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train(diffusion, loader, optimizer, device, args.epochs, start_epoch=start_epoch,
         n_classes=labels["n_subclasses"])


if __name__ == "__main__":
    main()
