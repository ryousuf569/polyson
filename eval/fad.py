# python eval/fad.py [--checkpoint PATH] [--per-class N] [--guidance W] [--splits N]
# Encoder choice follows Layer 6's dgm-eval (Stein et al. 2023)
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from models.check_gen import (CHECKPOINT_DEFAULT, CLAP_SAMPLE_RATE, LABELS_PATH,
                              STATS_PATH, SAMPLE_RATE, embed_audio, generate,
                              load_bigvgan, load_clap, load_diffusion,
                              load_real_index, match_level)
from models.diffusion import DDIM_ETA, DDIM_STEPS, GUIDANCE_SCALE, denormalize_mel

CACHE_DIR = os.path.join("eval", "cache")
RESULTS_DIR = os.path.join("eval", "results")
REAL_CACHE = os.path.join(CACHE_DIR, "real.npz")

PER_CLASS = 100
GEN_BATCH = 25
BATCH = 16
# Subclasses have ~50 clips per split half, far too few for a 512-d covariance
PCA_DIMS = 32
SPLITS = 5

_models = {}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta():
    labels = json.load(open(LABELS_PATH, encoding="utf-8"))
    stats = json.load(open(STATS_PATH, encoding="utf-8"))
    return labels, stats


def class_names(labels):
    return [labels["idx_to_subclass"][str(i)] for i in range(labels["n_subclasses"])]


def get_models(device):
    if not _models:
        _models["vocoder"] = load_bigvgan().to(device)
        _models["clap"], _models["proc"] = load_clap()
    return _models


def vocode(mels, device):
    vocoder = get_models(device)["vocoder"]
    wavs = []
    for i in range(0, len(mels), BATCH):
        x = torch.from_numpy(np.asarray(mels[i:i + BATCH], dtype=np.float32)).to(device)
        with torch.no_grad():
            out = vocoder(x).squeeze(1).cpu().numpy()
        wavs.extend(match_level(w.astype(np.float64)) for w in out)
    return wavs


def embed_wavs(wavs, device):
    import librosa

    m = get_models(device)
    vecs = []
    for i in range(0, len(wavs), BATCH):
        batch = [match_level(librosa.resample(w.astype(np.float32), orig_sr=SAMPLE_RATE,
                                              target_sr=CLAP_SAMPLE_RATE))
                 for w in wavs[i:i + BATCH]]
        vecs.append(embed_audio(m["clap"], m["proc"], batch))
    return np.concatenate(vecs)


def embed_mels(mels, device):
    return embed_wavs(vocode(mels, device), device)


def load_real(labels, device):
    if os.path.exists(REAL_CACHE):
        return dict(np.load(REAL_CACHE))
    rows = []
    for key, key_rows in sorted(load_real_index().items()):
        if key in labels["subclass_to_idx"]:
            rows += key_rows
    print("embedding %d real clips..." % len(rows))
    emb = []
    for i in range(0, len(rows), 256):
        mels = np.stack([np.load(r["mel_path"]) for r in rows[i:i + 256]])
        emb.append(embed_mels(mels, device))
        print("  %d/%d" % (min(i + 256, len(rows)), len(rows)))
    real = {
        "emb": np.concatenate(emb),
        "cls": np.array([labels["subclass_to_idx"][r["subclass_key"]] for r in rows]),
        "sound_id": np.array([r["sound_id"] for r in rows]),
        "mel_path": np.array([r["mel_path"] for r in rows]),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(REAL_CACHE, **real)
    return real


def gen_cache_path(args):
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    return os.path.join(CACHE_DIR, "gen_%s_g%.1f_s%d_n%d_seed%d.npz" %
                        (name, args.guidance, args.steps, args.per_class, args.seed))


def load_generated(args, labels, stats, device):
    path = gen_cache_path(args)
    if os.path.exists(path):
        return dict(np.load(path))
    torch.manual_seed(args.seed)
    diffusion = load_diffusion(args.checkpoint, labels["n_subclasses"], device)
    shape = (1, stats["n_mels"], stats["n_frames"])
    mels, cls = [], []
    for idx, key in enumerate(class_names(labels)):
        print("sampling %d of %s..." % (args.per_class, key))
        done = 0
        while done < args.per_class:
            n = min(GEN_BATCH, args.per_class - done)
            x = generate(diffusion, idx, n, shape, device, args.steps, args.eta, args.guidance)
            mels.append(denormalize_mel(x, stats["mel_ref"], stats["mel_floor"]).cpu().numpy()[:, 0])
            cls += [idx] * n
            done += n
    del diffusion
    torch.cuda.empty_cache()
    mels = np.concatenate(mels)
    print("embedding generated clips...")
    gen = {"emb": embed_mels(mels, device), "cls": np.array(cls), "mels": mels.astype(np.float16)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **gen)
    return gen


# Half of each subclass is the reference set (real-vs-real floor, DSP sources),
# the other half is what every candidate set is scored against
def split_real(cls, seed):
    rng = np.random.default_rng(seed)
    in_a = np.zeros(len(cls), dtype=bool)
    for c in np.unique(cls):
        idx = rng.permutation(np.flatnonzero(cls == c))
        in_a[idx[:len(idx) // 2]] = True
    return in_a


# Symmetric form of the trace term, avoids sqrtm's complex output on
# near-singular covariances
def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(s1)
    root1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh(root1 @ s2 @ root1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def fad_report(real, fake, names, seeds, dims=PCA_DIMS):
    from sklearn.decomposition import PCA

    pca = PCA(dims, random_state=0).fit(real["emb"])
    r, f = pca.transform(real["emb"]), pca.transform(fake["emb"])
    rows = []
    pooled_a, pooled_g = [], []
    for c in np.unique(fake["cls"]):
        floor, score = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            in_a = split_real(real["cls"], seed)
            a = np.flatnonzero(in_a & (real["cls"] == c))
            b = np.flatnonzero(~in_a & (real["cls"] == c))
            g = np.flatnonzero(fake["cls"] == c)
            n = min(len(a), len(g))
            a, g = rng.choice(a, n, replace=False), rng.choice(g, n, replace=False)
            floor.append(frechet(r[a], r[b]))
            score.append(frechet(f[g], r[b]))
            if seed == seeds[0]:
                pooled_a.append(a)
                pooled_g.append(g)
        rows.append({"key": names[c], "n": int(n),
                     "real": float(np.mean(floor)), "real_std": float(np.std(floor)),
                     "fake": float(np.mean(score)), "fake_std": float(np.std(score)),
                     "ratio": float(np.mean(score) / np.mean(floor))})

    in_a = split_real(real["cls"], seeds[0])
    b = ~in_a & np.isin(real["cls"], np.unique(fake["cls"]))
    a, g = np.concatenate(pooled_a), np.concatenate(pooled_g)
    pooled = {"real": frechet(real["emb"][a], real["emb"][b]),
              "fake": frechet(fake["emb"][g], real["emb"][b])}
    return {"per_class": rows, "pooled": pooled}


def print_fad(report, title):
    print("\n%s: FAD (CLAP, PCA %d) against held-out real half" % (title, PCA_DIMS))
    print("%-16s %5s %14s %14s %7s" % ("subclass", "n", "real|real", "fake|real", "ratio"))
    for row in report["per_class"]:
        print("%-16s %5d %8.3f+-%-5.3f %8.3f+-%-5.3f %7.2f" %
              (row["key"], row["n"], row["real"], row["real_std"], row["fake"],
               row["fake_std"], row["ratio"]))
    ratios = [row["ratio"] for row in report["per_class"]]
    print("mean ratio %.2f (1.0 = indistinguishable from real at this n)" % np.mean(ratios))
    print("pooled 512-d FAD: real|real %.4f, fake|real %.4f" %
          (report["pooled"]["real"], report["pooled"]["fake"]))


def add_gen_args(parser):
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--per-class", type=int, default=PER_CLASS, dest="per_class")
    parser.add_argument("--steps", type=int, default=DDIM_STEPS)
    parser.add_argument("--eta", type=float, default=DDIM_ETA)
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=0)


def save_json(obj, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    print("\nwrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Per-subclass Frechet Audio Distance.")
    add_gen_args(parser)
    parser.add_argument("--splits", type=int, default=SPLITS)
    args = parser.parse_args()

    device = get_device()
    labels, stats = load_meta()
    real = load_real(labels, device)
    gen = load_generated(args, labels, stats, device)

    report = fad_report(real, gen, class_names(labels), list(range(args.splits)))
    print_fad(report, "diffusion")
    save_json(report, "fad_%s" % os.path.basename(gen_cache_path(args)).replace(".npz", ".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
