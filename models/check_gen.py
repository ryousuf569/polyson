# python models/check_gen.py <subclass_tag> [--checkpoint PATH] [--steps N] [--n N] [--seed N]
# python models/check_gen.py --compare [--per-class N] [--checkpoint PATH] [--seed N]

import argparse
import csv
import json
import os
import random
import sys


import numpy as np
import torch

sys.path.insert(0, ".")

from models.cunet import ConditionalUNet
from models.diffusion import (DDIM_ETA, DDIM_STEPS, DDIMSampler, GaussianDiffusion,
                              NoiseSchedule, denormalize_mel)

LABELS_PATH = os.path.join("data", "processed", "labels.json")
STATS_PATH = os.path.join("data", "processed", "mel_stats.json")
INDEX_PATH = os.path.join("data", "processed", "index.csv")
CHECKPOINT_DEFAULT = os.path.join("models", "checkpoints", "cunet_epoch40.pt")
OUT_DIR = os.path.join("models", "generated")
PLOT_PATH = os.path.join(OUT_DIR, "real_vs_generated.png")

SAMPLE_RATE = 22050
PEAK_LEVEL = 0.95

# How many clips per subclass the --compare sweep draws from each side
COMPARE_PER_CLASS = 20

# Real and generated are two fixed identities, so they keep the same two colors
# in every panel rather than being recolored per plot
REAL_COLOR = "#3b6fb6"
GEN_COLOR = "#c1663a"

# This checkpoint matches our mel settings exactly, fmax 8000 and hop 256
BIGVGAN_MODEL = "nvidia/bigvgan_22khz_80band"

CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000


# Scale to a fixed peak, matching how preprocess.py levels every clip
def match_level(audio):
    peak = float(np.abs(audio).max())
    return audio if peak <= 0.0 else audio * (PEAK_LEVEL / peak)


# Rebuild the CU-Net and wrap it in the same diffusion process it trained under
def load_diffusion(checkpoint_path, n_classes, device):
    net = ConditionalUNet(n_classes=n_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return GaussianDiffusion(net, NoiseSchedule()).to(device)


# Load BigVGAN, which ships its own model code rather than a transformers class
def load_bigvgan():
    import huggingface_hub

    path = huggingface_hub.snapshot_download(BIGVGAN_MODEL)
    if path not in sys.path:
        sys.path.insert(0, path)
    import bigvgan
    from env import AttrDict

    config = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    model = bigvgan.BigVGAN(AttrDict(config), use_cuda_kernel=False)
    weights = torch.load(os.path.join(path, "bigvgan_generator.pt"), map_location="cpu")
    try:
        model.load_state_dict(weights["generator"])
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(weights["generator"])
    model.eval()
    torch.set_grad_enabled(False)
    return model


# Run one mel (n_mels, n_frames) through the vocoder and return a mono waveform
def decode_mel(model, mel):
    with torch.no_grad():
        wav = model(torch.from_numpy(mel).unsqueeze(0))
    return wav.squeeze().cpu().numpy().astype(np.float64)


# Load CLAP for the subclass fit check
def load_clap():
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL)
    model.eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    torch.set_grad_enabled(False)
    return model, processor


# Embed a waveform already at CLAP_SAMPLE_RATE into a normalized CLAP vector
def embed_audio(model, processor, audios):
    inputs = processor(audio=audios, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    vecs = (out["pooler_output"] if hasattr(out, "keys") else out).numpy().astype(np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)


# Embed subclass description text into normalized CLAP vectors, one per class
def embed_text(model, processor, texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    vecs = (out["pooler_output"] if hasattr(out, "keys") else out).numpy().astype(np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)


# Sample one batch of mels for a subclass with the DDIM sampler
def generate(diffusion, class_idx, n, shape, device, steps, eta):
    sampler = DDIMSampler(diffusion)
    labels = torch.full((n,), class_idx, dtype=torch.long, device=device)
    return sampler.sample(labels, (n,) + shape, device, num_steps=steps, eta=eta)


# CLAP's zero-shot fit check: embed the generated audio and every subclass's
# text description, then rank by cosine similarity so we can see whether the
# subclass we asked for is actually the closest match
def check_fit(clap_model, clap_processor, wav, labels, target_key):
    import librosa

    clap_wav = librosa.resample(wav.astype(np.float32), orig_sr=SAMPLE_RATE,
                                target_sr=CLAP_SAMPLE_RATE)
    audio_vec = embed_audio(clap_model, clap_processor, [match_level(clap_wav)])

    # Descriptions repeat across coarse classes (eg "a dry organic recorded
    # thud" for explosion, footstep and impact), so the class name has to be
    # folded back in or CLAP has no way to tell those subclasses apart
    keys = sorted(labels["subclass_to_idx"])
    texts = ["%s: %s" % (key.rsplit("_sub", 1)[0], labels["subclass_description"][key])
            for key in keys]
    text_vecs = embed_text(clap_model, clap_processor, texts)

    scores = (audio_vec @ text_vecs.T)[0]
    order = np.argsort(-scores)
    target_pos = keys.index(target_key)
    rank = int(np.where(order == target_pos)[0][0]) + 1

    print("  target %s scored %.4f, ranked %d/%d by CLAP" %
         (target_key, scores[target_pos], rank, len(keys)))
    print("  top matches:")
    for pos in order[:5]:
        marker = "*" if keys[pos] == target_key else " "
        print("    %s %-16s %.4f" % (marker, keys[pos], scores[pos]))


# Group the preprocessed clips by subclass tag so we can draw real mels to hold
# the generated ones against
def load_real_index():
    by_subclass = {}
    with open(INDEX_PATH, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_subclass.setdefault(row["subclass_key"], []).append(row)
    return by_subclass


# Draw n real clips for a subclass, returning their mels and cached CLAP vectors.
# Sampling without replacement, but a thin subclass just gives back what it has
def sample_real(rows, n, rng):
    picked = rng.sample(rows, min(n, len(rows)))
    mels = np.stack([np.load(row["mel_path"]) for row in picked])
    claps = np.stack([np.load(row["clap_path"]) for row in picked])
    return mels, claps


# The per-mel-bin mean level, ie the average spectral envelope over a set of
# clips. This is the shape that tells us whether the model learned where each
# subclass puts its energy across frequency
def spectral_envelope(mels):
    return mels.mean(axis=(0, 2))


# The per-frame mean level, ie the average temporal envelope. Captures whether
# generated clips have the same attack and decay shape as the real recordings
def temporal_envelope(mels):
    return mels.mean(axis=(0, 1))


# Mean cosine similarity between every generated clip and every real clip of the
# same subclass, in CLAP's audio space. Vectors are already unit norm, so the
# matrix product is the cosine directly
def cross_similarity(gen_vecs, real_vecs):
    return float((gen_vecs @ real_vecs.T).mean())


# Real-vs-real similarity within a subclass, excluding each clip against itself.
# This is the ceiling the generated score should be read against, since a class
# whose own recordings only agree at 0.4 cannot expect a generated 0.9
def self_similarity(vecs):
    if len(vecs) < 2:
        return float("nan")
    sims = vecs @ vecs.T
    off_diagonal = ~np.eye(len(vecs), dtype=bool)
    return float(sims[off_diagonal].mean())


# Vocode a batch of generated mels and embed them with CLAP, so they land in the
# same space as the cached real vectors from preprocessing
def embed_generated(vocoder, clap_model, clap_processor, mels):
    import librosa

    wavs = []
    for mel in mels:
        wav = match_level(decode_mel(vocoder, mel))
        wavs.append(match_level(librosa.resample(wav.astype(np.float32),
                                                 orig_sr=SAMPLE_RATE,
                                                 target_sr=CLAP_SAMPLE_RATE)))
    return embed_audio(clap_model, clap_processor, wavs)


# Walk every subclass, generate a batch, pull an equally sized real batch, and
# collect the statistics the summary plot is built from
def collect_comparison(diffusion, vocoder, clap_model, clap_processor, labels,
                       stats, device, args, rng):
    by_subclass = load_real_index()
    shape = (1, stats["n_mels"], stats["n_frames"])
    results = []

    for key in sorted(labels["subclass_to_idx"]):
        rows = by_subclass.get(key, [])
        if not rows:
            print("no real clips indexed for %s, skipping" % key)
            continue

        print("\n%s (%s)" % (key, labels["subclass_description"][key]))
        print("  sampling %d generated, drawing %d real..." %
             (args.per_class, min(args.per_class, len(rows))))

        x = generate(diffusion, labels["subclass_to_idx"][key], args.per_class,
                     shape, device, args.steps, args.eta)
        gen_mels = denormalize_mel(x, stats["mel_mean"],
                                   stats["mel_std"]).cpu().numpy()[:, 0]
        real_mels, real_claps = sample_real(rows, args.per_class, rng)

        gen_claps = embed_generated(vocoder, clap_model, clap_processor, gen_mels)
        cross = cross_similarity(gen_claps, real_claps)
        real_self = self_similarity(real_claps)
        print("  CLAP gen-vs-real %.4f, real-vs-real %.4f" % (cross, real_self))

        results.append({
            "key": key,
            "n_real": len(real_mels),
            "n_gen": len(gen_mels),
            "real_mels": real_mels,
            "gen_mels": gen_mels,
            "cross": cross,
            "real_self": real_self,
        })
    return results


# One figure answering "do these look like Freesound data": the average spectral
# and temporal envelopes per subclass, the pooled value distribution, a matched
# pair of example mels, and the CLAP agreement per subclass against its own
# real-vs-real ceiling
def plot_comparison(results, stats, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig = plt.figure(figsize=(4.1 * min(n, 4), 15.5))
    grid = fig.add_gridspec(5, min(n, 4), hspace=0.55, wspace=0.28)

    hz_per_bin = stats["fmax"] / stats["n_mels"]
    sec_per_frame = stats["hop_length"] / stats["sample_rate"]
    bins = np.arange(stats["n_mels"]) * hz_per_bin
    times = np.arange(stats["n_frames"]) * sec_per_frame

    # Row 1 and 2: the two envelopes, one panel per subclass for the first few
    # subclasses so the individual curves stay readable
    shown = results[:min(n, 4)]
    for col, res in enumerate(shown):
        ax = fig.add_subplot(grid[0, col])
        ax.plot(bins, spectral_envelope(res["real_mels"]), color=REAL_COLOR,
                linewidth=2, label="real")
        ax.plot(bins, spectral_envelope(res["gen_mels"]), color=GEN_COLOR,
                linewidth=2, label="generated")
        ax.set_title(res["key"], fontsize=10)
        ax.set_xlabel("frequency (Hz)", fontsize=8)
        if col == 0:
            ax.set_ylabel("mean log-mel", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)

        ax = fig.add_subplot(grid[1, col])
        ax.plot(times, temporal_envelope(res["real_mels"]), color=REAL_COLOR,
                linewidth=2, label="real")
        ax.plot(times, temporal_envelope(res["gen_mels"]), color=GEN_COLOR,
                linewidth=2, label="generated")
        ax.set_xlabel("time (s)", fontsize=8)
        if col == 0:
            ax.set_ylabel("mean log-mel", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)

    # Row 3: pooled value histogram across every subclass. A model that collapsed
    # to a narrow range shows up here as a much tighter curve than the real one
    ax = fig.add_subplot(grid[2, :])
    all_real = np.concatenate([r["real_mels"].ravel() for r in results])
    all_gen = np.concatenate([r["gen_mels"].ravel() for r in results])
    edges = np.linspace(min(all_real.min(), all_gen.min()),
                        max(all_real.max(), all_gen.max()), 90)
    ax.hist(all_real, bins=edges, color=REAL_COLOR, alpha=0.55, density=True,
            label="real (%d clips)" % sum(r["n_real"] for r in results))
    ax.hist(all_gen, bins=edges, color=GEN_COLOR, alpha=0.55, density=True,
            label="generated (%d clips)" % sum(r["n_gen"] for r in results))
    ax.set_title("pooled log-mel value distribution, all subclasses", fontsize=10)
    ax.set_xlabel("log-mel value", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=False)

    # Row 4: one real and one generated mel side by side, on a shared color
    # scale so the two are actually comparable by eye
    for col, res in enumerate(shown):
        pair = np.concatenate([res["real_mels"][0], res["gen_mels"][0]], axis=1)
        sub = grid[3, col].subgridspec(1, 2, wspace=0.06)
        vmin, vmax = float(pair.min()), float(pair.max())
        for j, (mel, name) in enumerate([(res["real_mels"][0], "real"),
                                         (res["gen_mels"][0], "generated")]):
            ax = fig.add_subplot(sub[0, j])
            ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                      vmin=vmin, vmax=vmax,
                      extent=[0, times[-1], 0, stats["fmax"]])
            ax.set_title("%s\n%s" % (res["key"], name), fontsize=8)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            # Only the left panel of each pair keeps the frequency axis, since
            # both share one scale
            if j == 0 and col == 0:
                ax.set_ylabel("frequency (Hz)", fontsize=7)
            else:
                ax.set_yticklabels([])

    # Row 5: the CLAP verdict. Each subclass gets its generated agreement next to
    # the real-vs-real ceiling, so a low bar is only damning when the real bar
    # beside it is high
    ax = fig.add_subplot(grid[4, :])
    keys = [r["key"] for r in results]
    pos = np.arange(len(keys))
    width = 0.38
    ax.bar(pos - width / 2, [r["real_self"] for r in results], width,
           color=REAL_COLOR, label="real vs real (ceiling)")
    ax.bar(pos + width / 2, [r["cross"] for r in results], width,
           color=GEN_COLOR, label="generated vs real")
    ax.set_xticks(pos)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean CLAP cosine", fontsize=8)
    ax.set_title("CLAP audio-embedding agreement per subclass", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Generated mels vs real Freesound mels, %d per subclass" %
                 max(r["n_gen"] for r in results), fontsize=13, y=0.997)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# Print the same numbers the plot shows, since a table is easier to diff between
# checkpoints than a picture
def print_summary(results):
    print("\n%-16s %6s %6s %8s %8s %8s" %
         ("subclass", "n_gen", "n_real", "gen~real", "real~real", "ratio"))
    for res in results:
        ratio = res["cross"] / res["real_self"] if res["real_self"] else float("nan")
        print("%-16s %6d %6d %8.4f %8.4f %8.2f" %
             (res["key"], res["n_gen"], res["n_real"], res["cross"],
              res["real_self"], ratio))

    cross = float(np.mean([r["cross"] for r in results]))
    ceiling = float(np.mean([r["real_self"] for r in results]))
    print("\noverall gen-vs-real %.4f against a real-vs-real ceiling of %.4f "
          "(%.0f%% of ceiling)" % (cross, ceiling, 100 * cross / ceiling))


# The --compare sweep: generate a batch for every subclass, hold each against an
# equally sized real batch, and write the summary plot
def run_comparison(args, labels, stats, device):
    rng = random.Random(args.seed)

    print("loading checkpoint %s..." % args.checkpoint)
    diffusion = load_diffusion(args.checkpoint, labels["n_subclasses"], device)
    print("loading %s..." % BIGVGAN_MODEL)
    vocoder = load_bigvgan()
    print("loading %s..." % CLAP_MODEL)
    clap_model, clap_processor = load_clap()

    results = collect_comparison(diffusion, vocoder, clap_model, clap_processor,
                                 labels, stats, device, args, rng)
    if not results:
        print("nothing to compare", file=sys.stderr)
        return 1

    print_summary(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_comparison(results, stats, args.plot)
    print("\nwrote %s" % args.plot)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Sample a mel from the trained CU-Net "
                                                 "with DDIM, check it against its subclass "
                                                 "with CLAP, and vocode it with BigVGAN.")
    parser.add_argument("subclass", nargs="?", help="subclass tag, eg explosion_sub1 "
                                                    "(see data/processed/labels.json), "
                                                    "not needed with --compare")
    parser.add_argument("--compare", action="store_true",
                        help="generate a batch for every subclass, compare it against "
                             "real preprocessed mels, and write a summary plot")
    parser.add_argument("--per-class", type=int, default=COMPARE_PER_CLASS,
                        dest="per_class",
                        help="clips per subclass per side in --compare mode")
    parser.add_argument("--plot", default=PLOT_PATH, help="where --compare writes its plot")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--n", type=int, default=1, help="how many clips to generate")
    parser.add_argument("--steps", type=int, default=DDIM_STEPS)
    parser.add_argument("--eta", type=float, default=DDIM_ETA)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    labels = json.load(open(LABELS_PATH, encoding="utf-8"))
    stats = json.load(open(STATS_PATH, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.compare:
        return run_comparison(args, labels, stats, device)

    if args.subclass is None:
        parser.error("a subclass tag is required unless --compare is given")
    if args.subclass not in labels["subclass_to_idx"]:
        print("unknown subclass %r, choose from:" % args.subclass, file=sys.stderr)
        for key in sorted(labels["subclass_to_idx"]):
            print("  %s" % key, file=sys.stderr)
        return 1

    print("loading checkpoint %s..." % args.checkpoint)
    diffusion = load_diffusion(args.checkpoint, labels["n_subclasses"], device)
    class_idx = labels["subclass_to_idx"][args.subclass]
    shape = (1, stats["n_mels"], stats["n_frames"])

    print("sampling %d clip(s) of %s (%s), %d ddim steps..." %
         (args.n, args.subclass, labels["subclass_description"][args.subclass], args.steps))
    x = generate(diffusion, class_idx, args.n, shape, device, args.steps, args.eta)
    mel = denormalize_mel(x, stats["mel_mean"], stats["mel_std"]).cpu().numpy()

    print("loading %s..." % BIGVGAN_MODEL)
    vocoder = load_bigvgan()
    print("loading %s..." % CLAP_MODEL)
    clap_model, clap_processor = load_clap()

    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(args.n):
        print("\nclip %d/%d" % (i + 1, args.n))
        wav = match_level(decode_mel(vocoder, mel[i, 0]))
        check_fit(clap_model, clap_processor, wav, labels, args.subclass)

        import soundfile as sf
        out_path = os.path.join(OUT_DIR, "%s_%d.wav" % (args.subclass, i))
        sf.write(out_path, wav, SAMPLE_RATE)
        print("  wrote %s" % out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
