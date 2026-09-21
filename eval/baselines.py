# python eval/baselines.py [--checkpoint PATH] [--per-class N] [--guidance W] [--pairs K]

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from eval.classify import accuracy_report, fit_classifier, print_accuracy
from eval.fad import (CACHE_DIR, add_gen_args, class_names, embed_wavs, fad_report,
                      gen_cache_path, get_device, load_generated, load_meta, load_real,
                      print_fad, save_json, split_real, vocode)
from models.check_gen import SAMPLE_RATE, match_level

PAIRS_DIR = os.path.join("eval", "nn_pairs")
AUG_SPLIT_SEED = 0
PITCH_SEMITONES = 2.0
STRETCH_RANGE = (0.85, 1.15)
SNR_RANGE_DB = (20.0, 40.0)
PAIRS = 5


def augment(wav, rng):
    import librosa

    y = librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE,
                                    n_steps=rng.uniform(-PITCH_SEMITONES, PITCH_SEMITONES))
    y = librosa.effects.time_stretch(y, rate=rng.uniform(*STRETCH_RANGE))
    y = librosa.util.fix_length(y, size=len(wav))
    rms = np.sqrt(np.mean(y ** 2)) + 1e-10
    noise = rng.standard_normal(len(y)) * rms / 10 ** (rng.uniform(*SNR_RANGE_DB) / 20)
    return match_level(y + noise)


# Sources come only from the reference half of split AUG_SPLIT_SEED, so the
# FAD target half never contains a clip the augmentation started from
def load_augmented(real, names, per_class, seed, device):
    path = os.path.join(CACHE_DIR, "aug_n%d_seed%d.npz" % (per_class, seed))
    if os.path.exists(path):
        return dict(np.load(path))
    rng = np.random.default_rng(seed)
    in_a = split_real(real["cls"], AUG_SPLIT_SEED)
    embs, cls, refs = [], [], []
    for c in np.unique(real["cls"]):
        print("augmenting %d of %s..." % (per_class, names[c]))
        src = np.resize(rng.permutation(np.flatnonzero(in_a & (real["cls"] == c))), per_class)
        wavs = vocode(np.stack([np.load(p) for p in real["mel_path"][src]]), device)
        embs.append(embed_wavs([augment(w, rng) for w in wavs], device))
        cls += [c] * per_class
        refs.append(src)
    aug = {"emb": np.concatenate(embs), "cls": np.array(cls), "ref": np.concatenate(refs)}
    np.savez(path, **aug)
    return aug


def nearest(train, x):
    sims = x @ train.T
    j = sims.argmax(1)
    return sims[np.arange(len(x)), j], j


def real_loo_nearest(real):
    sims = real["emb"] @ real["emb"].T
    np.fill_diagonal(sims, -np.inf)
    j = sims.argmax(1)
    return sims[np.arange(len(sims)), j], j


# The model trained on every real clip, so all of them count as training data.
# Real clips' leave-one-out NN similarity is how close two genuinely different
# recordings get; generated clips sitting well above that are replays
def memorization_report(real, sets, names):
    loo_sim, _ = real_loo_nearest(real)
    out = {}
    for name, fake in sets.items():
        sim, j = nearest(real["emb"], fake["emb"])
        rows = []
        for c in np.unique(fake["cls"]):
            f, r = fake["cls"] == c, real["cls"] == c
            thresh = np.percentile(loo_sim[r], 95)
            rows.append({"key": names[c],
                         "real_loo_median": float(np.median(loo_sim[r])),
                         "median": float(np.median(sim[f])),
                         "above_real_p95": float((sim[f] > thresh).mean()),
                         "nn_same_class": float((real["cls"][j[f]] == c).mean())})
        entry = {"per_class": rows, "median": float(np.median(sim)),
                 "real_loo_median": float(np.median(loo_sim)),
                 "above_real_p95": float(np.mean([r["above_real_p95"] for r in rows]))}
        if "ref" in fake:
            entry["nn_is_own_source"] = float((j == fake["ref"]).mean())
        out[name] = entry
    return out


def print_memorization(report):
    for name, entry in report.items():
        print("\n%s: nearest training clip (CLAP cosine)" % name)
        print("%-16s %9s %9s %11s %9s" % ("subclass", "real loo", "median", ">real p95",
                                           "same cls"))
        for row in entry["per_class"]:
            print("%-16s %9.3f %9.3f %11.3f %9.3f" %
                  (row["key"], row["real_loo_median"], row["median"],
                   row["above_real_p95"], row["nn_same_class"]))
        line = "median %.3f vs real loo %.3f, %.1f%% above real p95" % (
            entry["median"], entry["real_loo_median"], 100 * entry["above_real_p95"])
        if "nn_is_own_source" in entry:
            line += ", nn is own source clip %.1f%%" % (100 * entry["nn_is_own_source"])
        print(line)


# The closest generated/train pairs as wavs, to listen for replayed clips
def write_pairs(real, gen, names, k, device):
    import soundfile as sf

    sim, j = nearest(real["emb"], gen["emb"])
    top = np.argsort(-sim)[:k]
    gen_wavs = vocode(gen["mels"][top].astype(np.float32), device)
    train_wavs = vocode(np.stack([np.load(real["mel_path"][j[i]]) for i in top]), device)
    os.makedirs(PAIRS_DIR, exist_ok=True)
    for rank, i in enumerate(top):
        stem = os.path.join(PAIRS_DIR, "%02d_%s_%.3f" % (rank, names[gen["cls"][i]], sim[i]))
        sf.write(stem + "_gen.wav", gen_wavs[rank], SAMPLE_RATE)
        sf.write(stem + "_train_%s.wav" % real["sound_id"][j[i]], train_wavs[rank], SAMPLE_RATE)
    print("\nwrote %d nearest pairs to %s" % (k, PAIRS_DIR))


def main():
    parser = argparse.ArgumentParser(description="DSP augmentation and nearest-neighbor "
                                                 "baselines against the diffusion model.")
    add_gen_args(parser)
    parser.add_argument("--pairs", type=int, default=PAIRS)
    args = parser.parse_args()

    device = get_device()
    labels, stats = load_meta()
    names = class_names(labels)
    real = load_real(labels, device)
    gen = load_generated(args, labels, stats, device)
    aug = load_augmented(real, names, args.per_class, args.seed, device)

    seeds = [AUG_SPLIT_SEED]
    fad = {"diffusion": fad_report(real, gen, names, seeds),
           "dsp_aug": fad_report(real, aug, names, seeds)}
    for name, report in fad.items():
        print_fad(report, name)

    clf, cv_pred = fit_classifier(real)
    acc = {"diffusion": accuracy_report(clf, cv_pred, real, gen, names),
           "dsp_aug": accuracy_report(clf, cv_pred, real, aug, names)}
    for name, report in acc.items():
        print_accuracy(report, name)

    mem = memorization_report(real, {"diffusion": gen, "dsp_aug": aug}, names)
    print_memorization(mem)

    print("\n%-10s %10s %10s %10s %12s" % ("", "fad ratio", "acc", "nn median", ">real p95"))
    for name in fad:
        print("%-10s %10.2f %10.3f %10.3f %12.3f" %
              (name, np.mean([r["ratio"] for r in fad[name]["per_class"]]),
               acc[name]["fake_acc"], mem[name]["median"], mem[name]["above_real_p95"]))

    if args.pairs:
        write_pairs(real, gen, names, args.pairs, device)
    save_json({"fad": fad, "accuracy": acc, "memorization": mem}, "baselines_%s" %
              os.path.basename(gen_cache_path(args)).replace(".npz", ".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
