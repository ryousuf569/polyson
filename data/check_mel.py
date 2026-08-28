#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as P

OUT_DIR = os.path.join("data", "check")
# This checkpoint matches our mel settings exactly, fmax 8000 and hop 256
BIGVGAN_MODEL = "nvidia/bigvgan_22khz_80band"


# Load BigVGAN, which ships its own model code rather than a transformers class
def load_bigvgan():
    import huggingface_hub
    import torch

    path = huggingface_hub.snapshot_download(BIGVGAN_MODEL)
    # Its modules import each other by bare name, so the snapshot has to be
    # importable as a directory rather than loaded through transformers
    if path not in sys.path:
        sys.path.insert(0, path)
    import bigvgan
    from env import AttrDict

    # Built straight from the config and checkpoint because the hub mixin on
    # this repo expects an older huggingface_hub calling convention
    config = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    model = bigvgan.BigVGAN(AttrDict(config), use_cuda_kernel=False)
    weights = torch.load(os.path.join(path, "bigvgan_generator.pt"),
                         map_location="cpu")
    try:
        model.load_state_dict(weights["generator"])
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(weights["generator"])
    model.eval()
    torch.set_grad_enabled(False)
    return model


# Run one mel through the vocoder and return a mono waveform
def decode_mel(model, mel):
    import torch

    with torch.no_grad():
        wav = model(torch.from_numpy(mel).unsqueeze(0))
    return wav.squeeze().cpu().numpy().astype(np.float64)


# Peak normalize so the two waveforms sit at the same level for comparison
def match_level(audio):
    peak = float(np.abs(audio).max())
    return audio if peak <= 0.0 else audio * (P.PEAK_LEVEL / peak)


# Spectral distance between two waveforms, the honest number for vocoder error
def mel_distance(reference, output, bank, window):
    n = min(len(reference), len(output))
    a = P.log_mel(reference[:n], bank, window)
    b = P.log_mel(output[:n], bank, window)
    frames = min(a.shape[1], b.shape[1])
    return float(np.abs(a[:, :frames] - b[:, :frames]).mean())


# Pick one processed clip, decode its mel, and compare against the source audio
def main():
    parser = argparse.ArgumentParser(description="Check a saved mel by running it "
                                                 "through BigVGAN.")
    parser.add_argument("--sound-id", default=None, help="specific clip (default: random)")
    parser.add_argument("--seed", type=int, default=None, help="seed for the random pick")
    args = parser.parse_args()

    if not os.path.exists(P.INDEX_PATH):
        print("no %s, run preprocess.py first" % P.INDEX_PATH, file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(P.INDEX_PATH, newline="", encoding="utf-8")))
    if args.sound_id:
        rows = [r for r in rows if r["sound_id"] == args.sound_id]
        if not rows:
            print("no clip with id %s" % args.sound_id, file=sys.stderr)
            return 1
    if args.seed is not None:
        random.seed(args.seed)
    row = random.choice(rows)

    mel = np.load(row["mel_path"])
    source = P.decode(os.path.join(P.RAW_DIR, row["class"], "%s.mp3" % row["sound_id"]),
                      P.SAMPLE_RATE)
    if source is None:
        print("could not decode the source mp3", file=sys.stderr)
        return 1

    # The mel came from this exact window, so compare against the same slice
    onset = int(row["onset_sample"])
    reference = match_level(P.align(source, onset))

    print("clip %s (%s %s)" % (row["sound_id"], row["class"], row["subclass_key"]))
    print("  %s" % row["name"])
    print("loading %s..." % BIGVGAN_MODEL)
    model = load_bigvgan()
    output = match_level(decode_mel(model, mel))

    bank = P.mel_filterbank()
    window = np.hanning(P.N_FFT + 1)[:-1]
    stats = json.load(open(P.STATS_PATH, encoding="utf-8"))

    os.makedirs(OUT_DIR, exist_ok=True)
    import soundfile as sf
    ref_path = os.path.join(OUT_DIR, "%s_original.wav" % row["sound_id"])
    out_path = os.path.join(OUT_DIR, "%s_bigvgan.wav" % row["sound_id"])
    sf.write(ref_path, reference, P.SAMPLE_RATE)
    sf.write(out_path, output, P.SAMPLE_RATE)

    print("\nmel  %s  range %.2f to %.2f" % (mel.shape, mel.min(), mel.max()))
    print("dataset mel mean %.3f std %.3f" % (stats["mel_mean"], stats["mel_std"]))
    print("original %d samples, decoded %d samples" % (len(reference), len(output)))
    print("mel L1 distance %.4f" % mel_distance(reference, output, bank, window))
    print("\nwrote %s" % ref_path)
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
