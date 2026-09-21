#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import re

import numpy as np


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RAW_DIR = os.path.join("data", "raw")
SUBCLASS_CSV = os.path.join("data", "subclass", "subclasses.csv")
OUT_DIR = os.path.join("data", "processed")
MEL_DIR = os.path.join(OUT_DIR, "mel")
CLAP_DIR = os.path.join(OUT_DIR, "clap")
STATS_PATH = os.path.join(OUT_DIR, "mel_stats.json")
INDEX_PATH = os.path.join(OUT_DIR, "index.csv")
LABELS_PATH = os.path.join(OUT_DIR, "labels.json")

# Subclasses need at least this many clips to be worth training on
MIN_SUBCLASS_SIZE = 100
EXCLUDED_SUBCLASSES = {"explosion_sub0", "whoosh_sub3"}

# Mel settings, sized so 176 frames at hop 256 covers about 2.03 seconds
SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
N_FRAMES = 176
FMIN = 0.0
FMAX = 8000.0
# One hop longer than N_FRAMES * HOP would suggest, because BigVGAN pads by
# (n_fft - hop) // 2 rather than n_fft // 2 and loses a frame otherwise
CLIP_SAMPLES = N_FRAMES * HOP

# Where the attack lands in the output window, as a fraction of the clip
ONSET_OFFSET = 0.1
ONSET_THRESHOLD = 0.15
PEAK_LEVEL = 0.95
LOG_FLOOR = 1e-5
# Mels are stored raw, with only BigVGAN's 1e-5 clamp on them. The range the
# trainer actually sees is cut from the top instead: everything more than
# TOP_DB under the loudest frame in the corpus is clamped off, which on
# peak-normalized audio puts the floor near -80 dBFS, ie below hearing.
# These are natural logs of a magnitude, so one dB is log(10) / 20 of a unit
TOP_DB = 80.0
DB = float(np.log(10.0)) / 20.0

# CLAP wants 48k audio and gives back one 512 dim vector per clip
CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000
CLAP_BATCH = 16

INDEX_HEADER = [
    "class",
    "subclass",
    "subclass_key",
    "class_idx",
    "coarse_idx",
    "subclass_label",
    "sound_id",
    "name",
    "mel_path",
    "clap_path",
    "onset_sample",
    "onset_found",
    "source_duration",
]


# Decode an audio file to mono float32 at the given rate with ffmpeg
def decode(path, sample_rate):
    proc = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path,
                           "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
                          capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    audio = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
    if audio.size == 0 or not np.isfinite(audio).all():
        return None
    return audio


# The same librosa slaney basis BigVGAN builds, so the scales cannot drift
def mel_filterbank():
    from librosa.filters import mel as librosa_mel

    return librosa_mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                       fmin=FMIN, fmax=FMAX)


# Frame energy over time, used to locate the attack
def frame_energy(audio):
    if len(audio) < WIN:
        return np.array([float((audio ** 2).mean())])
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP]
    return (frames ** 2).mean(axis=1)


# Locate the attack, returning the sample index and whether a real rise was seen
def find_onset(audio):
    energy = frame_energy(audio)
    if energy.max() <= 0.0:
        return 0, False
    # Threshold sits between the noise floor and the peak so quiet lead-ins
    # do not count as the attack
    floor = float(np.percentile(energy, 10))
    level = floor + ONSET_THRESHOLD * (float(energy.max()) - floor)
    hits = np.flatnonzero(energy >= level)
    if len(hits) == 0:
        return 0, False
    # Many clips are already trimmed to the attack, so a first frame hit is a
    # real onset at zero rather than a failure, and only a clip that starts
    # well below its peak and never rises counts as having no attack
    if hits[0] == 0:
        return 0, float(energy[0]) >= 0.5 * float(energy.max())
    return int(hits[0]) * HOP, True


# Cut a fixed length window with the attack sitting at the target offset
def align(audio, onset):
    target = int(ONSET_OFFSET * CLIP_SAMPLES)
    start = max(0, onset - target)
    window = audio[start:start + CLIP_SAMPLES]
    if len(window) < CLIP_SAMPLES:
        window = np.pad(window, (0, CLIP_SAMPLES - len(window)))
    return window


# Scale to a fixed peak so loudness does not leak into the mel values
def normalize(audio):
    peak = float(np.abs(audio).max())
    if peak <= 0.0:
        return audio
    return audio * (PEAK_LEVEL / peak)


# Turn a waveform into an 80 by 176 log mel, matching BigVGAN meldataset.py
def log_mel(audio, bank, window):
    # BigVGAN computes, in order:
    #   y = pad(y, (n_fft - hop) // 2, mode="reflect")
    #   S = |stft(y, n_fft, hop, win, hann_window, center=False)|
    #   mel = mel_basis @ S
    #   out = log(clamp(mel, min=1e-5))
    # Magnitude, not power, is what goes into the filterbank. Squaring it
    # doubles every value after the log and the vocoder reads the wrong scale.
    pad = (N_FFT - HOP) // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    frames = frames[:N_FRAMES] * window
    magnitude = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1))
    mel = magnitude @ bank.T
    return np.log(np.maximum(mel, LOG_FLOOR)).T.astype(np.float32)


# Load the CLAP model and processor
def load_clap():
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL, token=False, use_safetensors=True)
    model.eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL, token=False)
    torch.set_grad_enabled(False)
    return model, processor


# Embed a batch of waveforms into normalized CLAP vectors
def embed_clap(model, processor, audios):
    import torch

    inputs = processor(audio=audios, sampling_rate=CLAP_SAMPLE_RATE,
                       return_tensors="pt")
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    vecs = out["pooler_output"] if hasattr(out, "keys") else out
    vecs = vecs.numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)


# Build the string to integer label maps that training and eval both read
def build_label_maps(rows, min_size):
    keys = sorted({subclass_key(r) for r in rows})
    coarse = sorted({r["class"] for r in rows})
    descriptions = {}
    for row in rows:
        descriptions.setdefault(subclass_key(row), row["subclass_label"])
    return {
        "subclass_to_idx": {k: i for i, k in enumerate(keys)},
        "idx_to_subclass": {str(i): k for i, k in enumerate(keys)},
        "coarse_to_idx": {c: i for i, c in enumerate(coarse)},
        "idx_to_coarse": {str(i): c for i, c in enumerate(coarse)},
        "subclass_description": descriptions,
        "n_subclasses": len(keys),
        "n_coarse": len(coarse),
        "min_subclass_size": min_size,
    }


# Stable name for one subclass, used as the key in the label map
def subclass_key(row):
    return "%s_sub%s" % (row["class"], row["subclass"])


# Readable folder name for a subclass, so the layout says what the clips are
def subclass_dirname(row):
    words = [w for w in re.split(r"[^a-z0-9]+", row["subclass_label"].lower()) if w]
    return "sub%s_%s" % (row["subclass"], "_".join(words) or "unlabeled")


# Read the subclass table and keep only rows from big enough subclasses
def load_kept_rows(min_size):
    if not os.path.exists(SUBCLASS_CSV):
        print("no %s, run sound_subclass.py first" % SUBCLASS_CSV, file=sys.stderr)
        return None
    rows = list(csv.DictReader(open(SUBCLASS_CSV, newline="", encoding="utf-8")))
    sizes = {}
    for row in rows:
        key = (row["class"], row["subclass"])
        sizes[key] = sizes.get(key, 0) + 1

    kept = [r for r in rows
            if sizes[(r["class"], r["subclass"])] >= min_size
            and subclass_key(r) not in EXCLUDED_SUBCLASSES]
    small = sorted(k for k, v in sizes.items() if v < min_size)
    print("subclasses kept %d of %d, clips %d of %d"
          % (len(sizes) - len(small) - len(EXCLUDED_SUBCLASSES), len(sizes),
             len(kept), len(rows)))
    for class_name, subclass in small:
        print("  dropped %s sub%s (n=%d, under %d)"
              % (class_name, subclass, sizes[(class_name, subclass)], min_size))
    for key in sorted(EXCLUDED_SUBCLASSES):
        print("  excluded %s (duplicate label)" % key)
    return kept


# One pass over the stored mels for the moments and the corpus peak. The peak
# is what the training range is measured down from, and it has to come from the
# whole dataset rather than per clip or relative loudness gets flattened
def scan_mels(paths):
    # float64 so the totals do not drift over a few thousand clips
    total = 0
    mel_sum = 0.0
    mel_sq_sum = 0.0
    mel_max = -np.inf
    for path in paths:
        mel = np.load(path).astype(np.float64)
        mel_sum += float(mel.sum())
        mel_sq_sum += float((mel ** 2).sum())
        mel_max = max(mel_max, float(mel.max()))
        total += mel.size
    mean = mel_sum / total
    std = float(np.sqrt(max(mel_sq_sum / total - mean * mean, 0.0))) or 1.0
    return mean, std, mel_max


# What share of the corpus the training floor clamps away, which is the one
# number that says whether TOP_DB is set somewhere sane
def fraction_below(paths, floor):
    below = 0
    total = 0
    for path in paths:
        mel = np.load(path)
        below += int((mel < floor).sum())
        total += mel.size
    return below / float(total)


# mel_mean and mel_std are kept for check_mel.py and for reference. The trainer
# reads mel_ref and mel_floor instead and min-max scales that window to [-1, 1]
def build_stats(mean, std, mel_max, n_clips, n_subclasses, top_db=TOP_DB):
    return {
        "mel_mean": mean,
        "mel_std": std,
        "mel_ref": mel_max,
        "mel_floor": mel_max - top_db * DB,
        "top_db": top_db,
        "n_mels": N_MELS,
        "n_frames": N_FRAMES,
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "win_length": WIN,
        "fmin": FMIN,
        "fmax": FMAX,
        "log_floor": LOG_FLOOR,
        "onset_offset": ONSET_OFFSET,
        "peak_level": PEAK_LEVEL,
        "clip_samples": CLIP_SAMPLES,
        "n_clips": n_clips,
        "n_subclasses": n_subclasses,
        "labels_path": LABELS_PATH,
    }


# Shared by a full run and a --stats-only rebuild so the two cannot disagree
def write_stats(paths, n_subclasses, top_db=TOP_DB):
    mean, std, mel_max = scan_mels(paths)
    stats = build_stats(mean, std, mel_max, len(paths), n_subclasses, top_db)
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    clamped = fraction_below(paths, stats["mel_floor"])
    print("mel mean %.4f std %.4f over %d clips" % (mean, std, len(paths)))
    print("training range [%.4f, %.4f], %.0f dB, clamps %.1f%% of the corpus"
          % (stats["mel_floor"], stats["mel_ref"], top_db, 100.0 * clamped))
    print("wrote %s" % STATS_PATH)
    return stats


# Rebuild mel_stats.json from the mels already on disk. Retuning TOP_DB only
# needs this, not a full re-run over the source audio
def recompute_stats(top_db=TOP_DB):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(LABELS_PATH):
        print("no %s yet, run a full preprocess first" % INDEX_PATH, file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(INDEX_PATH, newline="", encoding="utf-8")))
    labels = json.load(open(LABELS_PATH, encoding="utf-8"))
    paths = [row["mel_path"] for row in rows]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        print("%d mels named in the index are missing, eg %s"
              % (len(missing), missing[0]), file=sys.stderr)
        return 1
    write_stats(paths, labels["n_subclasses"], top_db)
    return 0


# Preprocess every kept clip and write mels, CLAP vectors and the stats file
def main():
    parser = argparse.ArgumentParser(description="Preprocess kept clips into log "
                                                 "mel spectrograms and CLAP vectors.")
    parser.add_argument("--classes", default=None,
                        help="comma-separated class names (default: all kept)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many clips, for a test run")
    parser.add_argument("--min-size", type=int, default=MIN_SUBCLASS_SIZE,
                        help="subclass size floor (default: %d)" % MIN_SUBCLASS_SIZE)
    parser.add_argument("--stats-only", action="store_true", dest="stats_only",
                        help="rebuild mel_stats.json from the mels already on "
                             "disk, without touching the source audio")
    parser.add_argument("--top-db", type=float, default=TOP_DB, dest="top_db",
                        help="dynamic range kept under the corpus peak "
                             "(default: %.0f)" % TOP_DB)
    args = parser.parse_args()

    if args.stats_only:
        return recompute_stats(args.top_db)

    rows = load_kept_rows(args.min_size)
    if not rows:
        print("nothing to preprocess", file=sys.stderr)
        return 1

    # Built from every kept row before --classes or --limit narrow things down,
    # so a partial run cannot write a label map that disagrees with a full one
    labels = build_label_maps(rows, args.min_size)

    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        rows = [r for r in rows if r["class"] in wanted]
        if not rows:
            print("no kept clips in %s" % ", ".join(sorted(wanted)), file=sys.stderr)
            return 1
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(MEL_DIR, exist_ok=True)
    os.makedirs(CLAP_DIR, exist_ok=True)

    with open(LABELS_PATH, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
    print("label map: %d subclasses, %d coarse classes"
          % (labels["n_subclasses"], labels["n_coarse"]))

    print("loading CLAP (%s)..." % CLAP_MODEL)
    model, processor = load_clap()

    bank = mel_filterbank()
    # Periodic hann to match torch.hann_window rather than numpy's symmetric one
    window = np.hanning(WIN + 1)[:-1]

    index = []
    no_onset = 0
    failed = 0
    batch_audio = []
    batch_rows = []

    # CLAP runs on the same aligned window as the mel so the two views match
    def flush():
        if not batch_audio:
            return
        vectors = embed_clap(model, processor, batch_audio)
        for entry, vector in zip(batch_rows, vectors):
            np.save(entry["clap_path"], vector)
        batch_audio.clear()
        batch_rows.clear()

    for i, row in enumerate(rows, 1):
        class_name = row["class"]
        sound_id = row["sound_id"]
        src = os.path.join(RAW_DIR, class_name, "%s.mp3" % sound_id)
        if not os.path.exists(src):
            print("  missing audio: %s" % src, file=sys.stderr)
            failed += 1
            continue

        audio = decode(src, SAMPLE_RATE)
        if audio is None:
            print("  decode failed: %s" % src, file=sys.stderr)
            failed += 1
            continue

        onset, onset_found = find_onset(audio)
        if not onset_found:
            no_onset += 1
        clip = normalize(align(audio, onset))
        mel = log_mel(clip, bank, window)
        if mel.shape != (N_MELS, N_FRAMES):
            print("  bad mel shape %s for %s" % (mel.shape, sound_id), file=sys.stderr)
            failed += 1
            continue

        sub_dir = subclass_dirname(row)
        class_mel_dir = os.path.join(MEL_DIR, class_name, sub_dir)
        class_clap_dir = os.path.join(CLAP_DIR, class_name, sub_dir)
        os.makedirs(class_mel_dir, exist_ok=True)
        os.makedirs(class_clap_dir, exist_ok=True)
        mel_path = os.path.join(class_mel_dir, "%s.npy" % sound_id)
        clap_path = os.path.join(class_clap_dir, "%s.npy" % sound_id)
        np.save(mel_path, mel)

        key = subclass_key(row)
        entry = {
            "class": class_name,
            "subclass": row["subclass"],
            "subclass_key": key,
            "class_idx": labels["subclass_to_idx"][key],
            "coarse_idx": labels["coarse_to_idx"][class_name],
            "subclass_label": row["subclass_label"],
            "sound_id": sound_id,
            "name": row.get("name", ""),
            "mel_path": mel_path,
            "clap_path": clap_path,
            "onset_sample": onset,
            "onset_found": int(onset_found),
            "source_duration": round(len(audio) / float(SAMPLE_RATE), 4),
        }
        index.append(entry)

        # CLAP needs its own sample rate, so decode the source a second time
        clap_audio = decode(src, CLAP_SAMPLE_RATE)
        if clap_audio is None:
            clap_audio = np.zeros(CLAP_SAMPLE_RATE, dtype=np.float64)
        scale = CLAP_SAMPLE_RATE / float(SAMPLE_RATE)
        start = max(0, int(onset * scale)
                    - int(ONSET_OFFSET * CLIP_SAMPLES * scale))
        span = int(CLIP_SAMPLES * scale)
        clap_clip = clap_audio[start:start + span]
        if len(clap_clip) < span:
            clap_clip = np.pad(clap_clip, (0, span - len(clap_clip)))
        batch_audio.append(normalize(clap_clip).astype(np.float32))
        batch_rows.append(entry)
        if len(batch_audio) >= CLAP_BATCH:
            flush()
            print("processed %d/%d" % (i, len(rows)))
    flush()

    if not index:
        print("no clips processed", file=sys.stderr)
        return 1

    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_HEADER)
        writer.writeheader()
        writer.writerows(index)

    print("\nprocessed %d clips, %d failed" % (len(index), failed))
    print("no clear attack in %d clips, aligned from the start" % no_onset)
    print("wrote %s" % INDEX_PATH)
    print("wrote %s" % LABELS_PATH)

    # Mels are saved unnormalized, so the stats file is what lets training and
    # inference agree on the scaling BigVGAN has to read back
    write_stats([entry["mel_path"] for entry in index], labels["n_subclasses"],
                args.top_db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
