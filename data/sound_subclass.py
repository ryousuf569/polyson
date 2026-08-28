#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "subclass")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
MANIFEST_PATH = os.path.join(RAW_DIR, "manifest.csv")

CLAP_MODEL = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000
BATCH_SIZE = 16

# Cluster counts to try for each class
K_RANGE = range(2, 9)

# Nameability we give up for a finer split, and the smallest usable cluster
NAMEABILITY_TOLERANCE = 0.015
MIN_CLUSTER_FRAC = 0.05

# Descriptions each cluster gets matched against to pick its name
LABEL_PROBES = [
    "a single sharp instant crack",
    "a long sustained shattering with debris",
    "a deep low boom or rumble",
    "a bright high-pitched metallic ring",
    "a rising whoosh that builds up",
    "a falling whoosh that fades away",
    "a synthetic electronic sound effect",
    "a dry organic recorded thud",
    "a short clean tap or click",
    "a noisy distorted blast",
    "repeated rhythmic steps",
    "a soft quiet muffled sound",
    "a reverberant sound in a large space",
    "a crunchy gritty texture",
]


# Decode an mp3 to mono float32 audio with ffmpeg
def decode_mp3(path):
    proc = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path,
                           "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
                          capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        return None
    return audio


# Load the CLAP model, downloading about 2GB the first time
def load_clap():
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL)
    model.eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    torch.set_grad_enabled(False)
    return model, processor


# Normalize rows so dot products come out as cosine similarities
def unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


# Turn a list of waveforms into normalized CLAP embeddings
def embed_audio(model, processor, audios):
    import torch

    inputs = processor(audio=audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    # transformers 5 returns an output object where older versions gave a tensor
    vecs = out["pooler_output"] if hasattr(out, "keys") else out
    return unit_rows(vecs.numpy().astype(np.float32))


# Turn text into CLAP embeddings that share the audio space
def embed_text(model, processor, texts):
    import torch

    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    vecs = out["pooler_output"] if hasattr(out, "keys") else out
    return unit_rows(vecs.numpy().astype(np.float32))


# Embed every clip in a class, reusing the cache when it matches
def embed_class(class_name, model_pair, force=False):
    cache_path = os.path.join(CACHE_DIR, "%s.npz" % class_name)
    class_dir = os.path.join(RAW_DIR, class_name)

    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(".mp3"))
    if not files:
        return None, []

    if os.path.exists(cache_path) and not force:
        cached = np.load(cache_path, allow_pickle=True)
        cached_ids = list(cached["sound_ids"])
        if cached_ids == [os.path.splitext(f)[0] for f in files]:
            print("%s: %d embeddings (cached)" % (class_name, len(cached_ids)))
            return cached["embeddings"], cached_ids
        print("%s: cache stale, recomputing" % class_name)

    model, processor = model_pair
    vectors = []
    sound_ids = []
    batch_audio = []
    batch_ids = []

    def flush():
        if not batch_audio:
            return
        vectors.append(embed_audio(model, processor, batch_audio))
        sound_ids.extend(batch_ids)
        batch_audio.clear()
        batch_ids.clear()

    for i, filename in enumerate(files, 1):
        audio = decode_mp3(os.path.join(class_dir, filename))
        if audio is None:
            print("  skipped (decode failed): %s" % filename, file=sys.stderr)
            continue
        batch_audio.append(audio)
        batch_ids.append(os.path.splitext(filename)[0])
        if len(batch_audio) >= BATCH_SIZE:
            flush()
            print("%s: embedded %d/%d" % (class_name, len(sound_ids), len(files)))
    flush()

    if not sound_ids:
        return None, []

    embeddings = np.vstack(vectors)
    print("%s: embedded %d/%d" % (class_name, len(sound_ids), len(files)))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, sound_ids=np.array(sound_ids))
    return embeddings, sound_ids


# Pick the cluster count whose clusters match the text probes best
def choose_clusters(embeddings, k_range, probe_vectors):
    """Scored by text probes rather than silhouette, on purpose.

    Silhouette picks k=2 for glass, which buries sharp breaks and long
    shatters in one cluster. That is the varied training data we are trying
    to get away from. Loosen the threshold enough to escape k=2 and it runs
    to k=8, where clusters no longer match any description. CLAP shares an
    audio and text space, so asking how well a cluster matches a written
    description is a more direct test of whether it is one real sound.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    candidates = []
    for k in k_range:
        if k >= n:
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue

        sizes = np.bincount(labels)
        smallest = sizes.min() / float(n)
        if smallest < MIN_CLUSTER_FRAC:
            print("  k=%d smallest=%.0f%%  (too fragmented, skipped)"
                  % (k, 100 * smallest))
            continue

        centroids = unit_rows(np.vstack([embeddings[labels == c].mean(axis=0)
                                         for c in range(k)]))
        # Size weighted so one tiny well named cluster cannot carry a bad k
        best_per_cluster = (centroids @ probe_vectors.T).max(axis=1)
        nameability = float(np.average(best_per_cluster, weights=sizes))
        sil = silhouette_score(embeddings, labels, metric="cosine")
        print("  k=%d nameability=%.3f silhouette=%.3f smallest=%.0f%%"
              % (k, nameability, sil, 100 * smallest))
        candidates.append((labels, k, sil, nameability))

    if not candidates:
        return np.zeros(n, dtype=int), 1, 0.0

    best_name = max(c[3] for c in candidates)
    keep = [c for c in candidates if c[3] >= best_name - NAMEABILITY_TOLERANCE]
    chosen = max(keep, key=lambda c: c[1])
    print("  -> k=%d (nameability %.3f)" % (chosen[1], chosen[3]))
    return chosen[0], chosen[1], chosen[2]


# Name each cluster with the closest text probe, no two clusters sharing one
def name_clusters(embeddings, labels, probe_vectors, probes):
    cluster_ids = sorted(set(labels))
    centroids = unit_rows(np.vstack([embeddings[labels == c].mean(axis=0)
                                     for c in cluster_ids]))
    sims = centroids @ probe_vectors.T

    names = {}
    used = set()
    # Confident clusters claim their best name before the weaker ones
    order = sorted(range(len(cluster_ids)), key=lambda i: -sims[i].max())
    for i in order:
        for probe_idx in np.argsort(-sims[i]):
            if probe_idx not in used:
                used.add(probe_idx)
                names[cluster_ids[i]] = (probes[probe_idx], float(sims[i][probe_idx]))
                break
    return names


# Project embeddings to 2D with UMAP, falling back to PCA on failure
def project_2d(embeddings):
    n = len(embeddings)
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=min(15, max(2, n - 1)), min_dist=0.1,
                            metric="cosine", random_state=42)
        return reducer.fit_transform(embeddings), "UMAP"
    except Exception as exc:
        print("  UMAP failed (%s), falling back to PCA" % exc, file=sys.stderr)
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=0).fit_transform(embeddings), "PCA"


# Scatter one class in 2D with the biggest subclass highlighted
def plot_class(class_name, coords, labels, names, method, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cluster_ids = sorted(set(labels))
    counts = {c: int((labels == c).sum()) for c in cluster_ids}
    biggest = max(counts, key=counts.get)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, cluster in enumerate(cluster_ids):
        mask = labels == cluster
        label, score = names[cluster]
        is_top = cluster == biggest
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=64 if is_top else 30,
                   color=cmap(i % 10),
                   alpha=0.85 if is_top else 0.55,
                   edgecolors="black" if is_top else "none",
                   linewidths=1.0 if is_top else 0,
                   label="%s%s  n=%d  (%s, sim %.2f)"
                         % ("* " if is_top else "", "sub%d" % cluster,
                            counts[cluster], label, score))

    ax.set_title("%s - %d clips, %d subclasses (%s)\n"
                 "largest: sub%d with %d clips (%.0f%%) - %s"
                 % (class_name, len(labels), len(cluster_ids), method, biggest,
                    counts[biggest], 100.0 * counts[biggest] / len(labels),
                    names[biggest][0]),
                 fontsize=12)
    ax.set_xlabel("%s dim 1" % method)
    ax.set_ylabel("%s dim 2" % method)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Bar chart comparing subclass sizes across all classes
def plot_overview(summary, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(summary), figsize=(4.2 * len(summary), 5.5))
    if len(summary) == 1:
        axes = [axes]

    for ax, entry in zip(axes, summary):
        counts = entry["counts"]
        clusters = sorted(counts, key=lambda c: -counts[c])
        top = clusters[0]
        colors = ["#d62728" if c == top else "#aab" for c in clusters]
        ax.bar(range(len(clusters)), [counts[c] for c in clusters], color=colors)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(["sub%d" % c for c in clusters], fontsize=8)
        ax.set_title("%s\nbest: sub%d (n=%d)" % (entry["class"], top, counts[top]),
                     fontsize=11)
        ax.set_ylabel("clips")
        ax.grid(axis="y", alpha=0.15)

    fig.suptitle("Largest subclass per class (red = most samples)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Map each sound id to its clip name from the manifest
def load_manifest_names():
    names = {}
    if not os.path.exists(MANIFEST_PATH):
        return names
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names[str(row.get("sound_id"))] = row.get("name", "")
    return names


# Run the whole pipeline and report the biggest subclass per class
def main():
    parser = argparse.ArgumentParser(description="Discover acoustic subclasses "
                                                 "within each sound class.")
    parser.add_argument("--classes", default=None, help="comma-separated class names (default: all)")
    parser.add_argument("--k", type=int, default=None, help="fixed subclass count (default: auto)")
    parser.add_argument("--force", action="store_true", help="recompute embeddings, ignore cache")
    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        print("no %s - run pull_freesound.py first" % RAW_DIR, file=sys.stderr)
        return 1

    available = sorted(d for d in os.listdir(RAW_DIR)
                       if os.path.isdir(os.path.join(RAW_DIR, d)))
    if args.classes:
        selected = [c.strip() for c in args.classes.split(",") if c.strip()]
        unknown = [c for c in selected if c not in available]
        if unknown:
            print("unknown class(es): %s (available: %s)"
                  % (", ".join(unknown), ", ".join(available)),
                  file=sys.stderr)
            return 1
    else:
        selected = available

    # Warn about empty class folders instead of quietly skipping them
    nonempty = []
    for class_name in selected:
        path = os.path.join(RAW_DIR, class_name)
        if any(f.lower().endswith(".mp3") for f in os.listdir(path)):
            nonempty.append(class_name)
        else:
            print("%s: no mp3 files, skipping" % class_name, file=sys.stderr)
    if not nonempty:
        print("no audio to analyze", file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("loading CLAP (%s)..." % CLAP_MODEL)
    model_pair = load_clap()
    probe_vectors = embed_text(model_pair[0], model_pair[1], LABEL_PROBES)

    manifest_names = load_manifest_names()
    summary = []
    rows = []

    for class_name in nonempty:
        print("\n=== %s ===" % class_name)
        embeddings, sound_ids = embed_class(class_name, model_pair, force=args.force)
        if embeddings is None or len(embeddings) < 2:
            print("%s: not enough clips to cluster" % class_name, file=sys.stderr)
            continue

        k_range = [args.k] if args.k else K_RANGE
        labels, k, score = choose_clusters(embeddings, k_range, probe_vectors)
        names = name_clusters(embeddings, labels, probe_vectors, LABEL_PROBES)
        counts = {c: int((labels == c).sum()) for c in sorted(set(labels))}
        biggest = max(counts, key=counts.get)

        print("%s: %d subclasses (silhouette %.3f), largest sub%d with %d/%d clips"
              % (class_name, k, score, biggest, counts[biggest], len(labels)))
        for cluster in sorted(counts, key=lambda c: -counts[c]):
            label, sim = names[cluster]
            print("  sub%d  n=%-4d %-38s sim=%.2f%s"
                  % (cluster, counts[cluster], label, sim,
                     "   <-- most samples" if cluster == biggest else ""))

        coords, method = project_2d(embeddings)
        plot_path = os.path.join(OUT_DIR, "%s_umap.png" % class_name)
        plot_class(class_name, coords, labels, names, method, plot_path)
        print("  wrote %s" % plot_path)

        # Ranks how typical each clip is so the best ones are easy to pull
        centroids = {c: unit_rows(embeddings[labels == c].mean(axis=0)[None, :])[0]
                     for c in counts}
        for idx, sound_id in enumerate(sound_ids):
            cluster = int(labels[idx])
            rows.append({
                "class": class_name,
                "sound_id": sound_id,
                "name": manifest_names.get(sound_id, ""),
                "subclass": cluster,
                "subclass_label": names[cluster][0],
                "subclass_size": counts[cluster],
                "is_largest_subclass": int(cluster == biggest),
                "similarity_to_center": round(float(embeddings[idx]
                                                    @ centroids[cluster]), 4),
                "umap_x": round(float(coords[idx, 0]), 4),
                "umap_y": round(float(coords[idx, 1]), 4),
            })

        summary.append({
            "class": class_name,
            "counts": counts,
            "biggest": biggest,
            "label": names[biggest][0],
            "total": len(labels),
            "k": k,
            "silhouette": score,})

    if not summary:
        print("nothing analyzed", file=sys.stderr)
        return 1

    csv_path = os.path.join(OUT_DIR, "subclasses.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    overview_path = os.path.join(OUT_DIR, "overview.png")
    plot_overview(summary, overview_path)

    print("\n" + "=" * 62)
    print("LARGEST SUBCLASS PER CLASS")
    print("=" * 62)
    for entry in sorted(summary, key=lambda e: -e["counts"][e["biggest"]]):
        count = entry["counts"][entry["biggest"]]
        print("%-10s sub%d  %3d/%d clips (%2.0f%%)  %s"
              % (entry["class"], entry["biggest"], count, entry["total"],
                 100.0 * count / entry["total"], entry["label"]))

    best = max(summary, key=lambda e: e["counts"][e["biggest"]])
    print("\nmost single-character training data: %s sub%d with %d clips (%s)"
          % (best["class"], best["biggest"], best["counts"][best["biggest"]],
             best["label"]))
    print("\nwrote %s" % csv_path)
    print("wrote %s" % overview_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
