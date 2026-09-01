# python models/check_gen.py <subclass_tag> [--checkpoint PATH] [--steps N] [--n N] [--seed N]


import argparse
import json
import os
import sys

# Windows needs Developer Mode or admin rights to create the symlinks
# huggingface_hub's cache uses by default, so skip them instead
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import numpy as np
import torch

sys.path.insert(0, ".")

from models.cunet import ConditionalUNet
from models.diffusion import (DDIM_ETA, DDIM_STEPS, DDIMSampler, GaussianDiffusion,
                              NoiseSchedule, denormalize_mel)

LABELS_PATH = os.path.join("data", "processed", "labels.json")
STATS_PATH = os.path.join("data", "processed", "mel_stats.json")
CHECKPOINT_DEFAULT = os.path.join("models", "checkpoints", "cunet_epoch40.pt")
OUT_DIR = os.path.join("models", "generated")

SAMPLE_RATE = 22050
PEAK_LEVEL = 0.95

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


def main():
    parser = argparse.ArgumentParser(description="Sample a mel from the trained CU-Net "
                                                 "with DDIM, check it against its subclass "
                                                 "with CLAP, and vocode it with BigVGAN.")
    parser.add_argument("subclass", help="subclass tag, eg explosion_sub1 "
                                         "(see data/processed/labels.json)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--n", type=int, default=1, help="how many clips to generate")
    parser.add_argument("--steps", type=int, default=DDIM_STEPS)
    parser.add_argument("--eta", type=float, default=DDIM_ETA)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    labels = json.load(open(LABELS_PATH, encoding="utf-8"))
    if args.subclass not in labels["subclass_to_idx"]:
        print("unknown subclass %r, choose from:" % args.subclass, file=sys.stderr)
        for key in sorted(labels["subclass_to_idx"]):
            print("  %s" % key, file=sys.stderr)
        return 1

    stats = json.load(open(STATS_PATH, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

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
