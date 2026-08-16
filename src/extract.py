"""
Stage 3: turn downloaded videos into feature rows → features.parquet.

- Models load ONCE (loading per-video would be ~100x slower).
- Each video's result is cached to data/features/cache/{id}.json, so a crash
  midway costs nothing — rerun and it resumes.
- Device auto-detects: MPS on your Mac now, CUDA on a cloud GPU later, no edits.

Run:  python extract.py
      python extract.py --limit 50
"""
import os
# let MPS fall back to CPU for the few ops it doesn't support, instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sqlite3, json, argparse, re, subprocess, tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import librosa
import pandas as pd
import config

# ─── device ───────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ─── load models once ─────────────────────────────────────────────────────
print("Loading CLIP …")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model.eval().to(DEVICE)

print("Loading text encoder …")
text_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


# ─── frame sampling ───────────────────────────────────────────────────────
MAX_FRAMES = 600     # cap very long videos (~10 min at 1fps) to bound memory/time
CLIP_BATCH = 32      # encode this many frames per GPU call, not all at once

def sample_frames(path, fps=1, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(src_fps / fps), 1)
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


@torch.no_grad()
def clip_embed(frames):
    """Mean + std pooled CLIP embedding, encoded in small batches to fit in GPU memory."""
    chunks = []
    for i in range(0, len(frames), CLIP_BATCH):
        batch = torch.stack(
            [preprocess(Image.fromarray(f)) for f in frames[i:i + CLIP_BATCH]]
        ).to(DEVICE)
        chunks.append(clip_model.encode_image(batch).float().cpu().numpy())
        if DEVICE == "mps":
            torch.mps.empty_cache()
    feats = np.concatenate(chunks, axis=0)
    return feats.mean(0), feats.std(0)


def motion_magnitude(frames):
    """Mean optical-flow magnitude — proxy for how much movement is on screen."""
    mags = []
    for a, b in zip(frames, frames[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(flow, axis=2).mean()))
    return float(np.mean(mags)) if mags else 0.0


def scene_cut_rate(frames, thresh=0.5):
    """Cuts per minute, via color-histogram difference between frames."""
    cuts = 0
    prev = None
    for f in frames:
        hist = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        if prev is not None:
            d = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > thresh:
                cuts += 1
        prev = hist
    minutes = max(len(frames) / 60.0, 1e-6)  # frames sampled at 1fps ≈ seconds
    return cuts / minutes


# ─── NEW parameter blocks (metadata/timing, title stats, colour) ────────────
NICHES = ["mountain", "racing", "travel"]
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F]")


def video_duration(path):
    """True length in seconds (before the MAX_FRAMES cap)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps else 0.0


def title_features(title):
    """Surface stats about the title text — cheap and known to move virality."""
    t = title or ""
    words = t.split()
    letters = [c for c in t if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    return {
        "title_len":         float(len(t)),
        "title_words":       float(len(words)),
        "title_has_number":  float(any(c.isdigit() for c in t)),
        "title_is_question": float("?" in t),
        "title_exclaim":     float(t.count("!")),
        "title_caps_ratio":  len(caps) / max(len(letters), 1),
        "title_hashtags":    float(t.count("#")),
        "title_emoji":       float(len(EMOJI_RE.findall(t))),
    }


def colour_stats(frames):
    """Mean/variation of brightness, saturation, contrast across the video."""
    step = max(len(frames) // 60, 1)       # subsample to bound cost
    bri, sat, con = [], [], []
    for f in frames[::step]:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        bri.append(float(f.mean()))
        sat.append(float(hsv[:, :, 1].mean()))
        con.append(float(f.std()))
    return {
        "brightness_mean": float(np.mean(bri)),
        "brightness_std":  float(np.std(bri)),
        "saturation_mean": float(np.mean(sat)),
        "contrast_mean":   float(np.mean(con)),
        "contrast_std":    float(np.std(con)),
    }


def niche_onehot(niche):
    return {f"niche_{n}": float(niche == n) for n in NICHES}


# ─── audio parameters (librosa; no heavy transcription) ─────────────────────
AUDIO_KEYS = ["audio_present", "audio_loudness", "audio_loudness_var",
              "audio_silence_ratio", "audio_tempo", "audio_brightness",
              "audio_noisiness"]

def _empty_audio():
    return {k: 0.0 for k in AUDIO_KEYS}


def audio_features(path):
    """Loudness, dynamics, silence, tempo and tone of the soundtrack.
    Extracts a downsampled mono wav via ffmpeg first (robust across codecs),
    then analyses the first 2 minutes. Returns zeros if there's no audio."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             tmp.name], capture_output=True, check=True)
        y, sr = librosa.load(tmp.name, sr=16000, duration=120)
    except Exception:
        return _empty_audio()
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    if y.size < sr:                       # less than 1s of audio
        return _empty_audio()

    rms  = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr  = librosa.feature.zero_crossing_rate(y)[0]
    try:
        tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])   # librosa ≥0.10
    except AttributeError:
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])      # older librosa

    return {
        "audio_present":        1.0,
        "audio_loudness":       float(rms.mean()),
        "audio_loudness_var":   float(rms.std()),
        "audio_silence_ratio":  float((rms < 0.02).mean()),
        "audio_tempo":          tempo,
        "audio_brightness":     float(cent.mean()),   # spectral centroid (treble-ness)
        "audio_noisiness":      float(zcr.mean()),     # zero-crossing rate
    }


# ─── per-video extraction ─────────────────────────────────────────────────
# only these are real video containers — never pick up a .webp/.jpg thumbnail
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

def find_video(video_id):
    for p in config.VIDEO_DIR.glob(f"{video_id}.*"):
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def features_from_path(path, title="", niche=None):
    """Extract the feature dict from a video file path.
    THE SINGLE SOURCE OF TRUTH for features — used by both training extraction
    (below) and serving (predict.py). Never write a second extraction path, or
    training and serving will silently diverge."""
    frames = sample_frames(path, fps=config.FRAME_FPS)
    if len(frames) < 2:
        raise ValueError("too few frames")
    hook = frames[:config.HOOK_SECONDS] or frames[:1]

    fmean, fstd = clip_embed(frames)
    hmean, _    = clip_embed(hook)
    tvec        = text_model.encode(title or "")
    dur         = video_duration(path)

    feat = {
        # visual dynamics
        "motion": motion_magnitude(frames),
        "hook_motion": motion_magnitude(hook),
        "scene_cut_rate": scene_cut_rate(frames),
        "n_frames": float(len(frames)),
        # metadata / timing (from the video itself → available at serve time too)
        "duration_sec": dur,
        "is_short": float(dur < 60),
    }
    feat.update(title_features(title))          # title_len, title_caps_ratio, …
    feat.update(colour_stats(frames))           # brightness_mean, saturation_mean, …
    feat.update(audio_features(path))           # loudness, tempo, silence, tone
    feat.update(niche_onehot(niche))            # niche_mountain / _racing / _travel
    feat.update({f"clip_mean_{i}": float(v) for i, v in enumerate(fmean)})
    feat.update({f"clip_std_{i}":  float(v) for i, v in enumerate(fstd)})
    feat.update({f"hook_clip_{i}": float(v) for i, v in enumerate(hmean)})
    feat.update({f"titleemb_{i}":  float(v) for i, v in enumerate(tvec)})
    return feat


def extract_one(video_id, title, niche=None):
    path = find_video(video_id)
    if path is None:
        raise FileNotFoundError("no video file")
    feat = features_from_path(path, title, niche)
    feat["video_id"] = video_id
    return feat


def main(limit):
    con = sqlite3.connect(config.DB_PATH)
    rows = con.execute(
        "SELECT video_id, title, niche FROM videos WHERE channel_median_views IS NOT NULL"
    ).fetchall()
    if limit:
        rows = rows[:limit]

    feats, done, fail = [], 0, 0
    for vid, title, niche in rows:
        cache = config.CACHE_DIR / f"{vid}.json"
        if cache.exists():
            feats.append(json.loads(cache.read_text()))
            continue
        try:
            f = extract_one(vid, title, niche)
            cache.write_text(json.dumps(f))
            feats.append(f)
            done += 1
            print(f"  ✓ {vid}  ({done})")
        except Exception as e:
            fail += 1
            print(f"  ✗ {vid}: {e}")

    if feats:
        pd.DataFrame(feats).to_parquet(config.PARQUET)
        print(f"\nWrote {len(feats)} rows → {config.PARQUET}")
    print(f"Newly extracted: {done},  failed: {fail}")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.limit)
