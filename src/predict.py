"""
Serving: score a NEW video file → will it beat its channel's baseline?

Feature extraction (PyTorch) runs in a SEPARATE process from scoring (LightGBM),
because loading both in one process segfaults on macOS. The extraction itself is
the exact same code as training (extract.features_from_path), so no train/serve skew.
Explanation uses LightGBM's built-in SHAP-style contributions — no extra library.

Run:
  python predict.py path/to/video.mp4 --title "My epic mountain climb" --channel-median 10000

--channel-median is your channel's typical view count. It matters: the model
predicts performance RELATIVE to it. If you don't know it, leave the default.
"""
import os, sys, json, pickle, argparse, subprocess, tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from train import build_features, MODEL_DIR      # note: train.py does NOT import torch

# group the many PCA'd embedding columns into human-readable buckets
GROUPS = {
    "clip_mean_": "overall visual content",
    "clip_std_":  "visual variety over time",
    "hook_clip_": "the opening shot (first 3s)",
    "titleemb_":  "title wording (meaning)",
    "niche_":     "niche",
}
FRIENDLY = {
    "motion":               "overall motion / action",
    "hook_motion":          "motion in the first 3 seconds",
    "scene_cut_rate":       "editing pace (cuts per minute)",
    "n_frames":             "sampled frames",
    "duration_sec":         "video length",
    "is_short":             "short-form (<60s)",
    "channel_median_views": "channel size",
    "title_len":            "title length",
    "title_words":          "title word count",
    "title_has_number":     "number in title",
    "title_is_question":    "title is a question",
    "title_exclaim":        "exclamation marks in title",
    "title_caps_ratio":     "ALL-CAPS in title",
    "title_hashtags":       "hashtags in title",
    "title_emoji":          "emojis in title",
    "brightness_mean":      "brightness",
    "brightness_std":       "brightness variation",
    "saturation_mean":      "colour saturation",
    "contrast_mean":        "contrast",
    "contrast_std":         "contrast variation",
    "audio_present":        "has audio",
    "audio_loudness":       "loudness",
    "audio_loudness_var":   "loudness dynamics",
    "audio_silence_ratio":  "amount of silence",
    "audio_tempo":          "audio tempo (BPM)",
    "audio_brightness":     "audio brightness (treble)",
    "audio_noisiness":      "audio noisiness",
}


def readable(feat_name):
    for prefix, label in GROUPS.items():
        if feat_name.startswith(prefix):
            return label
    return FRIENDLY.get(feat_name, feat_name)


def extract_via_subprocess(video_path, title, niche=""):
    """Run feature extraction in its own process, return the feature dict."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    print("Extracting features (separate process) …")
    helper = os.path.join(os.path.dirname(__file__), "_extract_features.py")
    subprocess.run(
        [sys.executable, helper, video_path, title, tmp.name, niche or ""],
        check=True,
    )
    feat = json.load(open(tmp.name))
    os.unlink(tmp.name)
    return feat


def score_video(video_path, title, channel_median, niche=""):
    """Score one video and return a structured result. Used by the CLI (main)
    and the Streamlit app. Extraction runs in a child process (no torch here)."""
    model = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    pcas  = pickle.load(open(MODEL_DIR / "pca.pkl", "rb"))
    names = json.load(open(MODEL_DIR / "feature_names.json"))

    feat = extract_via_subprocess(video_path, title, niche)
    feat["channel_median_views"] = channel_median
    df = pd.DataFrame([feat])
    X, _, _ = build_features(df, pcas=pcas, fit=False)

    lift = float(model.predict(X)[0])
    verdict = "LIKELY TO OVERPERFORM" if lift > 0.7 else \
              "AROUND AVERAGE" if lift > -0.2 else "LIKELY TO UNDERPERFORM"

    contribs = model.predict(X, pred_contrib=True)[0][:-1]  # drop base value
    bucket = {}
    for name, c in zip(names, contribs):
        bucket[readable(name)] = bucket.get(readable(name), 0.0) + c
    factors = sorted(bucket.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {"lift": lift, "multiplier": 10 ** lift, "verdict": verdict,
            "factors": factors}


def main(video_path, title, channel_median, niche=""):
    r = score_video(video_path, title, channel_median, niche)
    print("\n─── Prediction ───")
    print(f"  Title: {title or '(none)'}")
    print(f"  Predicted lift: {r['lift']:+.2f}  →  ~{r['multiplier']:.1f}x the channel's normal views")
    print(f"  Verdict: {r['verdict']}")
    print("\n─── What drove this (biggest factors) ───")
    for label, c in r["factors"][:5]:
        arrow = "↑ helps" if c > 0 else "↓ hurts"
        print(f"  {arrow:8s}  {label}   ({c:+.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video_path")
    p.add_argument("--title", default="")
    p.add_argument("--channel-median", type=float, default=10000,
                   help="your channel's typical view count")
    p.add_argument("--niche", default="", choices=["", "mountain", "racing", "travel"],
                   help="video niche (helps the model)")
    args = p.parse_args()
    main(args.video_path, args.title, args.channel_median, args.niche)
