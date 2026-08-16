"""
Internal helper — not run directly. predict.py calls this in a SEPARATE process
so PyTorch's OpenMP runtime never shares a process with LightGBM's (that combo
segfaults on macOS). Extracts features from one video, writes them to a JSON file.

Usage (called by predict.py):  python _extract_features.py <video> <title> <out.json>
"""
import sys, json
from extract import features_from_path

video_path, title, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
niche = sys.argv[4] if len(sys.argv) > 4 else ""
feat = features_from_path(video_path, title, niche)
with open(out_path, "w") as f:
    json.dump(feat, f)
