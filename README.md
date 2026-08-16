# viral-predictor

Predicts whether a video will over- or under-perform **relative to its channel's
own baseline** — not raw views (which just measure channel size). Trained on
YouTube videos in three niches (mountain / racing / travel), using visual, audio,
title, and metadata features.

The label is **lift** = `log10(views / channel_median_views)`, so the model
learns what's *in the video*, not just how big the channel is.

## Setup

```bash
git clone <your-repo-url> viral-predictor
cd viral-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create your `.env` from the template and add your YouTube API key:

```bash
cp .env.example .env
# edit .env → YOUTUBE_API_KEY=...
```

Get a key: Google Cloud Console → enable **YouTube Data API v3** → Credentials → API key.

## Run order

All scripts live in `src/`. Run them from the project root:

```bash
python src/collect.py                # 1. metadata → data/metadata.db (cheap, no videos)
python src/download.py --limit 50    # 2. download videos (360p)
python src/extract.py --limit 50     # 3. videos → data/features/features.parquet
python src/label.py                  # 4. add the 'lift' label
python src/train.py                  # 5. train LightGBM → models/v1/
python src/check_labels.py           # (optional) inspect label balance from the DB

# score a new video:
python src/predict.py my_video.mp4 --title "I climbed the Alps" --niche mountain --channel-median 10000

# or launch the web app:
streamlit run src/app.py
```

Start with `--limit 50` to prove it works end-to-end, then drop the flag. Every
stage is resumable — rerun anytime, it skips finished work.

## Project layout

```
viral-predictor/
├── .env                 # your API key (gitignored — never committed)
├── .env.example         # template
├── requirements.txt
├── src/
│   ├── config.py            # settings + paths; loads .env
│   ├── collect.py           # search → video metadata (sqlite)
│   ├── collect_channels.py  # expand: full channel upload history (hits + flops)
│   ├── download.py          # metadata → 360p video files (yt-dlp)
│   ├── extract.py           # videos → features  (features_from_path = shared extractor)
│   ├── _extract_features.py # helper: extraction in a subprocess (avoids torch+lightgbm segfault)
│   ├── label.py             # views + baseline → lift label
│   ├── train.py             # LightGBM + PCA, channel-split cross-validation
│   ├── predict.py           # score a new video + plain-English reasons
│   ├── check_labels.py      # sanity-check label distribution
│   └── app.py               # Streamlit upload UI
├── data/                # gitignored (videos, thumbs, db, features)
├── models/              # trained model bundle (model.txt + pca.pkl + names)
└── tests/
```

## How it works (short version)

1. **Collect** video metadata + each channel's median views (the baseline).
2. **Download** the videos at 360p.
3. **Extract** features — CLIP embeddings (full video + first-3s hook), motion,
   scene-cut rate, colour, title stats, audio (loudness/tempo/silence/tone), niche.
4. **Label** each video with its `lift` vs the channel baseline.
5. **Train** LightGBM, evaluated with `GroupKFold` on `channel_id` so it can't
   cheat by memorising a creator's fingerprint.
6. **Predict** a new video by running the *same* extractor, then the model, then
   SHAP-style contributions for the "what drove this" explanation.

## Notes

- **Same extractor for train and serve** (`features_from_path`) — no train/serve skew.
- **`predict.py` extracts in a subprocess** because PyTorch + LightGBM loaded in
  one process segfault on macOS.
- Device auto-detects: MPS on Apple Silicon, CUDA on a cloud GPU — no code change.
