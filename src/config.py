"""
Central config. Edit this file — everything else reads from it.

The YouTube API key is loaded from a .env file at the project root (never
hard-coded here, so it's safe to push to GitHub). See .env.example.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths (auto-created) ─────────────────────────────────────────────────
# config.py lives in src/, so the project root is one level up.
ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data"
DB_PATH   = DATA / "metadata.db"
VIDEO_DIR = DATA / "videos"
THUMB_DIR = DATA / "thumbs"
FEAT_DIR  = DATA / "features"
CACHE_DIR = FEAT_DIR / "cache"
PARQUET   = FEAT_DIR / "features.parquet"

for d in (DATA, VIDEO_DIR, THUMB_DIR, FEAT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── YouTube API (from .env, not hard-coded) ──────────────────────────────
load_dotenv(ROOT / ".env")
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# Optional: browser to pull YouTube cookies from, to get past bot checks.
# Set in .env as COOKIES_FROM_BROWSER=chrome (or safari/firefox/edge), or leave unset.
COOKIES_FROM_BROWSER = os.environ.get("COOKIES_FROM_BROWSER") or None

# ─── What to collect ──────────────────────────────────────────────────────
NICHE_QUERIES = {
    "mountain": ["mountain hiking vlog", "mountain climbing", "alps trek"],
    "racing":   ["car racing", "track day", "drift compilation"],
    "travel":   ["travel vlog", "solo travel", "backpacking asia"],
}
RESULTS_PER_QUERY = 50   # videos per search term (50 = API max per call)
MIN_AGE_DAYS = 7         # only collect videos this old, so views have settled

# ─── Extraction settings ──────────────────────────────────────────────────
FRAME_FPS    = 1     # sample 1 frame per second of video
HOOK_SECONDS = 3     # first N seconds get their own feature block
