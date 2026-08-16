"""
Stage 2: download the actual video files for rows in the DB.
Resumable — skips anything already on disk. Small failures don't stop the batch.

Run:  python download.py
      python download.py --limit 50      (just grab 50, for a test run)
"""
import sqlite3, time, argparse
from pathlib import Path
import yt_dlp
import config


# Optional: use your browser's YouTube cookies to get past "confirm you're not a
# bot" blocks. To enable, add this line to config.py:  COOKIES_FROM_BROWSER = "chrome"
# (or "safari" / "firefox" / "edge" — whichever you're logged into YouTube on).
COOKIES_BROWSER = getattr(config, "COOKIES_FROM_BROWSER", None)


def already_have(video_id):
    return any(config.VIDEO_DIR.glob(f"{video_id}.*"))


def download_one(video_id):
    opts = {
        # prefer h264 mp4 (OpenCV reads it reliably); fall back if unavailable
        "format": ("worst[height>=360][ext=mp4][vcodec^=avc1]/"
                   "best[height<=480][ext=mp4]/best[ext=mp4]/best"),
        # route videos and thumbnails to separate folders so the extractor
        # never confuses a .webp thumbnail for a video file
        "outtmpl": {
            "default":   str(config.VIDEO_DIR / f"{video_id}.%(ext)s"),
            "thumbnail": str(config.THUMB_DIR / f"{video_id}.%(ext)s"),
        },
        "quiet": True, "noprogress": True, "retries": 3,
        "ignoreerrors": True,
        "writethumbnail": True,
    }
    if COOKIES_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_BROWSER,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        code = ydl.download([f"https://youtube.com/watch?v={video_id}"])
    return code == 0


def main(limit):
    con = sqlite3.connect(config.DB_PATH)
    # ORDER BY RANDOM() so a capped run pulls a representative mix of hits AND
    # flops across all channels, not just the first-inserted (search-hit) rows.
    rows = con.execute(
        "SELECT video_id FROM videos WHERE channel_median_views IS NOT NULL "
        "ORDER BY RANDOM()").fetchall()
    if limit:
        rows = rows[:limit]

    ok = fail = skip = 0
    for (vid,) in rows:
        if already_have(vid):
            skip += 1
            continue
        try:
            if download_one(vid):
                con.execute("UPDATE videos SET downloaded=1 WHERE video_id=?", (vid,))
                con.commit()
                ok += 1
                print(f"  ✓ {vid}")
            else:
                fail += 1
                print(f"  ✗ {vid} (unavailable)")
        except Exception as e:
            fail += 1
            print(f"  ✗ {vid}: {e}")
        time.sleep(1)  # rate-limit so we don't get throttled

    print(f"\nDone. {ok} downloaded, {skip} already had, {fail} failed.")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.limit)
