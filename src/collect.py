"""
Stage 1: collect video METADATA from the YouTube API into a SQLite DB.
No videos are downloaded here — this is cheap and decides what's worth keeping.

Run:  python collect.py
"""
import sqlite3, datetime as dt, time
from googleapiclient.discovery import build
import config

yt = build("youtube", "v3", developerKey=config.API_KEY)


def db():
    con = sqlite3.connect(config.DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            niche TEXT,
            channel_id TEXT,
            title TEXT,
            publish_date TEXT,
            duration_iso TEXT,
            views INTEGER,
            channel_median_views INTEGER,
            thumb_url TEXT,
            downloaded INTEGER DEFAULT 0
        )""")
    return con


# cache channel medians so we don't re-query the same channel repeatedly
_median_cache = {}

def channel_median_views(channel_id, sample=20):
    if channel_id in _median_cache:
        return _median_cache[channel_id]
    try:
        ch = yt.channels().list(part="contentDetails", id=channel_id).execute()
        uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        items = yt.playlistItems().list(
            part="contentDetails", playlistId=uploads, maxResults=sample).execute()
        vids = [i["contentDetails"]["videoId"] for i in items["items"]]
        stats = yt.videos().list(part="statistics", id=",".join(vids)).execute()
        views = sorted(int(s["statistics"].get("viewCount", 0)) for s in stats["items"])
        med = views[len(views) // 2] if views else None
    except Exception as e:
        print(f"  ! channel median failed for {channel_id}: {e}")
        med = None
    _median_cache[channel_id] = med
    return med


def search(query, niche, con):
    before = (dt.datetime.utcnow() - dt.timedelta(days=config.MIN_AGE_DAYS))
    resp = yt.search().list(
        q=query, part="id", type="video",
        maxResults=config.RESULTS_PER_QUERY,
        publishedBefore=before.isoformat("T") + "Z",
    ).execute()
    ids = [it["id"]["videoId"] for it in resp["items"]]
    if not ids:
        return 0

    stats = yt.videos().list(
        part="snippet,statistics,contentDetails", id=",".join(ids)).execute()

    added = 0
    for v in stats["items"]:
        ch_id = v["snippet"]["channelId"]
        con.execute("""
            INSERT OR IGNORE INTO videos
            (video_id, niche, channel_id, title, publish_date,
             duration_iso, views, channel_median_views, thumb_url)
            VALUES (?,?,?,?,?,?,?,?,?)""", (
            v["id"], niche, ch_id,
            v["snippet"]["title"],
            v["snippet"]["publishedAt"],
            v["contentDetails"]["duration"],
            int(v["statistics"].get("viewCount", 0)),
            channel_median_views(ch_id),
            v["snippet"]["thumbnails"]["high"]["url"],
        ))
        added += 1
    con.commit()
    return added


def main():
    if config.API_KEY == "PASTE_YOUR_KEY_HERE":
        raise SystemExit("Edit config.py and paste your YouTube API key first.")
    con = db()
    total = 0
    for niche, queries in config.NICHE_QUERIES.items():
        for q in queries:
            n = search(q, niche, con)
            total += n
            print(f"[{niche}] '{q}' → {n} videos")
            time.sleep(1)  # be polite to the API
    count = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"\nDone. {total} rows touched, {count} unique videos in DB.")
    con.close()


if __name__ == "__main__":
    main()
