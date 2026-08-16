"""
Stage 1b: expand the dataset by pulling each known channel's FULL recent upload
history — hits AND flops — instead of only search-surfaced hits. This fixes the
collection bias that made ~68% of videos look 'viral'.

Very cheap on quota: ~3 units per channel (a search costs 100), so 260 channels
costs well under 1,000 of your 10,000 daily units — and can yield 5,000+ videos.

For each channel it also recomputes channel_median_views from the fuller sample
and updates every row for that channel, so the 'lift' label is consistent.

Run:  python collect_channels.py
      python collect_channels.py --per-channel 50
"""
import sqlite3, datetime as dt, time, argparse, statistics
from googleapiclient.discovery import build
import config

yt = build("youtube", "v3", developerKey=config.API_KEY)
NOW = dt.datetime.now(dt.timezone.utc)


def uploads_playlist(channel_id):
    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items")
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def playlist_video_ids(playlist_id, limit):
    ids, token = [], None
    while len(ids) < limit:
        resp = yt.playlistItems().list(
            part="contentDetails", playlistId=playlist_id,
            maxResults=50, pageToken=token).execute()
        ids += [i["contentDetails"]["videoId"] for i in resp["items"]]
        token = resp.get("nextPageToken")
        if not token:
            break
    return ids[:limit]


def fetch_details(video_ids):
    out = []
    for i in range(0, len(video_ids), 50):
        resp = yt.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i + 50])).execute()
        out += resp["items"]
    return out


def age_days(published_at):
    d = dt.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    return (NOW - d).days


def main(per_channel):
    con = sqlite3.connect(config.DB_PATH)
    # niche per channel, taken from what we already collected
    channels = con.execute(
        "SELECT channel_id, niche FROM videos GROUP BY channel_id").fetchall()
    print(f"Expanding {len(channels)} channels, up to {per_channel} videos each …\n")

    total_added = 0
    for n, (ch_id, niche) in enumerate(channels, 1):
        try:
            pl = uploads_playlist(ch_id)
            if not pl:
                continue
            vids = playlist_video_ids(pl, per_channel)
            details = fetch_details(vids)

            # keep only settled (old enough) videos with a real view count
            aged = []
            for v in details:
                if age_days(v["snippet"]["publishedAt"]) < config.MIN_AGE_DAYS:
                    continue
                if "viewCount" not in v.get("statistics", {}):
                    continue
                aged.append(v)
            if not aged:
                continue

            # recompute the channel baseline from this fuller sample
            views = [int(v["statistics"]["viewCount"]) for v in aged]
            median = int(statistics.median(views))

            for v in aged:
                con.execute("""
                    INSERT OR IGNORE INTO videos
                    (video_id, niche, channel_id, title, publish_date,
                     duration_iso, views, channel_median_views, thumb_url)
                    VALUES (?,?,?,?,?,?,?,?,?)""", (
                    v["id"], niche, ch_id,
                    v["snippet"]["title"],
                    v["snippet"]["publishedAt"],
                    v["contentDetails"]["duration"],
                    int(v["statistics"]["viewCount"]),
                    median,
                    v["snippet"]["thumbnails"]["high"]["url"],
                ))
            # keep the baseline consistent for older rows of this channel too
            con.execute("UPDATE videos SET channel_median_views=? WHERE channel_id=?",
                        (median, ch_id))
            con.commit()
            total_added += len(aged)
            print(f"  [{n}/{len(channels)}] {ch_id[:12]}… +{len(aged)} "
                  f"(median {median:,})")
        except Exception as e:
            print(f"  [{n}/{len(channels)}] {ch_id[:12]}… failed: {e}")
        time.sleep(0.2)

    count = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"\nDone. Added {total_added} rows. DB now has {count} unique videos.")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-channel", type=int, default=50)
    args = p.parse_args()
    main(args.per_channel)
