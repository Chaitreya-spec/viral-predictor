"""
Stage 4: build the LABEL for each video and join it to the features.

The label is NOT raw views (that just measures channel size). It's "lift" —
how much a video over/under-performed its own channel's typical views:

    lift = log10( (views + 1) / (channel_median_views + 1) )

    lift = 0   → performed exactly like the channel's normal video
    lift = 1   → 10x the channel's normal (a hit)
    lift < 0   → underperformed

Output: data/features/training_table.parquet  (features + lift + viral + channel_id)

Run:  python label.py
"""
import sqlite3
import numpy as np
import pandas as pd
import config

VIRAL_THRESHOLD = 0.7   # lift > 0.7 ≈ 5x the channel's normal → "viral"


def main():
    feats = pd.read_parquet(config.PARQUET)

    con = sqlite3.connect(config.DB_PATH)
    meta = pd.read_sql_query(
        "SELECT video_id, channel_id, views, channel_median_views FROM videos", con)
    con.close()

    df = feats.merge(meta, on="video_id", how="inner")
    df = df[df["channel_median_views"].notna() & (df["channel_median_views"] > 0)]

    df["lift"] = np.log10((df["views"] + 1) / (df["channel_median_views"] + 1))
    df["viral"] = (df["lift"] > VIRAL_THRESHOLD).astype(int)

    out = config.FEAT_DIR / "training_table.parquet"
    df.to_parquet(out)

    print(f"Labeled {len(df)} videos → {out}")
    print(f"  viral (lift > {VIRAL_THRESHOLD}): {df['viral'].sum()} "
          f"({100*df['viral'].mean():.1f}%)")
    print(f"  lift range: {df['lift'].min():.2f} to {df['lift'].max():.2f}")


if __name__ == "__main__":
    main()
