"""
Quick sanity check on the label distribution, straight from the DB metadata —
no videos or features needed. Run this right after collect_channels.py to see
whether the collection-bias fix worked, BEFORE spending time downloading.

A healthy dataset: viral rate ~10-25% (not 68%), lift centered near 0.

Run:  python check_labels.py
"""
import sqlite3, numpy as np
import config

con = sqlite3.connect(config.DB_PATH)
rows = con.execute(
    "SELECT views, channel_median_views FROM videos "
    "WHERE channel_median_views IS NOT NULL AND channel_median_views > 0").fetchall()
con.close()

views = np.array([r[0] for r in rows], float)
med   = np.array([r[1] for r in rows], float)
lift  = np.log10((views + 1) / (med + 1))
viral = (lift > 0.7).mean()

print(f"videos with a label : {len(rows)}")
print(f"lift  mean / median : {lift.mean():+.2f} / {np.median(lift):+.2f}")
print(f"lift  range         : {lift.min():+.2f} to {lift.max():+.2f}")
print(f"viral rate (>0.7)   : {100*viral:.1f}%   (want ~10-25%, not ~68%)")
print()
print(f"under-performers (<-0.2): {100*(lift < -0.2).mean():.1f}%")
print(f"around average          : {100*((lift >= -0.2) & (lift <= 0.7)).mean():.1f}%")
print(f"over-performers  (>0.7) : {100*viral:.1f}%")
