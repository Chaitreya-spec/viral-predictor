"""
Stage 5: train a LightGBM model to predict 'lift', and check it honestly.

Key correctness choices baked in:
- Embedding columns (clip_*, hook_clip_*, title_*) are reduced with PCA before
  the model sees them — a tree model chokes on 1000+ raw embedding dims.
- The split is BY CHANNEL (GroupKFold), so the model can't cheat by memorizing
  a creator's fingerprint from one video and reusing it on another.
- We compare against baselines. If the full model doesn't beat "predict the
  mean", the features aren't earning their keep (or there's just too little data).

Saves the model bundle to models/v1/.

Run:  python train.py
"""
import json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import config

MODEL_DIR = config.ROOT / "models" / "v1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# embedding column prefixes → how many PCA dims to keep for each block
EMBED_BLOCKS = {"clip_mean_": 16, "clip_std_": 16, "hook_clip_": 16, "titleemb_": 16}
# columns that are NOT features (ids, raw stats, and the labels themselves)
NON_FEATURE = {"video_id", "channel_id", "views", "lift", "viral"}


def build_features(df, pcas=None, fit=False):
    """Reduce embedding blocks with PCA and concat with scalar features.
    fit=True learns the PCAs + records the scalar column list (training);
    fit=False applies saved ones (serving), so train and serve stay identical.
    Scalar features are auto-detected: any feature column that isn't part of an
    embedding block — so newly added parameters are picked up automatically."""
    parts, names = [], []
    pcas = {} if fit else pcas

    for prefix, n_dims in EMBED_BLOCKS.items():
        cols = sorted([c for c in df.columns if c.startswith(prefix)],
                      key=lambda c: int(c.split("_")[-1]))
        block = df[cols].values
        k = min(n_dims, block.shape[0], block.shape[1])  # can't exceed n_samples/dims
        if fit:
            pcas[prefix] = PCA(n_components=k).fit(block)
        reduced = pcas[prefix].transform(block)
        parts.append(reduced)
        names += [f"{prefix}pca{i}" for i in range(reduced.shape[1])]

    if fit:
        scal = [c for c in df.columns
                if c not in NON_FEATURE
                and not any(c.startswith(p) for p in EMBED_BLOCKS)]
        pcas["_scalar_cols"] = scal
    else:
        scal = pcas["_scalar_cols"]

    # reindex so serving has exactly the training columns, in order (missing → 0)
    scal_df = df.reindex(columns=scal, fill_value=0.0)
    parts.append(scal_df.values)
    names += scal

    X = np.hstack(parts)
    return X, names, pcas


def main():
    df = pd.read_parquet(config.FEAT_DIR / "training_table.parquet")
    print(f"Training on {len(df)} videos, {df['channel_id'].nunique()} channels")

    if len(df) < 50:
        print("\n⚠️  WARNING: fewer than 50 videos. Results will be meaningless — "
              "this run only proves the code works. Scale up the data before "
              "trusting any number below.\n")

    y = df["lift"].values
    groups = df["channel_id"].values
    X, names, pcas = build_features(df, fit=True)

    # cross-validated, split by channel
    n_splits = min(5, df["channel_id"].nunique())
    gkf = GroupKFold(n_splits=max(n_splits, 2))
    preds = np.zeros_like(y)

    for tr, te in gkf.split(X, y, groups):
        params = dict(objective="regression", metric="l2", learning_rate=0.03,
                      num_leaves=31, min_data_in_leaf=5, feature_fraction=0.8,
                      verbose=-1)
        dtrain = lgb.Dataset(X[tr], label=y[tr])
        model = lgb.train(params, dtrain, num_boost_round=200)
        preds[te] = model.predict(X[te])

    # ── honest evaluation ──
    rho = spearmanr(preds, y).correlation
    mae = np.mean(np.abs(preds - y))
    baseline_mae = np.mean(np.abs(y.mean() - y))  # "just predict the average"

    print("\n─── Results (channel-split cross-validation) ───")
    print(f"  Spearman corr (higher=better, 0=useless): {rho:.3f}")
    print(f"  MAE:              {mae:.3f}")
    print(f"  MAE predict-mean: {baseline_mae:.3f}   (must beat this)")
    print(f"  {'✓ model beats baseline' if mae < baseline_mae else '✗ no better than guessing the average'}")

    # ── retrain on ALL data and save the bundle ──
    final = lgb.train(dict(objective="regression", metric="l2", learning_rate=0.03,
                           num_leaves=31, min_data_in_leaf=5, verbose=-1),
                      lgb.Dataset(X, label=y), num_boost_round=200)
    final.save_model(str(MODEL_DIR / "model.txt"))
    pickle.dump(pcas, open(MODEL_DIR / "pca.pkl", "wb"))
    json.dump(names, open(MODEL_DIR / "feature_names.json", "w"))
    json.dump({"spearman": rho, "mae": mae, "n": len(df)},
              open(MODEL_DIR / "metrics.json", "w"))
    print(f"\nSaved model bundle → {MODEL_DIR}")


if __name__ == "__main__":
    main()
