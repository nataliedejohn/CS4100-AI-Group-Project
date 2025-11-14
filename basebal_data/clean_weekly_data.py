"""Clean and lag weekly MLB feature tables to avoid label leakage.


"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _lag_features(df: pd.DataFrame, cols: Iterable[str], group_key: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"prev_{col}"] = out.groupby(group_key)[col].shift(1)
    return out


def clean_weekly_features(weekly_path: Path, meta_path: Path, out_dir: Path) -> None:
    weekly = _load_table(weekly_path)
    meta = _load_table(meta_path) if meta_path and meta_path.exists() else pd.DataFrame()

    date_cols = [c for c in ("week_start", "week_end") if c in weekly]
    for col in date_cols:
        weekly[col] = pd.to_datetime(weekly[col])

    weekly.sort_values(["player_id", "week_start"], inplace=True)
    weekly.reset_index(drop=True, inplace=True)

    if "season" not in weekly:
        weekly["season"] = weekly["week_start"].dt.year

    feature_cols = [
        "hr_rate_7",
        "hr_rate_14",
        "hr_rate_28",
        "hr_rate_season",
        "avg_ev_7",
        "avg_la_7",
        "barrel_rate_7",
        "pa_roll_7",
        "pa_roll_14",
        "pa_roll_28",
        "games_played",
    ]
    feature_cols = [col for col in feature_cols if col in weekly.columns]
    weekly = _lag_features(weekly, feature_cols, "player_id")

    if {"prev_pa_roll_7", "prev_games_played"}.issubset(weekly.columns):
        weekly["prev_pa_per_game"] = weekly["prev_pa_roll_7"] / np.clip(weekly["prev_games_played"], 1, None)
        weekly["prev_pa_per_game"] = weekly["prev_pa_per_game"].replace([np.inf, -np.inf], np.nan)
    elif "prev_pa_per_game" in weekly.columns:
        weekly.drop(columns=["prev_pa_per_game"], inplace=True, errors="ignore")

    if not meta.empty:
        meta = meta.rename(columns={"PA": "season_total_pa"})
        join_cols = [c for c in ("player_id", "season") if c in meta.columns]
        if join_cols:
            weekly = weekly.merge(meta[join_cols + [c for c in meta.columns if c not in join_cols]], on=join_cols, how="left")

    lagged_cols = [f"prev_{c}" for c in feature_cols]
    key_cols = [col for col in lagged_cols if col in weekly.columns]
    weekly = weekly.dropna(subset=key_cols)

    keep_cols = [
        "player_id",
        "week_id" if "week_id" in weekly.columns else None,
        "week_start",
        "week_end" if "week_end" in weekly.columns else None,
        "season",
        "label_hr" if "label_hr" in weekly.columns else None,
        "reward" if "reward" in weekly.columns else None,
        "prev_games_played" if "prev_games_played" in weekly.columns else None,
        "prev_pa_roll_7" if "prev_pa_roll_7" in weekly.columns else None,
        "prev_pa_roll_14" if "prev_pa_roll_14" in weekly.columns else None,
        "prev_pa_roll_28" if "prev_pa_roll_28" in weekly.columns else None,
        "prev_pa_per_game" if "prev_pa_per_game" in weekly.columns else None,
    ] + key_cols
    keep_cols = [col for col in keep_cols if col and col in weekly.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    cleaned = weekly[keep_cols].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "player_week_features_clean.parquet"
    csv_path = out_dir / "player_week_features_clean.csv"

    cleaned.to_parquet(parquet_path, index=False)
    cleaned.to_csv(csv_path, index=False)

    print(f"Wrote cleaned weekly features to {parquet_path}")
    print(f"Wrote cleaned weekly features CSV to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lag and clean weekly MLB feature tables.")
    parser.add_argument("--weekly", type=Path, required=True, help="Path to raw weekly feature table (parquet or CSV).")
    parser.add_argument("--meta", type=Path, required=True, help="Path to player season metadata table (parquet or CSV).")
    parser.add_argument("--out", type=Path, required=True, help="Directory for cleaned outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_weekly_features(args.weekly, args.meta, args.out)
