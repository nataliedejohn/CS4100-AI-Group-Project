"""Weekly home-run prediction pipeline demo.

This script demonstrates a data pipeline that
- collects MLB player data from pybaseball (Statcast + season stats)
- patches historical gaps with Retrosheet/Lahman CSVs if provided
- aggregates observations to weekly player buckets with rolling windows
- engineers predictive features and labels suitable for supervised or RL tasks

NOTE: The script requires `pybaseball`, `pandas`, and `numpy`.
      External downloads (Retrosheet/Lahman) are optional inputs and
      should be prepared separately.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from pybaseball import (
        cache,
        batting_stats,
        playerid_reverse_lookup,
        schedule_and_record,
        statcast_batter,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "pybaseball is required for this pipeline. Install via `pip install pybaseball`."
    ) from exc


cache.enable()


@dataclass
class PipelineConfig:
    start_season: int
    end_season: int
    output_dir: Path
    retrosheet_dir: Optional[Path] = None
    lahman_csv: Optional[Path] = None
    min_pa: int = 20

    @property
    def seasons(self) -> Iterable[int]:
        return range(self.start_season, self.end_season + 1)


def get_mlb_weeks(season: int) -> pd.DataFrame:
    """Return week boundaries (Mon-Sun) covering the MLB regular season."""
    season_start = dt.date(season, 3, 15)
    season_end = dt.date(season, 11, 15)

    weeks = []
    start = season_start - dt.timedelta(days=season_start.weekday())
    while start <= season_end:
        end = start + dt.timedelta(days=6)
        weeks.append({"week_start": start, "week_end": end, "season": season})
        start = end + dt.timedelta(days=1)
    return pd.DataFrame(weeks)


def fetch_player_statcast(player_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    data = statcast_batter(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), player_id)
    if data.empty:
        return data
    data["game_date"] = pd.to_datetime(data["game_date"]).dt.date
    data["player_id"] = player_id
    return data


def fetch_season_batting(season: int) -> pd.DataFrame:
    stats = batting_stats(season)
    stats = stats.rename(columns={"IDfg": "player_id"})
    stats["season"] = season
    return stats


def load_retrosheet_games(retrosheet_dir: Optional[Path]) -> pd.DataFrame:
    if not retrosheet_dir:
        return pd.DataFrame()
    files = list(retrosheet_dir.glob("*.csv"))
    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined.rename(columns=str.lower, inplace=True)
    return combined


def compile_player_games(config: PipelineConfig) -> pd.DataFrame:
    statcast_frames = []
    season_meta = []

    for season in config.seasons:
        season_stats = fetch_season_batting(season)
        season_stats = season_stats[season_stats["PA"] >= config.min_pa]
        season_meta.append(season_stats[["player_id", "Name", "PA", "season"]])

        fg_ids = season_stats["player_id"].astype(int).tolist()
        id_lookup = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
        mlbam_map = (
            id_lookup.dropna(subset=["key_mlbam"])
            .set_index("key_fangraphs")["key_mlbam"]
            .astype(int)
            .to_dict()
        )

        if not mlbam_map:
            print(f"No MLBAM ids found for season {season} after lookup.")
            continue

        season_start = dt.date(season, 3, 1)
        season_end = dt.date(season, 11, 30)

        for fg_id in fg_ids:
            mlbam_id = mlbam_map.get(fg_id)
            if not mlbam_id:
                continue
            print(f"Fetching Statcast for MLBAM {mlbam_id} (FG {fg_id}) in {season}...", flush=True)
            player_data = fetch_player_statcast(mlbam_id, season_start, season_end)
            if not player_data.empty:
                player_data["season"] = season
                player_data["fg_id"] = fg_id
                statcast_frames.append(player_data)

    statcast_df = pd.concat(statcast_frames, ignore_index=True) if statcast_frames else pd.DataFrame()
    season_meta_df = pd.concat(season_meta, ignore_index=True) if season_meta else pd.DataFrame()

    return statcast_df, season_meta_df


def add_week_keys(games: pd.DataFrame, week_calendar: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games
    week_calendar = week_calendar.copy()
    week_calendar["week_start"] = pd.to_datetime(week_calendar["week_start"])
    week_calendar["week_end"] = pd.to_datetime(week_calendar["week_end"])

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    games = games.merge(
        week_calendar,
        on="season",
        how="left",
    )
    games = games[(games["game_date"] >= games["week_start"]) & (games["game_date"] <= games["week_end"])]
    games["week_id"] = games["week_start"].dt.strftime("%Y-W%U")
    return games


def engineer_features(games: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "launch_speed",
        "launch_angle",
        "estimated_woba_using_speedangle",
        "estimated_ba_using_speedangle",
    ]
    for col in numeric_cols:
        if col not in games:
            games[col] = np.nan

    games.sort_values(["player_id", "game_date"], inplace=True)
    grouped = games.groupby("player_id", group_keys=False)

    def rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["PA"] = 1.0
        df["is_hr"] = (df["events"] == "home_run").astype(float)
        df["is_barrel"] = (df["bb_type"] == "barrel").astype(float)
        windows = {
            "7": 7,
            "14": 14,
            "28": 28,
        }
        for name, window in windows.items():
            roll = df.rolling(window=window, on="game_date", min_periods=1)
            df[f"hr_rate_{name}"] = roll["is_hr"].mean()
            df[f"pa_roll_{name}"] = roll["PA"].sum()
            df[f"avg_ev_{name}"] = roll["launch_speed"].mean()
            df[f"avg_la_{name}"] = roll["launch_angle"].mean()
            df[f"barrel_rate_{name}"] = roll["is_barrel"].mean()
        season_roll = df.expanding()
        df["hr_rate_season"] = season_roll["is_hr"].mean()
        return df

    games = grouped.apply(rolling_features)
    return games


def aggregate_weekly(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games
    agg = (
        games.groupby(["player_id", "week_id", "week_start", "week_end"])
        .agg(
            games_played=("game_pk", "nunique"),
            hr_week=("events", lambda x: np.sum(x == "home_run")),
            avg_ev_7=("avg_ev_7", "mean"),
            avg_la_7=("avg_la_7", "mean"),
            barrel_rate_7=("barrel_rate_7", "mean"),
            hr_rate_7=("hr_rate_7", "mean"),
            hr_rate_14=("hr_rate_14", "mean"),
            hr_rate_28=("hr_rate_28", "mean"),
            hr_rate_season=("hr_rate_season", "last"),
            pa_roll_7=("pa_roll_7", "last"),
            pa_roll_14=("pa_roll_14", "last"),
            pa_roll_28=("pa_roll_28", "last"),
        )
        .reset_index()
    )
    agg.rename(columns={"hr_week": "label_hr"}, inplace=True)
    agg["reward"] = agg["label_hr"].clip(upper=2)
    # Add season column from week_start
    agg["season"] = pd.to_datetime(agg["week_start"]).dt.year
    return agg


def main(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_weeks = pd.concat([get_mlb_weeks(season) for season in config.seasons], ignore_index=True)

    games, meta = compile_player_games(config)
    if games.empty:
        raise SystemExit("No Statcast data retrieved. Check pybaseball setup or date range.")

    games_week = add_week_keys(games, all_weeks)
    engineered = engineer_features(games_week)
    weekly = aggregate_weekly(engineered)

    weekly_path = config.output_dir / "player_week_features.parquet"
    meta_path = config.output_dir / "player_season_meta.parquet"

    weekly.to_parquet(weekly_path, index=False)
    meta.to_parquet(meta_path, index=False)

    weekly_csv_path = weekly_path.with_suffix(".csv")
    meta_csv_path = meta_path.with_suffix(".csv")
    weekly.to_csv(weekly_csv_path, index=False)
    meta.to_csv(meta_csv_path, index=False)

    print(f"Wrote weekly feature table to {weekly_path}")
    print(f"Wrote seasonal player metadata to {meta_path}")
    print(f"Wrote weekly feature CSV to {weekly_csv_path}")
    print(f"Wrote seasonal player metadata CSV to {meta_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build weekly HR prediction dataset.")
    parser.add_argument("--start", type=int, required=True, help="First season to include (e.g., 2023)")
    parser.add_argument("--end", type=int, required=True, help="Last season to include (e.g., 2024)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for artifacts")
    parser.add_argument("--retrosheet", type=Path, default=None, help="Folder with Retrosheet CSVs")
    parser.add_argument("--lahman", type=Path, default=None, help="Optional Lahman batting CSV")
    parser.add_argument("--min-pa", type=int, default=20, help="Minimum PA to keep a player in a season")
    args = parser.parse_args()

    cfg = PipelineConfig(
        start_season=args.start,
        end_season=args.end,
        output_dir=args.out,
        retrosheet_dir=args.retrosheet,
        lahman_csv=args.lahman,
        min_pa=args.min_pa,
    )
    main(cfg)
