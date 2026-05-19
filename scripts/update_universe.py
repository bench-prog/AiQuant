#!/usr/bin/env python3
"""Update smallcap universe JSON for SmallCapMomentumStrategy."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "ai_engine"))
from data_fetcher import fetch_coinpaprika_tickers, filter_smallcap_universe


def update_config_pair_whitelist(config_path: Path, pairs: list[str]) -> None:
    """Update pair_whitelist in the Freqtrade config file."""
    if not config_path.exists():
        print(f"Config file not found at {config_path}, skipping whitelist update.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    config.setdefault("exchange", {})["pair_whitelist"] = pairs

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Updated {len(pairs)} pairs in {config_path} pair_whitelist.")


def main():
    parser = argparse.ArgumentParser(description="Update smallcap universe JSON.")
    parser.add_argument(
        "--output",
        default="freqtrade/user_data/data/smallcap_universe.json",
        help="Path to output universe JSON file.",
    )
    parser.add_argument(
        "--config",
        default="freqtrade/config_smallcap.json",
        help="Path to Freqtrade config to update pair_whitelist.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-cap", type=float, default=500_000_000)
    parser.add_argument("--min-volume", type=float, default=10_000_000)
    parser.add_argument("--min-turnover", type=float, default=100.0)
    parser.add_argument("--max-turnover", type=float, default=500.0)
    args = parser.parse_args()

    df = fetch_coinpaprika_tickers()
    if df.empty:
        print("Failed to fetch CoinPaprika data.")
        sys.exit(1)

    universe = filter_smallcap_universe(
        df,
        max_market_cap=args.max_cap,
        min_volume=args.min_volume,
        min_turnover=args.min_turnover,
        max_turnover=args.max_turnover,
        top_n=args.top_n,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = universe[
        [
            "symbol",
            "binance_pair",
            "market_cap",
            "volume_24h",
            "turnover_rate",
            "percent_change_7d",
            "percent_change_24h",
            "first_data_at",
        ]
    ].to_dict(orient="records")

    payload = {
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "count": len(records),
        "coins": records,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Universe saved to {output_path} ({len(records)} coins)")

    # Update config pair_whitelist
    pairs = [r["binance_pair"] for r in records if "binance_pair" in r]
    update_config_pair_whitelist(Path(args.config), pairs)


if __name__ == "__main__":
    main()
