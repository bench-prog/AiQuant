#!/usr/bin/env python3
"""
Independent script to read drift alerts and send Telegram notifications.

Usage:
    cd research
    python alert_cli.py --test          # Send a test message
    python alert_cli.py --check --limit 5  # Send last 5 alerts
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_telegram_config(config_path: str) -> tuple[str, str] | None:
    """Load token and chat_id from Freqtrade config."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        return None

    telegram_cfg = config.get("telegram", {})
    if not telegram_cfg.get("enabled"):
        logger.warning("Telegram is disabled in config.")
        return None
    token = telegram_cfg.get("token")
    chat_id = telegram_cfg.get("chat_id")
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id missing in config.")
        return None
    return token, chat_id


def send_message(token: str, chat_id: str, text: str, dry_run: bool = False) -> bool:
    """Send a message via Telegram Bot API."""
    if dry_run:
        logger.info(f"[DRY-RUN] Would send Telegram message:\n{text}")
        return True

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    req = Request(url, data=data, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                logger.info("Telegram message sent successfully.")
                return True
            else:
                logger.warning(f"Telegram API returned status {resp.status}")
                return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def read_alerts(log_file: str, limit: int = 10) -> list[dict]:
    """Read last N alerts from JSONL file."""
    path = Path(log_file)
    if not path.exists():
        logger.warning(f"Alert log file not found: {log_file}")
        return []

    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    alerts = []
    for line in lines[-limit:]:
        try:
            alerts.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return alerts


def main() -> int:
    parser = argparse.ArgumentParser(description="AiQuant Drift Telegram Alert Tool")
    parser.add_argument("--config", default="../freqtrade/config_ai_model.json", help="Path to Freqtrade config_ai_model.json")
    parser.add_argument("--log-file", default="../freqtrade/user_data/logs/drift_alerts.jsonl", help="Path to drift alerts JSONL")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent alerts to send (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print message instead of sending")
    parser.add_argument("--check", action="store_true", help="Read drift_alerts.jsonl and send recent alerts")
    parser.add_argument("--test", action="store_true", help="Send a test message")
    args = parser.parse_args()

    if not args.check and not args.test:
        parser.print_help()
        return 1

    creds = load_telegram_config(args.config)
    if creds is None and not args.dry_run:
        return 1

    token, chat_id = creds if creds else ("", "")

    if args.test:
        text = (
            "✅ <b>AiQuant Drift Alert Test</b>\n"
            "This is a test message from alert_cli.py.\n"
            "Your drift alert pipeline is working!"
        )
        ok = send_message(token, chat_id, text, dry_run=args.dry_run)
        return 0 if ok else 1

    if args.check:
        alerts = read_alerts(args.log_file, limit=args.limit)
        if not alerts:
            logger.info("No alerts found to send.")
            return 0

        sent = 0
        for alert in alerts:
            text = alert.get("message", str(alert))
            if send_message(token, chat_id, text, dry_run=args.dry_run):
                sent += 1
        logger.info(f"Sent {sent}/{len(alerts)} alerts.")
        return 0 if sent == len(alerts) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
