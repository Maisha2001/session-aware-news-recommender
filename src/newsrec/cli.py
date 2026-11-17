import argparse
from datetime import datetime
from pathlib import Path
import json

from .baseline import evaluate
from .data import ingest


def main():
    parser = argparse.ArgumentParser(description="MIND chronological data and baseline evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("ingest")
    data.add_argument("--news", type=Path, required=True)
    data.add_argument("--behaviors", type=Path, required=True)
    data.add_argument("--database", type=Path, required=True)
    data.add_argument("--cutoff", type=datetime.fromisoformat, required=True)
    baseline = commands.add_parser("evaluate")
    baseline.add_argument("--database", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest(args.news, args.behaviors, args.database, args.cutoff)
    else:
        result = evaluate(args.database, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
