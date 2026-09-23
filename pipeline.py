"""CLI for validated loading and base graph features only."""
import argparse
import json
from pathlib import Path

from analytics.features import basic_features, build_graph, network_diagnostics
from utils.validation import DataValidationError, load_data


def run_pipeline(data_dir: Path, out_dir: Path) -> dict[str, object]:
    data = load_data(data_dir)
    graph = build_graph(data)
    features = basic_features(graph)
    diagnostics = network_diagnostics(data, graph, features)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "node_features.csv", index=False, na_rep="NaN")
    (out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    try:
        report = run_pipeline(args.data, args.out)
    except (DataValidationError, ValueError, OSError) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Features saved to {args.out / 'node_features.csv'}")


if __name__ == "__main__":
    main()
