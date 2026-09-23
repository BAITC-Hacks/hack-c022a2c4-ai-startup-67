"""CLI for validated features and explainable observed-network roles."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from analytics.features import basic_features, build_graph, network_diagnostics
from analytics.graph_features import structural_features
from analytics.temporal import temporal_features
from analytics.roles import ROLE_CONFIG, classify_roles, role_diagnostics
from utils.validation import DataValidationError, load_data


def run_pipeline(data_dir: Path, out_dir: Path) -> dict[str, object]:
    start = perf_counter()
    data = load_data(data_dir)
    graph = build_graph(data)
    features = basic_features(graph)
    features, metric_report = structural_features(graph, features)
    temporal_start = perf_counter()
    features = features.merge(temporal_features(data.transactions, data.nodes),
                              on="gid", how="left", validate="one_to_one")
    temporal_seconds = perf_counter() - temporal_start
    if len(features) != len(data.nodes) or features.gid.duplicated().any():
        raise ValueError("Feature output must contain exactly one row per gid")
    if np.isinf(features.select_dtypes(include="number").to_numpy(dtype=float)).any():
        raise ValueError("Feature output contains infinity")
    role_start = perf_counter()
    features = classify_roles(features)
    role_seconds = perf_counter() - role_start
    diagnostics = network_diagnostics(data, graph, features)
    diagnostics.update(metric_report)
    diagnostics.update({
        "feature_shape": list(features.shape),
        "n_positive_betweenness": int(features.betweenness.gt(0).sum()),
        "n_reachable_from_multiple_seeds": int(features.reachable_seed_count.gt(1).sum()),
        "n_3plus_in_counterparties": int(features.in_deg.ge(3).sum()),
        "n_10plus_out_counterparties": int(features.out_deg.ge(10).sum()),
        "n_short_window_ratio_available": int(features.short_window_outflow_ratio.notna().sum()),
        "n_in_short_cycle": int(features.is_in_short_cycle.sum()),
        "temporal_seconds": temporal_seconds,
        "role_seconds": role_seconds,
        "roles": role_diagnostics(features),
        "role_config": ROLE_CONFIG,
        "n_transactions_below_5000_kzt": int(data.transactions.sum_kzt.lt(5000).sum()),
        "n_transactions_outside_july_2026": int((
            data.transactions.date.dt.year.ne(2026) | data.transactions.date.dt.month.ne(7)).sum()),
    })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "node_features.csv", index=False, na_rep="NaN")
    # Явные временные заглушки; кластеризация и приоритет ещё не реализованы.
    results = features.assign(cluster_id=-1, priority_score=0.0)
    required = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    results = results[required + [column for column in results if column not in required]]
    results.to_csv(out_dir / "nodes_roles.csv", index=False, na_rep="NaN")
    diagnostics["pipeline_seconds"] = perf_counter() - start
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
    print(f"Roles saved to {args.out / 'nodes_roles.csv'}")


if __name__ == "__main__":
    main()
