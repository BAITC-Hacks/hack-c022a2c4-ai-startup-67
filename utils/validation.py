"""Load and validate the observed transaction network without inferring roles."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

SCHEMAS = {
    "nodes": ("gid", "depth", "is_seed"),
    "edges": ("src", "dst", "sum_kzt", "n_tx", "depth"),
    "transactions": ("src", "dst", "date", "sum_kzt"),
}
# Absolute tolerance of one tiyn; no turnover-dependent relative allowance.
MONEY_ATOL = 0.01
MONEY_RTOL = 0.0


class DataValidationError(ValueError):
    """An input violates the documented data contract."""


@dataclass(frozen=True)
class NetworkData:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    transactions: pd.DataFrame


def _require_schema(frame: pd.DataFrame, name: str) -> None:
    if not frame.columns.is_unique:
        raise DataValidationError(f"{name}: duplicate column names")
    missing = sorted(set(SCHEMAS[name]) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{name}: missing required columns: {missing}")
    for column in SCHEMAS[name]:
        if frame[column].isna().any():
            raise DataValidationError(f"{name}.{column}: null values are not allowed")


def _integer(frame: pd.DataFrame, name: str, column: str) -> None:
    # Reject float identifiers, even integral-looking ones: precision may be lost.
    if not is_integer_dtype(frame[column].dtype) or is_bool_dtype(frame[column].dtype):
        raise DataValidationError(f"{name}.{column}: expected integer dtype, got {frame[column].dtype}")


def _money(frame: pd.DataFrame, name: str) -> None:
    values = frame["sum_kzt"]
    if (not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype)
            or np.iscomplexobj(values.to_numpy())):
        raise DataValidationError(f"{name}.sum_kzt: expected real numeric amounts")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
        raise DataValidationError(f"{name}.sum_kzt: amounts must be finite and nonnegative")


def validate_edge_aggregates(edges: pd.DataFrame, transactions: pd.DataFrame) -> None:
    """Require unique edges, identical pairs, equal counts and matching money."""
    if edges.duplicated(["src", "dst"]).any():
        rows = edges.loc[edges.duplicated(["src", "dst"], keep=False), ["src", "dst"]]
        raise DataValidationError(f"edges: duplicate directed pairs: {rows.head(10).to_dict('records')}")
    # Sorted amounts make floating aggregation reproducible under row permutations.
    ordered = transactions.sort_values(["src", "dst", "sum_kzt"], kind="stable")
    aggregated = ordered.groupby(["src", "dst"], as_index=False).agg(
        tx_sum_kzt=("sum_kzt", "sum"), tx_n_tx=("sum_kzt", "size")
    )
    comparison = edges.merge(aggregated, on=["src", "dst"], how="outer",
                             indicator=True, validate="one_to_one")
    missing = comparison["_merge"] != "both"
    if missing.any():
        columns = ["src", "dst", "sum_kzt", "tx_sum_kzt", "n_tx", "tx_n_tx", "_merge"]
        raise DataValidationError(
            "edges/transactions: directed pair mismatch (left_only=edge, right_only=transactions): "
            + str(comparison.loc[missing, columns].head(10).to_dict("records"))
        )
    issues = []
    for edge_column, aggregate_column, monetary in (
        ("sum_kzt", "tx_sum_kzt", True), ("n_tx", "tx_n_tx", False)
    ):
        left, right = comparison[edge_column], comparison[aggregate_column]
        matches = (np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                              rtol=MONEY_RTOL, atol=MONEY_ATOL) if monetary else left.eq(right))
        for row in comparison.loc[~matches].head(10).to_dict("records"):
            issues.append({"src": row["src"], "dst": row["dst"], "field": edge_column,
                           "edge_value": row[edge_column], "transaction_aggregate": row[aggregate_column],
                           "difference": row[edge_column] - row[aggregate_column]})
    if issues:
        raise DataValidationError(f"edges/transactions: aggregate mismatch (edge minus transactions): {issues}")


def validate_data(nodes: pd.DataFrame, edges: pd.DataFrame,
                  transactions: pd.DataFrame) -> NetworkData:
    """Validate before graph construction; return canonical copies, not mutations."""
    frames = {"nodes": nodes.copy(), "edges": edges.copy(), "transactions": transactions.copy()}
    for name, frame in frames.items():
        _require_schema(frame, name)
        for column in (("gid", "depth") if name == "nodes" else ("src", "dst")):
            _integer(frame, name, column)
    nodes, edges, transactions = (frames[name] for name in SCHEMAS)
    if nodes["gid"].duplicated().any():
        raise DataValidationError("nodes.gid: identifiers must be unique")
    if not nodes["depth"].between(0, 4).all():
        raise DataValidationError("nodes.depth: expected range 0..4")
    if not is_bool_dtype(nodes["is_seed"].dtype):
        raise DataValidationError("nodes.is_seed: expected boolean dtype")
    _integer(edges, "edges", "depth")
    _integer(edges, "edges", "n_tx")
    if not edges["depth"].between(1, 4).all():
        raise DataValidationError("edges.depth: expected range 1..4")
    if not (edges["n_tx"] > 0).all():
        raise DataValidationError("edges.n_tx: transaction counts must be positive")
    for name, frame in (("edges", edges), ("transactions", transactions)):
        _money(frame, name)
        for column in ("src", "dst"):
            unknown = frame.loc[~frame[column].isin(nodes["gid"]), column]
            if not unknown.empty:
                raise DataValidationError(f"{name}.{column}: unknown gids: {sorted(set(unknown))[:10]}")
    dates = transactions["date"]
    if is_numeric_dtype(dates.dtype) or is_bool_dtype(dates.dtype):
        raise DataValidationError("transactions.date: expected calendar dates, not numeric timestamps")
    try:
        transactions["date"] = pd.to_datetime(dates, errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError(f"transactions.date: invalid datetime: {exc}") from exc
    if transactions["date"].isna().any():
        raise DataValidationError("transactions.date: null/NaT values are not allowed")
    if not pd.api.types.is_datetime64_any_dtype(transactions["date"].dtype):
        raise DataValidationError("transactions.date: expected a consistent datetime timezone")
    validate_edge_aggregates(edges, transactions)
    return NetworkData(
        nodes.sort_values("gid", kind="stable").reset_index(drop=True),
        edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True),
        transactions.sort_values(["src", "dst", "date", "sum_kzt"], kind="stable").reset_index(drop=True),
    )


def load_data(data_dir: Path) -> NetworkData:
    """Load all required parquet files and return validated, deterministic data."""
    data_dir = Path(data_dir)
    paths = {name: data_dir / f"{name}.parquet" for name in SCHEMAS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DataValidationError(f"Missing required parquet files: {', '.join(missing)}")
    frames = {}
    for name, path in paths.items():
        try:
            frames[name] = pd.read_parquet(path, engine="pyarrow")
        except ImportError as exc:
            raise DataValidationError("Parquet support unavailable; install requirements.txt (including pyarrow)") from exc
        except Exception as exc:
            raise DataValidationError(f"Cannot read {path}: {exc}") from exc
    return validate_data(**frames)
