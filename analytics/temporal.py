"""Calendar-day features of observed transactions; no intraday ordering.

Incoming visibility is incomplete, especially for seeds; depth-4 outgoing
activity may be invisible. Only July 2026 transfers >=5000 KZT are in scope.
Window ratios are heuristics, not evidence that the same money moved onward.
"""
import numpy as np
import pandas as pd


def window_outflow_ratio(incoming: pd.Series, outgoing: pd.Series, days: int) -> float:
    """Sum min(in_D, out_[D,D+days]) / sum(in_D), inclusive calendar dates.

    Overlapping incoming windows may reuse outgoing amounts. This measures
    temporal coincidence, not matched money or a conserved flow allocation.
    """
    total = incoming.sum()
    if total <= 0:
        return np.nan
    matched = 0.0
    for date, amount in incoming.items():
        window = outgoing.loc[(outgoing.index >= date)
                              & (outgoing.index <= date + pd.DateOffset(days=days))]
        matched += min(amount, window.sum())
    return float(np.clip(matched / total, 0.0, 1.0))


def temporal_features(transactions: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per supplied gid. Undefined ratios/active-day means are NaN.

    Seed window ratios are deliberately unavailable even with visible inflow.
    Dates are normalized before grouping; no transaction order is inferred.
    """
    frame = nodes[["gid"]].copy()
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx.date).dt.normalize()
    daily_tables = {}
    for direction, owner, counterparty in (("in", "dst", "src"), ("out", "src", "dst")):
        label = "incoming" if direction == "in" else "outgoing"
        daily = tx.groupby([owner, "date"]).agg(
            amount=("sum_kzt", "sum"), count=("sum_kzt", "size"),
            counterparties=(counterparty, "nunique"))
        daily_tables[direction] = {gid: group.droplevel(0).amount
                                   for gid, group in daily.groupby(level=0)}
        grouped = daily.groupby(level=0)
        mappings = {
            f"active_days_{direction}": (grouped.size(), True),
            f"{label}_daily_max_kzt": (grouped.amount.max(), False),
            f"{label}_daily_mean_kzt": (grouped.amount.mean(), False),
            f"max_{direction}_tx_per_day": (grouped["count"].max(), True),
        }
        if direction == "in":
            mappings["max_unique_in_senders_per_day"] = (grouped.counterparties.max(), True)
            mappings["days_with_3plus_in_senders"] = (
                daily.counterparties.ge(3).groupby(level=0).sum(), True)
        pairs = tx.groupby([owner, counterparty]).size()
        mappings[f"repeated_{direction}_counterparties"] = (pairs.gt(1).groupby(level=0).sum(), True)
        party = "from_single_sender" if direction == "in" else "to_single_receiver"
        mappings[f"max_transactions_{party}"] = (pairs.groupby(level=0).max(), True)
        for column, (values, is_count) in mappings.items():
            mapped = frame.gid.map(values)
            frame[column] = (mapped.fillna(0).astype("int64") if is_count else
                             mapped.astype(float) if "mean" in column else
                             mapped.fillna(0).astype(float))
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    for days, name in ((2, "short_window_outflow_ratio"), (0, "same_day_outflow_ratio")):
        frame[name] = pd.Series([
            np.nan if gid in seeds else window_outflow_ratio(
                daily_tables["in"].get(gid, empty), daily_tables["out"].get(gid, empty), days)
            for gid in frame.gid], index=frame.index, dtype=float)
    return frame
