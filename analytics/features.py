"""Directed, observed-network features; no role or risk classification."""
import networkx as nx
import numpy as np
import pandas as pd

from utils.validation import NetworkData


def build_graph(data: NetworkData) -> nx.DiGraph:
    """Build from validated data, including every isolated client."""
    graph = nx.DiGraph()
    for row in data.nodes.sort_values("gid").itertuples(index=False):
        graph.add_node(row.gid, depth=int(row.depth), is_seed=bool(row.is_seed))
    for row in data.edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt),
                       n_tx=int(row.n_tx), depth=int(row.depth))
    return graph


def basic_features(graph: nx.DiGraph) -> pd.DataFrame:
    """One row per graph node, sorted by integer gid; isolates are retained."""
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError("basic_features requires a directed simple graph")
    gids = sorted(graph.nodes)
    frame = pd.DataFrame({
        "gid": pd.Series(gids, dtype="uint64" if gids and max(gids) > np.iinfo(np.int64).max else "int64"),
        "depth": pd.Series([graph.nodes[g]["depth"] for g in gids], dtype="int64"),
        "is_seed": pd.Series([graph.nodes[g]["is_seed"] for g in gids], dtype="bool"),
    })
    for prefix, degree in (("in", graph.in_degree), ("out", graph.out_degree)):
        for suffix, weight, dtype in (("deg", None, "int64"), ("kzt", "sum_kzt", "float64"),
                                       ("tx", "n_tx", "int64")):
            values = dict(degree(weight=weight))
            frame[f"{prefix}_{suffix}"] = frame.gid.map(values).astype(dtype)
    try:
        pagerank = nx.pagerank(graph, weight="sum_kzt", alpha=0.85, tol=1e-12, max_iter=1000)
    except nx.PowerIterationFailedConvergence as exc:
        raise ValueError("Weighted PageRank did not converge after 1000 iterations") from exc
    frame["pagerank"] = frame.gid.map(pagerank).astype(float)
    # Observed flow ratio only. Seed incoming flows are incomplete; never use this
    # naively for seed classification. Monthly balance does not prove fast transit.
    frame["pass_through"] = frame.out_kzt.div(frame.in_kzt.where(frame.in_kzt > 0))
    # The visible graph ends here; this is NOT a confirmed terminal recipient.
    frame["truncated_by_depth"] = frame.depth.eq(4) & frame.out_deg.eq(0)
    # Incident totals: a self-loop contributes once on each side (twice in total).
    frame["total_kzt"] = frame.in_kzt + frame.out_kzt
    frame["total_tx"] = frame.in_tx + frame.out_tx
    return frame


def network_diagnostics(data: NetworkData, graph: nx.DiGraph,
                        features: pd.DataFrame) -> dict[str, object]:
    isolates = list(nx.isolates(graph))
    dates = data.transactions["date"]
    return {
        "n_nodes": graph.number_of_nodes(), "n_edges": graph.number_of_edges(),
        "n_transactions": len(data.transactions), "n_seed": int(data.nodes.is_seed.sum()),
        "n_isolated": len(isolates),
        "n_isolated_seed": sum(bool(graph.nodes[g]["is_seed"]) for g in isolates),
        "n_weakly_connected_components": nx.number_weakly_connected_components(graph),
        "n_depth4_truncated": int(features.truncated_by_depth.sum()),
        "observed_edge_turnover_kzt": float(data.edges.sum_kzt.sum()),
        "transaction_date_min": None if dates.empty else dates.min().isoformat(),
        "transaction_date_max": None if dates.empty else dates.max().isoformat(),
    }
