"""Explainable features of the visible directed graph, never account balances."""
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd


def structural_features(graph: nx.DiGraph, base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extend existing base features; return frame and metric diagnostics.

    Seed reachability includes the seed itself (a zero-hop path). HITS uses
    unweighted topology; PageRank remains the money-weighted base calculation.
    """
    frame = base.copy()
    report = {}
    start = perf_counter()
    values = nx.betweenness_centrality(graph, weight=None, normalized=True)
    frame["betweenness"] = frame.gid.map(values).astype(float)
    report["betweenness_seconds"] = perf_counter() - start
    report["betweenness_method"] = "exact directed unweighted, NetworkX normalized=True"

    start = perf_counter()
    hubs = authorities = dict.fromkeys(graph, 0.0)
    report["hits_status"] = "zero: no edges"
    if graph.number_of_edges():
        try:
            # Explicit unit weights; deterministic positive initial vector.
            topology = nx.DiGraph()
            topology.add_nodes_from(sorted(graph))
            topology.add_edges_from(sorted(graph.edges))
            with np.errstate(divide="ignore", invalid="ignore"):
                hubs, authorities = nx.hits(
                    topology, nstart=dict.fromkeys(topology, 1.0),
                    max_iter=1000, tol=1e-10, normalized=True,
                )
            for scores in (hubs, authorities):
                array = np.array(list(scores.values()))
                if not np.isfinite(array).all() or (array < -1e-8).any():
                    raise ValueError("nonfinite or negative HITS scores")
                # Remove numerical noise and enforce exact zero for isolates.
                for gid in scores:
                    scores[gid] = max(0.0, scores[gid]) if graph.degree(gid) else 0.0
                total = sum(scores.values())
                if total <= 0:
                    raise ValueError("zero HITS normalization")
                for gid in scores:
                    scores[gid] /= total
            report["hits_status"] = "computed: unweighted topology"
        except Exception as exc:
            # Optional solver failures must not discard the other features.
            hubs = authorities = dict.fromkeys(graph, np.nan)
            for gid in nx.isolates(graph):
                hubs[gid] = authorities[gid] = 0.0
            report["hits_status"] = f"unavailable: {type(exc).__name__}: {exc}"
    frame["hub_score"] = frame.gid.map(hubs).astype(float)
    frame["authority_score"] = frame.gid.map(authorities).astype(float)
    report["hits_seconds"] = perf_counter() - start

    component_id, component_size = {}, {}
    # Stable IDs ordered by smallest gid, including singleton isolates.
    for index, members in enumerate(sorted(nx.weakly_connected_components(graph), key=min)):
        component_id.update(dict.fromkeys(members, index))
        component_size.update(dict.fromkeys(members, len(members)))
    frame["weak_component_id"] = frame.gid.map(component_id).astype("int64")
    frame["weak_component_size"] = frame.gid.map(component_size).astype("int64")
    seeds = {gid for gid, attrs in graph.nodes(data=True) if attrs["is_seed"]}
    reach = dict.fromkeys(graph, 0)
    distances = dict.fromkeys(graph, np.nan)
    for seed in sorted(seeds):
        for gid, distance in nx.single_source_shortest_path_length(graph, seed).items():
            reach[gid] += 1
            if np.isnan(distances[gid]) or distance < distances[gid]:
                distances[gid] = distance
    for direction, neighbors in (("in", graph.predecessors), ("out", graph.successors)):
        frame[f"direct_seed_{direction}"] = pd.Series(
            [len(seeds.intersection(neighbors(g))) for g in frame.gid], index=frame.index, dtype="int64")
    frame["reachable_seed_count"] = frame.gid.map(reach).astype("int64")
    frame["min_seed_distance"] = frame.gid.map(distances).astype(float)
    mismatch = frame.min_seed_distance.notna() & frame.min_seed_distance.ne(frame.depth)
    report["n_seed_distance_depth_mismatch"] = int(mismatch.sum())
    report["n_unreachable_from_seeds"] = int(frame.min_seed_distance.isna().sum())

    for direction, neighbors in (("in", graph.predecessors), ("out", graph.successors)):
        largest, hhi = [], []
        for gid in frame.gid:
            amounts = [graph[n][gid]["sum_kzt"] if direction == "in"
                       else graph[gid][n]["sum_kzt"] for n in neighbors(gid)]
            total = sum(amounts)
            shares = np.array(amounts, dtype=float) / total if total > 0 else np.array([])
            largest.append(float(shares.max()) if shares.size else 0.0)
            hhi.append(float(np.square(shares).sum()))
        counterparty = "sender" if direction == "in" else "receiver"
        frame[f"largest_{direction}_{counterparty}_share"] = largest
        label = "incoming" if direction == "in" else "outgoing"
        frame[f"{label}_amount_hhi"] = hhi
        frame[f"{direction}_avg_tx_kzt"] = frame[f"{direction}_kzt"].div(
            frame[f"{direction}_tx"].where(frame[f"{direction}_tx"] > 0))
    frame["counterparty_balance"] = frame.out_deg - frame.in_deg
    frame["degree_ratio"] = frame.out_deg.div(frame.in_deg.where(frame.in_deg > 0))

    start = perf_counter()
    reverse = graph.reverse(copy=False)
    cycle_flags = []
    for gid in frame.gid:
        # v -> successor -> ... -> v has <=6 edges iff successor can reach
        # v in <=5 hops. BFS is polynomial; no cycle enumeration or counting.
        if graph.in_degree(gid) and graph.out_degree(gid):
            back = nx.single_source_shortest_path_length(reverse, gid, cutoff=5)
            cycle_flags.append(any(n in back for n in graph.successors(gid)))
        else:
            cycle_flags.append(False)
    frame["is_in_short_cycle"] = pd.Series(cycle_flags, index=frame.index, dtype=bool)
    report["cycle_analysis_executed"] = True
    report["cycle_method"] = "exact membership in directed cycles of length <=6, bounded reverse BFS"
    report["cycle_seconds"] = perf_counter() - start
    return frame, report
