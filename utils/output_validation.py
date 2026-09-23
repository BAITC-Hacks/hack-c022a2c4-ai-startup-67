"""Проверка итоговых CSV и связей между файлами, включая чтение с диска."""
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.roles import ALLOWED_ROLES, validate_role_output
from analytics.priority import PRIORITY_CONFIG, PRIORITY_COMPONENTS, build_top_nodes
from analytics.clustering import CLUSTER_CONFIG

NODE_FIRST = ['gid', 'role', 'role_score', 'cluster_id', 'priority_score', 'evidence']
CLUSTER_FIRST = ['cluster_id', 'n_nodes', 'n_seed', 'sum_kzt_internal', 'top_gids', 'hypothesis']
TOP_FIRST = ['rank', 'gid', 'role', 'priority_score', 'why']


def require(condition, message):
    if not condition:
        raise ValueError(f'Output validation: {message}')


def validate_outputs(nodes, clusters, top, graph, priority_config=None, cluster_config=None):
    """Валидирует три итоговые таблицы против исходного направленного графа."""
    priority_config = PRIORITY_CONFIG if priority_config is None else priority_config
    cluster_config = CLUSTER_CONFIG if cluster_config is None else cluster_config
    for frame, first in ((nodes, NODE_FIRST), (clusters, CLUSTER_FIRST), (top, TOP_FIRST)):
        require(frame.columns.is_unique and frame.columns[:len(first)].tolist() == first, 'incorrect required column order')
    validate_role_output(nodes, pd.Series(list(graph), dtype=nodes.gid.dtype))
    for frame in (nodes, clusters):
        require(pd.api.types.is_integer_dtype(frame.cluster_id.dtype)
                and frame.cluster_id.notna().all() and frame.cluster_id.ge(0).all(), 'invalid cluster_id')
    require(clusters.cluster_id.is_unique, 'duplicate cluster_id')
    require(set(nodes.cluster_id) == set(clusters.cluster_id), 'node/cluster ID mismatch')
    require(set(clusters.cluster_id) == set(range(len(clusters))), 'cluster IDs must be contiguous')
    for column in ['priority_score', *[f'priority_{part}' for part in PRIORITY_COMPONENTS]]:
        require(nodes[column].between(0, 1).all(), f'{column} outside [0,1] or missing')
    weighted = sum(nodes[f'priority_{part}'] * weight for part, weight in priority_config['weights'].items())
    require(np.allclose(weighted, nodes.priority_score, atol=1e-12, rtol=0), 'priority formula mismatch')
    require(nodes.priority_why.map(lambda v: isinstance(v, str) and bool(v.strip())).all(), 'empty node priority explanation')
    for name in ('n_nodes', 'n_seed'):
        require(pd.api.types.is_integer_dtype(clusters[name].dtype) and clusters[name].ge(0).all(), f'invalid {name}')
    require(clusters.n_nodes.sum() == len(nodes) and clusters.n_nodes.gt(0).all(), 'cluster node totals mismatch')
    require(clusters.hypothesis.map(lambda v: isinstance(v, str) and bool(v.strip())).all(), 'empty hypothesis')
    require(np.isfinite(clusters.sum_kzt_internal).all() and clusters.sum_kzt_internal.ge(0).all(), 'invalid internal turnover')
    assignment = dict(zip(nodes.gid, nodes.cluster_id))
    amounts = {cid: [] for cid in clusters.cluster_id}
    for src, dst, attrs in graph.edges(data=True):
        if assignment[src] == assignment[dst]:
            amounts[assignment[src]].append(float(attrs['sum_kzt']))
    groups = {cid: group for cid, group in nodes.groupby('cluster_id')}
    for row in clusters.itertuples():
        group = groups[row.cluster_id]
        require(row.n_nodes == len(group), 'cluster size mismatch')
        require(row.n_seed == sum(bool(graph.nodes[g]['is_seed']) for g in group.gid), 'cluster seed count mismatch')
        require(np.isclose(row.sum_kzt_internal, fsum(amounts[row.cluster_id]), atol=.01, rtol=0), 'directed internal turnover mismatch')
        try:
            ids = [int(value) for value in row.top_gids.split(',')]
        except (ValueError, AttributeError):
            raise ValueError('Output validation: malformed top_gids') from None
        expected = group.sort_values(['betweenness', 'pagerank', 'gid'], ascending=[False, False, True]).head(cluster_config['top_n']).gid.tolist()
        require(ids == expected, 'top_gids membership or structural order mismatch')
        for role in ALLOWED_ROLES:
            require(getattr(row, f'n_{role}') == int(group.role.eq(role).sum()), 'cluster role count mismatch')
    expected = build_top_nodes(nodes, priority_config['top_n'])
    require(len(top) == len(expected), 'wrong number of top nodes')
    require(top.gid.is_unique and top.gid.tolist() == expected.gid.tolist(), 'top gids or deterministic sorting mismatch')
    require(pd.api.types.is_integer_dtype(top['rank'].dtype) and top['rank'].tolist() == list(range(1, len(top) + 1)), 'ranks must start at 1 and be sequential')
    require(top.role.tolist() == expected.role.tolist(), 'top role mismatch')
    require(top.priority_score.between(0, 1).all() and top.priority_score.is_monotonic_decreasing, 'invalid top scores')
    require(np.allclose(top.priority_score, expected.priority_score, atol=1e-12, rtol=0), 'top score mismatch')
    require(top.why.map(lambda v: isinstance(v, str) and bool(v.strip())).all(), 'empty priority explanation')
    require(top.why.tolist() == expected.why.tolist(), 'top explanation mismatch')
    return {'valid': True, 'n_nodes': len(nodes), 'n_clusters': len(clusters), 'n_top_nodes': len(top)}


def validate_output_files(out_dir, graph, priority_config=None, cluster_config=None):
    """Читает именно записанные CSV; крупные gid никогда не проходят через float."""
    root = Path(out_dir)
    gid_dtype = 'uint64' if graph and max(graph) > np.iinfo(np.int64).max else 'int64'
    nodes = pd.read_csv(root / 'nodes_roles.csv', dtype={'gid': gid_dtype, 'cluster_id': 'int64'})
    clusters = pd.read_csv(root / 'clusters.csv', dtype={'cluster_id': 'int64', 'n_nodes': 'int64', 'n_seed': 'int64', 'top_gids': str})
    top = pd.read_csv(root / 'top_nodes.csv', dtype={'gid': gid_dtype, 'rank': 'int64', 'priority_score': float})
    # Пустые CSV не несут dtype для чисел: явно задаём их из контракта.
    if nodes.empty:
        for column in ['role_score', 'priority_score', *[f'{role}_score' for role in ALLOWED_ROLES],
                       *[f'priority_{part}' for part in PRIORITY_COMPONENTS]]:
            nodes[column] = nodes[column].astype(float)
        for column in ('is_seed', 'truncated_by_depth'):
            nodes[column] = nodes[column].astype(bool)
    if clusters.empty:
        clusters['sum_kzt_internal'] = clusters.sum_kzt_internal.astype(float)
    return validate_outputs(nodes, clusters, top, graph, priority_config, cluster_config)
