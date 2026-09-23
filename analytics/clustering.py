"""Объяснимая кластеризация; финансовый граф остаётся направленным."""
from copy import deepcopy
from math import fsum
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd

from analytics.roles import ALLOWED_ROLES

CLUSTER_CONFIG = {'seed': 42, 'resolution': 1.0, 'threshold': 1e-7, 'top_n': 5}
CLUSTER_COLUMNS = ['cluster_id', 'n_nodes', 'n_seed', 'sum_kzt_internal', 'top_gids',
                   'hypothesis', 'seed_ratio', 'dominant_role', 'is_isolated',
                   *[f'n_{role}' for role in ALLOWED_ROLES]]


def build_undirected_projection(graph: nx.DiGraph) -> nx.Graph:
    """Новый Graph: встречные суммы и n_tx складываются; петля учитывается один раз.

    Сортировка узлов и рёбер делает порядок обхода независимым от порядка ввода.
    Исходный DiGraph и его атрибуты не изменяются.
    """
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError('Clustering requires a directed simple graph')
    projection = nx.Graph()
    projection.add_nodes_from(sorted(graph.nodes))
    for src, dst, attrs in sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1])):
        amount = float(attrs['sum_kzt'])
        if not np.isfinite(amount) or amount < 0:
            raise ValueError('Projection requires finite nonnegative amounts')
        if not projection.has_edge(src, dst):
            projection.add_edge(src, dst, sum_kzt=0.0, n_tx_total=0)
        edge = projection[src][dst]
        edge['sum_kzt'] += amount
        edge['n_tx_total'] += int(attrs['n_tx'])
        if not np.isfinite(edge['sum_kzt']):
            raise ValueError('Projection edge amount overflow')
    return projection


def detect_communities(projection: nx.Graph, config: dict) -> list[set]:
    """Louvain независимо по компонентам; число сообществ не задаётся.

    Компоненты с нулевой суммой весов разбиваются на отдельные узлы: денежного
    сигнала нет, а взвешенная модулярность при нулевой сумме не определена.
    """
    communities = []
    for members in sorted(nx.connected_components(projection), key=min):
        component = projection.subgraph(members)
        if len(members) == 1 or component.size(weight='sum_kzt') == 0:
            communities.extend({gid} for gid in sorted(members))
        else:
            communities.extend(nx.community.louvain_communities(
                component, weight='sum_kzt', seed=config['seed'],
                resolution=config['resolution'], threshold=config['threshold']))
    return sorted(communities, key=lambda members: (-len(members), min(members)))


def cluster_hypothesis(metrics: dict) -> str:
    """Гипотеза описывает наблюдаемые метрики, не общие намерения участников."""
    if metrics['is_isolated']:
        return f"Изолированный узел без наблюдаемых рёбер; seed: {metrics['n_seed']}."
    text = f"Сообщество: {metrics['n_nodes']} узлов; seed: {metrics['n_seed']}."
    patterns = [
        ('coordinator', 'Структурно центральных кандидатов'),
        ('consolidator', 'Паттернов сбора'),
        ('distributor', 'Паттернов распределения'),
        ('transit', 'Наблюдаемых транзитных паттернов'),
    ]
    parts = [f"{label}: {metrics['n_' + role]}" for role, label in patterns if metrics['n_' + role]]
    if parts:
        text += ' ' + '; '.join(parts) + '.'
    elif metrics['n_terminal']:
        text += f" Без видимых исходящих переводов: {metrics['n_terminal']} узлов с ролью terminal."
    else:
        text += ' Выраженных ролевых паттернов не выявлено.'
    return text


def summarize_clusters(graph: nx.DiGraph, features: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Оборот берётся только из исходных направленных рёбер, включая петли."""
    if (not features.gid.is_unique or set(features.gid) != set(graph)
            or features.cluster_id.isna().any() or not features.cluster_id.ge(0).all()):
        raise ValueError('Every graph node must have exactly one nonnegative cluster ID')
    if not features.role.isin(ALLOWED_ROLES).all():
        raise ValueError('Unknown role in cluster summary')
    if top_n < 1:
        raise ValueError('top_n must be positive')
    assignments = dict(zip(features.gid, features.cluster_id))
    amounts = {int(cid): [] for cid in features.cluster_id.unique()}
    for src, dst, attrs in sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1])):
        if assignments[src] == assignments[dst]:
            amounts[assignments[src]].append(float(attrs['sum_kzt']))
    rows = []
    for cid, group in features.groupby('cluster_id', sort=True):
        counts = group.role.value_counts()
        # Настоящее большинство, включая peripheral; равенство — по имени роли.
        dominant = min(ALLOWED_ROLES, key=lambda role: (-int(counts.get(role, 0)), role))
        top = group.sort_values(['betweenness', 'pagerank', 'gid'], ascending=[False, False, True]).head(top_n)
        metrics = {
            'cluster_id': int(cid), 'n_nodes': len(group),
            'n_seed': sum(bool(graph.nodes[gid]['is_seed']) for gid in group.gid),
            'sum_kzt_internal': fsum(amounts[cid]),
            'top_gids': ','.join(str(gid) for gid in top.gid),
            'dominant_role': dominant,
            'is_isolated': len(group) == 1 and graph.degree(group.gid.iloc[0]) == 0,
            **{f'n_{role}': int(counts.get(role, 0)) for role in ALLOWED_ROLES},
        }
        metrics['seed_ratio'] = metrics['n_seed'] / metrics['n_nodes']
        metrics['hypothesis'] = cluster_hypothesis(metrics)
        rows.append(metrics)
    result = pd.DataFrame(rows, columns=CLUSTER_COLUMNS)
    for column in ['cluster_id', 'n_nodes', 'n_seed', *[f'n_{role}' for role in ALLOWED_ROLES]]:
        result[column] = result[column].astype('int64')
    for column in ('sum_kzt_internal', 'seed_ratio'):
        result[column] = result[column].astype(float)
    result['is_isolated'] = result.is_isolated.astype(bool)
    return result


def cluster_network(graph: nx.DiGraph, features: pd.DataFrame, config: dict = None):
    """Возвращает признаки с cluster_id, сводку и диагностику кластеризации."""
    start = perf_counter()
    config = deepcopy(CLUSTER_CONFIG if config is None else config)
    if not isinstance(config['seed'], int):
        raise ValueError('Louvain seed must be an integer')
    for name in ('resolution', 'threshold'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not isinstance(config['top_n'], int) or config['top_n'] < 1:
        raise ValueError('top_n must be a positive integer')
    if not features.gid.is_unique or features.gid.isna().any() or set(features.gid) != set(graph):
        raise ValueError('Clustering input must contain every gid exactly once')
    projection = build_undirected_projection(graph)
    communities = detect_communities(projection, config)
    membership = {}
    for cid, members in enumerate(communities):
        if not members or set(members).intersection(membership):
            raise ValueError('Communities must be nonempty and disjoint')
        membership.update(dict.fromkeys(members, cid))
    if set(membership) != set(graph):
        raise ValueError('Communities must cover every graph node')
    result = features.copy()
    result['cluster_id'] = result.gid.map(membership).astype('int64')
    clusters = summarize_clusters(graph, result, top_n=config['top_n'])
    if int(clusters.n_nodes.sum()) != len(graph):
        raise ValueError('Cluster sizes do not match graph size')
    sizes = clusters.n_nodes
    diagnostics = {
        'config': config,
        'method': 'weighted Louvain per weak component on a separate undirected projection',
        'n_clusters': len(clusters),
        'largest_cluster_size': int(sizes.max()) if len(sizes) else 0,
        'smallest_cluster_size': int(sizes.min()) if len(sizes) else 0,
        'n_singleton_clusters': int(sizes.eq(1).sum()),
        'n_isolated_clusters': int(clusters.is_isolated.sum()),
        'n_clusters_with_multiple_seeds': int(clusters.n_seed.ge(2).sum()),
        'cluster_size_distribution': {str(int(size)): int(count) for size, count in sizes.value_counts().sort_index().items()},
        'total_internal_observed_turnover': fsum(clusters.sum_kzt_internal),
        'n_unassigned_nodes': int(result.cluster_id.isna().sum() + result.cluster_id.lt(0).sum()),
        'clustering_seconds': perf_counter() - start,
    }
    return result, clusters, diagnostics
