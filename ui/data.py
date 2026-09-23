"""Кэшируемое чтение готовых файлов. Аналитический pipeline здесь не запускается."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from ui.formatting import ROLE_COLORS
from utils.provenance import source_fingerprints

REQUIRED_OUTPUTS = ('nodes_roles.csv', 'clusters.csv', 'top_nodes.csv')


class DashboardDataError(ValueError):
    pass


def fingerprint(directory, filenames):
    """Размер и mtime входят в ключ кэша: новый экспорт виден без перезапуска."""
    return tuple((name, (Path(directory) / name).stat().st_mtime_ns,
                  (Path(directory) / name).stat().st_size)
                 for name in filenames if (Path(directory) / name).is_file())


def missing_outputs(directory):
    return [name for name in REQUIRED_OUTPUTS if not (Path(directory) / name).is_file()]


@st.cache_data(show_spinner=False, max_entries=8)
def verify_sources(directory, signature, expected, names):
    """Проверяется только при новой версии файлов; CSV-аналитика не пересчитывается."""
    if not expected:
        return  # Совместимость со старыми экспортами без provenance.
    actual = source_fingerprints(directory, names)
    if any(actual[name] != expected.get(name) for name in names):
        raise DashboardDataError('Parquet изменились после расчёта CSV. Повторите pipeline.')


def _booleans(frame):
    for col in ('is_seed', 'truncated_by_depth', 'is_isolated'):
        if col in frame:
            converted = frame[col].astype(str).str.lower().map({'true': True, 'false': False})
            if converted.isna().any():
                raise DashboardDataError(f'Некорректные логические значения: {col}')
            frame[col] = converted.astype(bool)
    return frame


@st.cache_data(show_spinner=False, max_entries=8)
def load_outputs(directory, signature):
    root = Path(directory)
    nodes = pd.read_csv(root / 'nodes_roles.csv', dtype={'gid': str})
    clusters = pd.read_csv(root / 'clusters.csv', dtype={'top_gids': str})
    top = pd.read_csv(root / 'top_nodes.csv', dtype={'gid': str})
    for frame, required in ((nodes, ('gid', 'role', 'role_score', 'cluster_id', 'priority_score', 'evidence')),
                            (clusters, ('cluster_id', 'n_nodes', 'n_seed', 'sum_kzt_internal', 'top_gids', 'hypothesis')),
                            (top, ('rank', 'gid', 'role', 'priority_score', 'why'))):
        if not set(required).issubset(frame.columns):
            raise DashboardDataError('В CSV отсутствуют обязательные столбцы. Повторите pipeline.')
    if nodes.gid.isna().any() or not nodes.gid.str.fullmatch(r'\d+').all() or not nodes.gid.is_unique:
        raise DashboardDataError('Некорректные или повторяющиеся gid.')
    if not nodes.role.isin(ROLE_COLORS).all():
        raise DashboardDataError('В CSV найдена неизвестная роль.')
    for frame in (nodes, top):
        for column in ('priority_score', 'role_score'):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors='raise')
                if not frame[column].between(0, 1).all():
                    raise DashboardDataError(f'Некорректная оценка: {column}')
    if not clusters.cluster_id.is_unique:
        raise DashboardDataError('Повторяющиеся номера кластеров.')
    if not top.gid.isin(nodes.gid).all() or not nodes.cluster_id.isin(clusters.cluster_id).all():
        raise DashboardDataError('CSV не согласованы. Повторите pipeline для одного набора данных.')
    comparison = top.merge(nodes[['gid', 'role', 'priority_score']], on='gid', suffixes=('_top', '_node'), validate='one_to_one')
    if (not comparison.role_top.eq(comparison.role_node).all()
            or not np.allclose(comparison.priority_score_top, comparison.priority_score_node, atol=1e-12, rtol=0)):
        raise DashboardDataError('Рейтинг и таблица узлов относятся к разным расчётам. Повторите pipeline.')
    features_path = root / 'node_features.csv'
    if features_path.is_file():
        features = pd.read_csv(features_path, dtype={'gid': str})
        if set(features.gid) != set(nodes.gid) or not features.gid.is_unique:
            raise DashboardDataError('node_features.csv не соответствует nodes_roles.csv.')
        extra = [c for c in features if c not in nodes and c != 'gid']
        if extra:
            nodes = nodes.merge(features[['gid'] + extra], on='gid', validate='one_to_one', how='left')
    nodes = _booleans(nodes)
    clusters = _booleans(clusters)
    diagnostics = {}
    path = root / 'diagnostics.json'
    if path.is_file():
        try:
            diagnostics = json.loads(path.read_text())
            if not isinstance(diagnostics, dict):
                diagnostics = {}
        except (ValueError, OSError):
            diagnostics = {}
    return nodes, clusters, top, diagnostics


@st.cache_data(show_spinner=False, max_entries=12)
def download_bytes(directory, filename, signature):
    return (Path(directory) / filename).read_bytes()


@st.cache_data(show_spinner=False, max_entries=8)
def load_edges(directory, signature):
    root = Path(directory)
    edges = pd.read_parquet(root / 'edges.parquet')
    nodes = pd.read_parquet(root / 'nodes.parquet')
    if not {'src', 'dst', 'sum_kzt', 'n_tx'}.issubset(edges) or 'gid' not in nodes:
        raise DashboardDataError('Некорректная схема исходного графа.')
    for col in ('src', 'dst'):
        edges[col] = edges[col].map(str)
    nodes['gid'] = nodes.gid.map(str)
    return nodes, edges


@st.cache_resource(show_spinner=False, max_entries=4)
def load_graph(directory, signature):
    nodes, edges = load_edges(directory, signature)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.gid)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt), n_tx=int(row.n_tx))
    if set(graph) != set(nodes.gid):
        raise DashboardDataError('В рёбрах присутствуют gid, отсутствующие в nodes.parquet.')
    return nx.freeze(graph)


@st.cache_data(show_spinner=False, max_entries=4)
def load_transactions(directory, signature):
    tx = pd.read_parquet(Path(directory) / 'transactions.parquet')
    if not {'src', 'dst', 'date', 'sum_kzt'}.issubset(tx):
        raise DashboardDataError('Некорректная схема транзакций.')
    tx['src'] = tx.src.map(str)
    tx['dst'] = tx.dst.map(str)
    tx['date'] = pd.to_datetime(tx.date, errors='raise').dt.normalize()
    return tx


def find_node(nodes, query):
    """Точное строковое сравнение — без потери gid выше 2**53 в JavaScript."""
    gid = str(query).strip()
    matches = nodes[nodes.gid.eq(gid)]
    return None if matches.empty else matches.iloc[0]


def filter_nodes(nodes, roles=(), clusters=(), seed_only=False, truncated_only=False, minimum=0.):
    mask = nodes.priority_score.ge(minimum)
    if roles:
        mask &= nodes.role.isin(roles)
    if clusters:
        mask &= nodes.cluster_id.isin(clusters)
    if seed_only:
        mask &= nodes['is_seed'].eq(True) if 'is_seed' in nodes else False
    if truncated_only:
        mask &= nodes['truncated_by_depth'].eq(True) if 'truncated_by_depth' in nodes else False
    return nodes.loc[mask]


def node_warnings(row, graph=None):
    warnings = []
    if row.get('is_seed', False):
        warnings.append('Входящая активность seed-клиента может быть неполной: граф собран по исходящим переводам seed.')
    if row.get('truncated_by_depth', False) or (row.get('depth') == 4 and row.get('out_deg') == 0):
        warnings.append('Переводы за пределами 4-го перехода не наблюдаются. Ноль видимых исходящих не подтверждает конечное состояние узла.')
    isolated = graph.degree(row['gid']) == 0 if graph is not None and row['gid'] in graph else row.get('in_deg') == 0 and row.get('out_deg') == 0
    if isolated:
        warnings.append('В предоставленном графе для этого узла нет рёбер транзакций.')
    return warnings


def counterparties(edges, gid, incoming):
    owner, other = ('dst', 'src') if incoming else ('src', 'dst')
    return edges.loc[edges[owner].eq(gid), [other, 'sum_kzt', 'n_tx']].rename(
        columns={other: 'gid'}).sort_values(['sum_kzt', 'gid'], ascending=[False, True])


def daily_activity(tx, gid):
    incoming = tx.loc[tx.dst.eq(gid)].groupby('date').sum_kzt.sum()
    outgoing = tx.loc[tx.src.eq(gid)].groupby('date').sum_kzt.sum()
    frame = pd.concat([incoming.rename('Входящие'), outgoing.rename('Исходящие')], axis=1).fillna(0.)
    if not frame.empty:
        frame = frame.reindex(pd.date_range(tx.date.min(), tx.date.max(), freq='D'), fill_value=0.)
    frame.index.name = 'Дата'
    return frame
