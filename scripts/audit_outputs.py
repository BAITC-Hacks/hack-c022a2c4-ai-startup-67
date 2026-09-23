"""Воспроизводимый аудит CSV против исходных данных, без изменения оценок."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd

from analytics.features import build_graph
from analytics.roles import classify_roles
from analytics.priority import calculate_priority, PRIORITY_CONFIG
from analytics.temporal import temporal_features
from utils.output_validation import validate_output_files
from utils.validation import load_data


def audit(data_dir, out_dir):
    data = load_data(data_dir)
    graph = build_graph(data)
    result = validate_output_files(out_dir, graph)
    nodes = pd.read_csv(out_dir / 'nodes_roles.csv', dtype={'gid': data.nodes.gid.dtype})
    features = pd.read_csv(out_dir / 'node_features.csv', dtype={'gid': data.nodes.gid.dtype})
    clusters = pd.read_csv(out_dir / 'clusters.csv', dtype={'top_gids': str})
    top = pd.read_csv(out_dir / 'top_nodes.csv', dtype={'gid': data.nodes.gid.dtype})
    pd.testing.assert_frame_equal(nodes.sort_index(axis=1), features.sort_index(axis=1))
    checked_roles = classify_roles(features)
    assert checked_roles.role.tolist() == nodes.role.tolist()
    assert checked_roles.evidence.tolist() == nodes.evidence.tolist()
    assert np.allclose(checked_roles.role_score, nodes.role_score, atol=1e-12, rtol=0)
    checked_priority = calculate_priority(checked_roles)
    assert np.allclose(checked_priority.priority_score, nodes.priority_score, atol=1e-12, rtol=0)
    assert checked_priority.priority_why.tolist() == nodes.priority_why.tolist()
    temporal = temporal_features(data.transactions, data.nodes).set_index('gid')
    indexed = nodes.set_index('gid').reindex(temporal.index)
    for column in temporal:
        assert np.allclose(indexed[column], temporal[column], atol=1e-8, rtol=1e-12, equal_nan=True), column
    for prefix, degree in [('in', graph.in_degree), ('out', graph.out_degree)]:
        for suffix, weight in [('deg', None), ('kzt', 'sum_kzt'), ('tx', 'n_tx')]:
            assert np.allclose(nodes[f'{prefix}_{suffix}'], nodes.gid.map(dict(degree(weight=weight))), atol=.01, rtol=0)
    isolated = set(nx.isolates(graph))
    assert isolated <= set(features.gid) and isolated <= set(nodes.gid)
    assert nodes.loc[nodes.gid.isin(isolated), 'priority_score'].eq(0).all()
    assert nodes.loc[nodes.is_seed, ['priority_window', 'priority_same_day']].eq(0).all().all()
    assert nodes.loc[nodes.is_seed, ['short_window_outflow_ratio', 'same_day_outflow_ratio']].isna().all().all()
    for column in ['short_window_outflow_ratio', 'same_day_outflow_ratio']:
        assert nodes[column].dropna().between(0, 1).all()
    components = {gid: index for index, members in enumerate(nx.weakly_connected_components(graph)) for gid in members}
    disconnected = {}
    for cid, group in nodes.groupby('cluster_id'):
        # Louvain must not merge different original weak components. Its own
        # communities may nevertheless have disconnected induced subgraphs.
        assert len({components[gid] for gid in group.gid}) == 1
        sizes = sorted(map(len, nx.weakly_connected_components(graph.subgraph(group.gid))), reverse=True)
        if len(sizes) > 1:
            disconnected[int(cid)] = sizes
    required = nodes[['gid', 'role', 'role_score', 'cluster_id', 'priority_score', 'evidence']]
    assert not required.isna().any().any()
    assert not np.isinf(nodes.select_dtypes('number').to_numpy(dtype=float)).any()
    text = pd.concat([nodes.evidence, nodes.priority_why, clusters.hypothesis, top.why]).str.lower()
    assert not text.str.contains(r'преступник|виновен|доказанный организатор|criminal|guilty|proven organizer', regex=True).any()
    samples = nodes.groupby('role', group_keys=False).apply(lambda group: group.sample(min(3, len(group)), random_state=42), include_groups=False)
    # Preserve integer identifiers as strings for JSON consumers.
    selected = nodes.loc[samples.index].copy()
    selected['gid'] = selected.gid.map(str)
    sample_cols = ['gid', 'role', 'role_score', 'evidence', 'in_deg', 'out_deg', 'in_kzt', 'out_kzt',
                   'depth', 'is_seed', 'reachable_seed_count', 'betweenness', 'priority_score', 'cluster_id']
    quantiles = {'min': 0, 'p25': .25, 'median': .5, 'p75': .75, 'p90': .9, 'p95': .95, 'max': 1}
    top_details = nodes.set_index('gid').loc[top.head(20).gid].reset_index()
    top_details['gid'] = top_details.gid.map(str)
    components = [f'priority_{part}' for part in PRIORITY_CONFIG['weights']]
    result.update({
        'n_edges': len(data.edges), 'n_transactions': len(data.transactions), 'n_seed': int(nodes.is_seed.sum()),
        'n_isolated': len(isolated), 'n_truncated': int(nodes.truncated_by_depth.sum()),
        'n_truncated_terminal': int((nodes.truncated_by_depth & nodes.role.eq('terminal')).sum()),
        'n_seed_transit': int((nodes.is_seed & nodes.role.eq('transit')).sum()),
        'roles': nodes.groupby('role').role_score.agg(['count', 'mean', 'median']).to_dict('index'),
        'priority_quantiles': {name: float(nodes.priority_score.quantile(q)) for name,q in quantiles.items()},
        'largest_cluster': int(clusters.n_nodes.max()), 'singleton_clusters': int(clusters.n_nodes.eq(1).sum()),
        'multi_seed_clusters': int(clusters.n_seed.ge(2).sum()), 'disconnected_clusters': disconnected,
        'cluster_role_totals': {col: int(clusters[col].sum()) for col in clusters if col.startswith('n_')},
        'nullable_features': {col: int(count) for col, count in nodes.isna().sum().items() if count},
        'samples': selected[sample_cols].to_dict('records'),
        'top20': top_details[['gid', 'role', 'role_score', 'priority_score', 'cluster_id', 'in_kzt', 'out_kzt',
                              'betweenness', 'reachable_seed_count', 'priority_main_factors', *components]].to_dict('records'),
        'mean_top20_weighted_components': {name: float((top_details['priority_'+name]*weight).mean())
                                           for name,weight in PRIORITY_CONFIG['weights'].items()},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--out', type=Path, default=ROOT/'out')
    args = parser.parse_args()
    report = audit(args.data, args.out)
    target = args.out/'final_audit.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','top20')},ensure_ascii=False,indent=2))
    print(f'Audit saved to {target}')


if __name__ == '__main__':
    main()
