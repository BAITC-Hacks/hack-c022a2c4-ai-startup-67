"""Синтетические тесты проекции, сообществ и направленных денежных сумм."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from analytics.clustering import (CLUSTER_CONFIG, CLUSTER_COLUMNS,
                                  build_undirected_projection, cluster_network,
                                  summarize_clusters)


def fixture():
    graph = nx.DiGraph()
    for gid in range(1, 9):
        graph.add_node(gid, is_seed=gid in (1, 4, 7), depth=0 if gid in (1, 4, 7) else 1)
    for src, dst, amount in ((1, 2, 100), (2, 1, 50), (2, 3, 30), (3, 1, 80),
                             (4, 5, 90), (5, 6, 90), (6, 4, 90)):
        graph.add_edge(src, dst, sum_kzt=float(amount), n_tx=1)
    return graph


def features(graph):
    gids = sorted(graph)
    return pd.DataFrame({
        'gid': pd.Series(gids, dtype='uint64' if gids and max(gids) > 2 ** 63 - 1 else 'int64'),
        'is_seed': pd.Series([graph.nodes[g]['is_seed'] for g in gids], dtype=bool),
        'role': ['peripheral'] * len(gids),
        'role_score': [0.] * len(gids),
        'betweenness': pd.Series(dict(nx.betweenness_centrality(graph))).reindex(gids).to_numpy(),
        'pagerank': [1 / len(gids) for _ in gids],
    })


class ClusteringTests(unittest.TestCase):
    def test_bidirectional_projection_and_original_unchanged(self):
        graph = fixture()
        before = deepcopy(graph)
        projection = build_undirected_projection(graph)
        self.assertIsInstance(projection, nx.Graph)
        self.assertFalse(projection.is_directed())
        self.assertEqual(projection[1][2]['sum_kzt'], 150)
        self.assertEqual(projection[1][2]['n_tx_total'], 2)
        self.assertEqual(set(projection), set(graph))
        projection[1][2]['sum_kzt'] = 999
        self.assertTrue(nx.utils.graphs_equal(graph, before))
        self.assertTrue(graph.is_directed())

    def test_disconnected_components_never_mix(self):
        graph = fixture()
        before = deepcopy(graph)
        f = features(graph)
        result, summary, _ = cluster_network(graph, f)
        self.assertTrue(nx.utils.graphs_equal(graph, before))
        pd.testing.assert_frame_equal(result[f.columns], f)
        mapping = result.set_index('gid').cluster_id
        groups = list(nx.weakly_connected_components(graph))
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertTrue(set(mapping.loc[list(left)]).isdisjoint(mapping.loc[list(right)]))
        self.assertEqual(list(summary.columns), CLUSTER_COLUMNS)

    def test_isolates_are_separate_unique_clusters(self):
        graph = fixture()
        result, summary, report = cluster_network(graph, features(graph))
        mapping = result.set_index('gid').cluster_id
        self.assertNotEqual(mapping[7], mapping[8])
        for gid in (7, 8):
            row = summary.set_index('cluster_id').loc[mapping[gid]]
            self.assertEqual(row.n_nodes, 1)
            self.assertEqual(row.sum_kzt_internal, 0)
            self.assertTrue(row.is_isolated)
            self.assertIn('Изолированный узел', row.hypothesis)
        self.assertEqual(report['n_isolated_clusters'], 2)

    def test_every_gid_and_seed_counts(self):
        graph = fixture()
        result, summary, report = cluster_network(graph, features(graph))
        self.assertTrue(result.gid.is_unique)
        self.assertEqual(set(result.gid), set(graph))
        self.assertTrue(result.cluster_id.ge(0).all())
        self.assertEqual(summary.n_nodes.sum(), len(graph))
        self.assertEqual(summary.n_seed.sum(), 3)
        for row in summary.itertuples():
            members = result[result.cluster_id.eq(row.cluster_id)]
            self.assertEqual(row.n_seed, members.is_seed.sum())
            self.assertEqual(row.n_nodes, len(members))
            self.assertEqual(row.seed_ratio, row.n_seed / row.n_nodes)
        self.assertEqual(report['n_unassigned_nodes'], 0)
        self.assertEqual(sum(int(k) * v for k, v in report['cluster_size_distribution'].items()), len(graph))

    def test_internal_turnover_uses_directed_edges_excludes_crossing(self):
        graph = fixture()
        graph.add_edge(3, 3, sum_kzt=11., n_tx=1)
        f = features(graph)
        f['cluster_id'] = [0, 0, 1, 2, 2, 2, 3, 4]
        summary = summarize_clusters(graph, f).set_index('cluster_id')
        self.assertEqual(summary.loc[0, 'sum_kzt_internal'], 150)
        self.assertEqual(summary.loc[1, 'sum_kzt_internal'], 11)
        self.assertEqual(summary.loc[2, 'sum_kzt_internal'], 270)
        self.assertEqual(summary.sum_kzt_internal.sum(), 431)
        self.assertEqual(build_undirected_projection(graph)[3][3]['sum_kzt'], 11)
        self.assertFalse(summary.loc[1, 'is_isolated'])
        self.assertNotIn('Изолированный', summary.loc[1, 'hypothesis'])

    def test_deterministic_runs_insertion_order_and_stable_ids(self):
        graph = fixture()
        expected, summary, _ = cluster_network(graph, features(graph))
        for repeat in range(3):
            reordered = nx.DiGraph()
            reordered.add_nodes_from(reversed(list(graph.nodes(data=True))))
            reordered.add_edges_from(reversed(list(graph.edges(data=True))))
            actual, actual_summary, _ = cluster_network(reordered, features(graph).sample(frac=1, random_state=repeat))
            pd.testing.assert_frame_equal(actual.sort_values('gid').reset_index(drop=True), expected)
            pd.testing.assert_frame_equal(summary, actual_summary, check_exact=True)
        keys = [(-int(row.n_nodes), min(expected.loc[expected.cluster_id.eq(row.cluster_id), 'gid']))
                for row in summary.itertuples()]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(summary.cluster_id.tolist(), list(range(len(summary))))

    def test_top_gids_membership_order_and_large_ids(self):
        graph = nx.relabel_nodes(fixture(), {g: 2 ** 63 + g for g in fixture()})
        f = features(graph)
        # Равенство метрик разрешается по gid, без преобразования gid во float.
        f['betweenness'] = 0.
        f['pagerank'] = 0.5
        result, summary, _ = cluster_network(graph, f, {**CLUSTER_CONFIG, 'top_n': 2})
        for row in summary.itertuples():
            top = [int(gid) for gid in row.top_gids.split(',')]
            members = sorted(result.loc[result.cluster_id.eq(row.cluster_id), 'gid'])
            self.assertEqual(top, members[:2])
            self.assertLessEqual(len(top), 2)
        self.assertEqual(result.gid.dtype, f.gid.dtype)

    def test_role_composition_and_hypothesis_not_dominated_by_peripheral(self):
        graph = fixture()
        f = features(graph)
        f.loc[f.gid.eq(2), 'role'] = 'consolidator'
        f['cluster_id'] = 0
        summary = summarize_clusters(graph, f)
        row = summary.iloc[0]
        self.assertEqual(row.n_consolidator, 1)
        self.assertEqual(row.n_peripheral, 7)
        self.assertEqual(row.dominant_role, 'peripheral')
        self.assertIn('Паттернов сбора: 1', row.hypothesis)
        self.assertIn('seed: 3', row.hypothesis)
        self.assertEqual(sum(row[f'n_{r}'] for r in ('consolidator', 'transit', 'distributor', 'terminal', 'coordinator', 'peripheral')), 8)

    def test_empty_edgeless_and_zero_amount_components(self):
        for kind in ('empty', 'edgeless', 'zero', 'mixed'):
            with self.subTest(kind=kind):
                graph = nx.DiGraph()
                if kind != 'empty':
                    graph.add_nodes_from([(g, {'is_seed': False}) for g in (1, 2, 3)])
                if kind in ('zero', 'mixed'):
                    graph.add_edge(1, 2, sum_kzt=0., n_tx=1)
                if kind == 'mixed':
                    graph.add_edge(2, 3, sum_kzt=100., n_tx=1)
                result, summary, report = cluster_network(graph, features(graph))
                self.assertEqual(summary.n_nodes.sum(), len(graph))
                self.assertEqual(report['n_unassigned_nodes'], 0)
                self.assertTrue(np.isfinite(summary.sum_kzt_internal).all())
                if kind in ('edgeless', 'zero'):
                    self.assertEqual(len(summary), 3)
                if kind == 'zero':
                    self.assertFalse(summary.loc[summary.top_gids.isin(['1', '2']), 'is_isolated'].any())
                if kind == 'empty':
                    self.assertEqual(report['largest_cluster_size'], 0)
                    self.assertEqual(report['smallest_cluster_size'], 0)

    def test_single_self_loop(self):
        graph = nx.DiGraph()
        graph.add_node(1, is_seed=True)
        graph.add_edge(1, 1, sum_kzt=30., n_tx=2)
        _, summary, _ = cluster_network(graph, features(graph))
        self.assertEqual(summary.iloc[0].sum_kzt_internal, 30)
        self.assertFalse(summary.iloc[0].is_isolated)

    def test_louvain_arguments_and_no_fixed_cluster_count(self):
        graph = fixture()
        with patch('analytics.clustering.nx.community.louvain_communities',
                   wraps=nx.community.louvain_communities) as algorithm:
            cluster_network(graph, features(graph))
        self.assertEqual(algorithm.call_count, 2)
        for call in algorithm.call_args_list:
            self.assertFalse(call.args[0].is_directed())
            self.assertEqual(call.kwargs['weight'], 'sum_kzt')
            self.assertEqual(call.kwargs['seed'], 42)

    def test_invalid_input_rejected(self):
        graph = fixture()
        for bad in (features(graph).iloc[:-1], pd.concat([features(graph), features(graph)])):
            with self.assertRaises(ValueError):
                cluster_network(graph, bad)
        with self.assertRaises(ValueError):
            build_undirected_projection(graph.to_undirected())
        graph[1][2]['sum_kzt'] = np.inf
        with self.assertRaises(ValueError):
            build_undirected_projection(graph)


if __name__ == '__main__':
    unittest.main()
