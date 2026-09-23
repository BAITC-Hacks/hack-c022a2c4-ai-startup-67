"""Synthetic structural and date-only regression cases; no real client IDs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from analytics.features import basic_features, build_graph
from analytics.graph_features import structural_features
from analytics.temporal import temporal_features, window_outflow_ratio
from pipeline import run_pipeline
from tests.fixtures import fixture
from utils.validation import validate_data


class StructuralTests(unittest.TestCase):
    def graph(self):
        graph = nx.DiGraph()
        for gid, depth in ((1, 0), (2, 0), (3, 1), (4, 2), (5, 4)):
            graph.add_node(gid, depth=depth, is_seed=gid in (1, 2))
        for src, dst, amount in ((1, 3, 30), (2, 3, 70), (3, 4, 40)):
            graph.add_edge(src, dst, sum_kzt=amount, n_tx=2)
        return graph

    def test_seeds_direction_and_distances(self):
        frame, report = structural_features(self.graph(), basic_features(self.graph()))
        frame = frame.set_index('gid')
        self.assertEqual(frame.loc[3, 'direct_seed_in'], 2)
        self.assertEqual(frame.loc[3, 'direct_seed_out'], 0)
        self.assertEqual(frame.loc[4, 'reachable_seed_count'], 2)
        self.assertEqual(frame.loc[1, 'reachable_seed_count'], 1)
        self.assertEqual(frame.loc[4, 'min_seed_distance'], 2)
        self.assertEqual(frame.loc[1, 'min_seed_distance'], 0)
        self.assertTrue(np.isnan(frame.loc[5, 'min_seed_distance']))
        self.assertEqual(report['n_seed_distance_depth_mismatch'], 0)
        self.assertEqual(report['n_unreachable_from_seeds'], 1)
        self.assertGreater(frame.loc[3, 'betweenness'], 0)
        self.assertEqual(frame.loc[4, 'betweenness'], 0)

    def test_concentration_and_diversity(self):
        graph = self.graph()
        frame, _ = structural_features(graph, basic_features(graph))
        row = frame.set_index('gid').loc[3]
        self.assertAlmostEqual(row.largest_in_sender_share, .7)
        self.assertAlmostEqual(row.incoming_amount_hhi, .58)
        self.assertEqual(row.largest_out_receiver_share, 1)
        self.assertEqual(row.outgoing_amount_hhi, 1)
        self.assertEqual(row.in_avg_tx_kzt, 25)
        self.assertEqual(row.out_avg_tx_kzt, 20)
        self.assertEqual(row.counterparty_balance, -1)
        self.assertEqual(row.degree_ratio, .5)

    def test_direct_seed_out_and_zero_money(self):
        graph = self.graph()
        graph.add_edge(4, 1, sum_kzt=0, n_tx=1)
        frame, _ = structural_features(graph, basic_features(graph))
        row = frame.set_index('gid').loc[4]
        self.assertEqual(row.direct_seed_out, 1)
        self.assertEqual(row.outgoing_amount_hhi, 0)
        self.assertEqual(row.largest_out_receiver_share, 0)
        self.assertEqual(row.out_avg_tx_kzt, 0)

    def test_directed_unweighted_betweenness_and_depth_diagnostic(self):
        graph = self.graph()
        graph.add_edge(1, 4, sum_kzt=1e9, n_tx=1)
        frame, report = structural_features(graph, basic_features(graph))
        expected = nx.betweenness_centrality(graph, weight=None)
        self.assertEqual(frame.betweenness.tolist(), [expected[g] for g in frame.gid])
        self.assertEqual(report['n_seed_distance_depth_mismatch'], 1)
        self.assertEqual(frame.set_index('gid').loc[4, 'depth'], 2)

    def test_hits_failure_retains_frame(self):
        graph = self.graph()
        with patch('analytics.graph_features.nx.hits', side_effect=nx.PowerIterationFailedConvergence(1000)):
            frame, report = structural_features(graph, basic_features(graph))
        self.assertTrue(report['hits_status'].startswith('unavailable'))
        self.assertEqual(len(frame), 5)
        self.assertTrue(frame.loc[frame.gid.eq(3), 'hub_score'].isna().all())
        self.assertEqual(frame.set_index('gid').loc[5, 'hub_score'], 0)

    def test_cycles_length_bound_and_self_loop(self):
        for length in (1, 2, 6, 7):
            with self.subTest(length=length):
                graph = nx.DiGraph()
                for gid in range(length):
                    graph.add_node(gid, depth=0, is_seed=False)
                    graph.add_edge(gid, (gid + 1) % length, sum_kzt=1, n_tx=1)
                frame, report = structural_features(graph, basic_features(graph))
                self.assertTrue(frame.is_in_short_cycle.eq(length <= 6).all())
                self.assertTrue(report['cycle_analysis_executed'])

    def test_empty_and_edgeless(self):
        for count in (0, 2):
            graph = nx.DiGraph()
            for gid in range(count):
                graph.add_node(gid, depth=0, is_seed=True)
            frame, _ = structural_features(graph, basic_features(graph))
            self.assertEqual(len(frame), count)
            self.assertTrue(frame.hub_score.eq(0).all())
            self.assertTrue(frame.incoming_amount_hhi.eq(0).all())


class TemporalTests(unittest.TestCase):
    def data(self):
        tx = pd.DataFrame({
            'src': [1, 1, 2, 3, 4, 4, 4, 4],
            'dst': [4, 4, 4, 4, 5, 5, 6, 6],
            'date': pd.to_datetime(['2026-07-01'] * 4 +
                                   ['2026-07-01', '2026-07-03', '2026-07-04', '2026-06-30']),
            'sum_kzt': [10., 20., 30., 40., 10., 30., 100., 100.],
        })
        nodes = pd.DataFrame({'gid': range(1, 8), 'is_seed': [True] + [False] * 6})
        return tx, nodes

    def test_daily_and_repeated_relationships(self):
        frame = temporal_features(*self.data()).set_index('gid')
        row = frame.loc[4]
        self.assertEqual(row.active_days_in, 1)
        self.assertEqual(row.active_days_out, 4)
        self.assertEqual(row.incoming_daily_max_kzt, 100)
        self.assertEqual(row.incoming_daily_mean_kzt, 100)
        self.assertEqual(row.outgoing_daily_max_kzt, 100)
        self.assertEqual(row.outgoing_daily_mean_kzt, 60)
        self.assertEqual(row.max_in_tx_per_day, 4)
        self.assertEqual(row.max_out_tx_per_day, 1)
        self.assertEqual(row.max_unique_in_senders_per_day, 3)
        self.assertEqual(row.days_with_3plus_in_senders, 1)
        self.assertEqual(row.repeated_in_counterparties, 1)
        self.assertEqual(row.repeated_out_counterparties, 2)
        self.assertEqual(row.max_transactions_from_single_sender, 2)
        self.assertEqual(row.max_transactions_to_single_receiver, 2)

    def test_window_excludes_before_and_after(self):
        frame = temporal_features(*self.data()).set_index('gid')
        self.assertAlmostEqual(frame.loc[4, 'short_window_outflow_ratio'], .4)
        self.assertAlmostEqual(frame.loc[4, 'same_day_outflow_ratio'], .1)
        self.assertTrue(frame.short_window_outflow_ratio.dropna().between(0, 1).all())
        self.assertTrue(np.isnan(frame.loc[7, 'short_window_outflow_ratio']))
        tx, nodes = self.data()
        nodes.loc[nodes.gid.eq(4), 'is_seed'] = True
        self.assertTrue(np.isnan(temporal_features(tx, nodes).set_index('gid').loc[4, 'short_window_outflow_ratio']))

    def test_overlapping_windows_cap_and_month_boundary(self):
        incoming = pd.Series([20., 20.], index=pd.to_datetime(['2026-07-30', '2026-07-31']))
        outgoing = pd.Series([100.], index=pd.to_datetime(['2026-08-01']))
        self.assertEqual(window_outflow_ratio(incoming, outgoing, 2), 1)
        self.assertEqual(window_outflow_ratio(incoming, outgoing, 0), 0)
        self.assertTrue(np.isnan(window_outflow_ratio(incoming * 0, outgoing, 2)))

    def test_date_normalization_and_empty_transactions(self):
        tx, nodes = self.data()
        tx.loc[0, 'date'] += pd.Timedelta(hours=12)
        self.assertEqual(temporal_features(tx, nodes).set_index('gid').loc[4, 'active_days_in'], 1)
        frame = temporal_features(tx.iloc[:0], nodes)
        self.assertTrue(frame.active_days_in.eq(0).all())
        self.assertTrue(frame.incoming_daily_mean_kzt.isna().all())
        self.assertTrue(temporal_features(tx.iloc[:0], nodes.iloc[:0]).empty)

    def test_integrated_output_isolates_and_base_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = fixture()
            for name, table in zip(('nodes', 'edges', 'transactions'), tables):
                table.to_parquet(root / f'{name}.parquet', index=False)
            report = run_pipeline(root, root / 'out')
            frame = pd.read_csv(root / 'out/node_features.csv')
            base = basic_features(build_graph(validate_data(*tables)))
            pd.testing.assert_frame_equal(frame[base.columns], base)
            self.assertEqual(frame.gid.nunique(), len(tables[0]))
            row = frame.set_index('gid').loc[5]
            for column in ('betweenness', 'hub_score', 'authority_score', 'active_days_in',
                           'incoming_amount_hhi', 'outgoing_daily_max_kzt', 'repeated_in_counterparties'):
                self.assertEqual(row[column], 0)
            self.assertEqual(row.weak_component_size, 1)
            self.assertFalse(np.isinf(frame.select_dtypes('number')).any().any())
            self.assertEqual(report['feature_shape'], list(frame.shape))

    def test_shuffle_and_large_ids(self):
        tables = fixture()
        for table in tables:
            for column in ('gid', 'src', 'dst'):
                if column in table:
                    table[column] = table[column].astype('uint64') + 2 ** 63
        data = validate_data(*tables)
        graph = build_graph(data)
        expected, _ = structural_features(graph, basic_features(graph))
        shuffled = validate_data(*(table.sample(frac=1, random_state=7) for table in tables))
        actual, _ = structural_features(build_graph(shuffled), basic_features(build_graph(shuffled)))
        pd.testing.assert_frame_equal(actual, expected, atol=1e-12)
        left = temporal_features(data.transactions, data.nodes)
        right = temporal_features(shuffled.transactions, shuffled.nodes)
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(left.gid.tolist(), data.nodes.gid.tolist())


if __name__ == '__main__':
    unittest.main()
