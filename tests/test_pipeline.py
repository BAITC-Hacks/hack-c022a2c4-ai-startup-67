"""Run with python -m unittest discover -s tests -v."""
from tests.fixtures import fixture
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analytics.features import basic_features, build_graph, network_diagnostics
from pipeline import run_pipeline
from utils.validation import DataValidationError, load_data, validate_data

class PipelineTests(unittest.TestCase):

    def setUp(self):
        (self.nodes, self.edges, self.tx) = fixture()

    def data(self):
        return validate_data(self.nodes, self.edges, self.tx)

    def test_isolated_node_and_metadata_retained(self):
        graph = build_graph(self.data())
        features = basic_features(graph).set_index('gid')
        self.assertTrue(graph.is_directed())
        self.assertEqual(set(graph), set(self.nodes.gid))
        self.assertEqual(len(features), len(self.nodes))
        self.assertEqual(graph.nodes[5], {'depth': 0, 'is_seed': True})
        self.assertEqual(features.loc[5, 'in_deg'], 0)
        self.assertGreater(features.loc[5, 'pagerank'], 0)
        self.assertTrue(np.isfinite(features.pagerank).all())
        self.assertAlmostEqual(features.pagerank.sum(), 1.0)
        self.assertFalse(graph.has_edge(2, 1))

    def test_matching_aggregates_and_features(self):
        data = self.data()
        frame = basic_features(build_graph(data)).set_index('gid')
        self.assertEqual(frame.loc[2, 'in_kzt'], 100.0)
        self.assertEqual(frame.loc[2, 'in_tx'], 2)
        self.assertEqual(frame.loc[2, 'out_tx'], 1)
        self.assertEqual(frame.loc[2, 'pass_through'], 0.5)
        self.assertEqual(frame.loc[2, 'total_kzt'], 150.0)
        self.assertEqual(frame.loc[2, 'total_tx'], 3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data.transactions.date))
        self.assertEqual(self.tx.date.dtype, object)

    def test_zero_incoming_and_boundary(self):
        frame = basic_features(build_graph(self.data())).set_index('gid')
        self.assertTrue(np.isnan(frame.loc[1, 'pass_through']))
        self.assertTrue(np.isnan(frame.loc[5, 'pass_through']))
        self.assertTrue(frame.loc[3, 'truncated_by_depth'])
        self.assertFalse(frame.loc[4, 'truncated_by_depth'])
        self.assertFalse(np.isinf(frame.pass_through).any())

    def test_large_integer_identifiers_preserved(self):
        offset = 2 ** 53
        self.nodes['gid'] += offset
        for frame in (self.edges, self.tx):
            frame['src'] += offset
            frame['dst'] += offset
        result = basic_features(build_graph(self.data()))
        self.assertEqual(result.gid.tolist(), self.nodes.gid.tolist())

    def test_deterministic_under_row_shuffle(self):
        expected = basic_features(build_graph(self.data()))
        shuffled = validate_data(*(frame.sample(frac=1, random_state=7) for frame in fixture()))
        pd.testing.assert_frame_equal(expected, basic_features(build_graph(shuffled)), check_exact=True)

    def test_diagnostics(self):
        data = self.data()
        graph = build_graph(data)
        report = network_diagnostics(data, graph, basic_features(graph))
        self.assertEqual(report['n_nodes'], 5)
        self.assertEqual(report['n_edges'], 3)
        self.assertEqual(report['n_transactions'], 4)
        self.assertEqual(report['n_seed'], 2)
        self.assertEqual(report['n_isolated'], 1)
        self.assertEqual(report['n_isolated_seed'], 1)
        self.assertEqual(report['n_weakly_connected_components'], 2)
        self.assertEqual(report['n_depth4_truncated'], 1)
        self.assertEqual(report['observed_edge_turnover_kzt'], 175.0)

    def test_empty_network(self):
        data = validate_data(self.nodes.iloc[:0], self.edges.iloc[:0], self.tx.iloc[:0])
        graph = build_graph(data)
        frame = basic_features(graph)
        self.assertTrue(frame.empty)
        self.assertEqual(network_diagnostics(data, graph, frame)['n_nodes'], 0)

    def test_parquet_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for (name, frame) in zip(('nodes', 'edges', 'transactions'), fixture()):
                frame.to_parquet(root / f'{name}.parquet', index=False)
            run_pipeline(root, root / 'out')
            output = pd.read_csv(root / 'out/node_features.csv')
            self.assertEqual(output.gid.tolist(), [1, 2, 3, 4, 5])
            self.assertTrue((root / 'out/diagnostics.json').is_file())
            self.assertFalse((root / 'out/nodes_roles.csv').exists())
if __name__ == '__main__':
    unittest.main()
