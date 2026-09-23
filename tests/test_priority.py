"""Синтетические проверки прозрачного приоритета и итоговых CSV."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analytics.priority import (PRIORITY_CONFIG, PRIORITY_COMPONENTS, calculate_priority,
                                build_top_nodes, priority_diagnostics)
from analytics.roles import classify_roles
from analytics.features import build_graph
from pipeline import run_pipeline
from tests.test_roles import feature_fixture, chain
from tests.fixtures import fixture
from utils.validation import validate_data
from utils.output_validation import validate_outputs, validate_output_files


def ranked_fixture():
    transfers = [(1, 3, '2026-07-01', 100.), (2, 3, '2026-07-01', 100.),
                 (3, 4, '2026-07-02', 90.), (3, 5, '2026-07-02', 90.),
                 (4, 6, '2026-07-03', 90.), (5, 7, '2026-07-03', 90.),
                 (8, 9, '2026-07-01', 1e12)]
    return classify_roles(feature_fixture(transfers, seeds=(1, 2, 8), isolates=(10,)))


class PriorityTests(unittest.TestCase):
    def test_structure_and_huge_amount(self):
        f = calculate_priority(ranked_fixture()).set_index('gid')
        self.assertGreater(f.loc[3, 'priority_structural'], f.loc[10, 'priority_structural'])
        self.assertGreater(f.loc[3, 'priority_score'], f.loc[8, 'priority_score'])
        self.assertGreater(f.loc[3, 'priority_score'], f.loc[9, 'priority_score'])
        self.assertEqual(f.loc[10, 'priority_score'], 0)
        self.assertIn('изолированный', f.loc[10, 'priority_why'])

    def test_multiple_seeds_above_one(self):
        f = calculate_priority(ranked_fixture()).set_index('gid')
        self.assertGreater(f.loc[3, 'priority_seed_relevance'], f.loc[9, 'priority_seed_relevance'])
        self.assertEqual(f.loc[9, 'priority_seed_relevance'], 0)

    def test_bounds_weighted_formula_and_preserved_input(self):
        source = ranked_fixture()
        f = calculate_priority(source)
        pd.testing.assert_frame_equal(source, f[source.columns])
        for column in [f'priority_{p}' for p in PRIORITY_COMPONENTS] + ['priority_score']:
            self.assertTrue(f[column].between(0, 1).all())
        expected = sum(f['priority_' + p] * w for p, w in PRIORITY_CONFIG['weights'].items())
        np.testing.assert_allclose(expected, f.priority_score)
        self.assertTrue(f.priority_why.str.len().gt(0).all())

    def test_deterministic_repeated_and_shuffled(self):
        source = ranked_fixture()
        expected = calculate_priority(source)
        pd.testing.assert_frame_equal(expected, calculate_priority(source), check_exact=True)
        actual = calculate_priority(source.sample(frac=1, random_state=67)).sort_values('gid').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        top = build_top_nodes(expected)
        self.assertEqual(top['rank'].tolist(), list(range(1, len(source) + 1)))
        self.assertTrue(top.priority_score.is_monotonic_decreasing)
        lookup = expected.set_index('gid')
        for row in top.itertuples():
            self.assertEqual(row.role, lookup.loc[row.gid, 'role'])
            self.assertEqual(row.priority_score, lookup.loc[row.gid, 'priority_score'])
            self.assertEqual(row.why, lookup.loc[row.gid, 'priority_why'])

    def test_ties_role_score_then_gid(self):
        f = calculate_priority(classify_roles(chain()))
        f['priority_score'] = .5
        f['role_score'] = [.6, .8, .8, .1]
        self.assertEqual(build_top_nodes(f.sample(frac=1, random_state=2)).gid.tolist(), [2, 3, 1, 4])

    def test_seed_ignores_flow_balance_and_windows(self):
        source = ranked_fixture()
        source.loc[source.gid.eq(3), 'is_seed'] = True
        first = calculate_priority(source)
        source.loc[source.gid.eq(3), ['pass_through', 'short_window_outflow_ratio', 'same_day_outflow_ratio']] = [1e9, 1., 1.]
        second = calculate_priority(source)
        self.assertEqual(first.set_index('gid').loc[3, 'priority_score'], second.set_index('gid').loc[3, 'priority_score'])
        self.assertEqual(second.set_index('gid').loc[3, 'priority_window'], 0)
        self.assertIn('Входящие seed', second.set_index('gid').loc[3, 'priority_why'])

    def test_depth_flag_does_not_adjust_priority(self):
        source = ranked_fixture()
        first = calculate_priority(source)
        source.loc[source.gid.eq(9), ['depth', 'truncated_by_depth']] = [4, True]
        second = calculate_priority(source)
        self.assertEqual(first.set_index('gid').loc[9, 'priority_score'], second.set_index('gid').loc[9, 'priority_score'])
        self.assertIn('исходящие могут быть невидимы', second.set_index('gid').loc[9, 'priority_why'])

    def test_role_strength_is_not_priority(self):
        source = ranked_fixture()
        source.loc[source.gid.eq(3), 'role_score'] = .4
        source.loc[source.gid.eq(9), 'role_score'] = .99
        f = calculate_priority(source).set_index('gid')
        self.assertGreater(f.loc[3, 'priority_score'], f.loc[9, 'priority_score'])

    def test_empty_missing_temporal_and_large_gids(self):
        source = ranked_fixture()
        empty = calculate_priority(source.iloc[:0])
        self.assertTrue(build_top_nodes(empty).empty)
        self.assertEqual(priority_diagnostics(empty.assign(cluster_id=pd.Series(dtype='int64')), build_top_nodes(empty))['quantiles']['min'], None)
        source['gid'] = source.gid.astype('uint64') + 2 ** 63
        source[['short_window_outflow_ratio', 'same_day_outflow_ratio']] = np.nan
        f = calculate_priority(source)
        self.assertEqual(f.gid.tolist(), source.gid.tolist())
        self.assertTrue(f.priority_score.between(0, 1).all())

    def test_bad_config_rejected(self):
        config = deepcopy(PRIORITY_CONFIG)
        config['weights']['structural'] = .8
        with self.assertRaises(ValueError):
            calculate_priority(ranked_fixture(), config)
        with self.assertRaises(ValueError):
            build_top_nodes(calculate_priority(ranked_fixture()), 5)

    def test_unsigned_zero_counts_do_not_underflow(self):
        source = ranked_fixture()
        for column in ('reachable_seed_count', 'direct_seed_in', 'max_unique_in_senders_per_day'):
            source[column] = pd.Series(0, index=source.index, dtype='uint64')
        f = calculate_priority(source)
        self.assertTrue(f.priority_seed_relevance.eq(0).all())
        self.assertTrue(f.priority_sync_pct.eq(0).all())

    def test_default_top50(self):
        source = calculate_priority(ranked_fixture())
        large = pd.concat([source] * 6, ignore_index=True)
        large['gid'] = range(len(large))
        top = build_top_nodes(large)
        self.assertEqual(len(top), 50)
        self.assertEqual(top['rank'].tolist(), list(range(1, 51)))


class OutputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        tables = fixture()
        for name, table in zip(('nodes', 'edges', 'transactions'), tables):
            table.to_parquet(cls.root / f'{name}.parquet', index=False)
        cls.graph = build_graph(validate_data(*tables))
        cls.report = run_pipeline(cls.root, cls.root / 'out')
        cls.nodes = pd.read_csv(cls.root / 'out/nodes_roles.csv')
        cls.clusters = pd.read_csv(cls.root / 'out/clusters.csv', dtype={'top_gids': str})
        cls.top = pd.read_csv(cls.root / 'out/top_nodes.csv')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saved_files_validated(self):
        report = validate_output_files(self.root / 'out', self.graph)
        self.assertTrue(report['valid'])
        self.assertEqual(report['n_nodes'], 5)
        self.assertEqual(self.report['output_validation'], report)

    def test_cross_file_corruptions_rejected(self):
        cases = [('nodes', 'cluster_id', 999), ('nodes', 'priority_score', np.nan),
                 ('clusters', 'n_nodes', 999), ('clusters', 'n_seed', 999),
                 ('clusters', 'sum_kzt_internal', 999999.), ('clusters', 'top_gids', '999999'),
                 ('clusters', 'hypothesis', ''), ('top', 'role', 'invalid'),
                 ('top', 'rank', 0), ('top', 'gid', 999999), ('top', 'why', ''),
                 ('top', 'priority_score', .999)]
        for table, column, value in cases:
            with self.subTest(table=table, column=column):
                frames = {'nodes': self.nodes.copy(), 'clusters': self.clusters.copy(), 'top': self.top.copy()}
                frames[table].loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_outputs(frames['nodes'], frames['clusters'], frames['top'], self.graph)

    def test_empty_and_large_gid_pipeline(self):
        for mode in ('empty', 'large'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tables = fixture()
                if mode == 'empty':
                    tables = tuple(t.iloc[:0] for t in tables)
                else:
                    for table in tables:
                        for col in ('gid', 'src', 'dst'):
                            if col in table:
                                table[col] = table[col].astype('uint64') + 2 ** 63
                for name, table in zip(('nodes', 'edges', 'transactions'), tables):
                    table.to_parquet(root / f'{name}.parquet', index=False)
                report = run_pipeline(root, root / 'out')
                self.assertTrue(report['output_validation']['valid'])
                self.assertEqual(report['output_validation']['n_nodes'], len(tables[0]))


if __name__ == '__main__':
    unittest.main()
