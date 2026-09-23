"""Validation-only tests: python -m unittest tests.test_validation -v."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from utils.validation import DataValidationError, validate_data, load_data
from tests.fixtures import fixture

class ValidationTests(unittest.TestCase):

    def setUp(self):
        (self.nodes, self.edges, self.tx) = fixture()

    def data(self):
        return validate_data(self.nodes, self.edges, self.tx)

    def test_mismatched_money_diagnostic(self):
        self.edges.loc[0, 'sum_kzt'] = 101.0
        with self.assertRaisesRegex(DataValidationError, 'sum_kzt.*edge_value.*transaction_aggregate.*difference'):
            self.data()

    def test_mismatched_count_diagnostic(self):
        self.edges.loc[0, 'n_tx'] = 3
        with self.assertRaisesRegex(DataValidationError, 'n_tx.*edge_value.*transaction_aggregate.*difference'):
            self.data()

    def test_money_tolerance(self):
        self.edges.loc[0, 'sum_kzt'] += 0.001
        self.data()

    def test_pair_mismatch(self):
        self.tx.loc[0, 'dst'] = 3
        with self.assertRaisesRegex(DataValidationError, 'pair mismatch'):
            self.data()

    def test_duplicate_edges_rejected(self):
        self.edges = pd.concat([self.edges, self.edges.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(DataValidationError, 'duplicate directed pairs'):
            self.data()

    def test_invalid_inputs(self):
        cases = [('nodes', 'gid', None, 'null'), ('nodes', 'depth', 5, '0..4'), ('nodes', 'is_seed', None, 'null'), ('edges', 'src', 999, 'unknown gids'), ('transactions', 'dst', 999, 'unknown gids'), ('edges', 'sum_kzt', -1.0, 'nonnegative'), ('transactions', 'sum_kzt', np.inf, 'finite'), ('edges', 'n_tx', 0, 'positive'), ('transactions', 'date', 'invalid', 'invalid datetime'), ('transactions', 'date', None, 'null')]
        for (table, column, value, message) in cases:
            with self.subTest(table=table, column=column, value=value):
                (nodes, edges, tx) = fixture()
                frames = {'nodes': nodes, 'edges': edges, 'transactions': tx}
                if value is None:
                    frames[table][column] = frames[table][column].astype(object)
                frames[table].loc[0, column] = value
                with self.assertRaisesRegex(DataValidationError, message):
                    validate_data(**frames)

    def test_float_identifiers_rejected(self):
        self.nodes['gid'] = self.nodes.gid.astype(float)
        with self.assertRaisesRegex(DataValidationError, 'integer dtype'):
            self.data()

    def test_duplicate_gid_and_missing_column(self):
        self.nodes.loc[1, 'gid'] = 1
        with self.assertRaisesRegex(DataValidationError, 'unique'):
            self.data()
        (self.nodes, self.edges, self.tx) = fixture()
        self.edges = self.edges.drop(columns='depth')
        with self.assertRaisesRegex(DataValidationError, 'missing required columns'):
            self.data()

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataValidationError, 'Missing required parquet files'):
                load_data(Path(directory))
