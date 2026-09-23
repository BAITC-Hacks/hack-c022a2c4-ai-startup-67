"""UI и помощники проверяются на синтетических данных, без реальных gid."""
import importlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd
from streamlit.testing.v1 import AppTest

from pipeline import run_pipeline
from tests.fixtures import fixture
from ui.data import (load_outputs, fingerprint, missing_outputs, load_graph, load_edges,
                     find_node, filter_nodes, node_warnings, counterparties, daily_activity)
from ui.formatting import ROLE_COLORS
from visualization.graph_view import neighborhood, bounded_view, render_graph_html, MAX_NODES, MAX_EDGES

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.data_dir = cls.root / 'data'
        cls.out = cls.root / 'out'
        cls.data_dir.mkdir()
        for name, frame in zip(('nodes', 'edges', 'transactions'), fixture()):
            frame.to_parquet(cls.data_dir / f'{name}.parquet', index=False)
        run_pipeline(cls.data_dir, cls.out)
        cls.nodes, cls.clusters, cls.top, _ = load_outputs(str(cls.out), fingerprint(cls.out, ('nodes_roles.csv', 'clusters.csv', 'top_nodes.csv', 'node_features.csv')))
        cls.graph = load_graph(str(cls.data_dir), fingerprint(cls.data_dir, ('nodes.parquet', 'edges.parquet')))
        _, cls.edges = load_edges(str(cls.data_dir), fingerprint(cls.data_dir, ('nodes.parquet', 'edges.parquet')))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.environment = patch.dict(os.environ, {'MONEY_GRAPH_DATA_DIR': str(self.data_dir), 'MONEY_GRAPH_OUT_DIR': str(self.out)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def app(self):
        return AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()

    def test_app_import_and_all_tabs(self):
        self.assertTrue(callable(importlib.import_module('app').main))
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual([t.label for t in app.tabs], ['Dashboard', 'Client Analysis', 'Network', 'Clusters'])
        self.assertFalse(any('Визуализация временно' in warning.value for warning in app.warning))

    def test_csv_loading_and_exact_gid_search(self):
        self.assertEqual(len(self.nodes), 5)
        self.assertEqual(find_node(self.nodes, '2')['gid'], '2')
        self.assertEqual(find_node(self.nodes, ' 5 ')['gid'], '5')
        self.assertIsNone(find_node(self.nodes, '999'))
        self.assertIsNone(find_node(self.nodes, '2.0'))
        self.assertIsNone(find_node(self.nodes, '<script>'))

    def test_local_directions_and_hops(self):
        local = neighborhood(self.graph, '2', 1)
        self.assertEqual(set(local), {'1', '2', '3'})
        self.assertTrue(local.has_edge('1', '2'))
        self.assertTrue(local.has_edge('2', '3'))
        self.assertFalse(local.has_edge('2', '1'))
        self.assertEqual(set(neighborhood(self.graph, '2', 2)), {'1', '2', '3', '4'})
        self.assertEqual(set(neighborhood(self.graph, '5')), {'5'})
        self.assertEqual(len(neighborhood(self.graph, 'invalid')), 0)
        with self.assertRaises(ValueError):
            neighborhood(self.graph, '2', 3)
        self.assertTrue(nx.is_frozen(self.graph))

    def test_filters_are_applied(self):
        cid = find_node(self.nodes, '5')['cluster_id']
        f = filter_nodes(self.nodes, clusters=[cid])
        self.assertEqual(f.gid.tolist(), ['5'])
        self.assertEqual(set(filter_nodes(self.nodes, seed_only=True).gid), {'1', '5'})
        self.assertEqual(filter_nodes(self.nodes, truncated_only=True).gid.tolist(), ['3'])
        self.assertTrue(filter_nodes(self.nodes, minimum=1.).empty)
        f = filter_nodes(self.nodes, roles=['peripheral'])
        self.assertTrue(f.role.eq('peripheral').all())

    def test_role_colors_and_node_warnings(self):
        self.assertEqual(set(ROLE_COLORS), {'consolidator', 'transit', 'distributor', 'terminal', 'coordinator', 'peripheral'})
        self.assertTrue(any('seed' in text for text in node_warnings(find_node(self.nodes, '1'), self.graph)))
        self.assertTrue(any('4-го' in text for text in node_warnings(find_node(self.nodes, '3'), self.graph)))
        self.assertTrue(any('нет рёбер' in text for text in node_warnings(find_node(self.nodes, '5'), self.graph)))
        self.assertFalse(node_warnings(find_node(self.nodes, '2'), self.graph))

    def test_graph_cap_and_selected_preserved(self):
        graph = nx.complete_graph(180, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph, lambda g: str(g))
        nx.set_edge_attributes(graph, 100., 'sum_kzt')
        nx.set_edge_attributes(graph, 1, 'n_tx')
        frame = pd.DataFrame({'gid': list(graph), 'priority_score': [0.] * len(graph), 'role_score': [0.] * len(graph)})
        bounded, info = bounded_view(graph, frame, graph.nodes, selected='179')
        self.assertIn('179', bounded)
        self.assertEqual(len(bounded), MAX_NODES)
        self.assertEqual(bounded.number_of_edges(), MAX_EDGES)
        self.assertEqual(graph.number_of_edges(), 180 * 179)
        self.assertEqual(info['candidate_nodes'], 180)

    def test_offline_html_arrows_escaping_and_large_ids(self):
        first, second = '18446744073709551610', '18446744073709551611'
        records = [{'gid': first, 'role': 'coordinator', 'priority_score': .8},
                   {'gid': second, 'role': '<script>alert(1)</script>', 'priority_score': .1}]
        html = render_graph_html(records, [{'src': first, 'dst': second, 'sum_kzt': 100., 'n_tx': 2}], first)
        self.assertNotRegex(html, r'<script[^>]+src=[\"\']https?://')
        self.assertNotRegex(html, r'<link[^>]+href=[\"\']https?://')
        node_json = re.search(r'nodes = new vis.DataSet\((\[.*?\])\);', html).group(1)
        edge_json = re.search(r'edges = new vis.DataSet\((\[.*?\])\);', html).group(1)
        nodes, edges = json.loads(node_json), json.loads(edge_json)
        self.assertEqual(nodes[0]['id'], first)
        self.assertEqual(nodes[1]['label'], '')
        self.assertNotIn('<script>', nodes[1]['title'])
        self.assertEqual(edges[0]['from'], first)
        self.assertEqual(edges[0]['to'], second)
        self.assertTrue(edges[0]['arrows']['to']['enabled'])
        self.assertIn('ResizeObserver', html)
        self.assertIn('"physics": {"enabled": false}', html)

    def test_counterparties_and_daily_dates(self):
        self.assertEqual(counterparties(self.edges, '2', True).gid.tolist(), ['1'])
        self.assertEqual(counterparties(self.edges, '2', False).gid.tolist(), ['3'])
        tx = pd.read_parquet(self.data_dir / 'transactions.parquet')
        tx['src'], tx['dst'], tx['date'] = tx.src.map(str), tx.dst.map(str), pd.to_datetime(tx.date)
        daily = daily_activity(tx, '2')
        self.assertEqual(daily['Входящие'].sum(), 100)
        self.assertEqual(daily['Исходящие'].sum(), 50)
        self.assertEqual(len(daily), 4)
        self.assertTrue(daily_activity(tx, '5').empty)

    def test_app_search_seed_truncated_isolate_invalid(self):
        app = self.app()
        for gid, warning in [('1', 'seed'), ('3', '4-го'), ('5', 'нет рёбер'), ('999', None)]:
            app.text_input(key='search_gid').set_value(gid).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['selected_gid'], gid)
            if warning:
                self.assertTrue(any(warning in item.value for item in app.warning))
            else:
                self.assertTrue(any('GID не найден' in item.value for item in app.info))
        app.text_input(key='node_query').set_value('2').run()
        self.assertEqual(app.session_state['search_gid'], '2')

    def test_priority_selection_and_top100_fallback(self):
        app = self.app()
        app.selectbox(key='priority_count').select(100).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('только 5 строк' in item.value for item in app.info))
        gid = self.top.gid.iloc[-1]
        app.selectbox(key='priority_pick').select(gid).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['selected_gid'], gid)

    def test_network_modes_cluster_and_empty_filters(self):
        app = self.app()
        for mode in ('Selected cluster', 'Top priority network'):
            app.selectbox(key='network_mode').select(mode).run()
            self.assertFalse(app.exception)
        cid = int(find_node(self.nodes, '5')['cluster_id'])
        app.selectbox(key='cluster_pick').select(cid).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.slider), 0)

    def test_missing_outputs_stop_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(missing_outputs(directory)), 3)
            with patch.dict(os.environ, {'MONEY_GRAPH_OUT_DIR': directory}):
                app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any('Analysis files are missing' in item.value for item in app.error))
            self.assertTrue(any('python pipeline.py' in item.value for item in app.code))

    def test_optional_columns_and_parquet_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minimal = self.nodes[['gid', 'role', 'role_score', 'cluster_id', 'priority_score', 'evidence']]
            minimal.to_csv(root / 'nodes_roles.csv', index=False)
            self.clusters.to_csv(root / 'clusters.csv', index=False)
            self.top.to_csv(root / 'top_nodes.csv', index=False)
            with patch.dict(os.environ, {'MONEY_GRAPH_OUT_DIR': directory, 'MONEY_GRAPH_DATA_DIR': str(root / 'missing')}):
                app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any('Компоненты приоритета отсутствуют' in item.value for item in app.info))

    def test_visualization_failure_has_fallback(self):
        with patch('ui.components.cached_graph_html', side_effect=ValueError('synthetic failure')):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any('Визуализация временно' in item.value for item in app.warning))

    def test_ui_does_not_recalculate_analytics(self):
        with patch('networkx.pagerank', side_effect=AssertionError('must use exports')), \
             patch('networkx.betweenness_centrality', side_effect=AssertionError('must use exports')), \
             patch('networkx.community.louvain_communities', side_effect=AssertionError('must use exports')):
            app = self.app()
            app.text_input(key='search_gid').set_value('3').run()
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
