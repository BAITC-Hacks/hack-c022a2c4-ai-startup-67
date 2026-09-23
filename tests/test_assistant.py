"""Provider-independent tool and assistant contract tests; no live API calls."""
import json
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from assistant.tools import AnalysisTools, TOOL_SCHEMAS, TOOL_ARGUMENTS
from assistant.service import answer_question, UNAVAILABLE, SYSTEM_PROMPT
from tests import test_dashboard as dashboard_fixture

ROOT = dashboard_fixture.ROOT


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dashboard_fixture.DashboardTests.setUpClass()
        cls.tools = AnalysisTools(dashboard_fixture.DashboardTests.nodes, dashboard_fixture.DashboardTests.clusters, dashboard_fixture.DashboardTests.top, dashboard_fixture.DashboardTests.edges)

    @classmethod
    def tearDownClass(cls):
        dashboard_fixture.DashboardTests.tearDownClass()

    def test_all_tools_json_and_exact_gid(self):
        for name, fields in TOOL_ARGUMENTS.items():
            args = {k: {'gid': '2', 'role': 'peripheral', 'limit': 20, 'cluster_id': 0}[k] for k in fields}
            result = self.tools.call(name, args)
            json.dumps(result, allow_nan=False)
        self.assertEqual(self.tools.get_node_profile('2')['node']['gid'], '2')
        self.assertIn('error', self.tools.get_node_profile('unknown'))
        self.assertEqual(self.tools.get_node_profile('5')['node']['priority_score'], 0.)

    def test_directions_and_warnings(self):
        self.assertEqual(self.tools.get_incoming_counterparties('2')['rows'][0]['gid'], '1')
        self.assertEqual(self.tools.get_outgoing_counterparties('2')['rows'][0]['gid'], '3')
        self.assertTrue(self.tools.get_node_profile('3')['warnings'])
        self.assertTrue(self.tools.get_node_profile('1')['warnings'])
        self.assertEqual(self.tools.get_incoming_counterparties('5')['total_counterparties'], 0)

    def test_allowlist_and_bounded_limits(self):
        for name, args in [('__init__', {}), ('get_top_nodes', {'limit': 500}), ('get_top_nodes', {'limit': True}),
                           ('get_node_profile', {'gid': 2}), ('get_top_nodes', {'limit': 20, 'sql': 'DROP'}),
                           ('get_node_profile', None)]:
            self.assertIn('error', self.tools.call(name, args))
        self.assertTrue(all(t['strict'] and not t['parameters']['additionalProperties'] for t in TOOL_SCHEMAS))

    def test_actual_priority_and_cluster(self):
        row = self.tools.nodes.iloc[0]
        result = self.tools.get_node_priority_breakdown(row.gid)['stored_components']
        self.assertAlmostEqual(result['priority_score'], row.priority_score)
        largest = self.tools.get_cluster_with_most_seeds()['cluster']
        self.assertEqual(largest['n_seed'], self.tools.clusters.n_seed.max())
        for item in self.tools.find_multi_seed_nodes()['rows']:
            self.assertGreaterEqual(item['reachable_seed_count'], 2)

    def test_missing_optional_data(self):
        tools = AnalysisTools(self.tools.nodes, self.tools.clusters, self.tools.top)
        self.assertIn('error', tools.get_incoming_counterparties('2'))
        self.assertIsNone(tools.get_node_temporal_profile('2')['daily'])
        tx = pd.read_parquet(dashboard_fixture.DashboardTests.data_dir / 'transactions.parquet')
        tx['src'], tx['dst'], tx['date'] = tx.src.map(str), tx.dst.map(str), pd.to_datetime(tx.date)
        tools.transactions = tx
        self.assertEqual(sum(r['Входящие'] for r in tools.get_node_temporal_profile('2')['daily']), 100.)

    def test_unavailable_and_provider_failure(self):
        self.assertEqual(answer_question('Why?', '2', self.tools)['error'], UNAVAILABLE)
        client = Mock()
        client.responses.create.side_effect = RuntimeError('secret-must-not-leak')
        result = answer_question('Why?', '2', self.tools, client=client)
        self.assertIn('error', result)
        self.assertNotIn('secret-must-not-leak', json.dumps(result))
        client.responses.create.assert_called_once()

    def test_function_call_roundtrip(self):
        client = Mock()
        call = NS(type='function_call', name='get_incoming_counterparties', arguments='{"gid":"2"}', call_id='incoming')
        client.responses.create.side_effect = [NS(output=[call], output_text=''), NS(output=[], output_text='Finding: observed incoming flow.')]
        result = answer_question('Who sends?', '2', self.tools, client=client)
        self.assertIn('answer', result)
        self.assertEqual([x['tool'] for x in result['evidence']], ['get_node_profile', 'get_incoming_counterparties'])
        args = client.responses.create.call_args.kwargs
        self.assertFalse(args['store'])
        outputs = [x for x in args['input'] if isinstance(x, dict) and x.get('type') == 'function_call_output']
        self.assertEqual(json.loads(outputs[-1]['output'])['rows'][0]['gid'], '1')
        self.assertIn('Never invent', SYSTEM_PROMPT)

    def test_bad_tool_arguments_and_budgets(self):
        client = Mock()
        call = NS(type='function_call', name='get_top_nodes', arguments='not-json', call_id='bad')
        client.responses.create.side_effect = [NS(output=[call], output_text=''), NS(output=[], output_text='Data unavailable.')]
        result = answer_question('Test', '2', self.tools, client=client)
        self.assertIn('error', result['evidence'][-1]['result'])
        client.responses.create.reset_mock()
        client.responses.create.side_effect = None
        client.responses.create.return_value = NS(output=[call] * 9, output_text='')
        self.assertIn('error', answer_question('Test', '2', self.tools, client=client))
        self.assertEqual(client.responses.create.call_count, 1)

    def test_no_request_for_invalid_question_or_gid(self):
        client = Mock()
        for question, gid in [('', '2'), ('x' * 2001, '2'), ('Why?', '999')]:
            self.assertIn('error', answer_question(question, gid, self.tools, client=client))
        client.responses.create.assert_not_called()

    def test_ui_no_key_and_explicit_submit_only(self):
        env = {'MONEY_GRAPH_DATA_DIR': str(dashboard_fixture.DashboardTests.data_dir), 'MONEY_GRAPH_OUT_DIR': str(dashboard_fixture.DashboardTests.out)}
        with patch.dict(os.environ, env), patch('ui.assistant.setting', return_value=''):
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            self.assertFalse(app.exception)
            self.assertTrue(any(UNAVAILABLE in x.value for x in app.info))
        with patch.dict(os.environ, env), patch('ui.assistant.setting', return_value='test-placeholder'), \
                patch('ui.assistant.answer_question', return_value={'answer': 'Stored evidence.', 'evidence': []}) as answer:
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            answer.assert_not_called()
            next(x for x in app.text_input if x.label == 'Вопрос').set_value('Why?')
            next(x for x in app.button if x.label == 'Объяснить').click().run()
            self.assertFalse(app.exception)
            answer.assert_called_once()
            app.text_input(key='node_query').set_value('3').run()
            self.assertFalse(any(x.value == 'Stored evidence.' for x in app.markdown))
