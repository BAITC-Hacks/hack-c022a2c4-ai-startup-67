"""Синтетические проверки ролей, ограничений видимости и объяснений."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from analytics.features import basic_features, build_graph
from analytics.graph_features import structural_features
from analytics.temporal import temporal_features
from analytics.roles import (ALLOWED_ROLES, ROLE_CONFIG, SCORED_ROLES, classify_roles,
                             positive_percentile, role_diagnostics, select_role,
                             validate_role_output)
from analytics.explanations import generate_evidence, number
from utils.validation import validate_data


def feature_fixture(transfers, seeds=(1,), depths=None, isolates=()):
    """Реальные расчёты признаков над вымышленными переводами."""
    tx = pd.DataFrame(transfers, columns=['src', 'dst', 'date', 'sum_kzt'])
    gids = sorted(set(tx.src) | set(tx.dst) | set(isolates))
    nodes = pd.DataFrame({'gid': gids, 'depth': [(depths or {}).get(g, 0 if g in seeds else 1) for g in gids],
                          'is_seed': [g in seeds for g in gids]})
    edges = tx.groupby(['src', 'dst'], as_index=False).agg(sum_kzt=('sum_kzt', 'sum'), n_tx=('sum_kzt', 'size'))
    edges['depth'] = 1
    data = validate_data(nodes, edges, tx)
    graph = build_graph(data)
    base, _ = structural_features(graph, basic_features(graph))
    return base.merge(temporal_features(data.transactions, data.nodes), on='gid', validate='one_to_one')


def chain():
    return feature_fixture([(1, 2, '2026-07-01', 100.), (2, 3, '2026-07-02', 90.)],
                           depths={1: 0, 2: 1, 3: 2}, isolates=(4,))


class RoleTests(unittest.TestCase):
    def test_consolidator_many_senders(self):
        f = feature_fixture([(g, 10, '2026-07-01', 100.) for g in range(1, 7)], seeds=tuple(range(1, 7)))
        row = classify_roles(f).set_index('gid').loc[10]
        self.assertEqual(row.role, 'consolidator')
        self.assertGreater(row.consolidator_score, .8)
        self.assertTrue(row.consolidator_eligible)
        self.assertIn('6 отправителей', row.evidence)
        self.assertIn('600 KZT', row.evidence)

    def test_distributor_many_receivers(self):
        f = feature_fixture([(1, g, '2026-07-01', 100.) for g in range(2, 12)])
        row = classify_roles(f).set_index('gid').loc[1]
        self.assertEqual(row.role, 'distributor')
        self.assertGreater(row.distributor_score, .8)
        self.assertIn('10 получателей', row.evidence)
        self.assertIn('seed: входящие видны не полностью', row.evidence)

    def test_one_huge_transfer_is_not_collection_or_distribution(self):
        f = feature_fixture([(1, 2, '2026-07-01', 1e15)])
        result = classify_roles(f)
        self.assertTrue(result.consolidator_score.eq(0).all())
        self.assertTrue(result.distributor_score.eq(0).all())
        self.assertFalse(result.coordinator_eligible.any())

    def test_transit_observed_with_window(self):
        row = classify_roles(chain()).set_index('gid').loc[2]
        self.assertEqual(row.role, 'transit')
        self.assertAlmostEqual(row.transit_score, .4 * .9 + .45 * .9)
        self.assertEqual(row.role_score, row.transit_score)
        self.assertIn('выход/вход=0.90', row.evidence)
        self.assertIn('эвристика D…D+2=90%', row.evidence)

    def test_late_outflow_and_missing_temporal_signal_block_transit(self):
        f = feature_fixture([(1, 2, '2026-07-01', 100.), (2, 3, '2026-07-04', 100.)])
        self.assertFalse(classify_roles(f).set_index('gid').loc[2, 'transit_eligible'])
        f.loc[f.gid.eq(2), 'short_window_outflow_ratio'] = np.nan
        self.assertEqual(classify_roles(f).set_index('gid').loc[2, 'transit_score'], 0)

    def test_terminal_inside_visible_depth(self):
        row = classify_roles(chain()).set_index('gid').loc[3]
        self.assertEqual(row.role, 'terminal')
        self.assertTrue(row.terminal_eligible)
        self.assertIn('исходящих переводов не видно', row.evidence)
        self.assertIn('depth=2', row.evidence)

    def test_truncated_is_ineligible_even_with_incorrect_flag(self):
        f = chain()
        f.loc[f.gid.eq(3), 'depth'] = 4
        for flag in (False, True):
            f.loc[f.gid.eq(3), 'truncated_by_depth'] = flag
            row = classify_roles(f).set_index('gid').loc[3]
            self.assertEqual(row.role, 'peripheral')
            self.assertFalse(row.terminal_eligible)
            self.assertEqual(row.terminal_score, 0)
            self.assertIn('исходящие могут быть невидимы', row.evidence)

    def test_truncated_can_still_be_consolidator(self):
        f = feature_fixture([(g, 10, '2026-07-01', 100.) for g in range(1, 7)],
                            seeds=tuple(range(1, 7)), depths={10: 4})
        row = classify_roles(f).set_index('gid').loc[10]
        self.assertEqual(row.role, 'consolidator')
        self.assertIn('depth=4', row.evidence)
        self.assertFalse(row.terminal_eligible)

    def test_coordinator_two_seed_branches(self):
        f = feature_fixture([(1, 3, '2026-07-01', 100.), (2, 3, '2026-07-01', 100.),
                             (3, 4, '2026-07-04', 10.), (3, 5, '2026-07-04', 10.),
                             (4, 6, '2026-07-05', 10.), (5, 7, '2026-07-05', 10.)], seeds=(1, 2))
        row = classify_roles(f).set_index('gid').loc[3]
        self.assertEqual(row.role, 'coordinator')
        self.assertGreater(row.coordinator_score, .8)
        self.assertEqual(row.reachable_seed_count, 2)
        self.assertIn('betweenness p100', row.evidence)
        self.assertIn('достижим от 2 seed', row.evidence)

    def test_peripheral_isolate_and_zero_money(self):
        f = feature_fixture([(1, 2, '2026-07-01', 0.)], isolates=(3,))
        result = classify_roles(f)
        self.assertTrue(result.role.eq('peripheral').all())
        self.assertTrue(result.role_score.eq(0).all())
        self.assertEqual(len(result), 3)

    def test_seed_ignores_misleading_flow_balance(self):
        f = chain()
        f.loc[f.gid.eq(2), 'is_seed'] = True
        for ratio in (.01, 1., 1000., np.nan):
            f.loc[f.gid.eq(2), 'pass_through'] = ratio
            # Даже если вызывающий код передал seed временную эвристику.
            f.loc[f.gid.eq(2), 'short_window_outflow_ratio'] = 1
            row = classify_roles(f).set_index('gid').loc[2]
            self.assertFalse(row.transit_eligible)
            self.assertEqual(row.transit_score, 0)
            self.assertEqual(row.transit_balance, 0)
            self.assertEqual(row.transit_window, 0)
            self.assertNotEqual(row.role, 'transit')

    def test_normalization_ties_zero_missing_extreme(self):
        values = pd.Series([0., 0., 1., 1., 2., 1e30, np.nan])
        expected = pd.Series([0., 0., .375, .375, .75, 1., 0.])
        pd.testing.assert_series_equal(positive_percentile(values), expected)
        self.assertEqual(positive_percentile(pd.Series([7., 7.])).tolist(), [.75, .75])
        for invalid in (-1., np.inf):
            with self.assertRaises(ValueError):
                positive_percentile(pd.Series([invalid]))

    def test_selection_ties_margin_threshold_and_exact_score(self):
        scores = dict.fromkeys(SCORED_ROLES, 0.)
        eligible = dict.fromkeys(SCORED_ROLES, True)
        self.assertEqual(select_role(scores, eligible), ('peripheral', 0.))
        scores.update(transit=.80, consolidator=.805, coordinator=.83)
        self.assertEqual(select_role(scores, eligible), ('transit', .80))
        # Координатор должен превысить лучший другой score минимум на .05.
        scores['coordinator'] = .86
        self.assertEqual(select_role(scores, eligible), ('coordinator', .86))
        eligible['coordinator'] = False
        self.assertEqual(select_role(scores, eligible), ('transit', .80))
        scores['consolidator'] = .83
        self.assertEqual(select_role(scores, eligible), ('consolidator', .83))
        scores = dict.fromkeys(SCORED_ROLES, 0.)
        scores['terminal'] = ROLE_CONFIG['min_role_score']
        self.assertEqual(select_role(scores, eligible), ('terminal', .65))

    def test_row_shuffle_preserves_results_and_raw_features(self):
        f = chain()
        expected = classify_roles(f)
        actual = classify_roles(f.sample(frac=1, random_state=19)).sort_values('gid').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(expected[f.columns], f)
        self.assertTrue(expected.role.isin(ALLOWED_ROLES).all())
        self.assertTrue(expected.role_score.between(0, 1).all())
        self.assertTrue(expected.evidence.str.len().between(1, 200).all())

    def test_empty_and_large_identifiers(self):
        f = chain()
        self.assertTrue(classify_roles(f.iloc[:0]).empty)
        f['gid'] = f.gid.astype('uint64') + 2 ** 63
        self.assertEqual(classify_roles(f).gid.tolist(), f.gid.tolist())
        report = role_diagnostics(classify_roles(f.iloc[:0]))
        self.assertTrue(all(value == 0 for value in report['counts'].values()))
        self.assertTrue(all(value is None for value in report['mean_role_score'].values()))

    def test_config_and_output_validation(self):
        config = deepcopy(ROLE_CONFIG)
        config['weights']['transit']['transit_window'] = -1
        with self.assertRaises(ValueError):
            classify_roles(chain(), config)
        config = deepcopy(ROLE_CONFIG)
        config['min_role_score'] = .99
        self.assertTrue(classify_roles(chain(), config).role.eq('peripheral').all())
        f = classify_roles(chain())
        for col, value in (('role', None), ('role_score', np.nan), ('evidence', ''),
                           ('evidence', 'x' * 201), ('role_score', .1234)):
            broken = f.copy()
            broken.loc[0, col] = value
            with self.assertRaises(ValueError):
                validate_role_output(broken, f.gid)
        with self.assertRaises(ValueError):
            classify_roles(pd.concat([chain(), chain()]))

    def test_evidence_matches_values_and_stays_concise(self):
        f = classify_roles(chain())
        for row in f.to_dict('records'):
            self.assertEqual(row['evidence'], generate_evidence(row))
            self.assertNotIn('преступ', row['evidence'])
        row = f[f.role.eq('terminal')].iloc[0].to_dict()
        self.assertIn(number(row['in_kzt']), row['evidence'])
        row.update(in_kzt=1e300, in_deg=10 ** 40, is_seed=True, depth=4, truncated_by_depth=True)
        text = generate_evidence(row)
        self.assertLessEqual(len(text), 200)
        self.assertIn('исходящие могут быть невидимы', text)
        self.assertIn('входящие видны не полностью', text)


if __name__ == '__main__':
    unittest.main()
