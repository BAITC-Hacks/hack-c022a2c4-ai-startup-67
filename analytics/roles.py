"""Детерминированные роли наблюдаемого графа; оценки — не вероятности риска."""
from copy import deepcopy

import numpy as np
import pandas as pd

from analytics.explanations import generate_evidence

SCORED_ROLES = ('consolidator', 'transit', 'distributor', 'terminal', 'coordinator')
ALLOWED_ROLES = (*SCORED_ROLES, 'peripheral')
ROLE_CONFIG = {
    'min_role_score': 0.65,
    'min_in_counterparties': 3,
    'min_out_counterparties': 3,
    'min_transit_balance': 0.5,
    'min_transit_window': 0.5,
    'min_coordinator_betweenness_pct': 0.8,
    'min_coordinator_seeds': 2,
    'tie_tolerance': 0.02,
    'coordinator_override_margin': 0.05,
    # При близких оценках: совместное наличие потока и временного сигнала;
    # затем многосторонний сбор, раздача, наконец отсутствие видимого оттока.
    'tie_precedence': ('transit', 'consolidator', 'distributor', 'terminal'),
    'weights': {
        'consolidator': {'in_deg_pct': 0.40, 'in_tx_pct': 0.20, 'in_kzt_pct': 0.10,
                         'seed_support': 0.10, 'max_unique_in_senders_per_day_pct': 0.10,
                         'incoming_spread': 0.10},
        'distributor': {'out_deg_pct': 0.55, 'out_tx_pct': 0.20,
                        'out_kzt_pct': 0.10, 'outgoing_spread': 0.15},
        'transit': {'transit_balance': 0.40, 'transit_window': 0.45,
                    'repeat_support': 0.15},
        'terminal': {'terminal_observed_endpoint': 0.55, 'in_kzt_pct': 0.20,
                     'active_days_in_pct': 0.15, 'in_deg_pct': 0.10},
        'coordinator': {'betweenness_pct': 0.60, 'pagerank_pct': 0.20,
                        'reachable_seed_count_pct': 0.20},
    },
}
NORMALIZED_FEATURES = (
    'in_deg', 'out_deg', 'in_tx', 'out_tx', 'in_kzt', 'out_kzt',
    'betweenness', 'pagerank', 'reachable_seed_count', 'direct_seed_in',
    'max_unique_in_senders_per_day', 'active_days_in',
    'repeated_in_counterparties', 'repeated_out_counterparties',
)


def positive_percentile(values: pd.Series) -> pd.Series:
    """Средний ранг / число положительных значений; нули и NaN дают 0.

    Равные значения имеют одинаковый ранг. Нулевая betweenness не становится
    высоким перцентилем из-за массы нулей. Нормировка зависит от текущей выборки.
    """
    numeric = pd.to_numeric(values, errors='raise').astype(float)
    if np.isinf(numeric).any() or numeric.lt(0).any():
        raise ValueError('Role normalization requires nonnegative finite values or NaN')
    return numeric.where(numeric.gt(0)).rank(method='average', pct=True).fillna(0.0)


def validate_config(config: dict) -> None:
    for name in ('min_role_score', 'min_transit_balance', 'min_transit_window',
                 'min_coordinator_betweenness_pct', 'tie_tolerance', 'coordinator_override_margin'):
        if not np.isfinite(config[name]) or not 0 <= config[name] <= 1:
            raise ValueError(f'{name} must be in [0,1]')
    for name in ('min_in_counterparties', 'min_out_counterparties', 'min_coordinator_seeds'):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    precedence = config['tie_precedence']
    if len(precedence) != 4 or set(precedence) != set(SCORED_ROLES) - {'coordinator'}:
        raise ValueError('tie_precedence must contain each non-coordinator role once')
    if set(config['weights']) != set(SCORED_ROLES):
        raise ValueError('Each scored role requires weights')
    for role, weights in config['weights'].items():
        if (not weights or not all(np.isfinite(w) and w >= 0 for w in weights.values())
                or not np.isclose(sum(weights.values()), 1.0)):
            raise ValueError(f'{role}: weights must be nonnegative and sum to 1')


def select_role(scores: dict, eligible: dict, config: dict = ROLE_CONFIG) -> tuple[str, float]:
    """Максимальная допустимая оценка с явным правилом близких результатов.

    Coordinator заменяет другую роль только с запасом >= заданного margin.
    Среди остальных в пределах tolerance от максимума действует precedence.
    Peripheral имеет собственную нулевую оценку: сильный паттерн не установлен.
    """
    candidates = {role: float(scores[role]) for role in SCORED_ROLES
                  if eligible[role] and np.isfinite(scores[role])
                  and scores[role] >= config['min_role_score']}
    if not candidates:
        return 'peripheral', 0.0
    others = {role: score for role, score in candidates.items() if role != 'coordinator'}
    if 'coordinator' in candidates:
        if not others or candidates['coordinator'] >= max(others.values()) + config['coordinator_override_margin']:
            return 'coordinator', candidates['coordinator']
    maximum = max(others.values())
    for role in config['tie_precedence']:
        if role in others and maximum - others[role] <= config['tie_tolerance']:
            return role, others[role]
    raise AssertionError('No role selected from eligible candidates')


def classify_roles(features: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Сохраняет исходные признаки, добавляет компоненты, оценки, роли и объяснения."""
    config = deepcopy(ROLE_CONFIG if config is None else config)
    validate_config(config)
    frame = features.copy()
    if frame.gid.isna().any() or frame.gid.duplicated().any():
        raise ValueError('Role input requires unique non-null gids')
    for column in NORMALIZED_FEATURES:
        frame[f'{column}_pct'] = positive_percentile(frame[column])
    frame['seed_support'] = frame[['direct_seed_in_pct', 'reachable_seed_count_pct']].max(axis=1)
    for direction, amount in (('incoming', 'in_kzt'), ('outgoing', 'out_kzt')):
        frame[f'{direction}_spread'] = (1 - frame[f'{direction}_amount_hhi']).clip(0, 1).fillna(0).where(
            frame[amount].gt(0), 0.0)
    # Обе стороны должны быть видимы. Отношение 2 или 1/2 даёт 0.5;
    # огромный исходящий поток не превращается в «идеальный транзит».
    ratio = frame.pass_through.where(frame.pass_through.gt(0))
    frame['transit_balance'] = np.minimum(ratio, 1 / ratio).fillna(0).clip(0, 1)
    frame['transit_window'] = frame.short_window_outflow_ratio.fillna(0).clip(0, 1)
    frame.loc[frame.is_seed, ['transit_balance', 'transit_window']] = 0.0
    frame['repeat_support'] = frame[['repeated_in_counterparties_pct',
                                     'repeated_out_counterparties_pct']].mean(axis=1)
    # Проверяем и исходную глубину: ошибочный входной флаг не разрешит terminal.
    boundary = frame.truncated_by_depth | (frame.depth.eq(4) & frame.out_deg.eq(0))
    positive_in = frame.in_deg.gt(0) & frame.in_kzt.gt(0)
    positive_out = frame.out_deg.gt(0) & frame.out_kzt.gt(0)
    eligibility = {
        'consolidator': frame.in_deg.ge(config['min_in_counterparties']) & positive_in,
        'distributor': frame.out_deg.ge(config['min_out_counterparties']) & positive_out,
        'transit': (positive_in & positive_out & ~frame.is_seed
                    & frame.transit_balance.ge(config['min_transit_balance'])
                    & frame.transit_window.ge(config['min_transit_window'])),
        'terminal': positive_in & frame.out_deg.eq(0) & ~boundary,
        'coordinator': (frame.betweenness.gt(0)
                        & frame.betweenness_pct.ge(config['min_coordinator_betweenness_pct'])
                        & frame.reachable_seed_count.ge(config['min_coordinator_seeds'])
                        & frame.in_deg.gt(0) & frame.out_deg.gt(0)),
    }
    frame['terminal_observed_endpoint'] = eligibility['terminal'].astype(float)
    for role in SCORED_ROLES:
        frame[f'{role}_eligible'] = eligibility[role]
        score = sum(frame[component] * weight for component, weight in config['weights'][role].items())
        frame[f'{role}_score'] = score.where(eligibility[role], 0.0).clip(0, 1)
    frame['peripheral_score'] = 0.0
    selected = [select_role(
        {role: row[f'{role}_score'] for role in SCORED_ROLES},
        {role: row[f'{role}_eligible'] for role in SCORED_ROLES}, config)
        for row in frame.to_dict('records')]
    frame['role'] = pd.Series([role for role, _ in selected], index=frame.index, dtype='object')
    frame['role_score'] = pd.Series([score for _, score in selected], index=frame.index, dtype=float)
    frame['evidence'] = pd.Series([generate_evidence(row) for row in frame.to_dict('records')],
                                  index=frame.index, dtype='object')
    validate_role_output(frame, features.gid)
    return frame


def validate_role_output(frame: pd.DataFrame, expected_gids: pd.Series) -> None:
    """Проверка контракта перед экспортом; никаких пропавших узлов или пустых ролей."""
    if (len(frame) != len(expected_gids) or not frame.gid.is_unique
            or frame.gid.isna().any() or set(frame.gid) != set(expected_gids)):
        raise ValueError('Role output must preserve every gid exactly once')
    if not frame.role.isin(ALLOWED_ROLES).all():
        raise ValueError('Unknown or missing role')
    for column in [f'{role}_score' for role in ALLOWED_ROLES] + ['role_score']:
        if not frame[column].between(0, 1).all():
            raise ValueError(f'{column} must be finite and in [0,1]')
    selected = np.array([row[f'{row["role"]}_score'] for row in frame.to_dict('records')])
    if not np.array_equal(frame.role_score.to_numpy(), selected):
        raise ValueError('role_score differs from the selected role score')
    if not frame.evidence.map(lambda value: isinstance(value, str) and 0 < len(value.strip()) <= 200).all():
        raise ValueError('Evidence must contain 1..200 characters')
    boundary = frame.truncated_by_depth | (frame.depth.eq(4) & frame.out_deg.eq(0))
    if (boundary & frame.role.eq('terminal')).any():
        raise ValueError('Truncated nodes cannot be terminal')
    if (frame.is_seed & frame.role.eq('transit')).any():
        raise ValueError('Seed transit is disabled due to incomplete incoming visibility')


def role_diagnostics(frame: pd.DataFrame) -> dict:
    groups = frame.groupby('role').role_score
    return {
        'counts': {role: int(frame.role.eq(role).sum()) for role in ALLOWED_ROLES},
        'median_role_score': {role: float(groups.median()[role]) if role in groups.groups else None
                              for role in ALLOWED_ROLES},
        'mean_role_score': {role: float(groups.mean()[role]) if role in groups.groups else None
                            for role in ALLOWED_ROLES},
        'n_peripheral': int(frame.role.eq('peripheral').sum()),
        'n_truncated_terminal': int(((frame.truncated_by_depth | (frame.depth.eq(4) & frame.out_deg.eq(0)))
                                     & frame.role.eq('terminal')).sum()),
        'n_seed_transit': int((frame.is_seed & frame.role.eq('transit')).sum()),
    }
