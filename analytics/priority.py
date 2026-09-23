"""Приоритет аналитической проверки, не вероятность нарушения или виновности."""
from copy import deepcopy

import numpy as np
import pandas as pd

from analytics.roles import ALLOWED_ROLES, positive_percentile
from analytics.explanations import build_priority_explanation

PRIORITY_CONFIG = {
    'weights': {'structural': .35, 'seed_relevance': .25, 'role': .15, 'flow': .15, 'temporal': .10},
    'structural': {'betweenness_pct': .60, 'pagerank_pct': .30, 'coordinator_score': .10},
    'seed_relevance': {'priority_reachable_extra_pct': .75, 'priority_direct_extra_pct': .25},
    'role_relevance': {'coordinator': 1., 'consolidator': .9, 'distributor': .9,
                       'transit': .85, 'terminal': .35, 'peripheral': 0.},
    'flow': {'in_kzt_pct': .20, 'out_kzt_pct': .20, 'in_tx_pct': .15,
             'out_tx_pct': .15, 'in_deg_pct': .15, 'out_deg_pct': .15},
    'temporal': {'priority_window': .35, 'priority_same_day': .10,
                 'priority_sync_pct': .25, 'priority_sync_days_pct': .15, 'repeat_support': .15},
    'top_n': 50,
}
PRIORITY_COMPONENTS = tuple(PRIORITY_CONFIG['weights'])
TOP_COLUMNS = ['rank', 'gid', 'role', 'priority_score', 'why']


def validate_priority_config(config: dict) -> None:
    if set(config['weights']) != set(PRIORITY_COMPONENTS):
        raise ValueError('Priority requires all five dimensions')
    for name in ('weights', 'structural', 'seed_relevance', 'flow', 'temporal'):
        weights = config[name]
        if not weights or not all(np.isfinite(v) and v >= 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1, atol=1e-12, rtol=0):
            raise ValueError(f'{name}: weights must be nonnegative and sum to 1')
    if set(config['role_relevance']) != set(ALLOWED_ROLES) or not all(
            np.isfinite(v) and 0 <= v <= 1 for v in config['role_relevance'].values()):
        raise ValueError('Each role requires relevance in [0,1]')
    if not isinstance(config['top_n'], int) or config['top_n'] < 20:
        raise ValueError('top_n must be an integer >=20 (small datasets export all nodes)')


def calculate_priority(features: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Повторно использует готовые признаки; центральности не пересчитываются.

    Дополнительные seed сверх первого отличают многоветочную достижимость от
    обычного присутствия в собранном графе. Нормируются только эти новые сигналы.
    """
    config = deepcopy(PRIORITY_CONFIG if config is None else config)
    validate_priority_config(config)
    frame = features.copy()
    if not frame.gid.is_unique or frame.gid.isna().any() or not frame.role.isin(ALLOWED_ROLES).all():
        raise ValueError('Priority input requires unique gids and valid roles')
    for source, target in (('reachable_seed_count', 'priority_reachable_extra_pct'),
                           ('direct_seed_in', 'priority_direct_extra_pct'),
                           ('max_unique_in_senders_per_day', 'priority_sync_pct')):
        frame[target] = positive_percentile((frame[source].astype(float) - 1).clip(lower=0))
    frame['priority_sync_days_pct'] = positive_percentile(frame.days_with_3plus_in_senders)
    for source, target in (('short_window_outflow_ratio', 'priority_window'),
                           ('same_day_outflow_ratio', 'priority_same_day')):
        frame[target] = frame[source].fillna(0).clip(0, 1).where(~frame.is_seed, 0.)
    for dimension in PRIORITY_COMPONENTS:
        if dimension == 'role':
            frame['priority_role'] = frame.role_score * frame.role.map(config['role_relevance'])
        else:
            for column in config[dimension]:
                if not frame[column].between(0, 1).all():
                    raise ValueError(f'{column}: priority components require finite values in [0,1]')
            frame[f'priority_{dimension}'] = sum(frame[column] * weight
                                                for column, weight in config[dimension].items())
    # Базовый PageRank положителен и у изолятов; это не свидетельство активности.
    isolated = frame.in_deg.eq(0) & frame.out_deg.eq(0)
    columns = [f'priority_{dimension}' for dimension in PRIORITY_COMPONENTS]
    frame.loc[isolated, columns] = 0.
    frame['priority_score'] = sum(frame[f'priority_{dimension}'] * weight
                                   for dimension, weight in config['weights'].items()).clip(0, 1)
    if not frame[columns + ['priority_score']].apply(lambda series: series.between(0, 1).all()).all():
        raise ValueError('Priority output must be finite and in [0,1]')
    drivers = [sorted(PRIORITY_COMPONENTS,
                      key=lambda dimension: (-row[f'priority_{dimension}'] * config['weights'][dimension],
                                             PRIORITY_COMPONENTS.index(dimension)))[:2]
               for row in frame.to_dict('records')]
    frame['priority_main_factors'] = pd.Series([','.join(parts) for parts in drivers], index=frame.index, dtype=object)
    frame['priority_why'] = pd.Series([build_priority_explanation(row, parts)
                                     for row, parts in zip(frame.to_dict('records'), drivers)],
                                    index=frame.index, dtype=object)
    return frame


def build_top_nodes(features: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """При равенстве приоритета: role_score убывает, затем gid возрастает."""
    if not isinstance(top_n, int) or top_n < 20:
        raise ValueError('top_n must be >=20')
    top = features.sort_values(['priority_score', 'role_score', 'gid'],
                               ascending=[False, False, True]).head(top_n)
    result = top[['gid', 'role', 'priority_score', 'priority_why']].rename(columns={'priority_why': 'why'}).copy()
    result.insert(0, 'rank', np.arange(1, len(result) + 1, dtype='int64'))
    return result.reset_index(drop=True)


def priority_diagnostics(features: pd.DataFrame, top: pd.DataFrame) -> dict:
    first = top.head(20)
    selected = features[features.gid.isin(first.gid)]
    quantiles = {'min': 0., 'p25': .25, 'median': .5, 'p75': .75, 'p90': .9, 'p95': .95, 'max': 1.}
    return {
        'top_20': first.to_dict('records'),
        'quantiles': {name: float(features.priority_score.quantile(q)) if len(features) else None
                      for name, q in quantiles.items()},
        'mean_by_role': {role: float(group.priority_score.mean()) for role, group in features.groupby('role')},
        'n_seed_top20': int(selected.is_seed.sum()),
        'n_truncated_top20': int(selected.truncated_by_depth.sum()),
        'clusters_top20': {str(int(cid)): int(count) for cid, count in selected.cluster_id.value_counts().sort_index().items()},
        'n_top_exported': len(top),
    }
