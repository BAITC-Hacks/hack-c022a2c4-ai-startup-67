"""Краткие русскоязычные объяснения только по наблюдаемым числовым признакам."""


def number(value: float) -> str:
    """Компактное округление без ложной точности, включая очень большие суммы."""
    if abs(value) < 1_000_000:
        return f'{value:,.2f}'.rstrip('0').rstrip('.').replace(',', ' ')
    if abs(value) < 1_000_000_000:
        return f'{value / 1_000_000:.4g} млн'
    return f'{value / 1_000_000_000:.4g} млрд'


def generate_evidence(row: dict) -> str:
    role = row['role']
    suffix = ''
    if row['truncated_by_depth'] or (row['depth'] == 4 and row['out_deg'] == 0):
        suffix = '; depth=4: исходящие могут быть невидимы'
    if row['is_seed']:
        suffix += '; seed: входящие видны не полностью'
    if role == 'consolidator':
        text = (f"Сбор: {int(row['in_deg'])} отправителей, {int(row['in_tx'])} переводов, "
                f"{number(row['in_kzt'])} KZT входящих; достижим от {int(row['reachable_seed_count'])} seed")
    elif role == 'distributor':
        text = (f"Распределение: {int(row['out_deg'])} получателей, {int(row['out_tx'])} переводов, "
                f"{number(row['out_kzt'])} KZT исходящих; HHI={row['outgoing_amount_hhi']:.2f}")
    elif role == 'transit':
        text = (f"Наблюдаемый транзит: вход {number(row['in_kzt'])}, выход {number(row['out_kzt'])} KZT; "
                f"выход/вход={row['pass_through']:.2f}; эвристика D…D+2={row['short_window_outflow_ratio']:.0%}")
    elif role == 'terminal':
        text = (f"Наблюдаемый конец: {int(row['in_deg'])} отправителей, {number(row['in_kzt'])} KZT входящих; "
                f"исходящих переводов не видно; depth={int(row['depth'])}")
    elif role == 'coordinator':
        text = (f"Структурная связность: betweenness p{row['betweenness_pct'] * 100:.0f}, "
                f"PageRank p{row['pagerank_pct'] * 100:.0f}; достижим от {int(row['reachable_seed_count'])} seed; "
                f"связи вход/выход={int(row['in_deg'])}/{int(row['out_deg'])}")
    else:
        strongest = max(row[f'{name}_score'] for name in
                        ('consolidator', 'transit', 'distributor', 'terminal', 'coordinator'))
        text = (f"Сильный ролевой паттерн не выявлен: связи вход/выход={int(row['in_deg'])}/{int(row['out_deg'])}; "
                f"макс. допустимая оценка={strongest:.2f}")
    # Сохраняем ограничения видимости даже при необычно длинных числах.
    budget = 200 - len(suffix)
    if len(text) > budget:
        text = text[:budget - 1].rstrip() + '…'
    return text + suffix


def build_priority_explanation(row: dict, drivers: list) -> str:
    """Два крупнейших взвешенных вклада и реальные числовые основания."""
    if row['in_deg'] == 0 and row['out_deg'] == 0:
        return 'Приоритет 0: изолированный узел, наблюдаемых рёбер и переводов нет.'
    descriptions = {
        'structural': f"структура: betweenness p{row['betweenness_pct'] * 100:.0f}, PageRank p{row['pagerank_pct'] * 100:.0f}",
        'seed_relevance': f"достижим от {int(row['reachable_seed_count'])} seed; прямых seed-отправителей {int(row['direct_seed_in'])}",
        'role': f"роль {row['role']}, сила {row['role_score']:.2f}",
        'flow': f"контрагенты вход/выход {int(row['in_deg'])}/{int(row['out_deg'])}; переводы {int(row['in_tx'])}/{int(row['out_tx'])}",
        'temporal': f"до {int(row['max_unique_in_senders_per_day'])} отправителей/день; эвристика D…D+2={row['priority_window']:.0%}",
    }
    if row['is_seed']:
        descriptions['temporal'] = f"до {int(row['max_unique_in_senders_per_day'])} отправителей/день; дней с ≥3: {int(row['days_with_3plus_in_senders'])}"
    text = 'Приоритет проверки: ' + '; '.join(descriptions[part] for part in drivers) + '.'
    if row['is_seed']:
        text += ' Входящие seed видны не полностью.'
    if row['truncated_by_depth'] or (row['depth'] == 4 and row['out_deg'] == 0):
        text += ' Глубина 4: исходящие могут быть невидимы.'
    return text
