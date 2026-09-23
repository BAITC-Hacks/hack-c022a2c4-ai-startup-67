"""Generate a reproducible short demo using only actual pipeline outputs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    nodes = pd.read_csv(ROOT / 'out/nodes_roles.csv', dtype={'gid': str})
    clusters = pd.read_csv(ROOT / 'out/clusters.csv')
    ranked = nodes.sort_values(['priority_score', 'role_score', 'gid'], ascending=[False, False, True])
    groups = [ranked[ranked.role.isin(['consolidator', 'coordinator'])],
              ranked[ranked.role.isin(['distributor', 'transit'])],
              ranked[ranked.truncated_by_depth.eq(True)]]
    if any(group.empty for group in groups):
        raise ValueError('This dataset does not contain all three requested demo patterns.')
    lines = ['# 5-Minute Demo', '',
             'Автоматически выбрано из текущих CSV командой `python scripts/prepare_demo.py`. '
             'После нового расчёта обновите этот файл.', '',
             '## 0:00–0:30 — проблема', '',
             'Покажите задачу: сократить ручной поиск интересных структур, сохранив направление переводов и объяснимость. '
             'Результат — очередь проверки, не обвинение.', '',
             '## 0:30–1:00 — Dashboard', '',
             f'{len(nodes)} клиентов, {int(nodes.out_tx.sum())} переводов, {len(clusters)} кластеров, '
             f'{int(nodes.is_seed.sum())} seed. Покажите Top 20 и причину приоритета.', '']
    for index, (title, frame) in enumerate(zip(['1:00–2:00 — клиент 1', '2:00–3:00 — клиент 2', '3:00–4:00 — Network / Cluster и граница наблюдения'], groups)):
        row = frame.iloc[0]
        lines += [f'## {title}', '', f"GID **{row.gid}**, роль **{row.role}**, priority **{row.priority_score:.6f}**, кластер **{row.cluster_id}**.", '',
                  f'- Сохранённое объяснение: {row.evidence}',
                  f'- Входящих / исходящих контрагентов: {int(row.in_deg)} / {int(row.out_deg)}.',
                  f'- Наблюдаемые суммы: входящие {row.in_kzt:,.2f} KZT; исходящие {row.out_kzt:,.2f} KZT.',
                  f'- Достижим от {int(row.reachable_seed_count)} seed; betweenness {row.betweenness:.6f}.', '',
                  'В UI: вставьте GID в поиск Dashboard → Client Analysis; покажите роль и причину, контрагентов и стрелки локального графа.', '']
        if index == 2:
            lines += [f'Откройте кластер {row.cluster_id} в Network или Clusters. На больших кластерах покажите предупреждение о сокращении до 100 узлов.', '']
        caveats = ['Наблюдается только июль 2026 и выбранная сеть переводов.']
        if row.is_seed:
            caveats.append('Входящая активность seed может быть неполной; достижимость включает сам seed.')
        if row.truncated_by_depth:
            caveats.append('Глубина 4: отсутствие исходящих переводов не подтверждает terminal; дальнейший отток не наблюдается.')
        if row.role == 'coordinator':
            caveats.append('Структурная связность не устанавливает организатора или общий контроль.')
        lines += ['**Оговорка:** ' + ' '.join(caveats), '']
    lines += ['## 4:00–4:40 — AI Assistant (опция)', '',
              'Спросите: «Почему этот клиент в приоритете и что проверить дальше?» '
              'Покажите фактические результаты инструментов под ответом.', '',
              '**Fallback:** если ключа нет или API недоступен, покажите сохранённые evidence, priority_why '
              'и компоненты приоритета. Все основные результаты доступны без AI.', '',
              '## 4:40–5:00 — ограничения и ценность', '',
              'Только июль, исходящий обход seed, 4 перехода, порог 5 000 KZT; нет персональных атрибутов и ground truth. '
              'Ценность: воспроизводимая очередь проверки и объяснимое исследование наблюдаемых связей.', '']
    (ROOT / 'DEMO.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
