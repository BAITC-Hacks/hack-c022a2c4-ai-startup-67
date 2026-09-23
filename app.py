"""Локальный интерфейс аналитика. Запуск: streamlit run app.py."""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from analytics.priority import PRIORITY_CONFIG
from ui.data import (fingerprint, missing_outputs, load_outputs,
                     load_edges, load_graph, load_transactions, find_node,
                     node_warnings, counterparties, daily_activity)
from ui.data import download_bytes, verify_sources
from ui.formatting import ROLE_LABELS, LIMITATIONS, kzt, numeric
from ui.components import graph_panel, role_chart, role_legend, priority_breakdown, table
from visualization.graph_view import neighborhood

ROOT = Path(__file__).resolve().parent
TABS = ['Dashboard', 'Client Analysis', 'Network', 'Clusters']


def choose_gid(gid):
    st.session_state['selected_gid'] = str(gid).strip()
    st.session_state['search_gid'] = str(gid).strip()
    st.session_state['node_query'] = str(gid).strip()


def search_changed(key):
    choose_gid(st.session_state[key])


def ranking_changed():
    if st.session_state.get('priority_pick'):
        choose_gid(st.session_state['priority_pick'])


def overview(nodes, clusters, top, edges, weights, diagnostics):
    fields = [('Клиенты', len(nodes)), ('Транзакции', edges.n_tx.sum() if edges is not None else nodes.in_tx.sum() if 'in_tx' in nodes else None),
              ('Кластеры', len(clusters)), ('Seed', nodes.is_seed.sum() if 'is_seed' in nodes else None)]
    for col, (label, value) in zip(st.columns(4), fields):
        col.metric(label, numeric(value))
    st.subheader('Быстрый поиск')
    st.text_input('Search GID', key='search_gid', on_change=search_changed, args=('search_gid',))
    row = find_node(nodes, st.session_state.get('selected_gid', ''))
    if row is None:
        st.info('GID не найден. Проверьте идентификатор.')
    else:
        st.write(f"GID {row['gid']} · {row['role']} · Priority {row['priority_score']:.3f} · Cluster {row['cluster_id']}")
        st.write(row['evidence'])
        if st.button('Open full analysis'):
            st.info('Клиент выбран. Откройте вкладку Client Analysis выше.')
    priority_tab(nodes, nodes, top, weights)
    st.subheader('Распределение ролей')
    role_chart(nodes)
    with st.expander('How it works'):
        st.write('Parquet → направленный граф → структурные и временные признаки → роли → Louvain → приоритет. '
                 'Интерфейс читает готовые оценки. Неориентированная проекция используется только для Louvain.')
        for role, description in ROLE_LABELS.items():
            st.write(f'**{role}** — {description}.')
        st.write('Приоритет: структура 35%, связи с seed 25%, роль 15%, активность 15%, время 10%. '
                 'Высокая оценка — повод для проверки, а не вероятность нарушения.')
        for line in LIMITATIONS:
            st.write('• ' + line)


def network_tab(nodes, graph, selected):
    st.subheader('Направленные связи')
    mode = st.selectbox('Режим визуализации', ['Top priority network', 'Selected cluster'], key='network_mode')
    if mode == 'Selected cluster':
        cid = st.selectbox('Кластер для графа', sorted(nodes.cluster_id.unique()), key='network_cluster')
        candidates = nodes.loc[nodes.cluster_id.eq(cid), 'gid']
    else:
        count = st.selectbox('Максимум узлов', [20, 50], key='network_top_n')
        candidates = nodes.sort_values(['priority_score', 'role_score', 'gid'], ascending=[False, False, True]).head(count).gid
        st.caption('Узлы с наибольшим сохранённым приоритетом и только рёбра между ними.')
    role_legend()
    graph_panel(graph, nodes, candidates, selected if selected in set(candidates) else None)


def priority_tab(nodes, visible, top, weights):
    st.subheader('Очередь аналитической проверки')
    amount = st.selectbox('Показать', [20, 50, 100], index=0, format_func=lambda x: f'Top {x}', key='priority_count')
    subset = top[top.gid.isin(visible.gid)].sort_values('rank').head(amount)
    st.caption(f'Источник: top_nodes.csv · В файле {len(top)} строк, показано {len(subset)}. '
               'Ранги и оценки взяты из готового расчёта.')
    if amount > len(top):
        st.info(f'В экспорте только {len(top)} строк. Новые оценки в интерфейсе не рассчитываются.')
    if subset.empty:
        st.info('В сохранённом рейтинге нет строк, соответствующих фильтрам.')
        return
    table(subset[['rank', 'gid', 'role', 'priority_score', 'why']])
    selected = st.selectbox('Выбрать GID из рейтинга', subset.gid.tolist(), index=None,
                            placeholder='Выберите узел для разбора', key='priority_pick', on_change=ranking_changed)
    if selected:
        row = find_node(nodes, selected)
        st.caption('Узел выбран для вкладки Client Analysis. Там доступны все связи, метрики и контрагенты.')
        st.write(row.get('priority_why', subset.set_index('gid').loc[selected, 'why']))
        priority_breakdown(row, weights)


def node_tab(nodes, graph, edges, transactions, weights):
    st.subheader('Карточка клиента')
    st.text_input('Search GID / Поиск клиента', key='node_query', on_change=search_changed, args=('node_query',),
                  help='Точное совпадение. Можно найти любой gid, включая изолированный.')
    st.caption('Поиск доступен для всех клиентов, включая изолированные узлы.')
    row = find_node(nodes, st.session_state.get('selected_gid', ''))
    if row is None:
        st.info('GID не найден. Проверьте идентификатор или выберите узел из Dashboard.')
        return
    gid = row['gid']
    st.markdown(f'#### GID `{gid}`')
    left, right = st.columns(2)
    with left:
        st.markdown('#### Обзор клиента')
        for label, value in [('Роль', row['role']), ('Role score', numeric(row['role_score'], 3)),
                             ('Priority', numeric(row['priority_score'], 3)), ('Cluster', row['cluster_id']),
                             ('Глубина', row.get('depth', '—')), ('Seed', row.get('is_seed', '—'))]:
            st.write(f'**{label}:** {value}')
    with right:
        st.markdown('#### Наблюдаемые потоки')
        for label, value in [('Входящие', kzt(row.get('in_kzt'))), ('Исходящие', kzt(row.get('out_kzt'))),
                             ('Входящих контрагентов', numeric(row.get('in_deg'))), ('Исходящих контрагентов', numeric(row.get('out_deg'))),
                             ('Входящих переводов', numeric(row.get('in_tx'))), ('Исходящих переводов', numeric(row.get('out_tx')))]:
            st.write(f'**{label}:** {value}')
    for warning in node_warnings(row, graph):
        st.warning(warning)
    left, right = st.columns(2)
    with left:
        st.markdown('#### Почему эта роль?')
        st.write(row['evidence'])
    with right:
        st.markdown('#### Почему этот приоритет?')
        explanation = row.get('priority_why', row.get('why'))
        st.write(explanation if isinstance(explanation, str) else 'Объяснение приоритета отсутствует в этом экспорте.')
    with st.expander('Компоненты приоритета'):
        priority_breakdown(row, weights)
    with st.expander('Структурные и дополнительные метрики'):
        fields = ['pagerank', 'betweenness', 'hub_score', 'authority_score', 'reachable_seed_count',
                  'direct_seed_in', 'weak_component_id', 'weak_component_size', 'short_window_outflow_ratio',
                  'same_day_outflow_ratio', 'incoming_amount_hhi', 'outgoing_amount_hhi']
        records = [{'Метрика': name, 'Значение': numeric(row[name], 6)} for name in fields if name in row]
        if records:
            table(pd.DataFrame(records))
        else:
            st.info('Дополнительные метрики не экспортированы.')
    if edges is not None:
        incoming, outgoing = st.columns(2)
        with incoming:
            st.markdown('#### Входящие контрагенты')
            frame = counterparties(edges, gid, True)
            if frame.empty:
                st.caption('Входящих рёбер нет.')
            else:
                table(frame.head(10))
                if len(frame) > 10:
                    with st.expander(f'Все контрагенты ({len(frame)})'):
                        table(frame)
        with outgoing:
            st.markdown('#### Исходящие контрагенты')
            frame = counterparties(edges, gid, False)
            if frame.empty:
                st.caption('Исходящих рёбер нет.')
            else:
                table(frame.head(10))
                if len(frame) > 10:
                    with st.expander(f'Все контрагенты ({len(frame)})'):
                        table(frame)
    st.markdown('#### Активность по календарным дням')
    if transactions is None:
        st.info('Для дневного графика нужен transactions.parquet. Агрегаты выше доступны из CSV.')
    else:
        daily = daily_activity(transactions, gid)
        if daily.empty:
            st.caption('Для узла нет транзакций в предоставленном периоде.')
        else:
            long = daily.reset_index().melt('Дата', var_name='Направление', value_name='KZT')
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x=alt.X('Дата:T', title='Календарная дата'), y=alt.Y('KZT:Q', title='Наблюдаемая сумма, KZT'),
                color=alt.Color('Направление:N', scale=alt.Scale(domain=['Входящие', 'Исходящие'], range=['#57BEB4', '#AD8BF5'])),
                tooltip=[alt.Tooltip('Дата:T', format='%Y-%m-%d'), 'Направление', alt.Tooltip('KZT:Q', format=',.2f')]
            ).properties(height=240), use_container_width=True)
            st.caption('Источник: transactions.parquet. Только календарные даты; порядок переводов внутри дня неизвестен.')

    from ui.assistant import assistant_panel
    assistant_panel(st.session_state['analysis_tools'], gid, st.session_state['data_version'])
    st.markdown('#### Локальный направленный граф · 1 hop')
    role_legend()
    graph_panel(graph, nodes, neighborhood(graph, gid, 1).nodes if graph is not None else [], gid)


def clusters_tab(nodes, visible, clusters, graph):
    st.subheader('Сообщества сети')
    shown = clusters[clusters.cluster_id.isin(visible.cluster_id)]
    st.caption('Размеры и сводка относятся ко всему кластеру; граф ограничен 100 узлами.')
    table(shown[['cluster_id', 'n_nodes', 'n_seed', 'sum_kzt_internal', 'top_gids', 'hypothesis']])
    if shown.empty:
        st.info('Нет кластеров, соответствующих фильтрам.')
        return
    cid = st.selectbox('Выберите cluster_id', shown.cluster_id.tolist(), key='cluster_pick')
    summary = shown[shown.cluster_id.eq(cid)].iloc[0]
    members = nodes[nodes.cluster_id.eq(cid)]
    cols = st.columns(3)
    cols[0].metric('Узлы', numeric(summary['n_nodes']))
    cols[1].metric('Seed', numeric(summary['n_seed']))
    cols[2].metric('Внутренний оборот', kzt(summary['sum_kzt_internal']))
    st.write(summary['hypothesis'])
    st.caption('Структурные представители: ' + str(summary['top_gids']))
    role_chart(members)
    representatives = [gid for gid in str(summary['top_gids']).split(',') if gid in set(members.gid)]
    if representatives:
        target = st.selectbox('Представитель для Client Analysis', representatives, key='cluster_gid')
        st.button('Открыть карточку выбранного представителя', on_click=choose_gid, args=(target,), key='open_cluster_node')
        st.caption('Карточка доступна во вкладке Client Analysis; выбранный gid сохраняется между вкладками.')
    role_legend()
    graph_panel(graph, nodes, members.gid)


def main():
    st.set_page_config(page_title='AML Graph Analytics', page_icon='◈', layout='wide')
    st.markdown('<style>[data-testid="stMetricValue"]{font-size:1.35rem}h1{font-size:2rem!important}</style>', unsafe_allow_html=True)
    st.title('AML Graph Analytics')
    st.caption('Transaction network analysis for AML investigation')
    st.caption('Аналитические индикаторы для дополнительной проверки; не установление факта нарушения.')
    with st.sidebar:
        st.markdown('### Money Graph')
        with st.expander('Источники данных', expanded=False):
            out_dir = st.text_input('Папка результатов', value=os.environ.get('MONEY_GRAPH_OUT_DIR', str(ROOT / 'out')), key='out_directory')
            data_dir = st.text_input('Папка parquet', value=os.environ.get('MONEY_GRAPH_DATA_DIR', str(ROOT / 'data')), key='data_directory')
            st.caption('Пути локальные. Новый экспорт определяется по размеру и времени изменения файлов.')
    missing = missing_outputs(out_dir)
    if missing:
        st.error('Analysis files are missing. Сначала запустите аналитический pipeline.')
        st.write('Отсутствуют: ' + ', '.join(missing))
        st.code('python pipeline.py --data ./data --out ./out', language='bash')
        st.stop()
    try:
        nodes, clusters, top, diagnostics = load_outputs(out_dir, fingerprint(out_dir, (*('nodes_roles.csv', 'clusters.csv', 'top_nodes.csv'), 'node_features.csv', 'diagnostics.json')))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Не удалось прочитать результаты: {exc}')
        st.code('python pipeline.py --data ./data --out ./out', language='bash')
        st.stop()
    if nodes.empty:
        st.info('В экспортированном наборе нет узлов. Загрузите данные и выполните pipeline.')
        st.stop()
    graph = edges = transactions = None
    try:
        if all((Path(data_dir) / name).is_file() for name in ('nodes.parquet', 'edges.parquet')):
            signature = fingerprint(data_dir, ('nodes.parquet', 'edges.parquet'))
            verify_sources(data_dir, signature, diagnostics.get('source_fingerprints'), ('nodes.parquet', 'edges.parquet'))
            graph = load_graph(data_dir, signature)
            _, edges = load_edges(data_dir, signature)
            if set(graph) != set(nodes.gid):
                graph = edges = None
                st.warning('GID исходного графа не совпадают с CSV. Выберите соответствующий набор parquet; графы отключены.')
        else:
            st.warning('Исходные parquet графа не найдены. Таблицы доступны; укажите папку данных для направленных графов.')
    except Exception as exc:
        graph = edges = None
        st.warning(f'Исходный граф недоступен ({type(exc).__name__}). Проверьте parquet и повторите pipeline для выбранных данных.')
    if (Path(data_dir) / 'transactions.parquet').is_file():
        try:
            verify_sources(data_dir, fingerprint(data_dir, ('transactions.parquet',)),
                           diagnostics.get('source_fingerprints'), ('transactions.parquet',))
            transactions = load_transactions(data_dir, fingerprint(data_dir, ('transactions.parquet',)))
            if not transactions.src.isin(nodes.gid).all() or not transactions.dst.isin(nodes.gid).all():
                transactions = None
                st.warning('Транзакции относятся к другому набору gid; дневной график отключён.')
        except Exception as exc:
            transactions = None
            st.warning(f'Дневные транзакции недоступны ({type(exc).__name__}). Проверьте parquet и повторите pipeline.')
    if 'selected_gid' not in st.session_state:
        choose_gid(top.gid.iloc[0] if len(top) else nodes.gid.iloc[0])
    with st.sidebar:
        st.caption('Скачать исходный экспорт целиком')
        for name in ('nodes_roles.csv', 'clusters.csv', 'top_nodes.csv'):
            st.download_button(name, data=download_bytes(out_dir, name, fingerprint(out_dir, (name,))), file_name=name, mime='text/csv', key='download_' + name)
    selected = st.session_state.get('selected_gid', '')
    if selected and find_node(nodes, selected) is None:
        st.warning('Указанный GID не найден в экспорте.')
    saved_config = diagnostics.get('priority_config')
    weights = saved_config.get('weights') if isinstance(saved_config, dict) else None
    if not isinstance(weights, dict) or set(weights) != set(PRIORITY_CONFIG['weights']):
        weights = PRIORITY_CONFIG['weights']
    from assistant.tools import AnalysisTools
    st.session_state['analysis_tools'] = AnalysisTools(nodes, clusters, top, edges, transactions)
    st.session_state['data_version'] = (fingerprint(out_dir, ('nodes_roles.csv', 'clusters.csv', 'top_nodes.csv')),
                                        fingerprint(data_dir, ('edges.parquet', 'transactions.parquet')))
    dashboard, node, network, cluster = st.tabs(TABS)
    with dashboard:
        overview(nodes, clusters, top, edges, weights, diagnostics)
    with node:
        node_tab(nodes, graph, edges, transactions, weights)
    with network:
        network_tab(nodes, graph, selected)
    with cluster:
        clusters_tab(nodes, nodes, clusters, graph)


if __name__ == '__main__':
    main()
