"""Переиспользуемые элементы Streamlit без расчёта аналитических оценок."""
from html import escape

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui.formatting import ROLE_COLORS
from visualization.graph_view import bounded_view, render_graph_html as cached_graph_html


def graph_panel(graph, nodes, candidates, selected=None, limit=100):
    if graph is None:
        st.info('Для графа нужны nodes.parquet и edges.parquet в выбранной папке данных.')
        return
    if not set(candidates).intersection(graph):
        st.info('В выбранном представлении нет узлов. Измените фильтры или выберите другой gid.')
        return
    try:
        view, info = bounded_view(graph, nodes, candidates, selected, limit)
        records = nodes[nodes.gid.isin(view)].sort_values('gid').to_dict('records')
        edges = [dict(src=s, dst=d, **attrs) for s, d, attrs in view.edges(data=True)]
        st.caption(f"Показано {info['shown_nodes']} из {info['candidate_nodes']} узлов; "
                   f"{info['shown_edges']} направленных рёбер. Стрелка: отправитель → получатель. "
                   'Перетаскивание, масштабирование и подсказки доступны.')
        if info['shown_nodes'] < info['candidate_nodes'] or info['shown_edges'] < info['candidate_edges']:
            st.warning('Граф сокращён для читаемости: узлы по готовому приоритету, рёбра по сумме. Это не полный граф выбранной группы.')
        components.html(cached_graph_html(records, edges, selected), height=505, scrolling=False)
    except Exception as exc:
        st.warning(f'Визуализация временно недоступна ({type(exc).__name__}). Таблицы и показатели остаются доступны.')


def role_legend():
    labels = ' &nbsp; '.join(f'<span style="color:{color}">●</span> {escape(role)}'
                             for role, color in ROLE_COLORS.items())
    st.markdown(labels, unsafe_allow_html=True)


def role_chart(nodes):
    if nodes.empty:
        st.info('Нет узлов для диаграммы ролей.')
        return
    counts = nodes.role.value_counts().rename_axis('Роль').reset_index(name='Узлы')
    chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X('Узлы:Q', title='Количество узлов'), y=alt.Y('Роль:N', sort='-x', title=None),
        color=alt.Color('Роль:N', scale=alt.Scale(domain=list(ROLE_COLORS), range=list(ROLE_COLORS.values())), legend=None),
        tooltip=['Роль', 'Узлы']).properties(height=230)
    st.altair_chart(chart, use_container_width=True)


def priority_breakdown(row, weights):
    labels = {'structural': 'Структура', 'seed_relevance': 'Связи с seed', 'role': 'Сила роли',
              'flow': 'Активность', 'temporal': 'Временные признаки'}
    values = [{'Компонент': label, 'Оценка': row['priority_' + name], 'Вес': weights.get(name)}
              for name, label in labels.items() if 'priority_' + name in row and pd.notna(row['priority_' + name])]
    if not values:
        st.info('Компоненты приоритета отсутствуют в этом экспорте.')
        return
    st.altair_chart(alt.Chart(pd.DataFrame(values)).mark_bar(color='#57BEB4').encode(
        x=alt.X('Оценка:Q', scale=alt.Scale(domain=[0, 1]), title='Сохранённая оценка компонента, 0…1'),
        y=alt.Y('Компонент:N', sort=list(labels.values()), title=None),
        tooltip=['Компонент', alt.Tooltip('Оценка:Q', format='.3f'), alt.Tooltip('Вес:Q', format='.0%')]
    ).properties(height=180), use_container_width=True)
    st.caption('Это сохранённые компоненты до применения весов, а не доли итоговой оценки.')


def table(frame, **kwargs):
    return st.dataframe(frame, hide_index=True, width="stretch", column_config={
        'gid': st.column_config.TextColumn('GID'),
        'priority_score': st.column_config.NumberColumn('Priority', format='%.3f'),
        'role_score': st.column_config.NumberColumn('Role score', format='%.3f'),
        'sum_kzt': st.column_config.NumberColumn('Сумма, KZT', format='localized'),
        'sum_kzt_internal': st.column_config.NumberColumn('Внутренний оборот, KZT', format='localized'),
    }, **kwargs)
