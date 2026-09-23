"""Ограниченные направленные представления. Исходный граф не изменяется."""
from html import escape
import json
import math
import re

import networkx as nx
import streamlit as st
from pyvis.network import Network

from ui.formatting import ROLE_COLORS, kzt, numeric

MAX_NODES = 100
MAX_EDGES = 500


def neighborhood(graph, gid, hops=1):
    """Расширение по входящим и исходящим связям, но результат — DiGraph."""
    if hops not in (1, 2):
        raise ValueError('Допустимы только 1 или 2 перехода')
    if gid not in graph:
        return nx.DiGraph()
    seen, frontier = {gid}, {gid}
    for _ in range(hops):
        adjacent = set()
        for current in frontier:
            adjacent.update(graph.predecessors(current))
            adjacent.update(graph.successors(current))
        frontier = adjacent - seen
        seen.update(frontier)
    return graph.subgraph(seen).copy()


def bounded_view(graph, nodes, candidates, selected=None, limit=MAX_NODES):
    """Отбор по готовому приоритету; все оставшиеся рёбра сохраняют направление."""
    candidates = set(candidates).intersection(graph)
    records = nodes[nodes.gid.isin(candidates)].sort_values(
        ['priority_score', 'role_score', 'gid'], ascending=[False, False, True])
    ordered = records.gid.tolist()
    if selected in candidates:
        ordered = [selected] + [gid for gid in ordered if gid != selected]
    kept = ordered[:min(limit, MAX_NODES)]
    view = graph.subgraph(kept).copy()
    total_edges = view.number_of_edges()
    if total_edges > MAX_EDGES:
        strongest = sorted(view.edges(data=True), key=lambda e: (-e[2]['sum_kzt'], e[0], e[1]))[:MAX_EDGES]
        view.remove_edges_from(list(view.edges))
        view.add_edges_from(strongest)
    return view, {'candidate_nodes': len(candidates), 'shown_nodes': len(view),
                  'candidate_edges': total_edges, 'shown_edges': view.number_of_edges()}


def node_tooltip(row):
    fields = [('GID', row['gid']), ('Роль', row.get('role', '—')),
              ('Сила роли', numeric(row.get('role_score'), 3)),
              ('Приоритет', numeric(row.get('priority_score'), 3)),
              ('Кластер', row.get('cluster_id', '—')), ('Глубина', row.get('depth', '—')),
              ('Seed', row.get('is_seed', '—')),
              ('Входящих контрагентов', row.get('in_deg', '—')),
              ('Исходящих контрагентов', row.get('out_deg', '—')),
              ('Входящая сумма', kzt(row.get('in_kzt'))), ('Исходящая сумма', kzt(row.get('out_kzt'))),
              ('Betweenness', numeric(row.get('betweenness'), 6)),
              ('Достижим от seed', row.get('reachable_seed_count', '—'))]
    return '<br>'.join(f'{escape(label)}: {escape(str(value))}' for label, value in fields)


@st.cache_data(show_spinner=False, max_entries=32)
def render_graph_html(node_records, edge_records, selected=None):
    """Inline JS/CSS, без внешнего CDN, без физики; идентификаторы только строки."""
    net = Network(height='490px', width='100%', directed=True, bgcolor='#0E141C',
                  font_color='#E7EDF4', cdn_resources='in_line')
    # Компактная детерминированная раскладка только отображаемой части графа.
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(row['gid'] for row in node_records)
    layout_graph.add_edges_from((row['src'], row['dst']) for row in edge_records)
    positions = nx.spring_layout(layout_graph, seed=42, iterations=30, scale=380, weight=None) if len(layout_graph) > 1 else {gid: (0., 0.) for gid in layout_graph}
    for row in node_records:
        gid = str(row['gid'])
        score = row.get('priority_score', 0.)
        score = min(1., max(0., score)) if math.isfinite(score) else 0.
        x, y = positions[gid]
        net.add_node(gid, label=gid if gid == selected else '', title=node_tooltip(row),
                     color={'background': ROLE_COLORS.get(row.get('role'), ROLE_COLORS['peripheral']),
                            'border': '#FFFFFF' if gid == selected else '#243549'},
                     size=12 + 16 * score, borderWidth=4 if gid == selected else 1,
                     x=float(x), y=float(y), shape='dot')
        # PyVis заменяет пустую label на gid; убираем подписи неселектированных
        # узлов явно, иначе десятки длинных идентификаторов перекрывают стрелки.
        net.nodes[-1]['label'] = gid if gid == selected else ''
    logs = [math.log1p(row['sum_kzt']) for row in edge_records]
    maximum = max(logs, default=1.) or 1.
    for row, log_amount in zip(edge_records, logs):
        src, dst = str(row['src']), str(row['dst'])
        title = (f'{escape(src)} → {escape(dst)}<br>Сумма: {kzt(row["sum_kzt"])}'
                 f'<br>Переводов: {int(row["n_tx"])}')
        net.add_edge(src, dst, width=0.7 + 3.3 * log_amount / maximum, title=title,
                     arrows={'to': {'enabled': True, 'scaleFactor': .8}},
                     color={'color': '#697F96', 'highlight': '#E7EDF4'},
                     smooth={'enabled': True, 'type': 'curvedCW', 'roundness': .12})
    net.set_options(json.dumps({
        'physics': {'enabled': False},
        'interaction': {'hover': True, 'tooltipDelay': 120, 'dragNodes': True,
                        'zoomView': True, 'hideEdgesOnDrag': False},
        'nodes': {'font': {'size': 12, 'color': '#E7EDF4', 'strokeWidth': 3, 'strokeColor': '#0E141C'}},
        'edges': {'selectionWidth': 2},
    }))
    html = net.generate_html(notebook=False)
    # Шаблон PyVis включает Bootstrap CDN даже с in_line. Он не нужен графу.
    html = re.sub(r'<link\b[^>]*href=["\']https?://[^>]*>', '', html, flags=re.I)
    html = re.sub(r'<script\b[^>]*src=["\']https?://[^>]*>\s*</script>', '', html, flags=re.I)
    # Пустая рамка шаблона не должна показывать светлый фон.
    html = html.replace('</head>', '<style>html,body{margin:0;background:#0E141C}.card{border:0!important}#mynetwork{border:0!important}</style></head>')
    # Вкладки Streamlit изначально скрыты. Повторно подгоняем масштаб, когда
    # iframe получает реальную ширину; ручной zoom/drag затем сохраняется.
    resize_script = '''<script>
    const graphBox = document.getElementById('mynetwork');
    let lastWidth = 0;
    new ResizeObserver(() => {
      const width = graphBox.getBoundingClientRect().width;
      if (width > 0 && width !== lastWidth) {
        lastWidth = width;
        network.redraw(); network.fit({animation: false});
      }
    }).observe(graphBox);
    </script>'''
    return html.replace('</body>', resize_script + '</body>')
