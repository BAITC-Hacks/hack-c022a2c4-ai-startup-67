"""Read-only, bounded tools. No analytics recomputation or unrestricted queries."""
import json
import pandas as pd
from ui.data import find_node, counterparties, daily_activity, node_warnings
from ui.formatting import ROLE_COLORS, LIMITATIONS


def clean(value):
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient='records', date_format='iso', double_precision=15))
    if isinstance(value, pd.Series):
        return json.loads(value.to_json(date_format='iso', double_precision=15))
    return value


class AnalysisTools:
    def __init__(self, nodes, clusters, top, edges=None, transactions=None):
        self.nodes, self.clusters, self.top = nodes, clusters, top
        self.edges, self.transactions = edges, transactions

    def get_node_profile(self, gid):
        row = find_node(self.nodes, gid)
        if row is None:
            return {'error': 'GID not found'}
        return {'source': 'nodes_roles.csv', 'node': clean(row),
                'limitations': list(LIMITATIONS), 'warnings': node_warnings(row)}

    def _counterparties(self, gid, incoming):
        if find_node(self.nodes, gid) is None:
            return {'error': 'GID not found'}
        if self.edges is None:
            return {'error': 'Original edges unavailable'}
        rows = counterparties(self.edges, gid, incoming)
        return {'source': 'edges.parquet', 'gid': gid, 'direction': 'incoming' if incoming else 'outgoing',
                'total_counterparties': len(rows), 'shown': min(20, len(rows)), 'rows': clean(rows.head(20))}

    def get_incoming_counterparties(self, gid):
        return self._counterparties(gid, True)

    def get_outgoing_counterparties(self, gid):
        return self._counterparties(gid, False)

    def get_node_priority_breakdown(self, gid):
        row = find_node(self.nodes, gid)
        if row is None:
            return {'error': 'GID not found'}
        return {'source': 'nodes_roles.csv', 'gid': gid,
                'stored_components': clean(row[[c for c in row.index if c.startswith('priority_')]])}

    def get_node_temporal_profile(self, gid):
        row = find_node(self.nodes, gid)
        if row is None:
            return {'error': 'GID not found'}
        columns = [c for c in row.index if any(x in c for x in ('day', 'window', 'repeat'))]
        result = {'source': 'nodes_roles.csv / transactions.parquet', 'gid': gid,
                  'stored_features': clean(row[columns]), 'limitation': 'Dates only; no intra-day ordering or attribution of funds.'}
        result['daily'] = clean(daily_activity(self.transactions, gid).reset_index()) if self.transactions is not None else None
        return result

    def get_cluster_profile(self, cluster_id):
        rows = self.clusters[self.clusters.cluster_id.eq(cluster_id)]
        if rows.empty:
            return {'error': 'Cluster not found'}
        return {'source': 'clusters.csv', 'cluster': clean(rows.iloc[0]),
                'roles': {str(k): int(v) for k, v in self.nodes[self.nodes.cluster_id.eq(cluster_id)].role.value_counts().items()}}

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError('limit must be an integer between 1 and 50')
        return limit

    def get_top_nodes(self, limit=20):
        return {'source': 'top_nodes.csv', 'available': len(self.top), 'rows': clean(self.top.head(self._limit(limit)))}

    def _ranked(self, rows, limit):
        cols = [c for c in ['gid', 'role', 'priority_score', 'role_score', 'cluster_id', 'reachable_seed_count', 'evidence'] if c in rows]
        return {'source': 'nodes_roles.csv', 'total_matches': len(rows),
                'rows': clean(rows.sort_values(['priority_score', 'gid'], ascending=[False, True]).head(self._limit(limit))[cols])}

    def find_nodes_by_role(self, role, limit=20):
        if role not in ROLE_COLORS:
            return {'error': 'Unknown role'}
        return self._ranked(self.nodes[self.nodes.role.eq(role)], limit)

    def find_multi_seed_nodes(self, limit=20):
        if 'reachable_seed_count' not in self.nodes:
            return {'error': 'Seed reachability unavailable'}
        result = self._ranked(self.nodes[self.nodes.reachable_seed_count.ge(2)], limit)
        result['limitation'] = 'Reachability includes the seed itself at zero hops; not evidence of coordination.'
        return result

    def get_cluster_with_most_seeds(self):
        if self.clusters.empty:
            return {'error': 'No clusters'}
        cid = self.clusters.sort_values(['n_seed', 'cluster_id'], ascending=[False, True]).iloc[0].cluster_id
        return self.get_cluster_profile(int(cid))

    def call(self, name, arguments):
        if name not in TOOL_ARGUMENTS or not isinstance(arguments, dict):
            return {'error': 'Tool or arguments not allowed'}
        expected = TOOL_ARGUMENTS[name]
        if set(arguments) != set(expected):
            return {'error': 'Invalid argument fields'}
        for key, kind in expected.items():
            value = arguments[key]
            if kind == 'string' and (not isinstance(value, str) or len(value) > 100):
                return {'error': 'Invalid string argument'}
            if kind == 'integer' and type(value) is not int:
                return {'error': 'Invalid integer argument'}
        try:
            return getattr(self, name)(**arguments)
        except (ValueError, TypeError):
            return {'error': 'Invalid arguments'}


TOOL_ARGUMENTS = {
    **{name: {'gid': 'string'} for name in ['get_node_profile', 'get_incoming_counterparties',
        'get_outgoing_counterparties', 'get_node_priority_breakdown', 'get_node_temporal_profile']},
    'get_cluster_profile': {'cluster_id': 'integer'}, 'get_top_nodes': {'limit': 'integer'},
    'find_nodes_by_role': {'role': 'string', 'limit': 'integer'}, 'find_multi_seed_nodes': {'limit': 'integer'},
    'get_cluster_with_most_seeds': {},
}
TOOL_SCHEMAS = [{'type': 'function', 'name': name,
                 'description': name.replace('_', ' ') + '. Read stored data only; limits 1–50, counterparties top 20.',
                 'strict': True, 'parameters': {'type': 'object', 'properties': {k: {'type': v} for k, v in args.items()},
                 'required': list(args), 'additionalProperties': False}} for name, args in TOOL_ARGUMENTS.items()]
