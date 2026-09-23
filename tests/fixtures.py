"""Small synthetic network shared by tests."""
import pandas as pd

def fixture():
    nodes = pd.DataFrame({'gid': [1, 2, 3, 4, 5], 'depth': [0, 1, 4, 2, 0], 'is_seed': [True, False, False, False, True]})
    edges = pd.DataFrame({'src': [1, 2, 1], 'dst': [2, 3, 4], 'sum_kzt': [100.0, 50.0, 25.0], 'n_tx': [2, 1, 1], 'depth': [1, 4, 2]})
    tx = pd.DataFrame({'src': [1, 1, 2, 1], 'dst': [2, 2, 3, 4], 'date': ['2026-07-01', '2026-07-02', '2026-07-03', '2026-07-04'], 'sum_kzt': [40.0, 60.0, 50.0, 25.0]})
    return (nodes, edges, tx)
