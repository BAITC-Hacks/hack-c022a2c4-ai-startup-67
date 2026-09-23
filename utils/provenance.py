"""Отпечатки исходных файлов связывают предрасчёт с parquet в интерфейсе."""
import hashlib
from pathlib import Path

SOURCE_FILES = ('nodes.parquet', 'edges.parquet', 'transactions.parquet')


def source_fingerprints(directory, names=SOURCE_FILES):
    result = {}
    for name in names:
        digest = hashlib.sha256()
        with (Path(directory) / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result
