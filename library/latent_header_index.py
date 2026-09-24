"""Persistent header cache; file changes invalidate individual entries."""
import atexit
import json
import os
import sqlite3
import threading

_lock = threading.Lock()
_connection = None
_pending = 0


def _flush():
    if _connection is not None:
        _connection.commit()


def read_header(path):
    global _connection, _pending
    stat = os.stat(path)
    key = os.path.normcase(os.path.abspath(path))
    with _lock:
        if _connection is None:
            directory = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'Krea2Trainer')
            os.makedirs(directory, exist_ok=True)
            _connection = sqlite3.connect(os.path.join(directory, 'latent_headers.sqlite3'), timeout=30, check_same_thread=False)
            _connection.execute('PRAGMA journal_mode=WAL')
            _connection.execute('CREATE TABLE IF NOT EXISTS headers (path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER, header TEXT)')
            _connection.commit()
            atexit.register(_flush)
        row = _connection.execute('SELECT mtime, size, header FROM headers WHERE path=?', (key,)).fetchone()
        if row and row[:2] == (stat.st_mtime_ns, stat.st_size):
            return json.loads(row[2])
        with open(path, 'rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            if not 0 < size <= min(16 * 1024 * 1024, stat.st_size - 8):
                raise ValueError(f'Invalid safetensors header: {path}')
            header = json.loads(stream.read(size))
        _connection.execute('INSERT OR REPLACE INTO headers VALUES (?, ?, ?, ?)',
                            (key, stat.st_mtime_ns, stat.st_size, json.dumps(header)))
        _pending += 1
        if _pending >= 256:
            _connection.commit()
            _pending = 0
        return header
