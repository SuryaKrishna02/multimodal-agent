"""One way to open the SQLite files: commit on success, roll back on error, and always close.

`with sqlite3.connect(path) as db:` commits but does not close the connection. A daemon that runs for
days must not leak one connection per event, so every module opens its database through `connect()`.
WAL mode lets the dashboard read while the daemon writes, from another process.
"""

import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

_ready: set[str] = set()  # database files whose tables this process has created
_init_lock = threading.Lock()  # two dashboard requests must not upgrade the same file at once


@contextmanager
def connect(path: Path, init: str | Callable[[sqlite3.Connection], None] = "") -> Iterator[sqlite3.Connection]:
    """Open a database, create its tables the first time, and close it on exit.

    Args:
        path (Path): The database file. Its folder is created.
        init (str | Callable, optional): `CREATE TABLE IF NOT EXISTS ...` statements, or a function that creates
            or upgrades the tables. It runs once per process for each file. Defaults to "".

    Yields:
        sqlite3.Connection: The open connection. It commits when the block ends without an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    try:
        with _init_lock:
            if str(path) not in _ready or not path.exists():  # a deleted file (tests, a reset) needs its tables again
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(init) if isinstance(init, str) else init(db)
                db.commit()
                _ready.add(str(path))
        with db:
            yield db
    finally:
        db.close()
