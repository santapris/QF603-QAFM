"""WRDS connection using the user's own credentials (never stored in the repo).

Username: env var ``WRDS_USERNAME`` if set, otherwise the WRDS entry in ``~/.pgpass``.
"""

from __future__ import annotations

import os
from pathlib import Path

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"


def wrds_username() -> str:
    if os.environ.get("WRDS_USERNAME"):
        return os.environ["WRDS_USERNAME"]
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        for line in pgpass.read_text().splitlines():
            parts = line.split(":")
            if len(parts) >= 4 and parts[0] == WRDS_HOST:
                return parts[3]
    raise RuntimeError("WRDS username not found: set WRDS_USERNAME or run wrds.Connection() once to create ~/.pgpass")


def connect():
    import wrds
    return wrds.Connection(wrds_username=wrds_username(), verbose=False)
