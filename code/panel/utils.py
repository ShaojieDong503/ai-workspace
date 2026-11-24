"""Shared helpers: logging, month iteration, exception monitoring.

Every long-running driver wraps a unit of work in `guard()`. A failure is
logged with a full traceback, recorded in a failures list, and the loop moves
on, so one bad month cannot kill a 180-month run. The run exits non-zero at the
end if anything failed, and the failure list is written next to the outputs.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

START = date(2010, 1, 1)
END = date(2024, 12, 31)

# lattice, fitted to the CanOSSEM crosswalk
X0, DX = 3688478.20, 4999.827
Y0, DY = 5239381.11, -4998.864
WIDTH = 1065

# the square, W S E N
SQUARE = (-98.2, 38.5, -66.9, 59.1)
ROOT = Path(r"D:\ai-workspace\project")


def setup_logging(path: Path, level=logging.INFO) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("panel")
    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


@dataclass
class Failures:
    items: list = field(default_factory=list)

    def guard(self, log: logging.Logger, label: str, fn, *args, **kwargs):
        """Run fn; on any exception log the traceback and return None."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - we want everything here
            log.error("%s failed: %s", label, exc)
            log.error("%s", traceback.format_exc())
            self.items.append({"unit": label, "error": f"{type(exc).__name__}: {exc}"})
            return None

    def write(self, path: Path) -> None:
        import json
        path.write_text(json.dumps(self.items, indent=2), encoding="utf-8")


def fingerprint(path: Path) -> str | None:
    """BLAKE2b of a file's bytes, or None when it does not exist.

    A content hash rather than mtime: a copy changes the timestamp, and two
    different rebuilds can share one.
    """
    if not path.exists():
        return None
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs(path: Path) -> dict | None:
    """The recorded inputs of a parquet file, or None for a legacy file."""
    import pyarrow.parquet as pq
    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"inputs")
    return json.loads(raw) if raw else None


def rebuild_reason(path: Path, current: dict) -> str | None:
    """Why this output must be rebuilt, or None when it is up to date.

    A recorded null predecessor that now exists counts as a change, so a month
    built cold is rebuilt once its predecessor appears - and not before, so it
    cannot loop.
    """
    if not path.exists():
        return "missing"
    recorded = read_inputs(path)
    if recorded is None:
        return "no recorded inputs (legacy file)"
    for key, now in current.items():
        was = recorded.get(key)
        if was == now:
            continue
        if was is None and now is not None:
            return f"{key} now exists"
        if was is not None and now is None:
            return f"{key} disappeared"
        return f"{key} changed"
    return None


def months(start: date = START, end: date = END):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def days_in_month(year: int, month: int):
    return [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def utc6_day(ts) -> "np.datetime64":
    """UTC-6 day of a UTC timestamp: shift six hours back, then floor to date."""
    return (np.asarray(ts, dtype="datetime64[h]") - np.timedelta64(6, "h")).astype("datetime64[D]")
