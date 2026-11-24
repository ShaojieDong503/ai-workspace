#!/usr/bin/env python3
"""Download raw hourly PM2.5 from the EPA AQS API, by state, 2010-2024.

Endpoint: https://aqs.epa.gov/data/api/sampleData/byState
Parameters: 88101 (PM2.5 FRM/FEM, local conditions) and 88502 (PM2.5 non-FRM/FEM,
acceptable PM2.5 AQI & speciation mass).

States: the 26 states plus DC with an AQS PM2.5 site inside 98.2W-66.9W, 38.5N-59.1N.
Whole states are downloaded; any spatial cut happens later, never here.

Raw means raw: every response is saved verbatim as gzipped JSON (AQS Header + Data),
one file per state x parameter x quarter. Rows carry date_local/time_local (local
standard time) AND date_gmt/time_gmt (UTC), so the UTC-6 day is GMT - 6 h.

API etiquette (AQS terms of service): one request at a time, at least 5 s between
requests, at most 10 per minute. A quarter that times out or fails is retried, then
split into its three months.

Credentials: AQS_API_EMAIL / AQS_API_KEY environment variables, or a JSON file
{"email": ..., "key": ...} at --credentials (default ~/.aqs_api.json). Get a key at
https://aqs.epa.gov/data/api/signup?email=YOUR_EMAIL (it is emailed to you).

    python download_aqs_hourly_api.py --test          # one small request, then stop
    python download_aqs_hourly_api.py                 # everything, resumable
    python download_aqs_hourly_api.py --params 88502  # only the missing parameter
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

API = "https://aqs.epa.gov/data/api/sampleData/byState"
STATES = {
    "09": "Connecticut", "10": "Delaware", "11": "District Of Columbia", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "29": "Missouri", "31": "Nebraska", "33": "New Hampshire", "34": "New Jersey",
    "36": "New York", "38": "North Dakota", "39": "Ohio", "42": "Pennsylvania",
    "44": "Rhode Island", "46": "South Dakota", "50": "Vermont", "51": "Virginia",
    "54": "West Virginia", "55": "Wisconsin",
}
QUARTERS = {1: ("0101", "0331"), 2: ("0401", "0630"), 3: ("0701", "0930"), 4: ("1001", "1231")}
MIN_GAP_S = 6.0          # >= 5 s required, and keeps us under 10 requests/minute
NO_DATA = "No data matched your selection"
PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "data" / "raw" / "AQS_api_hourly")
    ap.add_argument("--params", default="88101,88502")
    ap.add_argument("--states", default=",".join(STATES), help="comma-separated AQS state codes")
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--credentials", type=Path, default=Path.home() / ".aqs_api.json")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--test", action="store_true", help="fetch Rhode Island 88101 Jan 2010 only")
    return ap.parse_args()


def credentials(path: Path) -> tuple[str, str]:
    email, key = os.environ.get("AQS_API_EMAIL"), os.environ.get("AQS_API_KEY")
    if (not email or not key) and path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        email, key = c.get("email"), c.get("key")
    if not email or not key:
        raise SystemExit(f"No AQS credentials: set AQS_API_EMAIL and AQS_API_KEY, or write {path}")
    return email, key


class Client:
    def __init__(self, email: str, key: str, timeout: int, retries: int, log: Path):
        self.auth = {"email": email, "key": key}
        self.timeout, self.retries, self.log = timeout, retries, log
        self.last = 0.0

    def _wait(self) -> None:
        gap = time.monotonic() - self.last
        if gap < MIN_GAP_S:
            time.sleep(MIN_GAP_S - gap)

    def get(self, params: dict) -> dict | None:
        """Return the parsed AQS response, or None after all retries fail."""
        for attempt in range(1, self.retries + 1):
            self._wait()
            self.last = time.monotonic()
            try:
                r = requests.get(API, params=self.auth | params, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                status = body.get("Header", [{}])[0].get("status", "")
                if status in ("Success", NO_DATA):
                    return body
                raise RuntimeError(f"AQS status {status!r}: {body.get('Header')}")
            except Exception as exc:  # network, HTTP, JSON or AQS-reported failure
                self.note(f"retry {attempt}/{self.retries} {params} -> {type(exc).__name__}: {str(exc)[:300]}")
                time.sleep(30 * attempt)
        return None

    def note(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with self.log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def save(path: Path, body: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(body, f)
    tmp.replace(path)                      # atomic: a file that exists is complete
    return len(body.get("Data") or [])


def fetch_window(client: Client, out: Path, param: str, state: str, bdate: str, edate: str) -> int | None:
    if out.exists():
        return -1
    body = client.get({"param": param, "state": state, "bdate": bdate, "edate": edate})
    return None if body is None else save(out, body)


def month_ranges(year: int, q: int) -> list[tuple[str, str, str]]:
    out = []
    for m in range(3 * q - 2, 3 * q + 1):
        last = (date(year + (m == 12), m % 12 + 1, 1) - date.resolution).day
        out.append((f"{m:02d}", f"{year}{m:02d}01", f"{year}{m:02d}{last:02d}"))
    return out


def main() -> int:
    args = parse_args()
    email, key = credentials(args.credentials)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(email, key, args.timeout, args.retries, args.out_dir / "download.log")

    if args.test:
        body = client.get({"param": "88101", "state": "44", "bdate": "20100101", "edate": "20100131"})
        if body is None:
            return 1
        rows = body.get("Data") or []
        print(json.dumps(body["Header"], indent=1))
        print(f"{len(rows)} rows; columns: {list(rows[0]) if rows else '-'}")
        return 0

    params = [p.strip() for p in args.params.split(",")]
    states = [s.strip().zfill(2) for s in args.states.split(",")]
    unknown = set(states) - set(STATES)
    if unknown:
        raise SystemExit(f"unknown state codes: {sorted(unknown)}")
    failed: list[str] = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for param in params:
        for state in states:
            for year in range(args.start_year, args.end_year + 1):
                for q, (b, e) in QUARTERS.items():
                    base = args.out_dir / param / f"{state}_{STATES[state].replace(' ', '_')}"
                    qfile = base / f"{year}_Q{q}.json.gz"
                    months = [(base / f"{year}_{m}.json.gz", bd, ed) for m, bd, ed in month_ranges(year, q)]
                    if qfile.exists() or all(p.exists() for p, _, _ in months):
                        continue
                    n = fetch_window(client, qfile, param, state, f"{year}{b}", f"{year}{e}")
                    if n is not None:
                        client.note(f"ok  {param} {state} {STATES[state]} {year} Q{q}: {n} rows")
                        continue
                    client.note(f"split {param} {state} {year} Q{q} into months")
                    for path, bd, ed in months:
                        n = fetch_window(client, path, param, state, bd, ed)
                        if n is None:
                            failed.append(f"{param} {state} {bd}-{ed}")
                            client.note(f"FAIL {param} {state} {bd}-{ed}")
                        elif n >= 0:
                            client.note(f"ok  {param} {state} {STATES[state]} {bd[:6]}: {n} rows")

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "source": "EPA AQS API sampleData/byState",
        "api": API,
        "params": params,
        "states": {s: STATES[s] for s in states},
        "years": [args.start_year, args.end_year],
        "layout": "<param>/<state code>_<name>/<year>_Q<q>.json.gz (or <year>_<MM>.json.gz when a quarter was split)",
        "format": "verbatim AQS JSON response (Header + Data), gzipped",
        "time_fields": "date_local/time_local = local standard time; date_gmt/time_gmt = UTC",
        "run_started_utc": started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failed_windows": failed,
    }, indent=2), encoding="utf-8")
    print(f"done; {len(failed)} failed window(s)" + ("" if not failed else " -- rerun to retry them"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
