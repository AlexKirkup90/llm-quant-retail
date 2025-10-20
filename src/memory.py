import json
from pathlib import Path
from typing import Dict, List, Tuple

from .config import RUNS_DIR

MEM_PATH = Path("metrics_history.json")
PORT_PATH = RUNS_DIR / "last_portfolio.json"
BANDIT_TRACE_PATH = RUNS_DIR / "bandit_trace.jsonl"
BANDIT_TRACE_BAK_PATH = RUNS_DIR / "bandit_trace.bak.jsonl"
TELEMETRY_PATH = RUNS_DIR / "telemetry.jsonl"

def load_last_portfolio():
    if not PORT_PATH.exists(): return None
    return json.loads(PORT_PATH.read_text())

def save_portfolio(port: dict):
    PORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORT_PATH.write_text(json.dumps(port, indent=2))

def append_metrics(row: dict):
    payload: list[dict] = []
    if MEM_PATH.exists():
        try:
            existing = json.loads(MEM_PATH.read_text())
            if isinstance(existing, list):
                payload = existing
            elif isinstance(existing, dict):
                payload = [existing]
        except json.JSONDecodeError:
            payload = []
    entry = dict(row)
    entry.setdefault("version", "0.6")
    payload.append(entry)
    MEM_PATH.write_text(json.dumps(payload, indent=2))


def append_bandit_trace(record: Dict[str, object]) -> None:
    """Append a single JSON record to the persistent bandit trace."""

    BANDIT_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    with BANDIT_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_bandit_trace(limit: int | None = None) -> List[Dict[str, object]]:
    """Return the most recent bandit trace entries.

    Parameters
    ----------
    limit:
        Optional number of trailing rows to return. If ``None`` the entire
        trace is loaded. Returns an empty list when the trace does not exist
        or cannot be decoded.
    """

    if not BANDIT_TRACE_PATH.exists():
        return []
    try:
        lines = BANDIT_TRACE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if limit is not None and limit > 0:
        lines = lines[-limit:]
    records: List[Dict[str, object]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def load_bandit_posteriors() -> Dict[str, Tuple[float, float]]:
    """Return the last recorded posterior parameters for each universe."""

    records = load_bandit_trace(limit=1)
    if not records:
        return {}
    last = records[-1]
    posteriors = last.get("posteriors")
    if not isinstance(posteriors, dict):
        return {}
    parsed: Dict[str, Tuple[float, float]] = {}
    for name, params in posteriors.items():
        if isinstance(params, (list, tuple)) and len(params) >= 2:
            try:
                a = float(params[0])
                b = float(params[1])
            except (TypeError, ValueError):
                continue
            parsed[str(name)] = (a, b)
        elif isinstance(params, dict):
            try:
                a = float(params.get("alpha"))
                b = float(params.get("beta"))
            except (TypeError, ValueError):
                continue
            parsed[str(name)] = (a, b)
    return parsed


def reset_bandit_trace() -> None:
    """Backup and clear the bandit trace, restoring implicit (1,1) priors."""

    if BANDIT_TRACE_PATH.exists():
        try:
            BANDIT_TRACE_BAK_PATH.write_text(
                BANDIT_TRACE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
            )
        except OSError:
            pass
        try:
            BANDIT_TRACE_PATH.unlink()
        except OSError:
            pass


def append_telemetry(record: Dict[str, object]) -> None:
    """Append a record to the telemetry log (JSON lines)."""

    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    with TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_recent_telemetry(metric: str, limit: int = 8) -> List[float]:
    """Load the most recent numeric telemetry values for a given metric."""

    if not TELEMETRY_PATH.exists():
        return []
    try:
        lines = TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    values: List[float] = []
    for line in reversed(lines):
        if limit is not None and len(values) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != metric:
            continue
        try:
            val = float(obj.get("value"))
        except (TypeError, ValueError):
            continue
        values.append(val)
    return list(reversed(values))
