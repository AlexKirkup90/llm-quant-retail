import json
from pathlib import Path

MEM_PATH = Path("metrics_history.json")
PORT_PATH = Path("runs/last_portfolio.json")

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
