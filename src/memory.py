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
    arr = []
    if MEM_PATH.exists(): arr = json.loads(MEM_PATH.read_text())
    arr.append(row)
    MEM_PATH.write_text(json.dumps(arr, indent=2))
