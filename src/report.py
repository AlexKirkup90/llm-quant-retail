from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def generate_equity_curve_plot(equity_curve: pd.Series, output_path: str):
    """Generates and saves an equity curve plot."""
    plt.figure(figsize=(10, 6))
    equity_curve.plot(title="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.savefig(output_path)
    plt.close()

def write_markdown(note_md: str, out_dir: str = "runs") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"weekly_report_{ts}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note_md)
    return str(path)
