from datetime import datetime
from pathlib import Path

def write_markdown(note_md: str, out_dir: str = "runs") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"weekly_report_{ts}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note_md)
    return str(path)
