import json, sys, subprocess, os
from pathlib import Path
from .file_contract import SPEC_REQUIRED_KEYS, ALLOWED_DIRS

def validate_spec(spec):
    missing = SPEC_REQUIRED_KEYS - set(spec.keys())
    if missing: raise ValueError(f"Missing spec keys: {missing}")
    return True

def write_changes(changes):
    for path, content in changes.items():
        if not any(str(path).startswith(p) for p in ALLOWED_DIRS):
            raise PermissionError(f"Path not allowed: {path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

def run_ci_locally():
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"])

def _load_json(path):
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    raw = path_obj.read_text().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def main(spec_path="spec/current_spec.json", changes_path="spec/changes.json"):
    spec = _load_json(spec_path)
    if not spec:
        print(f"No spec found at {spec_path}; nothing to apply.")
        return
    validate_spec(spec)
    changes = _load_json(changes_path) or {}
    if not changes:
        print(f"No changes requested in {changes_path}; nothing to apply.")
        return
    write_changes(changes)
    run_ci_locally()
    print("OK")
if __name__ == "__main__":
    main()
