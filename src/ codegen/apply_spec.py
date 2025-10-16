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

def main(spec_path="spec/current_spec.json", changes_path="spec/changes.json"):
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    changes = json.loads(Path(changes_path).read_text())
    write_changes(changes)
    run_ci_locally()
    print("OK")
if __name__ == "__main__":
    main()
