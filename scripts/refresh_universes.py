#!/usr/bin/env python3
"""Force refresh all universes from Wikipedia."""

from __future__ import annotations

import json
from typing import Dict

from src.universe_registry import refresh_all


def main() -> None:
    results: Dict[str, str] = refresh_all(force=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
