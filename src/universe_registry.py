from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
from pandas import DataFrame
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry

from .config import REF_DIR

LOGGER = logging.getLogger(__name__)

# Silence only the insecure request warnings that can pop up on some Wikipedia mirrors
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)  # type: ignore[attr-defined]

ProviderFn = Callable[[Optional[Path]], DataFrame]


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
    )
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


@dataclass
class UniverseDefinition:
    name: str
    url: str
    csv_filename: str
    provider: Optional[ProviderFn]
    refresh_days: int = 90

    @property
    def csv_path(self) -> Path:
        return REF_DIR / self.csv_filename


_COLUMN_NORMALISATION = {
    "symbol": ["symbol", "ticker", "code", "tickers", "epic", "tidm", "ric"],
    "name": ["security", "name", "company", "constituent", "issuer"],
    "sector": ["gics sector", "sector", "industry", "icb sector", "icb industry", "gics sub-industry"],
}

_MIN_ROWS = {
    "SP500_FULL": 6,
    "R1000": 8,
    "NASDAQ_100": 6,
    "FTSE_350": 8,
}


class UniverseRegistryError(RuntimeError):
    """Raised when a universe cannot be loaded from either live or cached data."""


def registry_list() -> List[str]:
    return list(_UNIVERSES.keys())


def _read_html(url: str, html_path: Optional[Path]) -> List[DataFrame]:
    def _read_from_text(text: str) -> List[DataFrame]:
        for flavor in ("lxml", "bs4"):
            try:
                return pd.read_html(StringIO(text), flavor=flavor)
            except ImportError:
                continue
        return _parse_html_tables_basic(text)

    if html_path is not None:
        content = Path(html_path).read_text(encoding="utf-8")
        return _read_from_text(content)
    try:
        response: Response = _SESSION.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - exercised via ValueError path
        raise ValueError(f"Failed to download universe page from {url}: {exc}") from exc
    return _read_from_text(response.text)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[Tuple[List[str], List[str]]]] = []
        self._in_table = False
        self._in_row = False
        self._capture = False
        self._current_table: List[Tuple[List[str], List[str]]] = []
        self._current_row_cells: List[str] = []
        self._current_row_types: List[str] = []
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row_cells = []
            self._current_row_types = []
        elif self._in_table and self._in_row and tag in {"td", "th"}:
            self._capture = True
            self._buffer = []
            self._current_row_types.append(tag)

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag = tag.lower()
        if self._in_table and self._in_row and tag in {"td", "th"} and self._capture:
            text = "".join(self._buffer).strip()
            self._current_row_cells.append(text)
            self._buffer = []
            self._capture = False
        elif self._in_table and tag == "tr" and self._in_row:
            if self._current_row_cells:
                self._current_table.append((self._current_row_types, self._current_row_cells))
            self._in_row = False
        elif tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
            self._current_table = []


def _parse_html_tables_basic(text: str) -> List[DataFrame]:
    parser = _TableParser()
    parser.feed(text)
    frames: List[DataFrame] = []
    for table in parser.tables:
        if not table:
            continue
        header: Optional[List[str]] = None
        rows: List[List[str]] = []
        max_len = 0
        for cell_types, cells in table:
            max_len = max(max_len, len(cells))
            if header is None and any(t == "th" for t in cell_types):
                header = cells
            else:
                rows.append(cells)
        if not rows:
            continue
        if header is None:
            header = [f"col_{idx}" for idx in range(max_len)]
        header = [col or f"col_{idx}" for idx, col in enumerate(header)]
        normalised_rows = [row + [""] * (max_len - len(row)) for row in rows]
        frame = pd.DataFrame(normalised_rows, columns=header[:max_len])
        frames.append(frame)
    if not frames:
        raise ValueError("No HTML tables could be parsed without lxml/bs4")
    return frames


def _flatten_columns(df: DataFrame) -> DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join(str(level).strip() for level in col if str(level) != "nan").strip()
            for col in df.columns
        ]
    return df


def _find_column(df: DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {col.strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for col in df.columns:
        col_norm = col.strip().lower()
        for candidate in candidates:
            if candidate in col_norm:
                return col
    return None


def _to_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("").replace("nan", "").map(lambda x: x.strip())


def _append_ftse_suffix(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return symbol
    if symbol.endswith(".L"):
        return symbol
    return f"{symbol}.L"


def _normalize_universe_df(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    df = _flatten_columns(df)
    df = df.replace({pd.NA: "", None: ""})

    symbol_col = _find_column(df, _COLUMN_NORMALISATION["symbol"])
    if symbol_col is None:
        raise ValueError(f"No symbol-like column found while parsing {universe}")
    name_col = _find_column(df, _COLUMN_NORMALISATION["name"])
    sector_col = _find_column(df, _COLUMN_NORMALISATION["sector"])

    symbol = _to_str_series(df[symbol_col]).str.upper()
    name = _to_str_series(df[name_col]) if name_col else pd.Series("", index=df.index)
    sector = _to_str_series(df[sector_col]) if sector_col else pd.Series("", index=df.index)

    out = pd.DataFrame({"symbol": symbol, "name": name, "sector": sector})
    out = out.loc[out["symbol"] != ""].copy()

    if universe.upper() == "FTSE_350":
        out["symbol"] = out["symbol"].map(_append_ftse_suffix)

    out = out.drop_duplicates(subset="symbol", keep="first")

    return out[["symbol", "name", "sector"]].reset_index(drop=True)

def _extract_symbol_tables(url: str, html_path: Optional[Path]) -> List[DataFrame]:
    tables = _read_html(url, html_path)
    symbol_tables: List[DataFrame] = []
    for table in tables:
        table = _flatten_columns(table)
        if _find_column(table, _COLUMN_NORMALISATION["symbol"]) is not None:
            symbol_tables.append(table)
    if not symbol_tables:
        raise ValueError(f"No table with symbol-like column found for {url}")
    return symbol_tables


def fetch_sp500_full(html_path: Optional[Path] = None) -> DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = _extract_symbol_tables(url, html_path)
        table = max(tables, key=len)
        return _normalize_universe_df(table, "SP500_FULL")
    except ValueError as exc:
        raise ValueError(f"SP500_FULL provider failed: {exc}") from exc


def fetch_nasdaq_100(html_path: Optional[Path] = None) -> DataFrame:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    try:
        tables = _extract_symbol_tables(url, html_path)
        table = max(tables, key=len)
        return _normalize_universe_df(table, "NASDAQ_100")
    except ValueError as exc:
        raise ValueError(f"NASDAQ_100 provider failed: {exc}") from exc


def fetch_r1000(html_path: Optional[Path] = None) -> DataFrame:
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    try:
        tables = _extract_symbol_tables(url, html_path)
        table = max(tables, key=len)
        return _normalize_universe_df(table, "R1000")
    except ValueError as exc:
        raise ValueError(f"R1000 provider failed: {exc}") from exc


def fetch_ftse_350(html_path: Optional[Path] = None) -> DataFrame:
    url = "https://en.wikipedia.org/wiki/FTSE_350_Index"
    try:
        tables = _extract_symbol_tables(url, html_path)
        normalised = [_normalize_universe_df(table, "FTSE_350") for table in tables]
    except ValueError as exc:
        raise ValueError(f"FTSE_350 provider failed: {exc}") from exc
    combined = pd.concat(normalised, ignore_index=True)
    combined = combined.drop_duplicates(subset="symbol", keep="first")
    return combined.reset_index(drop=True)


_UNIVERSES: Dict[str, UniverseDefinition] = {
    "SP500_FULL": UniverseDefinition(
        name="SP500_FULL",
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        csv_filename="sp500_full.csv",
        provider=fetch_sp500_full,
        refresh_days=90,
    ),
    "R1000": UniverseDefinition(
        name="R1000",
        url="https://en.wikipedia.org/wiki/Russell_1000_Index",
        csv_filename="r1000.csv",
        provider=fetch_r1000,
        refresh_days=90,
    ),
    "NASDAQ_100": UniverseDefinition(
        name="NASDAQ_100",
        url="https://en.wikipedia.org/wiki/NASDAQ-100",
        csv_filename="nasdaq_100.csv",
        provider=fetch_nasdaq_100,
        refresh_days=60,
    ),
    "FTSE_350": UniverseDefinition(
        name="FTSE_350",
        url="https://en.wikipedia.org/wiki/FTSE_350_Index",
        csv_filename="ftse_350.csv",
        provider=fetch_ftse_350,
        refresh_days=60,
    ),
    "SP500_MINI": UniverseDefinition(
        name="SP500_MINI",
        url="",
        csv_filename="sp500_mini.csv",
        provider=None,
        refresh_days=0,
    ),
}


def _should_refresh(definition: UniverseDefinition, force: bool) -> bool:
    if force:
        return True
    path = definition.csv_path
    if not path.exists():
        return True
    if definition.refresh_days <= 0:
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(tz=timezone.utc) - modified
    return age > timedelta(days=definition.refresh_days)


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(definition: UniverseDefinition, df: DataFrame) -> None:
    _ensure_directory(definition.csv_path)
    df.to_csv(definition.csv_path, index=False)


def _load_csv(definition: UniverseDefinition) -> DataFrame:
    if not definition.csv_path.exists():
        raise UniverseRegistryError(
            f"No cached CSV found for {definition.name} at {definition.csv_path}"
        )
    df = pd.read_csv(definition.csv_path)
    try:
        normalized = _normalize_universe_df(df, definition.name)
    except ValueError as exc:
        raise UniverseRegistryError(
            f"Cached CSV for {definition.name} could not be parsed: {exc}"
        ) from exc
    expected_columns = {"symbol", "name", "sector"}
    missing = expected_columns.difference(normalized.columns)
    if missing:
        raise UniverseRegistryError(
            f"Cached CSV for {definition.name} is missing columns: {sorted(missing)}"
        )
    return normalized


def refresh_universe(name: str, force: bool = False) -> Tuple[DataFrame, str]:
    name = (name or "").upper()
    if name not in _UNIVERSES:
        raise ValueError(f"Unknown universe: {name}")
    definition = _UNIVERSES[name]

    if definition.provider is None:
        df = _load_csv(definition)
        return df, "cache"

    needs_refresh = _should_refresh(definition, force)
    if needs_refresh:
        try:
            df = definition.provider(None)
            _write_csv(definition, df)
            return df, "live"
        except ValueError as exc:
            LOGGER.warning("Live refresh failed for %s: %s", name, exc)
            try:
                cached = _load_csv(definition)
            except UniverseRegistryError:
                raise UniverseRegistryError(
                    f"Unable to refresh {name}: {exc}. No cached data available."
                ) from exc
            return cached, "cache"
    cached = _load_csv(definition)
    return cached, "cache"


def load_universe(name: str, force_refresh: bool = False) -> DataFrame:
    df, _ = refresh_universe(name, force=force_refresh)
    return df


def refresh_all(force: bool = False) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for name in registry_list():
        try:
            _, source = refresh_universe(name, force=force)
            results[name] = source
        except Exception as exc:  # pragma: no cover - used for runtime reporting
            results[name] = f"error: {exc}"
    return results


__all__ = [
    "UniverseRegistryError",
    "fetch_sp500_full",
    "fetch_nasdaq_100",
    "fetch_r1000",
    "fetch_ftse_350",
    "load_universe",
    "refresh_all",
    "refresh_universe",
    "registry_list",
]
