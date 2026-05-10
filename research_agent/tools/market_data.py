from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_agent.tools.base import ToolResult


class MarketDataTool:
    """Load OHLCV data from yfinance, with a synthetic fallback for offline demos."""

    def __init__(self, data_dir: Path, allow_live_downloads: bool = True) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.allow_live_downloads = allow_live_downloads

    def run(self, symbol: str, lookback_days: int = 365, interval: str = "1d") -> ToolResult:
        warnings: list[str] = []
        df: pd.DataFrame
        if self.allow_live_downloads:
            try:
                import yfinance as yf

                df = yf.download(symbol, period=f"{lookback_days}d", interval=interval, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df.rename(columns=str.title)
                df = df.reset_index()
                if df.empty:
                    raise ValueError("Downloaded dataframe is empty")
            except Exception as exc:
                warnings.append(f"Live market download failed; using synthetic sample data: {exc}")
                df = _synthetic_ohlcv(lookback_days)
        else:
            warnings.append("Live market downloads disabled; using synthetic sample data.")
            df = _synthetic_ohlcv(lookback_days)

        df = _normalize_ohlcv(df)
        path = self.data_dir / f"{symbol.replace('-', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        df.to_csv(path, index=False)
        summary = {
            "symbol": symbol,
            "rows": int(len(df)),
            "start": str(df["Date"].iloc[0]) if len(df) else None,
            "end": str(df["Date"].iloc[-1]) if len(df) else None,
            "last_close": float(df["Close"].iloc[-1]) if len(df) else None,
            "path": str(path),
        }
        return ToolResult(ok=True, data={"path": str(path), "summary": summary}, warnings=warnings)


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_ohlcv(df)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: col.title() for col in df.columns}
    df = df.rename(columns=rename)
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for column in required:
        if column not in df.columns:
            if column == "Volume":
                df[column] = 0.0
            else:
                raise ValueError(f"Market data missing required column: {column}")
    df = df[required].copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return df


def _synthetic_ohlcv(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = max(rows, 120)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=rows, freq="D")
    returns = rng.normal(loc=0.001, scale=0.035, size=rows)
    close = 2500 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.025, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.025, rows))
    volume = rng.integers(100_000, 1_000_000, rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
