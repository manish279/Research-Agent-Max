"""Tests for the vectorized backtester and signal coercion."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_agent.tools.backtesting import BacktestTool, _coerce_signals, _run_vector_backtest


# --- helpers ---

def _make_df(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.001, 0.02, rows)))
    open_ = np.r_[close[0], close[:-1]]
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": np.maximum(open_, close) * 1.005,
        "Low": np.minimum(open_, close) * 0.995,
        "Close": close,
        "Volume": rng.integers(100_000, 1_000_000, rows),
    })


def _write_csv(df: pd.DataFrame, tmp_dir: str) -> str:
    path = Path(tmp_dir) / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


# --- _coerce_signals ---

def test_coerce_series():
    s = pd.Series([1.0, 0.0, -1.0])
    result = _coerce_signals(s, 3)
    assert len(result) == 3
    assert result.iloc[0] == pytest.approx(1.0)


def test_coerce_list():
    result = _coerce_signals([0.5, -0.5, 0.0], 3)
    assert len(result) == 3


def test_coerce_dataframe_with_signal_column():
    df = pd.DataFrame({"signal": [1.0, 0.0, -1.0], "other": [0, 0, 0]})
    result = _coerce_signals(df, 3)
    assert list(result) == pytest.approx([1.0, 0.0, -1.0])


def test_coerce_clips_to_bounds():
    result = _coerce_signals([2.0, -3.0, 0.5], 3)
    assert result.max() <= 1.0
    assert result.min() >= -1.0


def test_coerce_length_mismatch_raises():
    with pytest.raises(ValueError, match="Signal length"):
        _coerce_signals([1.0, 0.0], 5)


def test_coerce_dataframe_missing_signal_column_raises():
    df = pd.DataFrame({"wrong": [1.0, 0.0, -1.0]})
    with pytest.raises(ValueError, match="'signal' column"):
        _coerce_signals(df, 3)


# --- _run_vector_backtest ---

def test_backtest_returns_expected_keys():
    df = _make_df(200)
    signals = pd.Series([1.0] * 200)
    result = _run_vector_backtest(df, signals, trading_cost_bps=8.0)
    for key in ["total_return", "sharpe", "max_drawdown", "trades", "win_rate",
                "mean_abs_exposure", "annualized_volatility", "rows", "research_only"]:
        assert key in result, f"Missing key: {key}"


def test_backtest_flat_signal_has_no_exposure():
    df = _make_df(200)
    signals = pd.Series([0.0] * 200)
    result = _run_vector_backtest(df, signals, trading_cost_bps=8.0)
    assert result["mean_abs_exposure"] == pytest.approx(0.0, abs=1e-9)
    assert result["total_return"] == pytest.approx(0.0, abs=1e-6)


def test_backtest_research_only_flag_always_true():
    df = _make_df(100)
    signals = pd.Series([1.0] * 100)
    result = _run_vector_backtest(df, signals, trading_cost_bps=8.0)
    assert result["research_only"] is True


def test_backtest_row_count_matches():
    df = _make_df(150)
    signals = pd.Series([1.0] * 150)
    result = _run_vector_backtest(df, signals, trading_cost_bps=8.0)
    assert result["rows"] == 150


# --- BacktestTool end-to-end (subprocess) ---

SIMPLE_STRATEGY = """
import pandas as pd

def generate_signals(df):
    close = df["Close"].astype(float)
    fast = close.rolling(10).mean()
    slow = close.rolling(30).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[fast > slow] = 1.0
    signal[fast < slow] = -1.0
    return signal.fillna(0.0)
"""

BLOCKED_STRATEGY = """
import os

def generate_signals(df):
    return [0.0] * len(df)
"""

MISSING_FUNCTION_STRATEGY = """
def not_generate_signals(df):
    return [0.0] * len(df)
"""


def test_backtest_tool_runs_valid_strategy():
    df = _make_df(200)
    tool = BacktestTool()
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv(df, tmp)
        result = tool.run(path, SIMPLE_STRATEGY)
    assert result.ok, f"Expected ok=True, got error: {result.error}"
    assert "sharpe" in result.data


def test_backtest_tool_rejects_blocked_import():
    df = _make_df(200)
    tool = BacktestTool()
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv(df, tmp)
        result = tool.run(path, BLOCKED_STRATEGY)
    assert not result.ok
    assert "Blocked import" in result.error


def test_backtest_tool_fails_on_missing_function():
    df = _make_df(200)
    tool = BacktestTool()
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_csv(df, tmp)
        result = tool.run(path, MISSING_FUNCTION_STRATEGY)
    assert not result.ok
