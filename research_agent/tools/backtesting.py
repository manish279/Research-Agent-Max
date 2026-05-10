from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_agent.tools.base import ToolResult
from research_agent.tools.market_data import load_ohlcv
from research_agent.tools.python_exec import PythonSafetyError, validate_python_safety


class BacktestTool:
    """Vectorized long/flat/short backtester for generated signal functions.

    Signal generation runs in an isolated subprocess (same approach as
    SafePythonExecutor) so LLM-generated code cannot affect the main process
    even if the AST safety check is bypassed.
    """

    def __init__(self, trading_cost_bps: float = 8.0, signal_timeout: int = 30) -> None:
        self.trading_cost_bps = trading_cost_bps
        self.signal_timeout = signal_timeout

    def run(self, market_data_path: str, strategy_code: str) -> ToolResult:
        try:
            validate_python_safety(strategy_code)
            df = load_ohlcv(market_data_path)
            raw_signals = _run_signals_subprocess(strategy_code, market_data_path, self.signal_timeout)
            signals = _coerce_signals(raw_signals, len(df))
            result = _run_vector_backtest(df, signals, self.trading_cost_bps)
            return ToolResult(ok=True, data=result)
        except (PythonSafetyError, Exception) as exc:
            return ToolResult(ok=False, error=str(exc), data={})


def _run_signals_subprocess(strategy_code: str, market_data_path: str, timeout: int) -> list[float]:
    """Run generate_signals(df) in an isolated subprocess and return signal values."""
    runner_parts = [
        "import json\n",
        "import sys\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "_df = pd.read_csv(" + repr(market_data_path) + ")\n",
        'for _col in ["Open", "High", "Low", "Close", "Volume"]:\n',
        "    if _col in _df.columns:\n",
        '        _df[_col] = pd.to_numeric(_df[_col], errors="coerce")\n',
        '_df = _df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)\n',
        "\n",
        strategy_code,
        "\n",
        "try:\n",
        "    _raw = generate_signals(_df.copy())\n",
        "    if isinstance(_raw, pd.DataFrame):\n",
        '        if "signal" not in _raw.columns:\n',
        "            raise ValueError(\"Signal dataframe must include a 'signal' column.\")\n",
        "        _series = _raw['signal']\n",
        "    elif isinstance(_raw, pd.Series):\n",
        "        _series = _raw\n",
        "    else:\n",
        "        _series = pd.Series(_raw)\n",
        "    print(json.dumps(_series.tolist()))\n",
        "except Exception as _exc:\n",
        "    import json as _json\n",
        '    print(_json.dumps({"error": str(_exc)}), file=sys.stderr)\n',
        "    sys.exit(1)\n",
    ]
    runner_code = "".join(runner_parts)

    with tempfile.TemporaryDirectory(prefix="odc_backtest_") as tmp:
        runner_path = Path(tmp) / "runner.py"
        runner_path.write_text(runner_code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ValueError(f"Signal generation timed out after {timeout}s")

    if completed.returncode != 0:
        stderr_snippet = completed.stderr.strip()[:500]
        raise ValueError(f"Signal generation subprocess failed: {stderr_snippet}")

    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse signal output from subprocess: {exc}") from exc


def _coerce_signals(raw: Any, rows: int) -> pd.Series:
    if isinstance(raw, pd.DataFrame):
        if "signal" not in raw.columns:
            raise ValueError("Signal dataframe must include a 'signal' column.")
        series = raw["signal"]
    elif isinstance(raw, pd.Series):
        series = raw
    else:
        series = pd.Series(raw)
    if len(series) != rows:
        raise ValueError(f"Signal length {len(series)} does not match market data length {rows}.")
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(-1.0, 1.0).reset_index(drop=True)


def _run_vector_backtest(df: pd.DataFrame, signal: pd.Series, trading_cost_bps: float) -> dict[str, Any]:
    close = df["Close"].astype(float).reset_index(drop=True)
    market_returns = close.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (trading_cost_bps / 10_000)
    strategy_returns = position * market_returns - costs
    equity = (1 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    trades = int((turnover > 0).sum())
    annualization = 365
    volatility = float(strategy_returns.std(ddof=0) * np.sqrt(annualization))
    sharpe = 0.0 if volatility == 0 else float(strategy_returns.mean() * annualization / volatility)
    total_return = float(equity.iloc[-1] - 1)
    max_drawdown = float(drawdown.min())
    win_rate = float((strategy_returns[strategy_returns != 0] > 0).mean()) if (strategy_returns != 0).any() else 0.0
    exposure = float(position.abs().mean())

    # Full equity curve for UI charting
    dates = df["Date"].astype(str).tolist() if "Date" in df.columns else [str(i) for i in range(len(df))]
    equity_curve = {"dates": dates, "values": [float(x) for x in equity]}

    return {
        "total_return": total_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "win_rate": win_rate,
        "mean_abs_exposure": exposure,
        "trading_cost_bps": trading_cost_bps,
        "rows": int(len(df)),
        "equity_tail": [float(x) for x in equity.tail(10)],
        "equity_curve": equity_curve,
        "research_only": True,
    }
