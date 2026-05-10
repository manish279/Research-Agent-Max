"""ODC Markets Research Agent — Streamlit UI."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ODC Markets Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"

# ── session state init ────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "page": "new_run",
    "run_status": "idle",   # idle | running | complete | error
    "run_dir": None,        # set by thread on completion
    "run_error": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        return ""


def _read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events


def _list_runs() -> list[Path]:
    if not LOGS_DIR.exists():
        return []
    return sorted(
        [d for d in LOGS_DIR.iterdir() if d.is_dir() and (d / "run_summary.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )


def _latest_run_dir() -> Path | None:
    """Return the most recently modified dir in logs/ (used while a run is active)."""
    if not LOGS_DIR.exists():
        return None
    candidates = sorted(
        [d for d in LOGS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

# ── agent background runner ───────────────────────────────────────────────────

def _run_agent(question: str, asset: str, lookback_days: int, timeframe: str) -> None:
    """Executes in a daemon thread; writes results back to st.session_state."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from domain_packs.odc_markets import get_domain_pack
        from research_agent.config import AgentSettings
        from research_agent.graph import ResearchGraph

        settings = AgentSettings.from_env(PROJECT_ROOT)
        graph = ResearchGraph(settings=settings, domain_pack=get_domain_pack())
        result = graph.run(
            question=question,
            asset=asset,
            lookback_days=lookback_days,
            timeframe=timeframe,
        )
        st.session_state.run_dir = Path(result["artifacts"]["run_dir"])
        st.session_state.run_status = "complete"
    except Exception as exc:
        st.session_state.run_error = str(exc)
        st.session_state.run_status = "error"

# ── chart & metric helpers ────────────────────────────────────────────────────

_STEP_LABELS: dict[str, str] = {
    "plan":               "📋 Planning",
    "retrieve_memory":    "🧠 Retrieving memory",
    "gather_research":    "🔍 Web research",
    "load_market_data":   "📈 Loading market data",
    "strategy_generated": "⚙️  Generating strategy",
    "backtest":           "🔬 Backtesting",
    "risk_evaluation":    "🛡️  Risk evaluation",
    "critique":           "🔍 Critique",
    "completed":          "✅ Complete",
}
_STEP_ORDER = list(_STEP_LABELS.keys())


def _metrics_row(backtest: dict) -> None:
    cols = st.columns(5)
    pairs = [
        ("Total Return",  f"{backtest.get('total_return', 0):.1%}"),
        ("Sharpe",        f"{backtest.get('sharpe', 0):.2f}"),
        ("Max Drawdown",  f"{backtest.get('max_drawdown', 0):.1%}"),
        ("Trades",        str(backtest.get("trades", 0))),
        ("Win Rate",      f"{backtest.get('win_rate', 0):.1%}"),
    ]
    for col, (label, val) in zip(cols, pairs):
        col.metric(label, val)


def _equity_chart(backtest: dict) -> go.Figure | None:
    curve = backtest.get("equity_curve", {})
    dates = curve.get("dates") or list(range(len(backtest.get("equity_tail", []))))
    values = curve.get("values") or backtest.get("equity_tail", [])
    if not values:
        return None

    # Equity line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines",
        name="Equity",
        line=dict(color="#4f8ef7", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.08)",
    ))

    # Drawdown shading (if full curve available)
    if curve.get("values"):
        import numpy as np
        vals = [float(v) for v in values]
        running_max = [max(vals[:i+1]) for i in range(len(vals))]
        dd = [(v / m - 1) * 100 for v, m in zip(vals, running_max)]
        fig.add_trace(go.Scatter(
            x=dates, y=dd,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#f74f4f", width=1, dash="dot"),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(
                title="Drawdown %",
                overlaying="y",
                side="right",
                showgrid=False,
                tickformat=".1f",
            )
        )

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity (1.0 = start)",
        height=340,
        margin=dict(l=50, r=60, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _risk_panel(risk: dict) -> None:
    grade = risk.get("risk_grade", "unknown")
    if grade == "research-pass":
        st.success(f"Risk grade: **{grade.upper()}**")
    else:
        st.warning(f"Risk grade: **{grade.upper()}** — review required before any live use")

    for check in risk.get("checks", []):
        icon = "✅" if check["passed"] else "⚠️"
        st.markdown(f"{icon} `{check['name']}` — {check['detail']}")

    for w in risk.get("warnings", []):
        st.caption(f"⚠️ {w}")

# ── results renderer (shared by both pages) ───────────────────────────────────

def _render_results(run_dir: Path) -> None:
    summary   = _read_json(run_dir / "run_summary.json")
    backtest  = _read_json(run_dir / "backtest_result.json")
    risk      = _read_json(run_dir / "risk_report.json")
    report    = _read_text(run_dir / "final_report.md")
    plan      = _read_text(run_dir / "plan.md")
    strategy  = _read_text(run_dir / "strategy_code.py")
    idea      = _read_text(run_dir / "strategy_idea.md")
    critique  = _read_text(run_dir / "critique.md")

    asset   = summary.get("asset") or summary.get("market_summary", {}).get("symbol", "?")
    run_id  = summary.get("run_id", run_dir.name)
    question = summary.get("question", "")

    st.subheader(f"📊 {asset}")
    if question:
        st.caption(f'"{question}"')
    st.caption(f"Run ID: `{run_id}`")

    warnings = summary.get("warnings", [])
    for w in warnings:
        st.warning(w)
    for e in summary.get("errors", []):
        st.error(e)

    # Key metrics
    if backtest and "total_return" in backtest:
        _metrics_row(backtest)
        st.divider()
        fig = _equity_chart(backtest)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    elif backtest.get("error"):
        st.error(f"Backtest error: {backtest['error']}")

    # Risk panel
    if risk:
        with st.expander("🛡️ Risk Report", expanded=True):
            _risk_panel(risk)

    st.divider()

    # Tabs
    tab_labels = ["📄 Final Report", "💻 Strategy", "🗺️ Plan", "🔍 Critique", "📋 Event Log"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if report:
            st.markdown(report)
        else:
            st.info("Final report not available.")

    with tabs[1]:
        if idea:
            st.markdown("**Strategy idea:**")
            st.markdown(idea)
            st.divider()
        if strategy:
            st.code(strategy, language="python")
        else:
            st.info("Strategy code not available.")

    with tabs[2]:
        if plan:
            st.markdown(plan)
        else:
            st.info("Plan not available.")

    with tabs[3]:
        if critique:
            st.markdown(critique)
        else:
            st.info("Critique not available.")

    with tabs[4]:
        events = _read_events(run_dir)
        if events:
            for evt in reversed(events):
                ts    = evt.get("ts", "")[:19]
                etype = evt.get("event_type", "")
                st.markdown(f"`{ts}` **{etype}**")
        else:
            st.info("No events logged.")

# ── new run page ──────────────────────────────────────────────────────────────

def page_new_run() -> None:
    st.title("🚀 New Research Run")

    status = st.session_state.run_status

    # ── idle: show the form ──
    if status == "idle":
        with st.form("run_form"):
            asset = st.text_input(
                "Asset (Yahoo Finance symbol)",
                value="ETH-USD",
                help="Examples: ETH-USD, BTC-USD, SOL-USD, AAPL, TSLA",
            )
            question = st.text_area(
                "Research question",
                value=(
                    "Analyze market behavior and generate a simple, auditable strategy "
                    "hypothesis that can be backtested for paper-trading research."
                ),
                height=100,
            )
            c1, c2 = st.columns(2)
            with c1:
                lookback_days = st.slider("Lookback (days)", 30, 730, 365, step=30)
            with c2:
                timeframe = st.selectbox("Timeframe", ["1d", "1wk", "1h"])

            submitted = st.form_submit_button("▶ Start Run", type="primary", use_container_width=True)

        if submitted:
            st.session_state.run_status = "running"
            st.session_state.run_dir    = None
            st.session_state.run_error  = None
            threading.Thread(
                target=_run_agent,
                args=(question, asset, lookback_days, timeframe),
                daemon=True,
            ).start()
            st.rerun()

    # ── running: live progress ──
    elif status == "running":
        st.info("⏳ Agent is running — grab a coffee, this takes 2–5 min depending on model speed.")

        # Find the active run dir (created at run start)
        active_dir = st.session_state.run_dir or _latest_run_dir()
        completed_steps: set[str] = set()
        if active_dir:
            for evt in _read_events(active_dir):
                completed_steps.add(evt.get("event_type", ""))

        # Progress checklist
        prog = st.container()
        with prog:
            for step in _STEP_ORDER:
                done = step in completed_steps
                icon = "✅" if done else ("⏳" if step == next(
                    (s for s in _STEP_ORDER if s not in completed_steps), None
                ) else "○")
                st.markdown(f"{icon} {_STEP_LABELS[step]}")

        # Auto-refresh every 3 s while running
        time.sleep(3)
        st.rerun()

    # ── complete: show results ──
    elif status == "complete":
        st.success("✅ Run complete!")
        run_dir = st.session_state.run_dir
        if run_dir and Path(run_dir).exists():
            _render_results(Path(run_dir))
        st.divider()
        if st.button("🔄 Start another run", type="primary"):
            st.session_state.run_status = "idle"
            st.rerun()

    # ── error ──
    elif status == "error":
        st.error(f"❌ Run failed: {st.session_state.run_error}")
        st.caption("Check that Ollama is running and the models are pulled.")
        if st.button("🔄 Try again"):
            st.session_state.run_status = "idle"
            st.rerun()

# ── past runs page ────────────────────────────────────────────────────────────

def page_past_runs() -> None:
    st.title("📁 Past Runs")

    runs = _list_runs()
    if not runs:
        st.info("No completed runs yet. Head over to **New Run** to start one.")
        return

    def _run_label(r: Path) -> str:
        summary = _read_json(r / "run_summary.json")
        asset   = summary.get("asset") or summary.get("market_summary", {}).get("symbol", "?")
        grade   = _read_json(r / "risk_report.json").get("risk_grade", "")
        ts      = r.name[:15]
        return f"{ts} — {asset}" + (f"  [{grade}]" if grade else "")

    selected_name = st.selectbox(
        "Select a run to inspect",
        options=[r.name for r in runs],
        format_func=lambda n: _run_label(LOGS_DIR / n),
    )
    st.divider()
    _render_results(LOGS_DIR / selected_name)

# ── sidebar & router ──────────────────────────────────────────────────────────

def main() -> None:
    with st.sidebar:
        st.markdown("## 📈 ODC Markets")
        st.caption("Research Agent")
        st.divider()

        if st.button(
            "🚀  New Run",
            use_container_width=True,
            type="primary" if st.session_state.page == "new_run" else "secondary",
        ):
            st.session_state.page = "new_run"
            st.rerun()

        if st.button(
            "📁  Past Runs",
            use_container_width=True,
            type="primary" if st.session_state.page == "past_runs" else "secondary",
        ):
            st.session_state.page = "past_runs"
            st.rerun()

        st.divider()
        if st.session_state.run_status == "running":
            st.warning("⏳ Run in progress…")

        run_count = len(_list_runs())
        st.metric("Completed runs", run_count)
        st.divider()
        st.caption("⚠️ Research & paper-trading only. Not financial advice.")

    if st.session_state.page == "new_run":
        page_new_run()
    else:
        page_past_runs()


if __name__ == "__main__":
    main()
