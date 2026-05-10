from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from research_agent.config import AgentSettings
from research_agent.domain import DomainPack
from research_agent.llm import invoke_text, make_ollama_chat
from research_agent.logging.experiment_logger import ExperimentLogger
from research_agent.memory.llama_memory import ResearchMemory
from research_agent.prompts import render_template
from research_agent.reports import fallback_report
from research_agent.risk.evaluator import TradingRiskEvaluator
from research_agent.state import AgentState
from research_agent.tools.registry import build_default_tools


class ResearchGraph:
    """Shared LangGraph kernel used by ODC Markets and future domain packs."""

    def __init__(self, settings: AgentSettings, domain_pack: DomainPack) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.domain_pack = domain_pack
        self.logger = ExperimentLogger(settings.log_dir)
        self.memory = ResearchMemory(settings.memory_dir / domain_pack.name, settings.embedding_model)
        self.tools = {**build_default_tools(settings), **domain_pack.tools}
        self.planner_llm = make_ollama_chat(settings.planner_model, settings, temperature=0.15)
        self.generator_llm = make_ollama_chat(settings.generator_model, settings, temperature=0.2)
        self.critic_llm = make_ollama_chat(settings.critic_model, settings, temperature=0.1)
        self.risk_evaluator = TradingRiskEvaluator(domain_pack.risk_limits)
        self.graph = self._build_graph()

    def run(self, question: str, asset: str = "ETH-USD", lookback_days: int = 365, timeframe: str = "1d") -> AgentState:
        run_id, run_dir = self.logger.start_run(self.domain_pack.name, question)
        initial: AgentState = {
            "run_id": run_id,
            "domain": self.domain_pack.name,
            "question": question,
            "asset": asset,
            "lookback_days": lookback_days,
            "timeframe": timeframe,
            "artifacts": {"run_dir": str(run_dir)},
            "warnings": [],
            "errors": [],
        }
        return self.graph.invoke(initial)

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve_memory", self._retrieve_memory)
        graph.add_node("gather_research", self._gather_research)
        graph.add_node("load_market_data", self._load_market_data)
        graph.add_node("generate_strategy", self._generate_strategy)
        graph.add_node("execute_code_probe", self._execute_code_probe)
        graph.add_node("backtest", self._backtest)
        graph.add_node("evaluate_risk", self._evaluate_risk)
        graph.add_node("critique", self._critique)
        graph.add_node("final_report", self._final_report)
        graph.add_node("log_run", self._log_run)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "retrieve_memory")
        graph.add_edge("retrieve_memory", "gather_research")
        graph.add_edge("gather_research", "load_market_data")
        graph.add_edge("load_market_data", "generate_strategy")
        graph.add_edge("generate_strategy", "execute_code_probe")
        graph.add_edge("execute_code_probe", "backtest")
        graph.add_edge("backtest", "evaluate_risk")
        graph.add_edge("evaluate_risk", "critique")
        graph.add_edge("critique", "final_report")
        graph.add_edge("final_report", "log_run")
        graph.add_edge("log_run", END)
        return graph.compile()

    def _run_dir(self, state: AgentState) -> Path:
        return Path(state["artifacts"]["run_dir"])

    def _plan(self, state: AgentState) -> AgentState:
        prompt = render_template(
            self.domain_pack.prompt("planner.md"),
            question=state["question"],
            asset=state["asset"],
            timeframe=state["timeframe"],
            lookback_days=state["lookback_days"],
        )
        plan = invoke_text(
            self.planner_llm,
            prompt,
            system="You are a careful research planner. Use concise numbered steps.",
            fallback=_fallback_plan(state),
        )
        self.logger.write_text(self._run_dir(state) / "plan.md", plan)
        self.logger.append_event(self._run_dir(state), "plan", {"plan": plan})
        return {"plan": plan}

    def _retrieve_memory(self, state: AgentState) -> AgentState:
        context = self.memory.retrieve(f"{state['asset']} {state['question']}", top_k=5)
        self.logger.write_json(self._run_dir(state) / "memory_context.json", context)
        warnings = list(state.get("warnings", []))
        if self.memory.init_warnings:
            warnings.extend(self.memory.init_warnings)
        return {"memory_context": context, "warnings": warnings}

    def _gather_research(self, state: AgentState) -> AgentState:
        search_tool = self.tools["web_search"]
        scrape_tool = self.tools["web_scraper"]
        # Use the research question (and a snippet of the plan if available) so
        # the search is scoped to what the planner actually decided to investigate.
        question_snippet = state.get("question", "")[:120]
        plan_keywords = " ".join(state.get("plan", "").split()[:20])  # first ~20 words of plan
        query = f"{state['asset']} {question_snippet} {plan_keywords}".strip()
        search_result = search_tool.run(query)
        web_results = search_result.data if search_result.ok else []
        warnings = list(state.get("warnings", []))
        if search_result.warnings:
            warnings.extend(search_result.warnings)
        if search_result.error:
            warnings.append(f"Web search unavailable: {search_result.error}")

        scraped_pages = []
        for item in web_results[:2]:
            url = item.get("url")
            if not url:
                continue
            scraped = scrape_tool.run(url)
            if scraped.ok:
                scraped_pages.append(scraped.data)
            elif scraped.error:
                warnings.append(f"Scrape failed for {url}: {scraped.error}")

        self.logger.write_json(self._run_dir(state) / "web_results.json", web_results)
        self.logger.write_json(self._run_dir(state) / "scraped_pages.json", scraped_pages)
        self.logger.append_event(self._run_dir(state), "research_tools", {"query": query, "result_count": len(web_results)})
        return {"web_results": web_results, "scraped_pages": scraped_pages, "warnings": warnings}

    def _load_market_data(self, state: AgentState) -> AgentState:
        market_tool = self.tools["market_data"]
        result = market_tool.run(state["asset"], lookback_days=state["lookback_days"], interval=state["timeframe"])
        warnings = list(state.get("warnings", [])) + result.warnings
        if not result.ok:
            return {"errors": state.get("errors", []) + [result.error or "Market data load failed"], "warnings": warnings}
        self.logger.write_json(self._run_dir(state) / "market_summary.json", result.data["summary"])
        return {
            "market_data_path": result.data["path"],
            "market_summary": result.data["summary"],
            "warnings": warnings,
        }

    def _generate_strategy(self, state: AgentState) -> AgentState:
        prompt = render_template(
            self.domain_pack.prompt("strategy_generator.md"),
            question=state["question"],
            asset=state["asset"],
            plan=state.get("plan", ""),
            memory_context=state.get("memory_context", []),
            web_results=state.get("web_results", []),
            scraped_pages=state.get("scraped_pages", []),
            market_summary=state.get("market_summary", {}),
        )
        fallback = _fallback_strategy()
        response = invoke_text(
            self.generator_llm,
            prompt,
            system="You generate auditable research strategy code. Return a clear idea and Python code.",
            fallback=fallback,
        )
        code = _extract_code(response) or _extract_code(fallback) or fallback
        idea = _extract_idea(response)
        self.logger.write_text(self._run_dir(state) / "strategy_idea.md", idea)
        self.logger.write_text(self._run_dir(state) / "strategy_code.py", code)
        self.logger.append_event(self._run_dir(state), "strategy_generated", {"idea": idea, "code_chars": len(code)})
        return {"strategy_idea": idea, "strategy_code": code}

    def _execute_code_probe(self, state: AgentState) -> AgentState:
        executor = self.tools["python_exec"]
        probe_code = f"""
{state.get('strategy_code', '')}
print("generate_signals defined:", callable(generate_signals))
"""
        result = executor.run(probe_code)
        payload = result.to_dict()
        self.logger.write_json(self._run_dir(state) / "code_output.json", payload)
        warnings = list(state.get("warnings", []))
        if not result.ok:
            warnings.append(f"Code probe failed before backtest: {result.error}")
        return {"code_output": payload, "warnings": warnings}

    def _backtest(self, state: AgentState) -> AgentState:
        market_data_path = state.get("market_data_path", "")
        if not market_data_path:
            error = "Backtest skipped: market data was not available (see earlier errors)."
            self.logger.append_event(self._run_dir(state), "backtest_skipped", {"reason": error})
            return {
                "backtest_result": {"error": error, "research_only": True},
                "errors": state.get("errors", []) + [error],
            }
        backtest_tool = self.tools["backtest"]
        result = backtest_tool.run(market_data_path, state.get("strategy_code", ""))
        payload = result.data if result.ok else {"error": result.error}
        self.logger.write_json(self._run_dir(state) / "backtest_result.json", payload)
        self.logger.append_event(self._run_dir(state), "backtest", payload)
        if result.ok:
            return {"backtest_result": payload}
        return {"backtest_result": payload, "errors": state.get("errors", []) + [result.error or "Backtest failed"]}

    def _evaluate_risk(self, state: AgentState) -> AgentState:
        risk_report = self.risk_evaluator.evaluate(state.get("backtest_result", {}))
        self.logger.write_json(self._run_dir(state) / "risk_report.json", risk_report)
        self.logger.append_event(self._run_dir(state), "risk_evaluation", risk_report)
        return {"risk_report": risk_report}

    def _critique(self, state: AgentState) -> AgentState:
        prompt = render_template(
            self.domain_pack.prompt("critic.md"),
            question=state["question"],
            strategy_idea=state.get("strategy_idea", ""),
            strategy_code=state.get("strategy_code", ""),
            backtest_result=json.dumps(state.get("backtest_result", {}), indent=2),
            risk_report=json.dumps(state.get("risk_report", {}), indent=2),
        )
        critique = invoke_text(
            self.critic_llm,
            prompt,
            system="You are a skeptical reviewer. Focus on methodological weaknesses and risk.",
            fallback="Critique unavailable from local Ollama. Treat this result as unreviewed and require manual review.",
        )
        self.logger.write_text(self._run_dir(state) / "critique.md", critique)
        self.logger.append_event(self._run_dir(state), "critique", {"critique": critique})
        return {"critique": critique}

    def _final_report(self, state: AgentState) -> AgentState:
        prompt = render_template(
            self.domain_pack.prompt("final_report.md"),
            question=state["question"],
            asset=state["asset"],
            plan=state.get("plan", ""),
            strategy_idea=state.get("strategy_idea", ""),
            backtest_result=json.dumps(state.get("backtest_result", {}), indent=2),
            risk_report=json.dumps(state.get("risk_report", {}), indent=2),
            critique=state.get("critique", ""),
            warnings=state.get("warnings", []),
            domain_warning=self.domain_pack.report_warning,
        )
        report = invoke_text(
            self.generator_llm,
            prompt,
            system="Write a concise research-only report with clear paper-trading warnings.",
            fallback=fallback_report(state, self.domain_pack.report_warning),
        )
        self.logger.write_text(self._run_dir(state) / "final_report.md", report)
        self.memory.add_text(
            f"Run {state['run_id']} for {state['asset']}\n\nIdea:\n{state.get('strategy_idea', '')}\n\nReport:\n{report}",
            metadata={"run_id": state["run_id"], "domain": self.domain_pack.name, "asset": state["asset"]},
        )
        return {"final_report": report}

    def _log_run(self, state: AgentState) -> AgentState:
        summary = {
            "run_id": state["run_id"],
            "domain": state["domain"],
            "asset": state["asset"],
            "question": state["question"],
            "market_summary": state.get("market_summary", {}),
            "backtest_result": state.get("backtest_result", {}),
            "risk_report": state.get("risk_report", {}),
            "warnings": state.get("warnings", []),
            "errors": state.get("errors", []),
            "artifacts": state.get("artifacts", {}),
        }
        self.logger.write_json(self._run_dir(state) / "run_summary.json", summary)
        self.logger.append_event(self._run_dir(state), "completed", summary)
        return {"artifacts": state.get("artifacts", {})}


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "def generate_signals" in text:
        return text[text.index("def generate_signals") :].strip()
    return ""


def _extract_idea(text: str) -> str:
    code = _extract_code(text)
    idea = text.replace(code, "").replace("```python", "").replace("```", "").strip()
    return idea or "Model generated a signal function without a separate idea narrative."


def _fallback_plan(state: AgentState) -> str:
    return f"""1. Retrieve prior research memory for {state['asset']}.
2. Gather current market context and relevant web snippets.
3. Load OHLCV data for the requested lookback window.
4. Generate one simple, testable strategy hypothesis without hardcoding outcomes.
5. Backtest the generated signal function and evaluate drawdown, trade count, exposure, and sample size.
6. Use the critic model to identify weaknesses.
7. Produce a research-only final report with paper-trading warnings."""


def _fallback_strategy() -> str:
    return """Idea: Test a volatility-aware moving-average momentum hypothesis. The strategy is long when short-term trend is above long-term trend and realized volatility is not extreme, short when the opposite trend appears, and flat during warmup or high-volatility stress.

```python
def generate_signals(df):
    close = df["Close"].astype(float)
    fast = close.rolling(10).mean()
    slow = close.rolling(30).mean()
    returns = close.pct_change()
    realized_vol = returns.rolling(20).std()
    vol_cap = realized_vol.rolling(90).quantile(0.75)
    signal = pd.Series(0.0, index=df.index)
    signal[(fast > slow) & (realized_vol <= vol_cap)] = 1.0
    signal[(fast < slow) & (realized_vol <= vol_cap)] = -0.5
    return signal.fillna(0.0)
```
"""
