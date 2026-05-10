# ODC Markets LangGraph Research Agent

Production-oriented research-agent scaffold for ODC Markets experiments. The core agent kernel is domain-neutral; ODC Markets and the placeholder CosmoTwin scientific domain are separate `domain_packs` that provide prompts, tools, metrics, and evaluation rules.

> Research-only warning: this project generates experimental trading research and paper-trading reports. It is not financial advice and should not be connected to live execution without independent validation, monitoring, and compliance review.

## Folder Structure

```text
odc_markets_research_agent/
  requirements.txt
  .env.example
  README.md
  research_agent/
    config.py                 # runtime settings and paths
    graph.py                  # LangGraph orchestration nodes
    llm.py                    # Ollama Qwen/DeepSeek model factory
    state.py                  # graph state contract
    reports.py                # final report helpers
    logging/
      experiment_logger.py    # run/idea/code/backtest/critique/report persistence
    memory/
      llama_memory.py         # LlamaIndex-backed memory with JSONL fallback
    risk/
      evaluator.py            # strategy risk evaluation layer
    tools/
      base.py
      registry.py
      search.py
      scrape.py
      python_exec.py
      market_data.py
      backtesting.py
  domain_packs/
    odc_markets/
      pack.py                 # ODC Markets domain config
      metrics.py
      prompts/
        planner.md
        strategy_generator.md
        critic.md
        final_report.md
    cosmotwin/
      pack.py                 # extension example for scientific research
      metrics.py
      prompts/
        planner.md
        critic.md
  examples/
    run_eth_strategy.py       # ETH strategy research run
  data/
  logs/
  memory/
```

## Setup

1. Install and run Ollama.
2. Pull local models:

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

3. Create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Optional: copy `.env.example` to `.env` and edit model names or paths.

## Example Run

```powershell
python examples/run_eth_strategy.py --asset ETH-USD --lookback-days 365
```

The run creates a timestamped directory under `logs/` containing:

- `events.jsonl`
- `plan.md`
- `strategy_code.py`
- `code_output.json`
- `backtest_result.json`
- `risk_report.json`
- `critique.md`
- `final_report.md`
- `run_summary.json`

## Extending To CosmoTwin

CosmoTwin should reuse `research_agent/graph.py`, `research_agent/memory/`, `research_agent/logging/`, and any generic tools. Add domain-specific scientific tools, prompts, and metrics under `domain_packs/cosmotwin/`.

The intended extension points are:

- `DomainPack.tools`: expose domain-native tools without modifying the graph kernel.
- `DomainPack.prompt_dir`: replace trading prompts with scientific hypothesis, literature, simulation, or validation prompts.
- `DomainPack.metrics`: replace return/drawdown metrics with scientific quality metrics such as reproducibility, uncertainty, citation quality, physical plausibility, or simulation fit.
- `DomainPack.risk_limits`: for non-trading domains, reinterpret this as safety/validity thresholds rather than market risk.

For a new domain, create a new `domain_packs/<name>/pack.py`, then instantiate `ResearchGraph(settings, get_<name>_pack())`.
