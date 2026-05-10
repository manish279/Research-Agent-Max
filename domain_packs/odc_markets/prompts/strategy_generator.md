You are generating an auditable ODC Markets trading-research hypothesis.

Question:
{question}

Asset:
{asset}

Plan:
{plan}

Prior memory context:
{memory_context}

Web research snippets:
{web_results}

Scraped page extracts:
{scraped_pages}

Market summary:
{market_summary}

Task:
1. Propose one simple strategy idea that can be falsified by a backtest.
2. Do not hardcode one fixed strategy forever; choose a hypothesis based on the context.
3. Avoid data leakage. Use only current and past rows in indicators.
4. Generate Python code that defines exactly:

```python
def generate_signals(df):
    ...
    return signal
```

Requirements for `generate_signals`:
- input dataframe columns are Date, Open, High, Low, Close, Volume;
- return a pandas Series/list/dataframe with one signal per input row;
- signal values must be between -1 and 1;
- 1 means long, 0 means flat, -1 means short;
- include warmup handling by returning 0 while indicators are unavailable;
- use only pandas/numpy objects already available as `pd` and `np`;
- do not import modules;
- do not read or write files;
- do not call network APIs.

Return:
- an "Idea:" paragraph;
- one fenced Python code block.
