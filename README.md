# sabre

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/000cf19e-2d25-4adb-a0bb-9407e8d1bf45" />

Explain Jupyter cell outputs with Claude — plots, DataFrames, tracebacks, arrays.

Most notebook AI tools (Jupyter AI, Copilot, Pretzel) focus on the *code*. `sabre` is the opposite: it looks at what your cell *produced* and narrates it. What does this chart actually show? Why does this DataFrame look weird? What broke in this traceback?

```python
df.describe()
explain()                # narrates the last output
explain(fig)             # explains a matplotlib figure (sent as an image)
explain(err)             # explains an exception / traceback
```

Works in JupyterLab, classic Jupyter, and VS Code notebooks. (Colab doesn't allow pip-installed extensions, so it won't work there.)

## Install

```bash
pip install git+https://github.com/arunvenkatadri/sabre.git
export ANTHROPIC_API_KEY=sk-ant-...
```

You need your own Anthropic API key — `sabre` uses it directly, no middleman.

## Use

```python
%load_ext sabre
from sabre import explain

# Narrate the last cell output
df.head()
explain()

# Explain a specific object
explain(df)
explain(fig)
explain(some_exception)

# Line magic — same as explain(), with optional expression
%explain
%explain df[df.price.isna()]
```

## What it handles

| Type | What Claude sees |
|------|------------------|
| `pandas.DataFrame` | shape, dtypes, `describe()`, head, null counts |
| `pandas.Series` | dtype, `describe()`, value counts (for categorical), head |
| `numpy.ndarray` | shape, dtype, basic stats, flat sample |
| `matplotlib.Figure` / `Axes` | the rendered PNG (via Claude's vision) |
| `BaseException` | full traceback text |
| `str` | the text itself (truncated past 20K chars) |
| anything else | `repr()` fallback |

## Try it

```bash
git clone https://github.com/arunvenkatadri/sabre.git
cd sabre
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
jupyter lab demo.ipynb
```

`demo.ipynb` generates its own data and walks through all the supported types.

## How it works

- Model: `claude-opus-4-7` with adaptive thinking.
- Streaming, so long explanations render progressively in the cell.
- Per-type system prompts (see `src/sabre/prompts.py`) with prompt-cache `cache_control` — repeat calls are cheap.
- Serialization lives in `src/sabre/serialize.py`. Each branch there is a type dispatch — add a new `isinstance` check + a text/image payload to teach it a new object type.

<img width="1185" height="642" alt="Screenshot 2026-04-22 at 9 54 55 PM" src="https://github.com/user-attachments/assets/ea99f345-989e-40b4-8594-72031bafa57f" />



<img width="1171" height="719" alt="Screenshot 2026-04-22 at 9 55 54 PM" src="https://github.com/user-attachments/assets/7005b944-509a-4e2e-9ab5-891fce17f2ae" />





## Configuration

```python
explain(df, model="claude-sonnet-4-6", max_tokens=4096)
```

Reads `ANTHROPIC_API_KEY` from the environment.

## Known gaps

- No one-click UI in the cell output yet — you have to call `explain()` explicitly. A future JupyterLab extension could wrap this library and add a button; this is the library layer it would sit on top of.
- Prompts in `prompts.py` are the easiest thing to tune; they're deliberately short and may be wrong for your domain.

## License

MIT
