# Environment

- **OS:** Ubuntu 24.04.2 LTS
- **CPU:** Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz
- **GPU:** None
- **Python:** 3.12.10

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

## Environment Variables

- `PYTHONPATH=src` – required for local imports and running tests.
- `OPENAI_API_KEY` or `OPENAI_KEY` – optional keys used by `OpenAIRegionInterpreter`.
