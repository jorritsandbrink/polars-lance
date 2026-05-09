# polars-lance

Polars plugin for reading Lance datasets into Polars dataframes.

## Installation

```bash
pip install polars-lance
```

## Usage

```python
from polars_lance import scan_lance

lf = scan_lance("data/example.lance")
df = lf.collect()
```
