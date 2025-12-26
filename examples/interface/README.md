# Interface examples (`torchrdit.examples.interface`)

These scripts reproduce the original examples in `torchrdit/examples/`, but using the regulated public API:

```python
import torchrdit as tr
results = tr.simulate(spec)
opt = tr.optimize(spec, objective="...", options={...})
```

## Running
From repo root:

- Run one example: `UV_CACHE_DIR=.uv-cache uv run python torchrdit/examples/interface/example_gmrf_rdit.py`
- Quick mode (reduced grids/harmonics/steps): `UV_CACHE_DIR=.uv-cache uv run python torchrdit/examples/interface/example_gmrf_rdit.py --quick`

Notes:
- Examples are written to be readable first; default parameters may be slow on CPU.
- Output images are saved next to the script (same behavior as the original examples).
