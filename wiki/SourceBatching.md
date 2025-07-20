# Source Batching Module

## Overview
The source batching feature in TorchRDIT v0.1.22 enables efficient processing of multiple incident sources simultaneously, providing significant performance improvements for parameter sweeps and multi-condition optimizations.

## Key Features
- Process multiple incident angles in a single solve() call
- Handle different polarization states simultaneously
- Full gradient support for optimization workflows
- 3-6x speedup compared to sequential processing
- Memory-efficient processing of large parameter sweeps
- Zero breaking changes - fully backward compatible

## Quick Start

```python
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import numpy as np

# Create solver
solver = create_solver(algorithm=Algorithm.RDIT, rdim=[512, 512], kdim=[7, 7])

# Create multiple sources for angle sweep
deg = np.pi / 180
sources = [
    solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
    for angle in np.linspace(0, 60, 13) * deg
]

# Batch solve - returns BatchedSolverResults
results = solver.solve(sources)

# Access results
print(f"Transmission for all angles: {results.transmission[:, 0]}")
best_idx = results.find_optimal_source('max_transmission')
print(f"Best angle: {sources[best_idx]['theta'] * 180/np.pi:.1f}°")
```

## Usage Examples

### Angle Sweep
```python
# Create sources for angle sweep
angles = np.linspace(-60, 60, 121) * np.pi/180
sources = [
    solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
    for angle in angles
]

# Batch solve
results = solver.solve(sources)

# Plot angular response
import matplotlib.pyplot as plt
plt.plot(angles*180/np.pi, results.transmission[:, 0].numpy())
plt.xlabel('Angle (degrees)')
plt.ylabel('Transmission')
```

### Polarization Analysis
```python
# Different polarization states at fixed angle
theta = 30 * np.pi/180
polarizations = [
    {"pte": 1.0, "ptm": 0.0},    # TE
    {"pte": 0.0, "ptm": 1.0},    # TM
    {"pte": 0.707, "ptm": 0.707}, # 45° linear
    {"pte": 0.707, "ptm": 0.707j} # RCP
]

sources = [
    solver.add_source(theta=theta, phi=0, **pol)
    for pol in polarizations
]

results = solver.solve(sources)
```

### Optimization with Multiple Sources
```python
# Optimize for uniform response across angles
mask = shape_gen.generate_circle_mask(radius=0.3)
mask.requires_grad = True

optimizer = torch.optim.Adam([mask], lr=0.01)

# Target angles
sources = [
    solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
    for angle in [0, 15, 30] * np.pi/180
]

for epoch in range(100):
    optimizer.zero_grad()
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    results = solver.solve(sources)
    loss = -results.transmission.mean()  # Maximize average
    
    loss.backward()
    optimizer.step()
```

## BatchedSolverResults API

The `BatchedSolverResults` class provides convenient access to results from multiple sources:

### Attributes
- `transmission`: Shape (n_sources, n_freqs) - Total transmission efficiency
- `reflection`: Shape (n_sources, n_freqs) - Total reflection efficiency
- `n_sources`: Number of sources in the batch
- `source_parameters`: List of source dictionaries

### Methods
- `__getitem__(idx)`: Get results for specific source(s)
- `__iter__()`: Iterate over individual source results
- `find_optimal_source(metric)`: Find source with best performance
- `get_parameter_sweep_data(parameter, metric)`: Extract sweep data

### Example Usage
```python
# Access individual results
result_30deg = results[1]  # Returns SolverResults

# Iterate through results
for i, result in enumerate(results):
    print(f"Source {i}: T={result.transmission[0]:.3f}")

# Find optimal source
best_idx = results.find_optimal_source('max_transmission')
worst_idx = results.find_optimal_source('min_reflection')

# Extract parameter sweep data
angles, trans = results.get_parameter_sweep_data('theta', 'transmission')
```

## Performance Considerations

### Memory-Efficient Processing
For very large parameter sweeps, use chunking:

```python
def process_large_sweep(solver, angles, chunk_size=100):
    all_results = []
    
    for i in range(0, len(angles), chunk_size):
        chunk = angles[i:i+chunk_size]
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in chunk
        ]
        results = solver.solve(sources)
        all_results.append(results.transmission[:, 0].numpy())
    
    return np.concatenate(all_results)
```

### Performance Tips
1. Batch similar calculations together
2. Use GPU acceleration when available
3. Consider memory vs speed tradeoffs
4. For very large batches (>100 sources), use chunking

## Gradient Support

Source batching maintains full gradient support for optimization:

```python
# Gradients work seamlessly
mask.requires_grad = True
results = solver.solve(sources)
loss = -results.transmission.mean()
loss.backward()  # Computes gradients for all sources
```

## Examples

For complete examples, see:
- `examples/source_batching_basic.py` - Basic usage patterns
- `examples/source_batching_advanced.py` - Optimization examples
- `examples/source_batching_performance.py` - Performance benchmarks
- `examples/example_source_batching.py` - Comprehensive demos

