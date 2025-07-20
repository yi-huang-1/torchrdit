"""
Source Batching Performance Comparison for TorchRDIT v0.1.22

This script provides comprehensive performance benchmarks comparing sequential
vs batched source processing across different scenarios.

Key benchmarks:
- Scaling with number of sources
- Impact of structure complexity
- Memory usage comparison
- GPU vs CPU performance
- Different batch sizes
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator
import time
import gc
from pathlib import Path


def benchmark_scaling():
    """Benchmark performance scaling with number of sources."""
    print("\n=== Benchmark 1: Performance Scaling with Number of Sources ===")
    
    # Create solver with moderate complexity
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials and structure
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Test different numbers of sources
    n_sources_list = [1, 2, 5, 10, 20, 40, 80]
    sequential_times = []
    batched_times = []
    speedups = []
    
    deg = np.pi / 180
    
    for n_sources in n_sources_list:
        # Create sources
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Clear cache
        gc.collect()
        if str(solver.device).startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Sequential processing
        start = time.time()
        for source in sources:
            _ = solver.solve(source)
        seq_time = time.time() - start
        sequential_times.append(seq_time)
        
        # Clear cache again
        gc.collect()
        if str(solver.device).startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Batched processing
        start = time.time()
        _ = solver.solve(sources)
        batch_time = time.time() - start
        batched_times.append(batch_time)
        
        speedup = seq_time / batch_time if batch_time > 0 else 1.0
        speedups.append(speedup)
        
        print(f"N={n_sources:3d}: Sequential={seq_time:6.3f}s, "
              f"Batched={batch_time:6.3f}s, Speedup={speedup:.2f}x")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Timing comparison
    ax1.plot(n_sources_list, sequential_times, 'ro-', linewidth=2, 
             markersize=8, label='Sequential')
    ax1.plot(n_sources_list, batched_times, 'bo-', linewidth=2, 
             markersize=8, label='Batched')
    ax1.set_xlabel('Number of Sources')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Processing Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Speedup factor
    ax2.plot(n_sources_list, speedups, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Sources')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Batched Processing Speedup')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(speedups) * 1.2)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'performance_scaling.png', dpi=150)
    plt.close()
    
    print(f"\nAverage speedup for N>1: {np.mean(speedups[1:]):.2f}x")


def benchmark_complexity():
    """Benchmark performance with different structure complexities."""
    print("\n=== Benchmark 2: Impact of Structure Complexity ===")
    
    kdim_values = [3, 5, 7, 9, 11]
    speedups = []
    
    # Fixed number of sources
    n_sources = 20
    deg = np.pi / 180
    angles = np.linspace(0, 60, n_sources) * deg
    
    for kdim in kdim_values:
        # Create solver with varying complexity
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[kdim, kdim],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Add simple structure
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Create sources
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Time sequential (sample only)
        start = time.time()
        for i in range(min(3, n_sources)):
            _ = solver.solve(sources[i])
        seq_time = (time.time() - start) * n_sources / 3
        
        # Time batched
        start = time.time()
        _ = solver.solve(sources)
        batch_time = time.time() - start
        
        speedup = seq_time / batch_time
        speedups.append(speedup)
        
        print(f"kdim={kdim}x{kdim}: Sequential≈{seq_time:.3f}s, "
              f"Batched={batch_time:.3f}s, Speedup={speedup:.2f}x")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(kdim_values, speedups, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Fourier Harmonics (kdim)')
    plt.ylabel('Speedup Factor')
    plt.title(f'Speedup vs Structure Complexity ({n_sources} sources)')
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(__file__).parent / 'complexity_impact.png', dpi=150)
    plt.close()


def benchmark_wavelength_angle_sweep():
    """Benchmark combined wavelength and angle sweeps."""
    print("\n=== Benchmark 3: Combined Wavelength and Angle Sweep ===")
    
    # Create solver with multiple wavelengths
    wavelengths = np.linspace(1.5, 1.6, 5)
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=wavelengths,
        rdim=[256, 256],
        kdim=[5, 5],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add thin film stack
    si = create_material(name="Si", permittivity=12.25)
    sio2 = create_material(name="SiO2", permittivity=2.25)
    solver.add_materials([si, sio2])
    
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.2, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    
    # Create sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 13) * deg
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in angles
    ]
    
    print(f"Processing {len(wavelengths)} wavelengths × {len(sources)} angles = "
          f"{len(wavelengths) * len(sources)} total calculations")
    
    # Sequential timing (estimate)
    start = time.time()
    for i in range(min(3, len(sources))):
        _ = solver.solve(sources[i])
    est_seq_time = (time.time() - start) * len(sources) / 3
    
    # Batched timing
    start = time.time()
    results = solver.solve(sources)
    batch_time = time.time() - start
    
    speedup = est_seq_time / batch_time
    print(f"\nEstimated sequential: {est_seq_time:.2f}s")
    print(f"Batched processing: {batch_time:.2f}s")
    print(f"Speedup factor: {speedup:.1f}x")
    
    # Visualize results as 2D map
    transmission = results.transmission.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(transmission, aspect='auto', origin='lower',
                    extent=[wavelengths[0], wavelengths[-1], 0, 60],
                    cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Incident Angle (degrees)')
    plt.title('Transmission Map (Wavelength × Angle)')
    plt.colorbar(im, label='Transmission')
    plt.savefig(Path(__file__).parent / 'wavelength_angle_map.png', dpi=150)
    plt.close()


def benchmark_memory_usage():
    """Compare memory usage patterns between sequential and batched processing."""
    print("\n=== Benchmark 4: Memory Usage Comparison ===")
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory benchmark")
        return
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[512, 512],
        kdim=[9, 9],
        device='cuda'
    )
    
    # Add structure
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Test configurations
    n_sources_list = [10, 20, 40, 80]
    peak_memory_seq = []
    peak_memory_batch = []
    
    deg = np.pi / 180
    
    for n_sources in n_sources_list:
        # Create sources
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Sequential memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        for source in sources:
            _ = solver.solve(source)
        
        peak_seq = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peak_memory_seq.append(peak_seq)
        
        # Batched memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        _ = solver.solve(sources)
        
        peak_batch = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peak_memory_batch.append(peak_batch)
        
        print(f"N={n_sources:3d}: Sequential={peak_seq:7.1f}MB, "
              f"Batched={peak_batch:7.1f}MB, "
              f"Ratio={peak_batch/peak_seq:.2f}x")
    
    # Plot memory usage
    plt.figure(figsize=(8, 6))
    plt.plot(n_sources_list, peak_memory_seq, 'ro-', linewidth=2, 
             markersize=8, label='Sequential')
    plt.plot(n_sources_list, peak_memory_batch, 'bo-', linewidth=2, 
             markersize=8, label='Batched')
    plt.xlabel('Number of Sources')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(__file__).parent / 'memory_usage.png', dpi=150)
    plt.close()


def objective_function_radius(solver, sources, radius):
    """
    Objective function for radius-based optimization.
    
    Args:
        solver: The solver instance
        sources: List of source configurations
        radius: The radius tensor to optimize
        
    Returns:
        torch.Tensor: The loss value (negative mean transmission)
    """
    # Generate circle mask using the radius
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius)
    
    # Update solver with the new mask
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Solve for all sources
    results = solver.solve(sources)
    
    # Return loss (negative transmission to minimize)
    return -results.transmission.mean()


def benchmark_optimization_overhead():
    """Benchmark overhead in gradient computation for optimization."""
    print("\n=== Benchmark 5: Optimization Overhead ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials and layer
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Test with different numbers of sources
    n_sources_list = [1, 5, 10, 20]
    forward_times = []
    backward_times = []
    
    deg = np.pi / 180
    
    for n_sources in n_sources_list:
        # Create sources
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Create radius tensor as optimization variable
        radius = torch.tensor(0.3, dtype=torch.float32, device=solver.device)
        radius.requires_grad = True
        
        # Forward pass timing
        start = time.time()
        loss = objective_function_radius(solver, sources, radius)
        forward_time = time.time() - start
        forward_times.append(forward_time)
        
        # Backward pass timing
        start = time.time()
        loss.backward()
        backward_time = time.time() - start
        backward_times.append(backward_time)
        
        print(f"N={n_sources:2d}: Forward={forward_time:.3f}s, "
              f"Backward={backward_time:.3f}s, "
              f"Ratio={backward_time/forward_time:.2f}x")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Timing breakdown
    width = 0.35
    x = np.arange(len(n_sources_list))
    ax1.bar(x - width/2, forward_times, width, label='Forward', alpha=0.7)
    ax1.bar(x + width/2, backward_times, width, label='Backward', alpha=0.7)
    ax1.set_xlabel('Number of Sources')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Forward vs Backward Pass Timing')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_sources_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Backward/Forward ratio
    ratios = [b/f for b, f in zip(backward_times, forward_times)]
    ax2.plot(n_sources_list, ratios, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Sources')
    ax2.set_ylabel('Backward/Forward Time Ratio')
    ax2.set_title('Gradient Computation Overhead')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'optimization_overhead.png', dpi=150)
    plt.close()


def main():
    """Run all performance benchmarks."""
    print("TorchRDIT Source Batching - Performance Benchmarks")
    print("=================================================")
    
    # Run benchmarks
    benchmark_scaling()
    benchmark_complexity()
    benchmark_wavelength_angle_sweep()
    benchmark_memory_usage()
    benchmark_optimization_overhead()
    
    print("\nAll performance benchmarks completed!")
    print("Generated plots:")
    print("  - performance_scaling.png")
    print("  - complexity_impact.png")
    print("  - wavelength_angle_map.png")
    print("  - memory_usage.png")
    print("  - optimization_overhead.png")


if __name__ == "__main__":
    main()