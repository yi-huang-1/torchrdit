"""
Source Batching Performance Benchmarks (interface simulate/optimize)

This reproduces `torchrdit/examples/source_batching_performance.py`, but uses the
regulated interface APIs and auto harmonics selection (`harmonics="auto"` + `maxG`).
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchrdit as tr
from torchrdit.shapes import ShapeGenerator


def _shape_generator_cartesian(grids, device="cpu"):
    vec_p = torch.linspace(-0.5, 0.5, grids[0], device=device)
    vec_q = torch.linspace(-0.5, 0.5, grids[1], device=device)
    mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")
    return ShapeGenerator(mesh_p, mesh_q, tuple(grids))


def benchmark_scaling():
    """Benchmark performance scaling with number of sources."""
    print("\n=== Benchmark 1: Performance Scaling with Number of Sources ===")

    grids = [256, 256]
    harmonics = [7, 7]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)

    base_spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG, "device": device},
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [{"material": "Si", "thickness": 0.5, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}}],
        "output": {"type": "torch"},
    }

    n_sources_list = [1, 2, 5, 10, 20, 40, 80]
    sequential_times = []
    batched_times = []
    speedups = []

    deg = np.pi / 180

    for n_sources in n_sources_list:
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(angles)]

        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = time.time()
        for src in sources:
            one = dict(base_spec)
            one["sources"] = {k: v for k, v in src.items() if k != "name"}
            _ = tr.simulate(one)
        seq_time = time.time() - start
        sequential_times.append(seq_time)

        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = time.time()
        batch_spec = dict(base_spec)
        batch_spec["sources"] = sources
        _ = tr.simulate(batch_spec)
        batch_time = time.time() - start
        batched_times.append(batch_time)

        speedup = seq_time / batch_time if batch_time > 0 else 1.0
        speedups.append(speedup)

        print(f"N={n_sources:3d}: Sequential={seq_time:6.3f}s, Batched={batch_time:6.3f}s, Speedup={speedup:.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(n_sources_list, sequential_times, "ro-", linewidth=2, markersize=8, label="Sequential")
    ax1.plot(n_sources_list, batched_times, "bo-", linewidth=2, markersize=8, label="Batched")
    ax1.set_xlabel("Number of Sources")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Processing Time Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.plot(n_sources_list, speedups, "go-", linewidth=2, markersize=8)
    ax2.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Number of Sources")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_title("Batched Processing Speedup")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(speedups) * 1.2)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "performance_scaling.png", dpi=150)
    plt.close()

    print(f"\nAverage speedup for N>1: {np.mean(speedups[1:]):.2f}x")


def benchmark_complexity():
    """Benchmark performance with different structure complexities."""
    print("\n=== Benchmark 2: Impact of Structure Complexity ===")

    harmonics_values = [3, 5, 7, 9, 11]
    speedups = []
    n_sources = 20

    deg = np.pi / 180
    angles = np.linspace(0, 60, n_sources) * deg

    for harmonics in harmonics_values:
        maxG = harmonics * harmonics
        grids = [256, 256]
        wavelengths = np.array([1.55])

        shape_gen = _shape_generator_cartesian(grids, device="cpu")
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)

        base_spec = {
            "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG},
            "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
            "vars": {"$mask": mask},
            "layers": [{"material": "Si", "thickness": 0.5, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}}],
            "output": {"type": "torch"},
        }

        sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(angles)]

        start = time.time()
        for src in sources[:3]:
            one = dict(base_spec)
            one["sources"] = {k: v for k, v in src.items() if k != "name"}
            _ = tr.simulate(one)
        seq_time = (time.time() - start) * n_sources / 3

        start = time.time()
        batch_spec = dict(base_spec)
        batch_spec["sources"] = sources
        _ = tr.simulate(batch_spec)
        batch_time = time.time() - start

        speedup = seq_time / batch_time if batch_time > 0 else 1.0
        speedups.append(speedup)

        print(f"harmonics={harmonics}x{harmonics}: Sequential≈{seq_time:.3f}s, Batched={batch_time:.3f}s, Speedup={speedup:.2f}x")

    plt.figure(figsize=(8, 6))
    plt.plot(harmonics_values, speedups, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Fourier Harmonics (harmonics)")
    plt.ylabel("Speedup Factor")
    plt.title(f"Speedup vs Structure Complexity ({n_sources} sources)")
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(__file__).parent / "complexity_impact.png", dpi=150)
    plt.close()


def benchmark_wavelength_angle_sweep():
    """Benchmark combined wavelength and angle sweeps."""
    print("\n=== Benchmark 3: Combined Wavelength and Angle Sweep ===")

    wavelengths = np.linspace(1.5, 1.6, 5)
    grids = [256, 256]
    harmonics = [5, 5]
    maxG = harmonics[0] * harmonics[1]

    deg = np.pi / 180
    angles = np.linspace(0, 60, 13) * deg
    sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(angles)]

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG},
        "materials": {"Si": {"permittivity": 12.25}, "SiO2": {"permittivity": 2.25}},
        "layers": [
            {"material": "SiO2", "thickness": 0.5, "is_homogeneous": True},
            {"material": "Si", "thickness": 0.1, "is_homogeneous": True},
            {"material": "SiO2", "thickness": 0.2, "is_homogeneous": True},
            {"material": "Si", "thickness": 0.1, "is_homogeneous": True},
            {"material": "SiO2", "thickness": 0.5, "is_homogeneous": True},
        ],
        "sources": sources,
        "output": {"type": "torch"},
    }

    print(f"Processing {len(wavelengths)} wavelengths x {len(sources)} angles = {len(wavelengths) * len(sources)} total calculations")

    start = time.time()
    for src in sources[:3]:
        one = dict(spec)
        one["sources"] = {k: v for k, v in src.items() if k != "name"}
        _ = tr.simulate(one)
    est_seq_time = (time.time() - start) * len(sources) / 3

    start = time.time()
    results = tr.simulate(spec)
    batch_time = time.time() - start

    speedup = est_seq_time / batch_time if batch_time > 0 else 1.0
    print(f"\nEstimated sequential: {est_seq_time:.2f}s")
    print(f"Batched processing: {batch_time:.2f}s")
    print(f"Speedup factor: {speedup:.1f}x")

    transmission = np.stack([results[f"s{i}"]["efficiency"]["transmission"].detach().cpu().numpy() for i in range(len(sources))], axis=0)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(transmission, aspect="auto", origin="lower", extent=[wavelengths[0], wavelengths[-1], 0, 60], cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Incident Angle (degrees)")
    plt.title("Transmission Map (Wavelength x Angle)")
    plt.colorbar(im, label="Transmission")
    plt.savefig(Path(__file__).parent / "wavelength_angle_map.png", dpi=150)
    plt.close()


def benchmark_memory_usage():
    """Compare memory usage patterns between sequential and batched processing."""
    print("\n=== Benchmark 4: Memory Usage Comparison ===")

    if not torch.cuda.is_available():
        print("GPU not available, skipping memory benchmark")
        return

    grids = [512, 512]
    harmonics = [9, 9]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])
    device = "cuda"

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)

    base_spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG, "device": device},
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [{"material": "Si", "thickness": 0.5, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}}],
        "output": {"type": "torch"},
    }

    n_sources_list = [10, 20, 40, 80]
    peak_memory_seq = []
    peak_memory_batch = []

    deg = np.pi / 180

    for n_sources in n_sources_list:
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(angles)]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        for src in sources:
            one = dict(base_spec)
            one["sources"] = {k: v for k, v in src.items() if k != "name"}
            _ = tr.simulate(one)

        peak_seq = torch.cuda.max_memory_allocated() / 1024**2
        peak_memory_seq.append(peak_seq)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        batch_spec = dict(base_spec)
        batch_spec["sources"] = sources
        _ = tr.simulate(batch_spec)

        peak_batch = torch.cuda.max_memory_allocated() / 1024**2
        peak_memory_batch.append(peak_batch)

        print(f"N={n_sources:3d}: Sequential={peak_seq:7.1f}MB, Batched={peak_batch:7.1f}MB, Ratio={peak_batch/peak_seq:.2f}x")

    plt.figure(figsize=(8, 6))
    plt.plot(n_sources_list, peak_memory_seq, "ro-", linewidth=2, markersize=8, label="Sequential")
    plt.plot(n_sources_list, peak_memory_batch, "bo-", linewidth=2, markersize=8, label="Batched")
    plt.xlabel("Number of Sources")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(__file__).parent / "memory_usage.png", dpi=150)
    plt.close()


def benchmark_optimization_overhead():
    """Benchmark optimization step time for different batch sizes."""
    print("\n=== Benchmark 5: Optimization Overhead ===")

    grids = [256, 256]
    harmonics = [5, 5]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_sources_list = [1, 5, 10, 20]
    step_times = []

    deg = np.pi / 180

    for n_sources in n_sources_list:
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(angles)]

        radius = torch.tensor(0.3, dtype=torch.float32, device=device, requires_grad=True)
        spec = {
            "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG, "device": device},
            "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
            "vars": {"$r": radius},
            "layers": [{"material": "Si", "thickness": 0.5, "is_homogeneous": False, "pattern": {"bg_material": "air", "shapes": [{"name": "c", "type": "circle", "center": [0, 0], "radius": "$r"}], "layer_shape": "c"}}],
            "sources": sources,
            "output": {"type": "torch"},
        }

        term_list = [f"Results['s{i}']['efficiency']['transmission'][0]" for i in range(len(sources))]
        objective = f"-mean([{', '.join(term_list)}])"

        start = time.time()
        _ = tr.optimize(spec, objective=objective, options={"steps": 1, "optimizer": {"name": "adam", "lr": 1e-2}})
        step_time = time.time() - start
        step_times.append(step_time)

        print(f"N={n_sources:2d}: Optimize step time={step_time:.3f}s")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(n_sources_list, step_times, "go-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Sources")
    ax.set_ylabel("Step Time (seconds)")
    ax.set_title("Optimization Step Time vs Batch Size")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "optimization_overhead.png", dpi=150)
    plt.close()


def main():
    """Run all performance benchmarks."""
    print("TorchRDIT Source Batching - Performance Benchmarks (interface)")
    print("=============================================================")

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
