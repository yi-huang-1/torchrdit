"""
Source Batching Basic Usage Examples (interface simulate)

This reproduces `torchrdit/examples/source_batching_basic.py`, but runs the
forward simulation through the regulated interface API (`tr.simulate`) and
uses auto harmonics selection (`harmonics="auto"` + `maxG`).
"""

from __future__ import annotations

import time
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


def example_single_vs_batched():
    """Compare single-source and batched processing via interface."""
    print("\n=== Example 1: Single vs Batched Processing ===")

    grids = [512, 512]
    harmonics = [7, 7]
    maxG = harmonics[0] * harmonics[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavelengths = np.array([1.55])

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)

    base_spec = {
        "solver": {
            "algorithm": "RDIT",
            "wavelengths": wavelengths,
            "grids": grids,
            "harmonics": "auto",
            "maxG": maxG,
            "device": device,
        },
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.5,
                "is_homogeneous": False,
                "pattern": {"bg_material": "air", "shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"},
            }
        ],
        "output": {"type": "torch"},
    }

    print(f"Solver spec uses device: {device}")

    print("\n--- Single Source Processing ---")
    single_spec = dict(base_spec)
    single_spec["sources"] = {"theta": 0, "phi": 0, "pte": 1.0, "ptm": 0.0}
    result = tr.simulate(single_spec)

    print("Single source result:")
    print(f"  Transmission: {result['efficiency']['transmission'][0].item():.4f}")
    print(f"  Reflection: {result['efficiency']['reflection'][0].item():.4f}")
    print(f"  Result type: {type(result).__name__}")

    print("\n--- Batched Source Processing ---")
    deg = np.pi / 180
    sources = [
        {"name": "s0", "theta": float(0 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
        {"name": "s1", "theta": float(30 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
        {"name": "s2", "theta": float(45 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
        {"name": "s3", "theta": float(60 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
    ]

    batch_spec = dict(base_spec)
    batch_spec["sources"] = sources
    results = tr.simulate(batch_spec)

    print("Batched results:")
    print(f"  Result type: {type(results).__name__}")
    print(f"  Number of sources: {len(sources)}")
    print("\nTransmission values:")
    for i, src in enumerate(sources):
        angle = src["theta"] * 180 / np.pi
        trans = results[f"s{i}"]["efficiency"]["transmission"][0].item()
        print(f"  θ={angle:3.0f}°: {trans:.4f}")


def example_angle_sweep():
    """Demonstrate angle sweep with batched sources."""
    print("\n=== Example 2: Angle Sweep with Batched Sources ===")

    grids = [512, 512]
    harmonics = [11, 11]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])

    mask = torch.zeros(512, 512)
    period = 128
    duty_cycle = 0.5
    for i in range(0, 512, period):
        mask[:, i : i + int(period * duty_cycle)] = 1.0

    deg = np.pi / 180
    angles = np.linspace(-60, 60, 121) * deg
    sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(angles)]

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG},
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.6,
                "is_homogeneous": False,
                "pattern": {"bg_material": "air", "shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"},
            }
        ],
        "sources": sources,
        "output": {"type": "torch"},
    }

    print(f"\nComparing performance for {len(sources)} sources:")

    start_time = time.time()
    for src in sources[:5]:
        one = dict(spec)
        one["sources"] = {k: v for k, v in src.items() if k != "name"}
        _ = tr.simulate(one)
    sequential_time = (time.time() - start_time) * len(sources) / 5
    print(f"Sequential processing (estimated): {sequential_time:.2f}s")

    start_time = time.time()
    results = tr.simulate(spec)
    batched_time = time.time() - start_time
    print(f"Batched processing: {batched_time:.2f}s")
    print(f"Speedup: {sequential_time / batched_time:.1f}x")

    transmissions = np.array([results[f"s{i}"]["efficiency"]["transmission"][0].detach().cpu().numpy() for i in range(len(sources))])
    reflections = np.array([results[f"s{i}"]["efficiency"]["reflection"][0].detach().cpu().numpy() for i in range(len(sources))])
    angles_deg = angles * 180 / np.pi

    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, transmissions, "b-", linewidth=2, label="Transmission")
    plt.plot(angles_deg, reflections, "r-", linewidth=2, label="Reflection")
    plt.plot(angles_deg, transmissions + reflections, "k--", linewidth=1, label="Sum", alpha=0.5)
    plt.xlabel("Incident Angle (degrees)")
    plt.ylabel("Efficiency")
    plt.title("Angular Response of Silicon Grating")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-60, 60)
    plt.ylim(0, 1.05)
    plt.savefig("angle_sweep_basic.png", dpi=150)
    plt.show()

    best_idx = int(np.argmax(transmissions))
    print(f"\nMaximum transmission: {transmissions[best_idx]:.4f}")
    print(f"Occurs at angle: {angles_deg[best_idx]:.1f}°")


def example_polarization_analysis():
    """Demonstrate polarization state analysis with batched sources."""
    print("\n=== Example 3: Polarization State Analysis ===")

    grids = [256, 256]
    harmonics = [7, 7]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask = shape_gen.generate_rectangle_mask(center=(0, 0), x_size=0.3, y_size=0.5, angle=0)

    theta = 30 * np.pi / 180
    polarizations = [
        {"name": "TE", "pte": 1.0, "ptm": 0.0},
        {"name": "TM", "pte": 0.0, "ptm": 1.0},
        {"name": "45° Linear", "pte": 0.7071, "ptm": 0.7071},
        {"name": "-45° Linear", "pte": 0.7071, "ptm": -0.7071},
        {"name": "RCP-like", "pte": 0.7071, "ptm": 0.5},
        {"name": "LCP-like", "pte": 0.7071, "ptm": -0.5},
    ]

    sources = [{"name": f"s{i}", "theta": float(theta), "phi": 0, "pte": pol["pte"], "ptm": pol["ptm"]} for i, pol in enumerate(polarizations)]

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG},
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [
            {"material": "air", "thickness": 0.5, "is_homogeneous": True},
            {"material": "Si", "thickness": 0.5, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}},
            {"material": "air", "thickness": 0.5, "is_homogeneous": True},
        ],
        "sources": sources,
        "output": {"type": "torch"},
    }

    results = tr.simulate(spec)
    transmission = np.array([results[f"s{i}"]["efficiency"]["transmission"][0].detach().cpu().numpy() for i in range(len(polarizations))])
    reflection = np.array([results[f"s{i}"]["efficiency"]["reflection"][0].detach().cpu().numpy() for i in range(len(polarizations))])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(polarizations))
    width = 0.35

    ax.bar(x - width / 2, transmission, width, label="Transmission", color="blue", alpha=0.7)
    ax.bar(x + width / 2, reflection, width, label="Reflection", color="red", alpha=0.7)
    ax.set_xlabel("Polarization State")
    ax.set_ylabel("Efficiency")
    ax.set_title("Polarization-Dependent Response at 30° Incidence")
    ax.set_xticks(x)
    ax.set_xticklabels([p["name"] for p in polarizations], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("polarization_analysis_basic.png", dpi=150)
    plt.show()

    print("\nPolarization analysis results:")
    for i, pol in enumerate(polarizations):
        print(f"{pol['name']:15s}: T={transmission[i]:.4f}, R={reflection[i]:.4f}")


def example_parameter_sweep():
    """Demonstrate mixed parameter sweeps in a single batch."""
    print("\n=== Example 4: Mixed Parameter Sweep ===")

    grids = [256, 256]
    harmonics = [5, 5]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.25)

    deg = np.pi / 180
    mixed_sources = [
        {"name": "s0", "theta": float(0 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
        {"name": "s1", "theta": float(30 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
        {"name": "s2", "theta": float(60 * deg), "phi": 0, "pte": 1.0, "ptm": 0.0},
        {"name": "s3", "theta": float(45 * deg), "phi": 0, "pte": 0.0, "ptm": 1.0},
        {"name": "s4", "theta": float(45 * deg), "phi": 0, "pte": 0.707, "ptm": 0.707},
        {"name": "s5", "theta": float(30 * deg), "phi": float(45 * deg), "pte": 1.0, "ptm": 0.0},
        {"name": "s6", "theta": float(30 * deg), "phi": float(90 * deg), "pte": 1.0, "ptm": 0.0},
    ]

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG},
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [{"material": "Si", "thickness": 0.4, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}}],
        "sources": mixed_sources,
        "output": {"type": "torch"},
    }

    results = tr.simulate(spec)
    theta_values = np.array([src["theta"] for src in mixed_sources])

    print("\nUnique theta values (degrees):", np.unique(theta_values) * 180 / np.pi)
    print("\nMixed parameter sweep results:")
    for i, src in enumerate(mixed_sources):
        trans = results[f"s{i}"]["efficiency"]["transmission"][0].item()
        print(
            f"  θ={src['theta']*180/np.pi:5.1f}°, "
            f"φ={src['phi']*180/np.pi:5.1f}°, "
            f"pte={src['pte']:.3f}, ptm={src['ptm']:.3f} "
            f"→ T={trans:.4f}"
        )


def main():
    """Run all basic examples."""
    print("TorchRDIT Source Batching - Basic Examples (interface)")
    print("====================================================")

    example_single_vs_batched()
    example_angle_sweep()
    example_polarization_analysis()
    example_parameter_sweep()

    print("\nAll basic examples completed!")
    print("Generated plots:")
    print("  - angle_sweep_basic.png")
    print("  - polarization_analysis_basic.png")


if __name__ == "__main__":
    main()
