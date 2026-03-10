"""
Source Batching Advanced Examples (interface optimize/simulate)

This reproduces `torchrdit/examples/source_batching_advanced.py`, but uses the
regulated interface APIs (`tr.simulate`, `tr.optimize`) and auto harmonics
selection (`harmonics="auto"` + `maxG`).
"""

from __future__ import annotations

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


def example_multi_angle_optimization():
    """Optimize a structure for uniform response across multiple angles."""
    print("\n=== Example 1: Multi-Angle Optimization ===")

    grids = [256, 256]
    harmonics = [7, 7]
    maxG = harmonics[0] * harmonics[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavelengths = np.array([1.55])

    radius = torch.tensor(0.25, requires_grad=True, device=device)

    deg = np.pi / 180
    target_angles = [0 * deg, 15 * deg, 30 * deg]
    sources = [{"name": f"s{i}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, angle in enumerate(target_angles)]

    spec = {
        "solver": {
            "algorithm": "RDIT",
            "wavelengths": wavelengths,
            "grids": grids,
            "harmonics": "auto",
            "maxG": maxG,
            "device": device,
        },
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$r": radius},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.6,
                "is_homogeneous": False,
                "pattern": {"bg_material": "air", "shapes": [{"name": "c", "type": "circle", "center": [0, 0], "radius": "$r"}], "layer_shape": "c"},
            }
        ],
        "sources": sources,
        "output": {"type": "torch"},
    }

    term_list = [f"Results['s{i}']['efficiency']['transmission'][0]" for i in range(len(sources))]
    mean_term = f"mean([{', '.join(term_list)}])"
    mean_sq_term = f"mean([{', '.join([f'({t})**2' for t in term_list])}])"
    objective = f"-{mean_term} + 0.5*({mean_sq_term} - ({mean_term})**2)"

    opt_out = tr.optimize(
        spec,
        objective=objective,
        options={"steps": 50, "optimizer": {"name": "adam", "lr": 0.02}, "return_best": True},
    )

    best = opt_out.get("best")
    best_results = best["results"] if best else opt_out.get("last_results", {})
    best_vars = best["vars"] if best else opt_out["vars"]
    radius_opt = best_vars["$r"].item()
    loss_history = opt_out["loss_history"]

    trans = np.array([best_results[f"s{i}"]["efficiency"]["transmission"][0].detach().cpu().numpy() for i in range(len(sources))])

    print(f"Final radius: {radius_opt:.4f}")
    print(f"Final average transmission: {trans.mean():.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax1.plot(loss_history, "b-", linewidth=2)
    ax1.set_ylabel("Loss")
    ax1.set_title("Optimization Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.array(target_angles) * 180 / np.pi, trans, "o-", linewidth=2)
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_ylabel("Transmission")
    ax2.set_title("Optimized Transmission at Target Angles")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "multi_angle_optimization.png", dpi=150)
    plt.close(fig)


def example_robust_design():
    """Design a structure robust to both angle and polarization variations."""
    print("\n=== Example 2: Robust Design for Angle and Polarization ===")

    grids = [256, 256]
    harmonics = [7, 7]
    maxG = harmonics[0] * harmonics[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavelengths = np.array([1.55])

    r1 = torch.tensor(0.15, requires_grad=True, device=device)
    r2 = torch.tensor(0.15, requires_grad=True, device=device)
    sep = torch.tensor(0.3, requires_grad=True, device=device)

    deg = np.pi / 180
    sources = []
    for angle in np.array([0, 20, 40]) * deg:
        sources.append({"name": f"a{len(sources)}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0})
    for pte, ptm in [(1.0, 0.0), (0.0, 1.0), (0.707, 0.707)]:
        sources.append({"name": f"a{len(sources)}", "theta": float(30 * deg), "phi": 0, "pte": pte, "ptm": ptm})

    spec = {
        "solver": {
            "algorithm": "RDIT",
            "wavelengths": wavelengths,
            "grids": grids,
            "harmonics": "auto",
            "maxG": maxG,
            "device": device,
        },
        "materials": {"Si": {"permittivity": 12.25}, "SiO2": {"permittivity": 2.25}},
        "vars": {"$r1": r1, "$r2": r2, "$sep": sep, "$sep2": "$sep/2", "$msep2": "-$sep/2"},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.5,
                "is_homogeneous": False,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [
                        {"name": "c1", "type": "circle", "center": ["$msep2", 0], "radius": "$r1"},
                        {"name": "c2", "type": "circle", "center": ["$sep2", 0], "radius": "$r2"},
                        {"name": "mask", "type": "op", "expr": "union(c1, c2)"},
                    ],
                    "layer_shape": "mask",
                },
            }
        ],
        "sources": sources,
        "output": {"type": "torch"},
    }

    term_list = [f"Results['{src['name']}']['efficiency']['transmission'][0]" for src in sources]
    mean_term = f"mean([{', '.join(term_list)}])"
    objective = f"-{mean_term} + 2.0*(max([{', '.join(term_list)}]) - min([{', '.join(term_list)}])) + 0.1*(r1**2 + r2**2)"

    opt_out = tr.optimize(
        spec,
        objective=objective,
        options={"steps": 40, "optimizer": {"name": "adam", "lr": 0.01}, "return_best": True},
    )

    best = opt_out.get("best")
    best_results = best["results"] if best else opt_out.get("last_results", {})
    best_vars = best["vars"] if best else opt_out["vars"]
    loss_history = opt_out["loss_history"]

    trans = np.array([best_results[src["name"]]["efficiency"]["transmission"][0].detach().cpu().numpy() for src in sources])
    r1_val = float(best_vars["$r1"])
    r2_val = float(best_vars["$r2"])
    sep_val = float(best_vars["$sep"])

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask1 = shape_gen.generate_circle_mask(center=(-sep_val / 2, 0), radius=r1_val)
    mask2 = shape_gen.generate_circle_mask(center=(sep_val / 2, 0), radius=r2_val)
    final_mask = torch.maximum(mask1, mask2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(loss_history, "b-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(np.arange(len(trans)), trans, color="green", alpha=0.7)
    ax.set_xlabel("Source Index")
    ax.set_ylabel("Transmission")
    ax.set_title("Final Transmission Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.bar(["r1", "r2", "sep"], [r1_val, r2_val, sep_val], color="purple", alpha=0.7)
    ax.set_title("Optimized Parameters")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.imshow(final_mask.detach().cpu().numpy(), cmap="hot", origin="lower")
    ax.set_title("Final Optimized Structure")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "robust_design_optimization.png", dpi=150)
    plt.close(fig)

    print("\nFinal design:")
    print(f"  Mean transmission: {trans.mean():.4f}")
    print(f"  Std transmission: {trans.std():.4f}")
    print(f"  Parameters: r1={r1_val:.3f}, r2={r2_val:.3f}, sep={sep_val:.3f}")


def example_large_sweep_chunking():
    """Demonstrate memory-efficient processing of large parameter sweeps."""
    print("\n=== Example 3: Memory-Efficient Large Sweep Processing ===")

    def process_large_sweep(base_spec, angles, chunk_size=100):
        all_transmissions = []
        all_reflections = []

        n_chunks = (len(angles) + chunk_size - 1) // chunk_size
        print(f"Processing {len(angles)} angles in {n_chunks} chunks...")

        for i in range(0, len(angles), chunk_size):
            chunk_angles = angles[i : i + chunk_size]
            sources = [{"name": f"s{j}", "theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0} for j, angle in enumerate(chunk_angles)]

            spec = dict(base_spec)
            spec["sources"] = sources
            results = tr.simulate(spec)
            all_transmissions.extend([results[f"s{j}"]["efficiency"]["transmission"][0].item() for j in range(len(sources))])
            all_reflections.extend([results[f"s{j}"]["efficiency"]["reflection"][0].item() for j in range(len(sources))])

            if base_spec["solver"].get("device", "cpu").startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  Processed chunk {i // chunk_size + 1}/{n_chunks}")

        return np.array(all_transmissions), np.array(all_reflections)

    grids = [256, 256]
    harmonics = [5, 5]
    maxG = harmonics[0] * harmonics[1]
    wavelengths = np.array([1.55])

    shape_gen = _shape_generator_cartesian(grids, device="cpu")
    mask = shape_gen.generate_rectangle_mask(x_size=0.4, y_size=0.2, angle=45)

    base_spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": grids, "harmonics": "auto", "maxG": maxG},
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "vars": {"$mask": mask},
        "layers": [{"material": "Si", "thickness": 0.5, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}}],
        "output": {"type": "torch"},
    }

    deg = np.pi / 180
    many_angles = np.linspace(0, 89, 1000) * deg

    start = time.time()
    trans_large, refl_large = process_large_sweep(base_spec, many_angles, chunk_size=100)
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results shape: {trans_large.shape}")
    print(f"Average transmission: {trans_large.mean():.4f}")
    print("Peak memory usage avoided by chunking!")

    stride = 10
    angles_deg = many_angles[::stride] * 180 / np.pi

    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, trans_large[::stride], "b-", linewidth=1, label="Transmission")
    plt.plot(angles_deg, refl_large[::stride], "r-", linewidth=1, label="Reflection")
    plt.xlabel("Incident Angle (degrees)")
    plt.ylabel("Efficiency")
    plt.title("Large Angle Sweep (1000 angles)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 89)
    plt.ylim(0, 1.05)
    plt.savefig(Path(__file__).parent / "large_sweep_chunking.png", dpi=150)
    plt.close(plt.gcf())


def example_gradient_validation():
    """Explain why gradient validation is not supported via interface solve."""
    print("\n=== Example 4: Gradient Validation for Batched Solving ===")
    print("Interface solve runs under torch.no_grad(); use solver-level APIs for gradient validation.")


def main():
    """Run all advanced examples."""
    print("TorchRDIT Source Batching - Advanced Examples (interface)")
    print("=======================================================")

    example_multi_angle_optimization()
    example_robust_design()
    example_large_sweep_chunking()
    example_gradient_validation()

    print("\nAll advanced examples completed!")
    print("Generated plots:")
    print("  - multi_angle_optimization.png")
    print("  - robust_design_optimization.png")
    print("  - large_sweep_chunking.png")


if __name__ == "__main__":
    main()
