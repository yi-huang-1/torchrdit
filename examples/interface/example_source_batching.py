"""
Example: Source Batching for Efficient Multi-Angle and Multi-Polarization Simulations (interface simulate)

This reproduces `torchrdit/examples/example_source_batching.py`, but uses the
regulated interface API for forward simulations:

```python
import torchrdit as tr
results = tr.simulate(spec)
```

Plotting parameters and numerical constants match the original script.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import torchrdit as tr
from torchrdit.shapes import ShapeGenerator


def _shape_generator_cartesian(rdim, device="cpu"):
    vec_p = torch.linspace(-0.5, 0.5, rdim[0], device=device)
    vec_q = torch.linspace(-0.5, 0.5, rdim[1], device=device)
    mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")
    return ShapeGenerator(mesh_p, mesh_q, tuple(rdim))


def example_angle_sweep():
    """Demonstrate angle sweep with batched sources."""
    print("\n=== Example 1: Angle Sweep ===")

    # Create a simple grating structure (same parameters as original)
    rdim = [512, 512]
    kdim = [5, 5]
    wavelengths = np.array([1.55])
    device = "cpu"

    mask = torch.zeros(512, 512)
    for i in range(0, 512, 128):
        mask[:, i : i + 64] = 1.0

    # Create multiple sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 13) * deg  # 0° to 60° in 5° steps

    sources = []
    for i, theta in enumerate(angles):
        sources.append({"name": f"s{i}", "theta": float(theta), "phi": 0, "pte": 1.0, "ptm": 0.0})

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": rdim, "harmonics": kdim, "device": device},
        "materials": {
            "Si": {"permittivity": 12.25},  # n=3.5
            "SiO2": {"permittivity": 2.25},  # n=1.5
            "air": {"permittivity": 1.0},
        },
        "vars": {"$mask": mask},
        "layers": [
            {"material": "air", "thickness": 0.5, "is_homogeneous": True},
            {
                "material": "Si",
                "thickness": 0.5,
                "is_homogeneous": False,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [{"name": "m", "type": "mask", "value": "$mask"}],
                    "layer_shape": "m",
                },
            },
            {"material": "SiO2", "thickness": 1.0, "is_homogeneous": True},
        ],
        "sources": sources,
        "output": {"type": "torch"},
    }

    print(f"\nComparing performance for {len(sources)} sources:")

    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for src in sources:
        one = dict(spec)
        one["sources"] = {k: v for k, v in src.items() if k != "name"}
        result = tr.simulate(one)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"Sequential processing: {sequential_time:.3f} seconds")

    # Batched processing
    start_time = time.time()
    batched_results = tr.simulate(spec)
    batched_time = time.time() - start_time
    print(f"Batched processing: {batched_time:.3f} seconds")
    print(f"Speedup: {sequential_time/batched_time:.2f}x")

    angles_deg = angles * 180 / np.pi
    transmission_te = np.array([batched_results[f"s{i}"]["efficiency"]["transmission"][0].detach().numpy() for i in range(len(sources))])
    reflection_te = np.array([batched_results[f"s{i}"]["efficiency"]["reflection"][0].detach().numpy() for i in range(len(sources))])

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(angles_deg, transmission_te, "b-", linewidth=2, label="Transmission")
    plt.plot(angles_deg, reflection_te, "r-", linewidth=2, label="Reflection")
    plt.plot(angles_deg, transmission_te + reflection_te, "k--", linewidth=1, label="Total", alpha=0.5)
    plt.xlabel("Incident Angle (degrees)")
    plt.ylabel("Efficiency")
    plt.title("TE Polarization Response")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)

    best_trans_idx = int(np.argmax(transmission_te))
    worst_trans_idx = int(np.argmin(reflection_te))

    plt.plot(
        angles_deg[best_trans_idx],
        transmission_te[best_trans_idx],
        "go",
        markersize=8,
        label=f"Max T @ {angles_deg[best_trans_idx]:.1f}°",
    )
    plt.plot(
        angles_deg[worst_trans_idx],
        reflection_te[worst_trans_idx],
        "rs",
        markersize=8,
        label=f"Min R @ {angles_deg[worst_trans_idx]:.1f}°",
    )

    plt.subplot(1, 2, 2)
    zero_order_trans = np.array(
        [batched_results[f"s{i}"]["diffraction_efficiency"]["transmission"][0, 2, 2].detach().numpy() for i in range(len(sources))]
    )

    plt.plot(angles_deg, zero_order_trans, "b-", linewidth=2, label="Zero Order")
    plt.plot(angles_deg, transmission_te, "k--", linewidth=1, label="Total", alpha=0.5)
    plt.xlabel("Incident Angle (degrees)")
    plt.ylabel("Transmission Efficiency")
    plt.title("Diffraction Order Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("example_angle_sweep.png", dpi=150)
    plt.show()

    return batched_results


def example_polarization_sweep():
    """Demonstrate polarization state analysis with batched sources."""
    print("\n=== Example 2: Polarization State Analysis ===")

    rdim = [256, 256]
    kdim = [5, 5]
    wavelengths = np.array([1.55])
    device = "cpu"

    shape_gen = _shape_generator_cartesian(rdim, device=device)
    mask = shape_gen.generate_rectangle_mask(center=(0, 0), x_size=0.3, y_size=0.5, angle=0)

    theta = 30 * np.pi / 180  # 30° incidence
    polarizations = [
        {"name": "TE", "pte": 1.0, "ptm": 0.0},
        {"name": "TM", "pte": 0.0, "ptm": 1.0},
        {"name": "45° Linear", "pte": 0.7071, "ptm": 0.7071},
        {"name": "-45° Linear", "pte": 0.7071, "ptm": -0.7071},
        {"name": "RCP-like", "pte": 0.7071, "ptm": 0.5},
        {"name": "LCP-like", "pte": 0.7071, "ptm": -0.5},
    ]

    sources = []
    for i, pol in enumerate(polarizations):
        sources.append({"name": f"s{i}", "theta": float(theta), "phi": 0, "pte": pol["pte"], "ptm": pol["ptm"]})

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": rdim, "harmonics": kdim, "device": device},
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(polarizations))
    width = 0.35

    ax1.bar(x - width / 2, transmission, width, label="Transmission", color="blue", alpha=0.7)
    ax1.bar(x + width / 2, reflection, width, label="Reflection", color="red", alpha=0.7)

    ax1.set_xlabel("Polarization State")
    ax1.set_ylabel("Efficiency")
    ax1.set_title("Polarization-Dependent Response")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p["name"] for p in polarizations], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1.05)

    ax2.set_aspect("equal")

    for i, (pol, trans) in enumerate(zip(polarizations, transmission)):
        Ex = pol["pte"]
        Ey = pol["ptm"]

        S0 = abs(Ex) ** 2 + abs(Ey) ** 2
        S1 = abs(Ex) ** 2 - abs(Ey) ** 2
        S2 = 2 * Ex * Ey

        if S0 > 0:
            s1 = S1 / S0
            s2 = S2 / S0

            color = plt.cm.viridis(trans)
            ax2.scatter(s1, s2, s=200, c=[color], alpha=0.8, edgecolors="black", linewidth=1)
            ax2.annotate(pol["name"], (s1, s2), xytext=(5, 5), textcoords="offset points", fontsize=9)

    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    ax2.add_artist(circle)

    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel("S₁/S₀")
    ax2.set_ylabel("S₂/S₀")
    ax2.set_title("Polarization States on Poincaré Sphere (Equatorial View)")
    ax2.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label("Transmission")

    plt.tight_layout()
    plt.savefig("example_polarization_sweep.png", dpi=150)
    plt.show()

    return results


def example_optimization_with_batched_sources():
    """Demonstrate optimization with multiple incident conditions."""
    print("\n=== Example 3: Multi-Angle Optimization ===")

    rdim = [256, 256]
    kdim = [5, 5]
    wavelengths = np.array([1.55])
    device = "cpu"

    shape_gen = _shape_generator_cartesian(rdim, device=device)

    solver_spec = {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": rdim, "harmonics": kdim, "device": device}
    layers = [
        {"material": "air", "thickness": 0.5, "is_homogeneous": True},
        {"material": "Si", "thickness": 0.3, "is_homogeneous": False, "pattern": {"shapes": [{"name": "m", "type": "mask", "value": "$mask"}], "layer_shape": "m"}},
        {"material": "air", "thickness": 0.5, "is_homogeneous": True},
    ]

    radius_init = 0.2
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_init)
    mask = mask.to(torch.float32)
    mask.requires_grad = True

    deg = np.pi / 180
    target_angles = [0 * deg, 15 * deg, 30 * deg]
    sources = [{"name": f"s{i}", "theta": float(th), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, th in enumerate(target_angles)]

    optimizer = torch.optim.Adam([mask], lr=0.02)

    history = {"loss": [], "transmission": []}

    print("\nOptimizing for uniform response across multiple angles...")
    n_iterations = 50

    base_spec = {
        "solver": solver_spec,
        "materials": {"Si": {"permittivity": 12.25}, "air": {"permittivity": 1.0}},
        "layers": layers,
        "sources": sources,
        "output": {"type": "torch"},
    }

    for iter in range(n_iterations):
        optimizer.zero_grad()

        spec = dict(base_spec)
        spec["vars"] = {"$mask": mask}

        results = tr.simulate(spec)
        trans_values = torch.stack([results[f"s{i}"]["efficiency"]["transmission"][0] for i in range(len(sources))])
        avg_trans = trans_values.mean()
        var_trans = trans_values.var()

        loss = -avg_trans + 0.5 * var_trans

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mask.data = torch.clamp(mask.data, 0, 1)

        history["loss"].append(loss.item())
        history["transmission"].append(trans_values.detach().numpy())

        if iter % 10 == 0:
            print(f"Iteration {iter}: Loss = {loss.item():.4f}, " f"Avg T = {avg_trans.item():.4f}, " f"Std T = {var_trans.sqrt().item():.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(history["loss"], "b-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    trans_history = np.array(history["transmission"])
    for i, angle in enumerate(target_angles):
        ax.plot(trans_history[:, i], linewidth=2, label=f"{angle*180/np.pi:.0f}°")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Transmission")
    ax.set_title("Transmission Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    initial_mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_init)
    im = ax.imshow(initial_mask.numpy(), cmap="binary")
    ax.set_title("Initial Pattern")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(mask.detach().numpy(), cmap="binary")
    ax.set_title("Optimized Pattern")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("example_optimization_batched.png", dpi=150)
    plt.show()

    print("\nFinal performance verification:")
    test_angles = np.linspace(0, 45, 10) * deg
    test_sources = [{"name": f"t{i}", "theta": float(th), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, th in enumerate(test_angles)]

    test_spec = dict(base_spec)
    test_spec["sources"] = test_sources
    test_spec["vars"] = {"$mask": mask.detach()}

    final_results = tr.simulate(test_spec)
    final_trans = np.array([final_results[f"t{i}"]["efficiency"]["transmission"][0].detach().numpy() for i in range(len(test_sources))])

    print(f"Average transmission: {final_trans.mean():.4f}")
    print(f"Standard deviation: {final_trans.std():.4f}")
    print(f"Min/Max transmission: {final_trans.min():.4f} / {final_trans.max():.4f}")

    return mask, final_results


def example_wavelength_and_angle_sweep():
    """Demonstrate combined wavelength and angle parameter sweep."""
    print("\n=== Example 4: Combined Wavelength and Angle Sweep ===")

    wavelengths = np.linspace(1.5, 1.6, 5)
    rdim = [256, 256]
    kdim = [5, 5]
    device = "cpu"

    deg = np.pi / 180
    angles = np.linspace(0, 60, 7) * deg
    sources = [{"name": f"s{i}", "theta": float(th), "phi": 0, "pte": 1.0, "ptm": 0.0} for i, th in enumerate(angles)]

    spec = {
        "solver": {"algorithm": "RDIT", "wavelengths": wavelengths, "grids": rdim, "harmonics": kdim, "device": device},
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

    results = tr.simulate(spec)

    transmission = np.stack([results[f"s{i}"]["efficiency"]["transmission"].detach().numpy() for i in range(len(sources))], axis=0)
    reflection = np.stack([results[f"s{i}"]["efficiency"]["reflection"].detach().numpy() for i in range(len(sources))], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(transmission, aspect="auto", origin="lower", extent=[wavelengths[0], wavelengths[-1], 0, 60], cmap="viridis", vmin=0, vmax=1)
    ax1.set_xlabel("Wavelength (μm)")
    ax1.set_ylabel("Incident Angle (degrees)")
    ax1.set_title("Transmission Map")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(reflection, aspect="auto", origin="lower", extent=[wavelengths[0], wavelengths[-1], 0, 60], cmap="plasma", vmin=0, vmax=1)
    ax2.set_xlabel("Wavelength (μm)")
    ax2.set_ylabel("Incident Angle (degrees)")
    ax2.set_title("Reflection Map")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig("example_wavelength_angle_sweep.png", dpi=150)
    plt.show()

    best_idx = int(np.argmax(transmission.mean(axis=1)))
    best_angle = angles[best_idx] * 180 / np.pi

    best_wl_idx = int(np.argmax(transmission[best_idx, :]))
    best_wl = wavelengths[best_wl_idx]

    print("\nOptimal operating point:")
    print(f"  Angle: {best_angle:.1f}°")
    print(f"  Wavelength: {best_wl:.3f} μm")
    print(f"  Transmission: {transmission[best_idx, best_wl_idx]:.4f}")

    return results


def main():
    """Run all examples."""
    print("TorchRDIT Source Batching Examples")
    print("==================================")

    example_angle_sweep()
    example_polarization_sweep()
    example_optimization_with_batched_sources()
    example_wavelength_and_angle_sweep()

    print("\nAll examples completed!")
    print("Generated plots: ")
    print("  - example_angle_sweep.png")
    print("  - example_polarization_sweep.png")
    print("  - example_optimization_batched.png")
    print("  - example_wavelength_angle_sweep.png")


if __name__ == "__main__":
    main()

