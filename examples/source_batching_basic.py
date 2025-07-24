"""
Source Batching Basic Usage Examples for TorchRDIT v0.1.22

This script demonstrates the basic usage of source batching feature that allows 
you to process multiple incident angles and polarizations simultaneously.

Key features demonstrated:
- Single vs batched source comparison
- Angle sweep simulations
- Polarization state analysis
- Basic performance comparison
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator
import time


def example_single_vs_batched():
    """Compare traditional single-source approach with new batched processing."""
    print("\n=== Example 1: Single vs Batched Processing ===")
    
    # Create a solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),  # 1.55 μm wavelength
        rdim=[512, 512],        # Real-space dimensions
        kdim=[7, 7],            # k-space dimensions
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)  # n=3.5
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Add a simple structure - silicon cylinder in air
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    print(f"Solver created with device: {solver.device}")
    
    # Single source example (backward compatible)
    print("\n--- Single Source Processing ---")
    source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
    result = solver.solve(source)
    
    print("Single source result:")
    print(f"  Transmission: {result.transmission[0].item():.4f}")
    print(f"  Reflection: {result.reflection[0].item():.4f}")
    print(f"  Result type: {type(result).__name__}")
    
    # Multiple sources example (new feature)
    print("\n--- Batched Source Processing ---")
    deg = np.pi / 180
    sources = [
        solver.add_source(theta=0*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=45*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=60*deg, phi=0, pte=1.0, ptm=0.0)
    ]
    
    # Batch solve
    results = solver.solve(sources)
    
    print("Batched results:")
    print(f"  Result type: {type(results).__name__}")
    print(f"  Number of sources: {results.n_sources}")
    print(f"  Transmission shape: {results.transmission.shape}")
    print("\nTransmission values:")
    for i, trans in enumerate(results.transmission[:, 0]):
        angle = results.source_parameters[i]['theta'] * 180 / np.pi
        print(f"  θ={angle:3.0f}°: {trans.item():.4f}")
    
    # Demonstrate accessing individual results
    print("\n--- Accessing Individual Results ---")
    result_45deg = results[2]  # Returns SolverResults for 45° incidence
    print(f"Result at 45°: {type(result_45deg).__name__}")
    print(f"  Transmission: {result_45deg.transmission[0].item():.4f}")
    
    # Iterate through all results
    print("\nIterating through results:")
    for i, single_result in enumerate(results):
        angle = results.source_parameters[i]['theta'] * 180 / np.pi
        print(f"  θ={angle:3.0f}°: T={single_result.transmission[0].item():.4f}")


def example_angle_sweep():
    """Demonstrate efficient angle sweep using source batching."""
    print("\n=== Example 2: Angle Sweep with Batched Sources ===")
    
    # Create a more complex structure - grating
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[512, 512],
        kdim=[11, 11]
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Add grating structure
    solver.add_layer(material_name="Si", thickness=0.6, is_homogeneous=False)
    
    # Create grating pattern
    mask = torch.zeros(512, 512)
    period = 128  # pixels
    duty_cycle = 0.5
    for i in range(0, 512, period):
        mask[:, i:i+int(period*duty_cycle)] = 1.0
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Angle sweep from -60° to 60°
    deg = np.pi / 180
    angles = np.linspace(-60, 60, 121) * deg
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in angles
    ]
    
    # Compare sequential vs batched processing
    print(f"\nComparing performance for {len(sources)} sources:")
    
    # Sequential processing (for comparison)
    start_time = time.time()
    sequential_results = []
    for i in range(min(5, len(sources))):  # Only do a few for time comparison
        result = solver.solve(sources[i])
        sequential_results.append(result)
    sequential_time = (time.time() - start_time) * len(sources) / 5  # Extrapolate
    print(f"Sequential processing (estimated): {sequential_time:.2f}s")
    
    # Batched processing
    start_time = time.time()
    results = solver.solve(sources)
    batched_time = time.time() - start_time
    print(f"Batched processing: {batched_time:.2f}s")
    print(f"Speedup: {sequential_time/batched_time:.1f}x")
    
    # Extract and plot results
    transmissions = results.transmission[:, 0].detach().cpu().numpy()
    reflections = results.reflection[:, 0].detach().cpu().numpy()
    angles_deg = angles * 180 / np.pi
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, transmissions, 'b-', linewidth=2, label='Transmission')
    plt.plot(angles_deg, reflections, 'r-', linewidth=2, label='Reflection')
    plt.plot(angles_deg, transmissions + reflections, 'k--', 
             linewidth=1, label='Sum (Conservation)', alpha=0.5)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Efficiency')
    plt.title('Angular Response of Silicon Grating')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-60, 60)
    plt.ylim(0, 1.05)
    plt.savefig('angle_sweep_basic.png', dpi=150)
    plt.show()
    
    # Find optimal angle
    best_idx = results.find_optimal_source(metric="max_transmission")
    best_params = results.source_parameters[best_idx]
    print(f"\nMaximum transmission: {transmissions[best_idx]:.4f}")
    print(f"Occurs at angle: {best_params['theta']*180/np.pi:.1f}°")


def example_polarization_analysis():
    """Demonstrate polarization state analysis with batched sources."""
    print("\n=== Example 3: Polarization State Analysis ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7]
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Create anisotropic structure (elliptical pillar)
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create elliptical pattern
    shape_gen = ShapeGenerator.from_solver(solver)
    # Create an elongated pattern using a rectangle to simulate ellipse
    mask = shape_gen.generate_rectangle_mask(center=(0, 0), width=0.3, height=0.5, angle=0)
    solver.update_er_with_mask(mask=mask, layer_index=1)
    
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    
    # Create sources with different polarization states
    theta = 30 * np.pi / 180  # 30° incidence
    
    # Define polarization states
    polarizations = [
        {"name": "TE", "pte": 1.0, "ptm": 0.0},
        {"name": "TM", "pte": 0.0, "ptm": 1.0},
        {"name": "45° Linear", "pte": 0.7071, "ptm": 0.7071},
        {"name": "-45° Linear", "pte": 0.7071, "ptm": -0.7071},
        # Note: Complex polarizations not supported in current implementation
        # Using elliptical approximations instead
        {"name": "RCP-like", "pte": 0.7071, "ptm": 0.5},
        {"name": "LCP-like", "pte": 0.7071, "ptm": -0.5},
    ]
    
    sources = []
    for pol in polarizations:
        source = solver.add_source(
            theta=theta,
            phi=0,
            pte=pol["pte"],
            ptm=pol["ptm"]
        )
        sources.append(source)
    
    # Solve for all polarizations
    results = solver.solve(sources)
    
    # Extract results
    transmission = results.transmission[:, 0].detach().cpu().numpy()
    reflection = results.reflection[:, 0].detach().cpu().numpy()
    
    # Visualize polarization response
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Bar plot of transmission/reflection
    x = np.arange(len(polarizations))
    width = 0.35
    
    ax.bar(x - width/2, transmission, width, label='Transmission', color='blue', alpha=0.7)
    ax.bar(x + width/2, reflection, width, label='Reflection', color='red', alpha=0.7)
    
    ax.set_xlabel('Polarization State')
    ax.set_ylabel('Efficiency')
    ax.set_title('Polarization-Dependent Response at 30° Incidence')
    ax.set_xticks(x)
    ax.set_xticklabels([p["name"] for p in polarizations], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('polarization_analysis_basic.png', dpi=150)
    plt.show()
    
    print("\nPolarization analysis results:")
    for i, pol in enumerate(polarizations):
        print(f"{pol['name']:15s}: T={transmission[i]:.4f}, R={reflection[i]:.4f}")


def example_parameter_sweep():
    """Demonstrate mixed parameter sweeps in a single batch."""
    print("\n=== Example 4: Mixed Parameter Sweep ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5]
    )
    
    # Add materials and simple structure
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.4, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.25)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Create mixed parameter sweep
    deg = np.pi / 180
    mixed_sources = [
        # Vary angle at TE polarization
        solver.add_source(theta=0*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=60*deg, phi=0, pte=1.0, ptm=0.0),
        # Vary polarization at fixed angle
        solver.add_source(theta=45*deg, phi=0, pte=0.0, ptm=1.0),
        solver.add_source(theta=45*deg, phi=0, pte=0.707, ptm=0.707),
        # Vary azimuthal angle
        solver.add_source(theta=30*deg, phi=45*deg, pte=1.0, ptm=0.0),
        solver.add_source(theta=30*deg, phi=90*deg, pte=1.0, ptm=0.0),
    ]
    
    # Solve batch
    results = solver.solve(mixed_sources)
    
    # Extract parameter sweep data
    theta_data, theta_values = results.get_parameter_sweep_data('theta', 'transmission')
    print("\nUnique theta values (degrees):", np.unique(theta_values.numpy()) * 180 / np.pi)
    
    # Display results
    print("\nMixed parameter sweep results:")
    for i, params in enumerate(results.source_parameters):
        t = results.transmission[i, 0].item()
        print(f"  θ={params['theta']*180/np.pi:5.1f}°, "
              f"φ={params['phi']*180/np.pi:5.1f}°, "
              f"pte={params['pte']:.3f}, ptm={params['ptm']:.3f} "
              f"→ T={t:.4f}")


def main():
    """Run all basic examples."""
    print("TorchRDIT Source Batching - Basic Examples")
    print("==========================================")
    
    # Run examples
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