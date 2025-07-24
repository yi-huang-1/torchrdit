"""
Example: Source Batching for Efficient Multi-Angle and Multi-Polarization Simulations

This example demonstrates how to use the source batching feature in TorchRDIT to
efficiently simulate multiple incident conditions simultaneously. Source batching
provides significant performance improvements when analyzing structures under
various illumination conditions.

Key features demonstrated:
1. Angle sweep simulations
2. Polarization state analysis
3. Parameter optimization with multiple sources
4. Performance comparison between sequential and batched processing
5. Visualization of batched results
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator


def example_angle_sweep():
    """Demonstrate angle sweep with batched sources."""
    print("\n=== Example 1: Angle Sweep ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),  # 1.55 μm wavelength
        rdim=[512, 512],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)  # n=3.5
    sio2 = create_material(name="SiO2", permittivity=2.25)  # n=1.5
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, sio2, air])
    
    # Create a simple grating structure
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create grating pattern
    mask = torch.zeros(512, 512)
    for i in range(0, 512, 128):
        mask[:, i:i+64] = 1.0
    solver.update_er_with_mask(mask=mask, layer_index=1, bg_material="SiO2")
    
    solver.add_layer(material_name="SiO2", thickness=1.0, is_homogeneous=True)
    
    # Create multiple sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 13) * deg  # 0° to 60° in 5° steps
    
    sources = []
    for theta in angles:
        source = solver.add_source(
            theta=theta,
            phi=0,
            pte=1.0,  # TE polarization
            ptm=0.0
        )
        sources.append(source)
    
    # Time batched vs sequential processing
    print(f"\nComparing performance for {len(sources)} sources:")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for source in sources:
        result = solver.solve(source)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"Sequential processing: {sequential_time:.3f} seconds")
    
    # Batched processing
    start_time = time.time()
    batched_results = solver.solve(sources)
    batched_time = time.time() - start_time
    print(f"Batched processing: {batched_time:.3f} seconds")
    print(f"Speedup: {sequential_time/batched_time:.2f}x")
    
    # Extract and plot results
    angles_deg = angles * 180 / np.pi
    
    # Get transmission for all angles
    transmission_te = batched_results.transmission[:, 0].detach().numpy()
    reflection_te = batched_results.reflection[:, 0].detach().numpy()
    
    # Plot angle dependence
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(angles_deg, transmission_te, 'b-', linewidth=2, label='Transmission')
    plt.plot(angles_deg, reflection_te, 'r-', linewidth=2, label='Reflection')
    plt.plot(angles_deg, transmission_te + reflection_te, 'k--', 
             linewidth=1, label='Total', alpha=0.5)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Efficiency')
    plt.title('TE Polarization Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)
    
    # Find and mark critical angle features
    best_trans_idx = batched_results.find_optimal_source('max_transmission')
    worst_trans_idx = batched_results.find_optimal_source('min_reflection')
    
    plt.plot(angles_deg[best_trans_idx], transmission_te[best_trans_idx], 
             'go', markersize=8, label=f'Max T @ {angles_deg[best_trans_idx]:.1f}°')
    plt.plot(angles_deg[worst_trans_idx], reflection_te[worst_trans_idx], 
             'rs', markersize=8, label=f'Min R @ {angles_deg[worst_trans_idx]:.1f}°')
    
    # Diffraction efficiency map
    plt.subplot(1, 2, 2)
    # Get zero-order transmission for all angles
    zero_order_trans = batched_results.transmission_diffraction[:, 0, 2, 2].detach().numpy()
    
    plt.plot(angles_deg, zero_order_trans, 'b-', linewidth=2, label='Zero Order')
    plt.plot(angles_deg, transmission_te, 'k--', linewidth=1, label='Total', alpha=0.5)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Transmission Efficiency')
    plt.title('Diffraction Order Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('example_angle_sweep.png', dpi=150)
    plt.show()
    
    return batched_results


def example_polarization_sweep():
    """Demonstrate polarization state analysis with batched sources."""
    print("\n=== Example 2: Polarization State Analysis ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Create anisotropic structure (elliptical pillar array)
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
        # Using 45° elliptical approximations instead
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
    transmission = results.transmission[:, 0].detach().numpy()
    reflection = results.reflection[:, 0].detach().numpy()
    
    # Visualize polarization response
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of transmission/reflection
    x = np.arange(len(polarizations))
    width = 0.35
    
    ax1.bar(x - width/2, transmission, width, label='Transmission', color='blue', alpha=0.7)
    ax1.bar(x + width/2, reflection, width, label='Reflection', color='red', alpha=0.7)
    
    ax1.set_xlabel('Polarization State')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('Polarization-Dependent Response')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p["name"] for p in polarizations], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    # Polarization ellipse visualization
    ax2.set_aspect('equal')
    
    # Plot Poincaré sphere projection
    for i, (pol, trans) in enumerate(zip(polarizations, transmission)):
        # Calculate Stokes parameters
        Ex = pol["pte"]
        Ey = pol["ptm"]
        
        S0 = abs(Ex)**2 + abs(Ey)**2
        S1 = abs(Ex)**2 - abs(Ey)**2
        S2 = 2 * Ex * Ey  # For real polarizations
        # S3 = 0  # For real polarizations, imaginary part is 0
        
        # Normalize
        if S0 > 0:
            s1 = S1 / S0
            s2 = S2 / S0
            # s3 = S3 / S0
            
            # Plot on unit circle (equatorial projection)
            color = plt.cm.viridis(trans)
            ax2.scatter(s1, s2, s=200, c=[color], alpha=0.8, edgecolors='black', linewidth=1)
            ax2.annotate(pol["name"], (s1, s2), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax2.add_artist(circle)
    
    # Draw axes
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel('S₁/S₀')
    ax2.set_ylabel('S₂/S₀')
    ax2.set_title('Polarization States on Poincaré Sphere (Equatorial View)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for transmission
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Transmission')
    
    plt.tight_layout()
    plt.savefig('example_polarization_sweep.png', dpi=150)
    plt.show()
    
    return results


def example_optimization_with_batched_sources():
    """Demonstrate optimization with multiple incident conditions."""
    print("\n=== Example 3: Multi-Angle Optimization ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Create structure with optimizable layer
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.3, is_homogeneous=False)
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    
    # Create initial pattern (to be optimized)
    radius_init = 0.2
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_init)
    mask = mask.to(torch.float32)
    mask.requires_grad = True
    
    solver.update_er_with_mask(mask=mask, layer_index=1)
    
    # Define target angles for optimization
    deg = np.pi / 180
    target_angles = [0*deg, 15*deg, 30*deg]  # Optimize for these angles
    
    sources = []
    for theta in target_angles:
        source = solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0)
        sources.append(source)
    
    # Optimization loop
    optimizer = torch.optim.Adam([mask], lr=0.02)
    
    history = {
        'loss': [],
        'transmission': []
    }
    
    print("\nOptimizing for uniform response across multiple angles...")
    n_iterations = 50
    
    for iter in range(n_iterations):
        optimizer.zero_grad()
        
        # Solve for all sources
        results = solver.solve(sources)
        
        # Objective: Maximize average transmission while minimizing variance
        trans_values = results.transmission[:, 0]
        avg_trans = trans_values.mean()
        var_trans = trans_values.var()
        
        # Combined loss
        loss = -avg_trans + 0.5 * var_trans  # Maximize avg, minimize variance
        
        loss.backward()
        optimizer.step()
        
        # Clamp mask values to [0, 1]
        with torch.no_grad():
            mask.data = torch.clamp(mask.data, 0, 1)
        
        # Update structure
        solver.update_er_with_mask(mask=mask, layer_index=1)
        
        # Store history
        history['loss'].append(loss.item())
        history['transmission'].append(trans_values.detach().numpy())
        
        if iter % 10 == 0:
            print(f"Iteration {iter}: Loss = {loss.item():.4f}, "
                  f"Avg T = {avg_trans.item():.4f}, "
                  f"Std T = {var_trans.sqrt().item():.4f}")
    
    # Plot optimization results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss history
    ax = axes[0, 0]
    ax.plot(history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Optimization Loss')
    ax.grid(True, alpha=0.3)
    
    # Transmission evolution
    ax = axes[0, 1]
    trans_history = np.array(history['transmission'])
    for i, angle in enumerate(target_angles):
        ax.plot(trans_history[:, i], linewidth=2, 
                label=f'{angle*180/np.pi:.0f}°')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Transmission')
    ax.set_title('Transmission Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Initial pattern
    ax = axes[1, 0]
    initial_mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_init)
    im = ax.imshow(initial_mask.numpy(), cmap='binary')
    ax.set_title('Initial Pattern')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Optimized pattern
    ax = axes[1, 1]
    im = ax.imshow(mask.detach().numpy(), cmap='binary')
    ax.set_title('Optimized Pattern')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('example_optimization_batched.png', dpi=150)
    plt.show()
    
    # Verify final performance with more angles
    print("\nFinal performance verification:")
    test_angles = np.linspace(0, 45, 10) * deg
    test_sources = [solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0) 
                    for theta in test_angles]
    
    final_results = solver.solve(test_sources)
    final_trans = final_results.transmission[:, 0].detach().numpy()
    
    print(f"Average transmission: {final_trans.mean():.4f}")
    print(f"Standard deviation: {final_trans.std():.4f}")
    print(f"Min/Max transmission: {final_trans.min():.4f} / {final_trans.max():.4f}")
    
    return mask, final_results


def example_wavelength_and_angle_sweep():
    """Demonstrate combined wavelength and angle parameter sweep."""
    print("\n=== Example 4: Combined Wavelength and Angle Sweep ===")
    
    # Create solver with multiple wavelengths
    wavelengths = np.linspace(1.5, 1.6, 5)  # 5 wavelengths
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=wavelengths,
        rdim=[256, 256],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)
    sio2 = create_material(name="SiO2", permittivity=2.25)
    solver.add_materials([si, sio2])
    
    # Create thin film stack
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.2, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    
    # Create sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 7) * deg
    
    sources = []
    for theta in angles:
        source = solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0)
        sources.append(source)
    
    # Solve
    results = solver.solve(sources)
    
    # Extract 2D data: (n_angles, n_wavelengths)
    transmission = results.transmission.detach().numpy()
    reflection = results.reflection.detach().numpy()
    
    # Create 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Transmission map
    im1 = ax1.imshow(transmission, aspect='auto', origin='lower', 
                     extent=[wavelengths[0], wavelengths[-1], 0, 60],
                     cmap='viridis', vmin=0, vmax=1)
    ax1.set_xlabel('Wavelength (μm)')
    ax1.set_ylabel('Incident Angle (degrees)')
    ax1.set_title('Transmission Map')
    plt.colorbar(im1, ax=ax1)
    
    # Reflection map
    im2 = ax2.imshow(reflection, aspect='auto', origin='lower',
                     extent=[wavelengths[0], wavelengths[-1], 0, 60],
                     cmap='plasma', vmin=0, vmax=1)
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('Incident Angle (degrees)')
    ax2.set_title('Reflection Map')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('example_wavelength_angle_sweep.png', dpi=150)
    plt.show()
    
    # Find optimal operating point
    best_idx = results.find_optimal_source('max_transmission')
    best_angle = angles[best_idx] * 180 / np.pi
    
    # Find best wavelength for this angle
    best_wl_idx = np.argmax(transmission[best_idx, :])
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
    
    # Run examples
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