"""
Source Batching Advanced Examples for TorchRDIT v0.1.22

This script demonstrates advanced optimization techniques using source batching,
including multi-angle optimization, robust design, and gradient-based inverse design.

Key features demonstrated:
- Optimization with multiple incident conditions
- Robust design across angle/polarization variations
- Memory-efficient chunking for large sweeps
- Gradient preservation in batched mode
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator
import time
from pathlib import Path

def example_multi_angle_optimization():
    """Optimize a structure for uniform response across multiple angles."""
    print("\n=== Example 1: Multi-Angle Optimization ===")
    
    # Create optimizer solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Add layer
    solver.add_layer(material_name="Si", thickness=0.6, is_homogeneous=False)
    
    # Parameterized structure - optimizable radius
    radius_param = torch.tensor(0.25, requires_grad=True, device=solver.device)
    optimizer = torch.optim.Adam([radius_param], lr=0.02)
    
    # Target multiple angles for robust design
    deg = np.pi / 180
    target_angles = np.array([0, 15, 30]) * deg
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in target_angles
    ]
    
    # Optimization loop
    n_epochs = 50
    history = {'loss': [], 'radius': [], 'transmissions': []}
    
    print("Starting optimization...")
    print("Target: Maximize average transmission across angles while minimizing variance")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Update structure with current radius
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_param)
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Batch solve for all angles
        results = solver.solve(sources)
        
        # Loss: maximize average transmission across all angles
        trans_values = results.transmission[:, 0]
        avg_transmission = trans_values.mean()
        variance = trans_values.var()
        
        # Combined loss: maximize average, minimize variance
        loss = -avg_transmission + 0.5 * variance
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Clamp radius to valid range
        with torch.no_grad():
            radius_param.clamp_(0.1, 0.45)
        
        # Record history
        history['loss'].append(loss.item())
        history['radius'].append(radius_param.item())
        history['transmissions'].append(trans_values.detach().cpu().numpy())
        
        if epoch % 10 == 0:
            trans_np = trans_values.detach().cpu().numpy()
            print(f"Epoch {epoch:3d}: Loss = {loss.item():7.4f}, "
                  f"Radius = {radius_param.item():.4f}, "
                  f"T = [{trans_np[0]:.3f}, {trans_np[1]:.3f}, {trans_np[2]:.3f}]")
    
    print("\nOptimization complete!")
    print(f"Final radius: {radius_param.item():.4f}")
    print(f"Final average transmission: {-history['loss'][-1]:.4f}")
    
    # Plot optimization history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Loss history
    ax1.plot(-np.array(history['loss']), 'b-', linewidth=2)
    ax1.set_ylabel('Average Transmission')
    ax1.set_title('Multi-Angle Optimization Progress')
    ax1.grid(True, alpha=0.3)
    
    # Transmission evolution
    trans_history = np.array(history['transmissions'])
    for i, angle in enumerate(target_angles):
        ax2.plot(trans_history[:, i], linewidth=2, 
                 label=f'θ = {angle*180/np.pi:.0f}°')
    ax2.plot(-np.array(history['loss']), 'k--', linewidth=2, 
             label='Average', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Transmission')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'multi_angle_optimization.png', dpi=150)
    plt.close(fig)


def example_robust_design():
    """Design a structure robust to both angle and polarization variations."""
    print("\n=== Example 2: Robust Design for Angle and Polarization ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    sio2 = create_material(name="SiO2", permittivity=2.25)
    solver.add_materials([si, sio2])
    
    # Add optimizable layer
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create parametric structure - two circles
    radius1 = torch.tensor(0.15, requires_grad=True, device=solver.device)
    radius2 = torch.tensor(0.15, requires_grad=True, device=solver.device)
    sep = torch.tensor(0.3, requires_grad=True, device=solver.device)
    
    optimizer = torch.optim.Adam([radius1, radius2, sep], lr=0.01)
    
    # Create sources for robustness - angles and polarizations
    deg = np.pi / 180
    sources = []
    
    # Multiple angles with TE
    for angle in np.array([0, 20, 40]) * deg:
        sources.append(solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0))
    
    # Multiple polarizations at 30°
    for pte, ptm in [(1.0, 0.0), (0.0, 1.0), (0.707, 0.707)]:
        sources.append(solver.add_source(theta=30*deg, phi=0, pte=pte, ptm=ptm))
    
    print(f"Optimizing for {len(sources)} different incident conditions")
    
    # Optimization
    n_epochs = 40
    history = {'loss': [], 'params': [], 'mean_trans': [], 'std_trans': []}
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Create two-circle pattern
        shape_gen = ShapeGenerator.from_solver(solver)
        mask1 = shape_gen.generate_circle_mask(center=(-sep/2, 0), radius=radius1)
        mask2 = shape_gen.generate_circle_mask(center=(sep/2, 0), radius=radius2)
        mask = torch.maximum(mask1, mask2)
        
        solver.update_er_with_mask(mask=mask, layer_index=0, bg_material="SiO2")
        
        # Batch solve
        results = solver.solve(sources)
        
        # Robust objective: high mean, low variance
        trans = results.transmission[:, 0]
        mean_trans = trans.mean()
        std_trans = trans.std()
        
        # Penalize large structures
        size_penalty = 0.1 * (radius1**2 + radius2**2)
        
        loss = -mean_trans + 2.0 * std_trans + size_penalty
        
        loss.backward()
        optimizer.step()
        
        # Constraints
        with torch.no_grad():
            radius1.clamp_(0.05, 0.25)
            radius2.clamp_(0.05, 0.25)
            sep.clamp_(0.1, 0.6)
        
        # Record
        history['loss'].append(loss.item())
        history['params'].append([radius1.item(), radius2.item(), sep.item()])
        history['mean_trans'].append(mean_trans.item())
        history['std_trans'].append(std_trans.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Mean T = {mean_trans.item():.4f}, "
                  f"Std T = {std_trans.item():.4f}, "
                  f"r1={radius1.item():.3f}, r2={radius2.item():.3f}, sep={sep.item():.3f}")
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss evolution
    ax1.plot(history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Optimization Loss')
    ax1.grid(True, alpha=0.3)
    
    # Mean vs Std transmission
    ax2.plot(history['mean_trans'], 'g-', linewidth=2, label='Mean T')
    ax2.plot(history['std_trans'], 'r-', linewidth=2, label='Std T')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Transmission')
    ax2.set_title('Transmission Statistics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Parameter evolution
    params = np.array(history['params'])
    ax3.plot(params[:, 0], label='Radius 1', linewidth=2)
    ax3.plot(params[:, 1], label='Radius 2', linewidth=2)
    ax3.plot(params[:, 2], label='Separation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Structure Parameters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final structure
    ax4.imshow(mask.detach().cpu().numpy(), cmap='hot', origin='lower')
    ax4.set_title('Final Optimized Structure')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'robust_design_optimization.png', dpi=150)
    plt.close(fig)
    
    print("\nFinal design:")
    print(f"  Mean transmission: {history['mean_trans'][-1]:.4f}")
    print(f"  Std transmission: {history['std_trans'][-1]:.4f}")
    print(f"  Parameters: r1={radius1.item():.3f}, r2={radius2.item():.3f}, sep={sep.item():.3f}")


def example_large_sweep_chunking():
    """Demonstrate memory-efficient processing of large parameter sweeps."""
    print("\n=== Example 3: Memory-Efficient Large Sweep Processing ===")
    
    def process_large_sweep(solver, angles, chunk_size=100):
        """Process large angle sweep in memory-efficient chunks."""
        all_transmissions = []
        all_reflections = []
        
        n_chunks = (len(angles) + chunk_size - 1) // chunk_size
        print(f"Processing {len(angles)} angles in {n_chunks} chunks...")
        
        for i in range(0, len(angles), chunk_size):
            # Get chunk of angles
            chunk_angles = angles[i:i+chunk_size]
            
            # Create sources for chunk
            sources = [
                solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
                for angle in chunk_angles
            ]
            
            # Process chunk
            results = solver.solve(sources)
            all_transmissions.extend(results.transmission[:, 0].detach().cpu().numpy())
            all_reflections.extend(results.reflection[:, 0].detach().cpu().numpy())
            
            # Clear GPU memory if needed
            if str(solver.device).startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Progress
            print(f"  Processed chunk {i//chunk_size + 1}/{n_chunks}")
        
        return np.array(all_transmissions), np.array(all_reflections)
    
    # Create solver with simple structure
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5]
    )
    
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_rectangle_mask(width=0.4, height=0.2, angle=45)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Process 1000 angles
    deg = np.pi / 180
    many_angles = np.linspace(0, 89, 1000) * deg
    
    start = time.time()
    trans_large, refl_large = process_large_sweep(solver, many_angles, chunk_size=100)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results shape: {trans_large.shape}")
    print(f"Average transmission: {trans_large.mean():.4f}")
    print("Peak memory usage avoided by chunking!")
    
    # Plot subset of results
    stride = 10  # Plot every 10th point
    angles_deg = many_angles[::stride] * 180 / np.pi
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, trans_large[::stride], 'b-', linewidth=1, label='Transmission')
    plt.plot(angles_deg, refl_large[::stride], 'r-', linewidth=1, label='Reflection')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Efficiency')
    plt.title('Large Angle Sweep (1000 angles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 89)
    plt.ylim(0, 1.05)
    plt.savefig(Path(__file__).parent / 'large_sweep_chunking.png', dpi=150)
    plt.close(plt.gcf())


def example_gradient_validation():
    """Validate gradient consistency between sequential and batched solving."""
    print("\n=== Example 4: Gradient Validation for Batched Solving ===")
    
    # Create two identical solvers
    def create_test_solver():
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        return solver
    
    solver_seq = create_test_solver()
    solver_batch = create_test_solver()
    
    # Create identical optimizable radius parameters
    radius_seq = torch.tensor(0.3, requires_grad=True, device=solver_seq.device)
    radius_batch = torch.tensor(0.3, requires_grad=True, device=solver_batch.device)
    
    # Create sources
    deg = np.pi / 180
    angles = np.array([0, 30, 60]) * deg
    sources = [
        solver_seq.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in angles
    ]
    
    # Sequential solving and gradient
    print("Computing sequential gradients...")
    trans_seq = []
    for source in sources:
        # Update mask for each source
        shape_gen = ShapeGenerator.from_solver(solver_seq)
        mask_seq = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_seq)
        solver_seq.update_er_with_mask(mask=mask_seq, layer_index=0)
        
        result = solver_seq.solve(source)
        trans_seq.append(result.transmission[0])
    
    loss_seq = -torch.stack(trans_seq).mean()
    loss_seq.backward()
    grad_seq = radius_seq.grad.clone()
    
    # Batched solving and gradient
    print("Computing batched gradients...")
    # Update mask once for batched solve
    shape_gen = ShapeGenerator.from_solver(solver_batch)
    mask_batch = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_batch)
    solver_batch.update_er_with_mask(mask=mask_batch, layer_index=0)
    
    results_batch = solver_batch.solve(sources)
    loss_batch = -results_batch.transmission[:, 0].mean()
    loss_batch.backward()
    grad_batch = radius_batch.grad
    
    # Compare gradients
    grad_diff = torch.abs(grad_seq - grad_batch).item()
    grad_rel_diff = grad_diff / (torch.abs(grad_seq).item() + 1e-8)
    
    print("\nGradient comparison:")
    print(f"  Sequential gradient: {grad_seq.item():.6f}")
    print(f"  Batched gradient: {grad_batch.item():.6f}")
    print(f"  Absolute difference: {grad_diff:.2e}")
    print(f"  Relative difference: {grad_rel_diff:.2e}")
    print(f"  Gradients match: {'YES' if grad_rel_diff < 1e-5 else 'NO'}")
    
    # Visualize radius optimization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Show the gradient values as a bar chart
    gradients = [grad_seq.item(), grad_batch.item()]
    labels = ['Sequential', 'Batched']
    colors = ['blue', 'orange']
    
    bars = ax.bar(labels, gradients, color=colors)
    ax.set_ylabel('Gradient Value')
    ax.set_title('Gradient Comparison: Sequential vs Batched')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, grad in zip(bars, gradients):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{grad:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'gradient_validation.png', dpi=150)
    plt.close(fig)


def main():
    """Run all advanced examples."""
    print("TorchRDIT Source Batching - Advanced Examples")
    print("=============================================")
    
    # Run examples
    example_multi_angle_optimization()
    example_robust_design()
    example_large_sweep_chunking()
    example_gradient_validation()
    
    print("\nAll advanced examples completed!")
    print("Generated plots:")
    print("  - multi_angle_optimization.png")
    print("  - robust_design_optimization.png")
    print("  - large_sweep_chunking.png")
    print("  - gradient_validation.png")


if __name__ == "__main__":
    main()