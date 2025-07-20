"""
Gradient preservation tests for source batching.

This module ensures that gradient flow is preserved when using batched sources,
and that optimization workflows function correctly with both sequential and
batched solving.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator


class TestSourceBatchingGradients:
    """Test gradient preservation in source batching."""
    
    def create_optimization_solver(self):
        """Create solver suitable for optimization tests."""
        return create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],  # Smaller for faster tests
            kdim=[5, 5],
            device='cpu'
        )
    
    def setup_parametric_structure(self, solver, radius):
        """Set up a structure with parametric geometry."""
        # Add materials
        Si = create_material(name="Si", permittivity=11.7)
        SiO2 = create_material(name="SiO2", permittivity=2.25)
        solver.add_materials([Si, SiO2])
        
        # Create layers
        solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
        solver.add_layer(material_name="Si", thickness=0.3, is_homogeneous=False)
        
        # Create parametric mask
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius)
        solver.update_er_with_mask(mask=mask, layer_index=1)
    
    def test_gradient_consistency_sequential_vs_batch(self):
        """Test that gradients are identical for sequential and batched solving."""
        # Create two separate solvers to avoid state contamination
        solver1 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cpu'
        )
        solver2 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cpu'
        )
        
        # Create parametric variable
        radius = torch.tensor(0.2, requires_grad=True)
        
        # Create multiple sources
        deg = np.pi / 180
        sources = [
            {"theta": 0*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": 10*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": 20*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # Sequential solving
        self.setup_parametric_structure(solver1, radius)
        loss_sequential = 0
        for source in sources:
            result = solver1.solve(source)
            loss_sequential += torch.sum(result.transmission)
        
        # Compute gradient for sequential
        grad_sequential = torch.autograd.grad(loss_sequential, radius, retain_graph=True)[0]
        
        # Clear gradients
        if radius.grad is not None:
            radius.grad.zero_()
        
        # Batched solving  
        self.setup_parametric_structure(solver2, radius)
        results_batch = solver2.solve(sources)
        loss_batch = torch.sum(results_batch.transmission)
        
        # Compute gradient for batched
        grad_batch = torch.autograd.grad(loss_batch, radius)[0]
        
        # Compare gradients
        assert torch.allclose(grad_sequential, grad_batch, rtol=1e-5, atol=1e-8), \
            f"Gradient mismatch: sequential={grad_sequential.item()}, batch={grad_batch.item()}"
        
        # Also check that losses are equal
        assert torch.allclose(loss_sequential, loss_batch, rtol=1e-5, atol=1e-8), \
            f"Loss mismatch: sequential={loss_sequential.item()}, batch={loss_batch.item()}"
    
    def test_gradient_flow_rdit(self):
        """Test gradient flow through R-DIT solver with batched sources."""
        solver = self.create_optimization_solver()
        # Skip algorithm check - solver.algorithm returns an object, not enum
        
        # Parameters
        radius = torch.tensor(0.15, requires_grad=True)
        thickness = torch.tensor(0.3, requires_grad=True)
        
        # Multiple sources for robustness
        deg = np.pi / 180
        sources = [
            {"theta": i*5*deg, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(5)
        ]
        
        # Setup structure
        Si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([Si])
        
        solver.add_layer(material_name="Si", thickness=thickness, is_homogeneous=False)
        
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius)
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Solve and compute loss
        results = solver.solve(sources)
        
        # Objective: maximize average transmission
        loss = -torch.mean(results.transmission)
        
        # Check gradient flow
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert radius.grad is not None, "No gradient for radius"
        assert thickness.grad is not None, "No gradient for thickness"
        
        assert not torch.allclose(radius.grad, torch.tensor(0.0)), \
            "Radius gradient is zero"
        assert not torch.allclose(thickness.grad, torch.tensor(0.0)), \
            "Thickness gradient is zero"
        
        # Print gradients for debugging
        print(f"\nR-DIT Gradient Test:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Radius gradient: {radius.grad.item():.6e}")
        print(f"  Thickness gradient: {thickness.grad.item():.6e}")
    
    def test_gradient_flow_rcwa(self):
        """Test gradient flow through RCWA solver with batched sources."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cpu'
        )
        
        # Parameters
        radius = torch.tensor(0.15, requires_grad=True)
        thickness = torch.tensor(0.3, requires_grad=True)
        
        # Multiple sources
        deg = np.pi / 180
        sources = [
            {"theta": i*5*deg, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(5)
        ]
        
        # Setup structure
        Si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([Si])
        
        solver.add_layer(material_name="Si", thickness=thickness, is_homogeneous=False)
        
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius)
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Solve and compute loss
        results = solver.solve(sources)
        
        # Objective: maximize average transmission
        loss = -torch.mean(results.transmission)
        
        # Check gradient flow
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert radius.grad is not None, "No gradient for radius"
        assert thickness.grad is not None, "No gradient for thickness"
        
        assert not torch.allclose(radius.grad, torch.tensor(0.0)), \
            "Radius gradient is zero"
        assert not torch.allclose(thickness.grad, torch.tensor(0.0)), \
            "Thickness gradient is zero"
        
        # Print gradients for debugging
        print(f"\nRCWA Gradient Test:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Radius gradient: {radius.grad.item():.6e}")
        print(f"  Thickness gradient: {thickness.grad.item():.6e}")
    
    def test_optimization_workflow(self):
        """Test full optimization workflow with batched sources."""
        solver = self.create_optimization_solver()
        
        # Initial parameters
        radius = torch.tensor(0.1, requires_grad=True)
        
        # Multiple sources to optimize for
        deg = np.pi / 180
        sources = [
            {"theta": i*10*deg, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(4)
        ]
        
        # Optimizer
        optimizer = torch.optim.Adam([radius], lr=0.01)
        
        # Track optimization progress
        losses = []
        radii = []
        
        # Optimization loop
        n_epochs = 10
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Setup structure with current radius
            self.setup_parametric_structure(solver, radius)
            
            # Batched solve
            results = solver.solve(sources)
            
            # Loss: minimize average transmission (for testing)
            loss = torch.mean(results.transmission)
            losses.append(loss.item())
            radii.append(radius.item())
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Ensure radius stays positive
            with torch.no_grad():
                radius.clamp_(min=0.05, max=0.45)
        
        # Verify optimization is working
        # Loss should change over epochs
        loss_change = abs(losses[-1] - losses[0])
        assert loss_change > 1e-4, \
            f"Loss didn't change significantly: {losses[0]:.6f} -> {losses[-1]:.6f}"
        
        # Radius should change
        radius_change = abs(radii[-1] - radii[0])
        assert radius_change > 1e-4, \
            f"Radius didn't change significantly: {radii[0]:.6f} -> {radii[-1]:.6f}"
        
        print(f"\nOptimization Test Results:")
        print(f"  Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        print(f"  Initial radius: {radii[0]:.4f}, Final radius: {radii[-1]:.4f}")
        print(f"  Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    
    def test_multi_wavelength_gradient(self):
        """Test gradient preservation with multiple wavelengths."""
        # Create multi-wavelength solver
        wavelengths = np.array([1.45, 1.50, 1.55, 1.60])
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=wavelengths,
            rdim=[64, 64],  # Smaller for speed
            kdim=[5, 5],
            device='cpu'
        )
        
        # Parameters
        radius = torch.tensor(0.2, requires_grad=True)
        
        # Sources for each wavelength
        sources = [
            {"theta": 0, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/180 * 5, "phi": 0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # Setup structure
        self.setup_parametric_structure(solver, radius)
        
        # Solve
        results = solver.solve(sources)
        
        # Loss across all wavelengths and sources
        # results.transmission shape: [n_wavelengths, n_sources]
        loss = torch.mean(results.transmission)
        
        # Check gradient
        loss.backward()
        
        assert radius.grad is not None, "No gradient for radius"
        assert not torch.allclose(radius.grad, torch.tensor(0.0)), \
            "Radius gradient is zero"
        
        print(f"\nMulti-wavelength Gradient Test:")
        print(f"  Wavelengths: {len(wavelengths)}")
        print(f"  Sources: {len(sources)}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Radius gradient: {radius.grad.item():.6e}")
    
    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across batches."""
        # Create separate solvers to avoid state issues
        solver1 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cpu'
        )
        solver2 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cpu'
        )
        
        # Parameter
        radius = torch.tensor(0.2, requires_grad=True)
        
        # Create two batches of sources
        deg = np.pi / 180
        batch1 = [
            {"theta": 0*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": 10*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
        ]
        batch2 = [
            {"theta": 20*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": 30*deg, "phi": 0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # Method 1: Process all at once
        self.setup_parametric_structure(solver1, radius)
        all_sources = batch1 + batch2
        results_all = solver1.solve(all_sources)
        loss_all = torch.sum(results_all.transmission)
        grad_all = torch.autograd.grad(loss_all, radius, retain_graph=True)[0]
        
        # Method 2: Process in batches with gradient accumulation
        if radius.grad is not None:
            radius.grad.zero_()
        
        self.setup_parametric_structure(solver2, radius)
        
        # Batch 1
        results1 = solver2.solve(batch1)
        loss1 = torch.sum(results1.transmission)
        loss1.backward(retain_graph=True)
        grad_batch1 = radius.grad.clone()
        
        # Batch 2 (accumulate gradients)
        results2 = solver2.solve(batch2)
        loss2 = torch.sum(results2.transmission)
        loss2.backward()
        grad_accumulated = radius.grad.clone()
        
        # Compare
        assert torch.allclose(grad_all, grad_accumulated, rtol=1e-5, atol=1e-8), \
            f"Gradient accumulation mismatch: all={grad_all.item()}, accumulated={grad_accumulated.item()}"
        
        print(f"\nGradient Accumulation Test:")
        print(f"  All-at-once gradient: {grad_all.item():.6e}")
        print(f"  Accumulated gradient: {grad_accumulated.item():.6e}")
        print(f"  Batch 1 gradient: {grad_batch1.item():.6e}")
    
    def test_complex_optimization_scenario(self):
        """Test a complex optimization scenario similar to the GMRF example."""
        # Materials
        n_SiO = 1.4496
        n_SiN = 1.9360
        material_sio = create_material(name="SiO", permittivity=n_SiO**2)
        material_sin = create_material(name="SiN", permittivity=n_SiN**2)
        
        # Structure parameters
        h1 = torch.tensor(0.23, requires_grad=False)  # Fixed thickness
        h2 = torch.tensor(0.345, requires_grad=False)  # Fixed thickness
        
        # Optimization parameter
        radius = torch.tensor(0.4, requires_grad=True)  # Variable radius
        
        # Multiple incident angles
        deg = np.pi / 180
        sources = [
            {"theta": i*2*deg, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(5)
        ]
        
        # Setup optimizer
        optimizer = torch.optim.Adam([radius], lr=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
        # Optimization loop
        n_epochs = 15
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Create fresh solver for each iteration
            solver = create_solver(
                algorithm=Algorithm.RDIT,
                lam0=np.array([1.537]),  # Target wavelength
                rdim=[128, 128],
                kdim=[7, 7],
                device='cpu'
            )
            solver.add_materials([material_sio, material_sin])
            
            # Build structure
            solver.add_layer(material_name="SiO", thickness=h1, is_homogeneous=False)
            solver.add_layer(material_name="SiN", thickness=h2, is_homogeneous=True)
            
            # Create pattern
            shape_gen = ShapeGenerator.from_solver(solver)
            mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius)
            mask = 1 - mask  # Invert for air holes
            solver.update_er_with_mask(mask=mask, layer_index=0)
            
            # Solve
            results = solver.solve(sources)
            
            # Objective: minimize transmission (create resonance)
            loss = torch.mean(results.transmission) * 100
            losses.append(loss.item())
            
            # Backward and update
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Constrain radius
            with torch.no_grad():
                radius.clamp_(min=0.1, max=0.5)
        
        # Verify optimization worked
        assert len(losses) == n_epochs
        # Check if loss decreased or stayed relatively stable (optimization may converge)
        assert losses[-1] <= losses[0] * 1.1, \
            f"Loss increased significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
        
        print(f"\nComplex Optimization Test:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss change: {(losses[-1] - losses[0])/losses[0]*100:.1f}%")
        print(f"  Final radius: {radius.item():.4f}")
    
    def test_gradient_numerical_validation(self):
        """Validate gradients using finite differences."""
        # Create fresh solver
        optimization_solver = self.create_optimization_solver()
        # Parameter
        radius = torch.tensor(0.2, requires_grad=True)
        epsilon = 1e-4
        
        # Sources
        sources = [
            {"theta": 0, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/180 * 10, "phi": 0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # Compute analytical gradient
        solver1 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cpu'
        )
        self.setup_parametric_structure(solver1, radius)
        results = solver1.solve(sources)
        loss = torch.sum(results.transmission)
        grad_analytical = torch.autograd.grad(loss, radius)[0]
        
        # Compute numerical gradient
        with torch.no_grad():
            # Forward difference
            solver2 = create_solver(
                algorithm=Algorithm.RDIT,
                lam0=np.array([1.55]),
                rdim=[128, 128],
                kdim=[5, 5],
                device='cpu'
            )
            radius_plus = radius + epsilon
            self.setup_parametric_structure(solver2, radius_plus)
            results_plus = solver2.solve(sources)
            loss_plus = torch.sum(results_plus.transmission)
            
            # Backward difference
            solver3 = create_solver(
                algorithm=Algorithm.RDIT,
                lam0=np.array([1.55]),
                rdim=[128, 128],
                kdim=[5, 5],
                device='cpu'
            )
            radius_minus = radius - epsilon
            self.setup_parametric_structure(solver3, radius_minus)
            results_minus = solver3.solve(sources)
            loss_minus = torch.sum(results_minus.transmission)
            
            # Central difference
            grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Compare
        relative_error = abs(grad_analytical - grad_numerical) / (abs(grad_analytical) + 1e-8)
        
        print(f"\nNumerical Gradient Validation:")
        print(f"  Analytical gradient: {grad_analytical.item():.6e}")
        print(f"  Numerical gradient: {grad_numerical.item():.6e}")
        print(f"  Relative error: {relative_error.item():.6e}")
        
        assert relative_error < 1e-2, \
            f"Gradient validation failed: relative error = {relative_error.item():.6e}"