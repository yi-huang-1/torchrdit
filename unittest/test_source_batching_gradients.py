"""
Gradient preservation tests for source batching.

Focus on core, lightweight checks that genuinely verify gradient behavior:
- Sequential vs batched gradient equivalence
- Gradient accumulation across micro-batches
- Multi-wavelength gradient flow
- Numerical gradient validation via finite differences
"""
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator


class TestSourceBatchingGradients:
    """Test gradient preservation in source batching."""
    
    def create_optimization_solver(self):
        """Create a small solver for gradient tests (kept light)."""
        return create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[64, 64],
            kdim=[3, 3],
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
            rdim=[64, 64],
            kdim=[3, 3],
            device='cpu'
        )
        solver2 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[64, 64],
            kdim=[3, 3],
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
    
    # Removed algorithm-specific gradient smoke tests (covered elsewhere).
    
    # Removed full optimization loop (integration-level and heavy, plus stateful layering).
    
    def test_multi_wavelength_gradient(self):
        """Test gradient preservation with multiple wavelengths."""
        # Create multi-wavelength solver
        wavelengths = np.array([1.45, 1.50, 1.55, 1.60])
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=wavelengths,
            rdim=[48, 48],  # Smaller for speed
            kdim=[3, 3],
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
            rdim=[64, 64],
            kdim=[3, 3],
            device='cpu'
        )
        solver2 = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[64, 64],
            kdim=[3, 3],
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
    
    # Removed complex optimization scenario (integration-level and heavy).
    
    def test_gradient_numerical_validation(self):
        """Validate gradients using finite differences."""
        # Create fresh solver
        _ = self.create_optimization_solver()
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
            rdim=[64, 64],
            kdim=[3, 3],
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
                rdim=[64, 64],
                kdim=[3, 3],
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
                rdim=[64, 64],
                kdim=[3, 3],
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
