"""
Test suite for the unified tensorized _process_layer method.

This test suite ensures mathematical equivalence between single and batched
source processing using the unified _process_layer implementation.
Following TDD principles, these tests verify the unified method behavior.
"""

import torch
import numpy as np
import pytest
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator


class TestProcessLayerTensorization:
    """Test suite ensuring mathematical equivalence between _process_layer implementations."""

    @pytest.fixture
    def setup_solver(self):
        """Create a solver with both homogeneous and non-homogeneous layers."""
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55, 1.31]),  # Multiple wavelengths
            rdim=[512, 512],
            kdim=[5, 5],
            device="cpu",  # Use CPU for consistent testing
        )

        # Add materials
        si = create_material(name="Si", permittivity=12.25)
        sio2 = create_material(name="SiO2", permittivity=2.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, sio2, air])

        # Add layers (mix of homogeneous and patterned)
        solver.add_layer(material_name="Si", thickness=0.5)  # Homogeneous layer
        
        # Add patterned layer (non-homogeneous)
        solver.add_layer(material_name="air", thickness=0.3, is_homogeneous=False)
        
        # Create pattern mask for the non-homogeneous layer
        shape_gen = ShapeGenerator.from_solver(solver)
        # Create a rectangular pattern
        mask = shape_gen.generate_rectangle_mask(
            center=(0, 0),
            x_size=0.4,
            y_size=0.3
        )
        # Update the layer with the pattern (SiO2 in air background)
        solver.update_er_with_mask(mask=mask, layer_index=1)

        return solver

    def test_single_source_equivalence(self, setup_solver):
        """Verify _process_layer handles single and batched sources identically."""
        solver = setup_solver
        solver.debug_tensorization = True  # Enable debug logging
        
        # Set up source and pre-solve
        source = solver.add_source(theta=30 * np.pi / 180, phi=45 * np.pi / 180, pte=1.0, ptm=0.0)
        solver.src = source
        solver._pre_solve()
        
        # Initialize k-vectors
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        
        # Get common matrices
        matrices_single = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        
        # Test both homogeneous and non-homogeneous layers
        for n_layer in range(len(solver.layer_manager.layers)):
            print(f"\nTesting layer {n_layer} (homogeneous: {solver.layer_manager.layers[n_layer].is_homogeneous})")
            
            # Process with single source method
            smat_single = solver._process_layer(n_layer, matrices_single)
            
            # Process with batched source (n_sources=1)
            solver._pre_solve([source])
            kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
            matrices_batched = solver._setup_common_matrices(kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b)
            smat_batched = solver._process_layer(n_layer, matrices_batched)
            
            # Compare S-matrices
            for key in ["S11", "S12", "S21", "S22"]:
                single_val = smat_single[key]
                batched_val = smat_batched[key]
                
                # For batched with n_sources=1, extract the single source
                if batched_val.dim() > single_val.dim():
                    batched_val = batched_val[0]
                
                assert single_val.shape == batched_val.shape, (
                    f"Shape mismatch for {key}: single {single_val.shape} vs batched {batched_val.shape}"
                )
                
                max_diff = torch.abs(single_val - batched_val).max().item()
                assert torch.allclose(single_val, batched_val, atol=1e-10, rtol=1e-6), (
                    f"{key} mismatch: max diff = {max_diff}"
                )
                
                print(f"  ✓ {key} matches (max diff: {max_diff:.2e})")

    def test_multiple_sources_sequential_match(self, setup_solver):
        """Verify batched processing matches sequential processing."""
        solver = setup_solver
        
        # Create multiple sources with different angles
        angles = [0, 30, 45, 60]
        sources = [
            solver.add_source(theta=angle * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Process each layer
        for n_layer in range(len(solver.layer_manager.layers)):
            print(f"\nTesting layer {n_layer} with {len(sources)} sources")
            
            # Process sequentially
            sequential_smats = {"S11": [], "S12": [], "S21": [], "S22": []}
            
            for source in sources:
                solver.src = source
                solver._pre_solve()
                kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
                matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
                
                smat = solver._process_layer(n_layer, matrices)
                for key in sequential_smats:
                    sequential_smats[key].append(smat[key].clone())
            
            # Stack sequential results
            for key in sequential_smats:
                sequential_smats[key] = torch.stack(sequential_smats[key], dim=0)
            
            # Process in batch
            solver._pre_solve(sources)
            kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
            matrices_batched = solver._setup_common_matrices(kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b)
            smat_batched = solver._process_layer(n_layer, matrices_batched)
            
            # Compare each S-matrix component
            for key in ["S11", "S12", "S21", "S22"]:
                seq_val = sequential_smats[key]
                batch_val = smat_batched[key]
                
                assert seq_val.shape == batch_val.shape, (
                    f"Shape mismatch for {key}: sequential {seq_val.shape} vs batched {batch_val.shape}"
                )
                
                max_diff = torch.abs(seq_val - batch_val).max().item()
                assert torch.allclose(seq_val, batch_val, atol=1e-10, rtol=1e-6), (
                    f"{key} mismatch: max diff = {max_diff}"
                )
                
                print(f"  ✓ {key} matches (max diff: {max_diff:.2e})")

    def test_single_vs_batched_solve_match(self, setup_solver):
        """Full-pipeline check: single-source solve equals indexed batched solve."""
        solver = setup_solver

        # Create sources with different angles
        angles = [0, 30, 45]
        sources = [solver.add_source(theta=a * np.pi / 180, phi=0, pte=1.0, ptm=0.0) for a in angles]

        # Compare single-source solve to corresponding entry from batched solve
        target_idx = 1  # 30 degrees
        result_single = solver.solve(sources[target_idx])
        results_batched = solver.solve(sources)

        result_from_batch = results_batched[target_idx]

        assert torch.allclose(result_single.transmission, result_from_batch.transmission, atol=1e-10, rtol=1e-6)
        assert torch.allclose(result_single.reflection, result_from_batch.reflection, atol=1e-10, rtol=1e-6)

    def test_edge_cases(self, setup_solver):
        """Test edge cases: grazing incidence, etc."""
        solver = setup_solver
        
        # Test case 1: Grazing incidence
        source_grazing = solver.add_source(theta=85 * np.pi / 180, phi=0, pte=0.5, ptm=0.5)
        solver.src = source_grazing
        solver._pre_solve()
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        
        # Process homogeneous layer at grazing angle
        n_layer = 0
        smat_grazing = solver._process_layer(n_layer, matrices)
        
        # Verify no NaN or Inf values
        for key, val in smat_grazing.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key} at grazing incidence"
        
        print("  ✓ Grazing incidence handled correctly")

    def test_gradient_consistency_wrt_mask(self):
        """Gradients w.r.t. a learnable mask should match: single vs 1-source batch.

        Uses a small problem size to keep backward pass lightweight.
        """
        torch.manual_seed(0)

        # Build a compact solver to keep grad test fast
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[32, 32],
            kdim=[3, 3],
            device="cpu",
        )

        # Materials
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])

        # Add a single patterned layer with foreground Si in air background
        solver.add_layer(material_name="Si", thickness=0.2, is_homogeneous=False)

        # Learnable mask parameter
        mask_param = torch.rand(solver.rdim[0], solver.rdim[1], dtype=torch.float32, requires_grad=True)
        solver.update_er_with_mask(mask=mask_param, layer_index=0)  # bg defaults to air

        # Source
        src = solver.add_source(theta=20 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)

        # Single-source solve + grad
        result_single = solver.solve(src)
        loss_single = result_single.transmission.sum()
        loss_single.backward(retain_graph=True)
        grad_single = mask_param.grad.detach().clone()

        # Reset gradients
        mask_param.grad.zero_()

        # Batched (n_sources=1) solve + grad
        results_batched = solver.solve([src])
        loss_batched = results_batched[0].transmission.sum()
        loss_batched.backward()
        grad_batched = mask_param.grad.detach().clone()

        # Compare gradients
        assert grad_single.shape == grad_batched.shape
        max_diff = torch.max(torch.abs(grad_single - grad_batched)).item()
        assert torch.allclose(grad_single, grad_batched, atol=1e-8, rtol=1e-6), f"Grad mismatch: {max_diff:.2e}"
        print(f"  ✓ Gradient match w.r.t. mask (max diff: {max_diff:.2e})")

    

    def test_numerical_stability(self, setup_solver):
        """Numerical sanity: thin layer produces finite S-matrix entries."""
        solver = setup_solver

        # Add very thin layer and ensure finite outputs
        solver.add_layer(material_name="Si", thickness=1e-6)

        source = solver.add_source(theta=45 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
        solver.src = source
        solver._pre_solve()
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)

        n_layer = len(solver.layer_manager.layers) - 1  # Thin layer
        smat_thin = solver._process_layer(n_layer, matrices)

        for key, val in smat_thin.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key} for thin layer"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
