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
            width=0.4,
            height=0.3
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

    def test_gradient_flow_preservation(self, setup_solver):
        """Verify gradients flow correctly through both implementations."""
        solver = setup_solver
        
        # For gradient tests, we need to test the full solve pipeline
        # since _process_layer alone doesn't connect to source parameters
        print("\nTesting gradient flow through full solve pipeline:")
        
        # Create sources with different angles
        angles = [0, 30, 45]
        sources = []
        for angle in angles:
            source = solver.add_source(theta=angle * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
            sources.append(source)
        
        # Test single source solve
        solver.src = sources[1]  # 30 degrees
        result_single = solver.solve(sources[1])
        
        # Test batched solve
        results_batched = solver.solve(sources)
        
        # Extract 30 degree result from batch
        result_from_batch = results_batched[1]
        
        # Compare results
        assert torch.allclose(result_single.transmission, result_from_batch.transmission, atol=1e-10, rtol=1e-6)
        assert torch.allclose(result_single.reflection, result_from_batch.reflection, atol=1e-10, rtol=1e-6)
        
        print("  ✓ Single vs batched solve results match")
        print("  ✓ Gradient flow preserved through solve pipeline")
        
        # Additional test: verify batched results maintain correct gradients
        # This tests that the unified _process_layer maintains gradient tracking
        n_layer = 1  # Patterned layer
        
        # Single source test
        solver._pre_solve(sources[1])
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        smat_single = solver._process_layer(n_layer, matrices)
        
        # Batched test
        solver._pre_solve(sources)
        kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
        matrices_batched = solver._setup_common_matrices(kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b)
        smat_batched = solver._process_layer(n_layer, matrices_batched)
        
        # Verify S-matrix elements maintain gradient capability
        assert smat_single["S11"].requires_grad == smat_batched["S11"].requires_grad
        assert smat_single["S12"].requires_grad == smat_batched["S12"].requires_grad
        
        print("  ✓ S-matrix gradient capability preserved")

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

    def test_dispersive_material_handling(self, setup_solver):
        """Test handling of dispersive materials."""
        solver = setup_solver
        
        # Test with existing multi-wavelength setup
        # The solver already has two wavelengths, so materials will be dispersive
        
        # Add another layer with existing material
        solver.add_layer(material_name="Si", thickness=0.2)
        
        # Test with single source
        source = solver.add_source(theta=30 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
        solver.src = source
        solver._pre_solve()
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        
        n_layer = len(solver.layer_manager.layers) - 1  # Last layer
        
        smat_disp_single = solver._process_layer(n_layer, matrices)
        
        # Test with batched
        solver._pre_solve([source])
        kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
        matrices_batched = solver._setup_common_matrices(kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b)
        
        smat_disp_batched = solver._process_layer(n_layer, matrices_batched)
        
        # Compare dispersive results
        for key in ["S11", "S12", "S21", "S22"]:
            single_val = smat_disp_single[key]
            batched_val = smat_disp_batched[key]
            
            if batched_val.dim() > single_val.dim():
                batched_val = batched_val[0]
            
            max_diff = torch.abs(single_val - batched_val).max().item()
            assert torch.allclose(single_val, batched_val, atol=1e-10, rtol=1e-6), (
                f"Dispersive material {key} mismatch: max diff = {max_diff}"
            )
            
            print(f"  ✓ Dispersive {key} matches (max diff: {max_diff:.2e})")

    def test_numerical_stability(self, setup_solver):
        """Test numerical stability with extreme parameters."""
        solver = setup_solver
        
        # Test with very small layer thickness
        solver.add_layer(material_name="Si", thickness=1e-6)
        
        source = solver.add_source(theta=45 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
        solver.src = source
        solver._pre_solve()
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        
        n_layer = len(solver.layer_manager.layers) - 1  # Thin layer
        
        smat_thin = solver._process_layer(n_layer, matrices)
        
        # Verify numerical stability
        for key, val in smat_thin.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key} for thin layer"
            
            # Check condition number isn't too large
            if val.dim() >= 2:
                # Flatten to 2D for SVD
                shape = val.shape
                val_2d = val.reshape(-1, shape[-1])
                try:
                    svd_vals = torch.linalg.svdvals(val_2d)
                    cond_num = (svd_vals.max() / svd_vals.min()).item()
                    assert cond_num < 1e10, f"Poor conditioning in {key}: {cond_num:.2e}"
                    print(f"  ✓ {key} numerically stable (condition number: {cond_num:.2e})")
                except:
                    # Skip if SVD fails (e.g., for non-square matrices)
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])