"""Test suite for tensorized source batching operations.

This module tests the mathematical equivalence between sequential and tensorized
processing of batched sources, ensuring that the optimization maintains correctness
while improving performance.
"""

import torch
import numpy as np
import pytest
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material


class TestTensorizedBatching:
    """Test suite for tensorized source batching operations."""
    
    @pytest.fixture
    def setup_solver(self):
        """Create a basic solver for testing."""
        # Create solver with parameters
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            kdim=[3, 3],
            rdim=[256, 256],
            device='cpu'
        )
        
        # Create and add materials
        air = create_material(name="Air", permittivity=1.0)
        si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([air, si])
        
        # Add layers with materials
        solver.add_layer(material_name="Air", thickness=0.0)
        solver.add_layer(material_name="Si", thickness=0.5)
        solver.add_layer(material_name="Air", thickness=0.0)
        
        return solver
    
    def test_single_source_equivalence(self, setup_solver):
        """Verify batched[0] == non-batched for single source."""
        solver = setup_solver
        
        # Single source
        source = solver.add_source(theta=30*np.pi/180, phi=0, pte=1.0, ptm=0.0)
        
        # Set source and run pre-solve to initialize
        solver.src = source
        solver._pre_solve()
        
        # Get k-vectors
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        
        # Non-batched matrices
        matrices_single = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        
        # Batched with single source
        kx_0_batched = kx_0.unsqueeze(0)
        ky_0_batched = ky_0.unsqueeze(0)
        kz_ref_0_batched = kz_ref_0.unsqueeze(0)
        kz_trn_0_batched = kz_trn_0.unsqueeze(0)
        
        matrices_batched = solver._setup_common_matrices(
            kx_0_batched, ky_0_batched, kz_ref_0_batched, kz_trn_0_batched
        )
        
        # Compare all matrices
        for key in matrices_single:
            single_val = matrices_single[key]
            batched_val = matrices_batched[key]
            
            # Handle the source dimension
            if batched_val.dim() > single_val.dim():
                batched_val = batched_val.squeeze(0)
            
            print(f"[DEBUG] Comparing {key}:")
            print(f"  Single shape: {single_val.shape}")
            print(f"  Batched shape: {batched_val.shape}")
            
            # Check shapes match
            assert single_val.shape == batched_val.shape, \
                f"Shape mismatch for {key}: {single_val.shape} vs {batched_val.shape}"
            
            # Check values match
            max_diff = torch.max(torch.abs(single_val - batched_val))
            print(f"  Max diff: {max_diff}")
            
            assert torch.allclose(single_val, batched_val, atol=1e-6, rtol=1e-5), \
                f"Mismatch in {key}: max diff = {max_diff}"
    
    def test_multiple_source_equivalence(self, setup_solver):
        """Verify each batched[i] == sequential result[i]."""
        solver = setup_solver
        
        # Multiple sources
        angles = [0, 30, 45, 60]
        sources = []
        for angle in angles:
            source = solver.add_source(theta=angle*np.pi/180, phi=0, pte=1.0, ptm=0.0)
            sources.append(source)
        
        # Process sequentially
        sequential_results = []
        for source in sources:
            solver.src = source
            solver._pre_solve()
            kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
            matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
            sequential_results.append(matrices)
        
        # Process batched using unified _pre_solve
        solver._pre_solve(sources)
        kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
        matrices_batched = solver._setup_common_matrices(
            kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b
        )
        
        # Compare each source
        for i, angle in enumerate(angles):
            print(f"\n[DEBUG] Comparing source {i} (angle={angle}°)")
            for key in sequential_results[0]:
                seq_val = sequential_results[i][key]
                batch_val = matrices_batched[key]
                
                # Extract the i-th source from batched result
                if batch_val.dim() > seq_val.dim():
                    batch_val = batch_val[i]
                
                max_diff = torch.max(torch.abs(seq_val - batch_val))
                print(f"  {key}: max diff = {max_diff}")
                
                assert torch.allclose(seq_val, batch_val, atol=1e-6, rtol=1e-5), \
                    f"Source {i}, {key}: max diff = {max_diff}"
    
    def test_numerical_precision(self, setup_solver):
        """Ensure torch.allclose with appropriate tolerances."""
        solver = setup_solver
        
        # Test with edge cases: very small and very large values
        # Near-normal incidence (small k-values)
        source_normal = solver.add_source(theta=1e-6, phi=0, pte=1.0, ptm=0.0)
        
        # Grazing incidence (large k-values)  
        source_grazing = solver.add_source(theta=89*np.pi/180, phi=0, pte=1.0, ptm=0.0)
        
        sources = [source_normal, source_grazing]
        
        # Process batched using unified _pre_solve
        solver._pre_solve(sources)
        kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
        matrices_batched = solver._setup_common_matrices(
            kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b
        )
        
        # Check numerical stability
        for key, val in matrices_batched.items():
            # Check for NaN or Inf
            assert not torch.any(torch.isnan(val)), f"{key} contains NaN values"
            assert not torch.any(torch.isinf(val)), f"{key} contains Inf values"
            
            # Check that values are within reasonable range
            if 'inv' not in key and 'mat_v0' not in key:
                assert torch.max(torch.abs(val)) < 1e10, f"{key} has unreasonably large values"
    
    def test_gradient_preservation(self, setup_solver):
        """Verify gradients flow correctly through vectorized ops."""
        solver = setup_solver
        
        # Create a simple scenario that tests gradient flow through matrix operations
        # We'll create a dummy tensor that requires grad and use it in matrix setup
        
        # Process batched sources normally first
        sources = []
        for angle in [0, 30, 45]:
            source = solver.add_source(theta=angle*np.pi/180, phi=0, pte=1.0, ptm=0.0)
            sources.append(source)
        
        solver._pre_solve(sources)
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        
        # Create a dummy parameter that requires grad and modify kx_0
        dummy_param = torch.tensor([1.0, 1.0, 1.0], requires_grad=True, dtype=torch.cfloat)
        
        # Modify kx_0 with the parameter to introduce gradient dependency
        kx_0_grad = kx_0 + dummy_param[:, None, None, None] * 0.01
        
        # Process through batched matrices
        matrices_batched = solver._setup_common_matrices(
            kx_0_grad, ky_0, kz_ref_0, kz_trn_0
        )
        
        # Compute a simple loss function
        loss = torch.sum(torch.abs(matrices_batched['mat_kx']))
        
        # Check that gradients can be computed
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert dummy_param.grad is not None, "Gradients not computed"
        assert not torch.allclose(dummy_param.grad, torch.zeros_like(dummy_param.grad)), \
            "Gradients are all zero"
        
        print(f"[DEBUG] Gradient values: {dummy_param.grad}")
    
    def test_shape_consistency_across_operations(self, setup_solver):
        """Test that shapes remain consistent through all tensorized operations."""
        solver = setup_solver
        
        n_sources = 5
        sources = []
        for i in range(n_sources):
            angle = i * 15 * np.pi/180
            source = solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            sources.append(source)
        
        # Process batched
        solver._pre_solve(sources)
        kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b = solver._initialize_k_vectors()
        
        # Check input shapes
        assert kx_0_b.shape[0] == n_sources
        assert ky_0_b.shape[0] == n_sources
        assert kz_ref_0_b.shape[0] == n_sources
        assert kz_trn_0_b.shape[0] == n_sources
        
        # Process through matrices
        matrices_batched = solver._setup_common_matrices(
            kx_0_b, ky_0_b, kz_ref_0_b, kz_trn_0_b
        )
        
        # Check output shapes
        n_harmonics_squared = solver.kdim[0] * solver.kdim[1]
        
        # Vector quantities should have shape (n_sources, n_freqs, n_harmonics_squared)
        for key in ['mat_kx', 'mat_ky', 'mat_kz_ref', 'mat_kz_trn', 'mat_kz']:
            assert matrices_batched[key].shape == (n_sources, solver.n_freqs, n_harmonics_squared), \
                f"{key} has wrong shape: {matrices_batched[key].shape}"
        
        # Diagonal matrices should have shape (n_sources, n_freqs, n_harmonics_squared, n_harmonics_squared)
        for key in ['mat_kx_diag', 'mat_ky_diag']:
            assert matrices_batched[key].shape == (n_sources, solver.n_freqs, n_harmonics_squared, n_harmonics_squared), \
                f"{key} has wrong shape: {matrices_batched[key].shape}"
        
        # Block matrices should have shape (n_sources, n_freqs, 2*n_harmonics_squared, 2*n_harmonics_squared)
        for key in ['mat_w0', 'mat_v0']:
            assert matrices_batched[key].shape == (n_sources, solver.n_freqs, 2*n_harmonics_squared, 2*n_harmonics_squared), \
                f"{key} has wrong shape: {matrices_batched[key].shape}"
    
    
