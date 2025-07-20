"""
Test suite for tensor-level batched matrix operations.

This test file validates that batched matrix operations (identity matrices, 
matrix multiplications) properly handle the source dimension and maintain 
mathematical correctness.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestBatchedMatrices:
    """Test batched matrix operations for tensor-level source processing."""

    def setup_solver(self, n_freqs=3, kdim=(3, 3)):
        """Create a basic solver for testing."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.linspace(0.8, 1.2, n_freqs),
            rdim=[256, 256],
            kdim=list(kdim),
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]]),
            device="cpu",
            is_use_FFF=False,
        )

        # Add materials and layers
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.5), is_homogeneous=True)

        return solver

    def test_batched_identity_matrix_creation(self):
        """Test creation of batched identity matrices."""
        n_sources = 4
        n_freqs = 3
        n_harmonics = 5
        device = "cpu"
        dtype = torch.complex64
        
        # Method 1: Create once and expand
        eye = torch.eye(n_harmonics, device=device, dtype=dtype)
        eye_batched = eye.unsqueeze(0).unsqueeze(0).expand(n_sources, n_freqs, -1, -1)
        
        # Verify shape
        assert eye_batched.shape == (n_sources, n_freqs, n_harmonics, n_harmonics)
        
        # Verify it's still an identity matrix for each source and frequency
        for i in range(n_sources):
            for j in range(n_freqs):
                torch.testing.assert_close(eye_batched[i, j], eye, rtol=1e-6, atol=1e-8)
                
        # Method 2: Using broadcasting directly
        eye_broadcast = torch.eye(n_harmonics, device=device, dtype=dtype)[None, None, :, :]
        
        # Should work with broadcasting
        test_tensor = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, dtype=dtype)
        result = eye_broadcast + test_tensor
        assert result.shape == (n_sources, n_freqs, n_harmonics, n_harmonics)

    def test_batched_matrix_multiplication(self):
        """Test batched matrix multiplication with source dimension."""
        n_sources = 3
        n_freqs = 2
        n_harmonics = 4
        device = "cpu"
        dtype = torch.complex64
        
        # Create test matrices with shape (n_sources, n_freqs, n_harmonics, n_harmonics)
        A = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, dtype=dtype, device=device)
        B = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, dtype=dtype, device=device)
        
        # Test @ operator
        C = A @ B
        assert C.shape == (n_sources, n_freqs, n_harmonics, n_harmonics)
        
        # Verify correctness by checking one element
        for i in range(n_sources):
            for j in range(n_freqs):
                # NaN values are bugs - matrix results should never contain NaN
                assert not torch.isnan(C[i, j]).any(), \
                    f"NaN values found in matrix multiplication result at [{i}, {j}] - this is a bug!"
                
                expected = A[i, j] @ B[i, j]
                assert not torch.isnan(expected).any(), \
                    f"NaN values found in expected matrix result at [{i}, {j}] - this is a bug!"
                
                torch.testing.assert_close(C[i, j], expected, rtol=1e-5, atol=1e-6)

    def test_batched_diagonal_matrix_operations(self):
        """Test operations with diagonal matrices in batched setting."""
        solver = self.setup_solver(n_freqs=2, kdim=(3, 3))
        n_sources = 4
        n_harmonics_squared = solver.kdim[0] * solver.kdim[1]
        
        # Create batched diagonal values
        # Shape: (n_sources, n_freqs, n_harmonics_squared)
        diag_values = torch.randn(n_sources, solver.n_freqs, n_harmonics_squared, 
                                  dtype=solver.tcomplex, device=solver.device)
        
        # Convert to diagonal matrices
        # Method 1: Using torch.diag_embed
        diag_matrices = torch.diag_embed(diag_values)
        assert diag_matrices.shape == (n_sources, solver.n_freqs, n_harmonics_squared, n_harmonics_squared)
        
        # Verify diagonal property
        for i in range(n_sources):
            for j in range(solver.n_freqs):
                # Check off-diagonal elements are zero
                mask = torch.eye(n_harmonics_squared, dtype=torch.bool, device=solver.device)
                off_diag = diag_matrices[i, j][~mask]
                assert torch.allclose(off_diag, torch.zeros_like(off_diag))
                
                # Check diagonal elements match
                diag_extracted = torch.diagonal(diag_matrices[i, j])
                torch.testing.assert_close(diag_extracted, diag_values[i, j], rtol=1e-6, atol=1e-8)

    def test_batched_k_vector_to_matrix_transformation(self):
        """Test transformation of k-vectors to diagonal matrices with batching."""
        solver = self.setup_solver(n_freqs=3, kdim=(5, 5))
        n_sources = 3
        
        # Create batched k-vectors
        # Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        kx_0 = torch.randn(n_sources, solver.n_freqs, solver.kdim[0], solver.kdim[1], 
                          dtype=solver.tcomplex, device=solver.device)
        ky_0 = torch.randn(n_sources, solver.n_freqs, solver.kdim[0], solver.kdim[1], 
                          dtype=solver.tcomplex, device=solver.device)
        
        # Transform to diagonal matrices (flatten spatial dimensions)
        # Current solver approach: transpose then flatten
        mat_kx = kx_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_ky = ky_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        
        # Verify shape
        n_harmonics_squared = solver.kdim[0] * solver.kdim[1]
        assert mat_kx.shape == (n_sources, solver.n_freqs, n_harmonics_squared)
        assert mat_ky.shape == (n_sources, solver.n_freqs, n_harmonics_squared)
        
        # Convert to diagonal matrices for use in matrix operations
        mat_kx_diag = torch.diag_embed(mat_kx)
        mat_ky_diag = torch.diag_embed(mat_ky)
        
        assert mat_kx_diag.shape == (n_sources, solver.n_freqs, n_harmonics_squared, n_harmonics_squared)
        assert mat_ky_diag.shape == (n_sources, solver.n_freqs, n_harmonics_squared, n_harmonics_squared)

    def test_batched_matrix_inverse(self):
        """Test batched matrix inverse operations."""
        n_sources = 2
        n_freqs = 3
        n_harmonics = 4
        device = "cpu"
        dtype = torch.complex64
        
        # Create invertible matrices
        A = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, dtype=dtype, device=device)
        # Make sure they're invertible by adding diagonal dominance
        eye = torch.eye(n_harmonics, dtype=dtype, device=device)
        A = A + 3 * eye[None, None, :, :]
        
        # Compute inverse
        A_inv = torch.linalg.inv(A)
        
        # Verify A @ A_inv = I
        identity = A @ A_inv
        expected_identity = eye[None, None, :, :].expand(n_sources, n_freqs, -1, -1)
        
        torch.testing.assert_close(identity, expected_identity, rtol=1e-5, atol=1e-6)

    def test_batched_matrix_solve(self):
        """Test batched linear system solving."""
        n_sources = 3
        n_freqs = 2
        n_harmonics = 5
        device = "cpu"
        dtype = torch.complex64
        
        # Create system A @ x = b
        A = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, dtype=dtype, device=device)
        # Make A well-conditioned
        eye = torch.eye(n_harmonics, dtype=dtype, device=device)
        A = A + 3 * eye[None, None, :, :]
        
        b = torch.randn(n_sources, n_freqs, n_harmonics, 1, dtype=dtype, device=device)
        
        # Solve using torch.linalg.solve
        x = torch.linalg.solve(A, b)
        
        # Verify solution
        b_reconstructed = A @ x
        torch.testing.assert_close(b_reconstructed, b, rtol=1e-5, atol=1e-6)

    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency with large batch sizes."""
        n_sources = 32  # Large batch
        n_freqs = 10
        n_harmonics = 25  # 5x5 grid
        device = "cpu"
        dtype = torch.complex64
        
        # Create identity matrix efficiently
        eye = torch.eye(n_harmonics, device=device, dtype=dtype)
        
        # Method 1: Expand (memory efficient - no copy)
        eye_expanded = eye.unsqueeze(0).unsqueeze(0).expand(n_sources, n_freqs, -1, -1)
        
        # Method 2: Repeat (creates copies - less memory efficient)
        eye_repeated = eye.unsqueeze(0).unsqueeze(0).repeat(n_sources, n_freqs, 1, 1)
        
        # Both should give same result
        torch.testing.assert_close(eye_expanded, eye_repeated)
        
        # Verify broadcasting works without creating full tensor
        test_diag = torch.randn(n_sources, n_freqs, n_harmonics, dtype=dtype, device=device)
        # This should work efficiently with broadcasting
        result = eye[None, None, :, :] * test_diag[:, :, :, None]
        assert result.shape == (n_sources, n_freqs, n_harmonics, n_harmonics)

    def test_batched_matrix_operations_with_solver_dimensions(self):
        """Test matrix operations using actual solver dimensions and setup."""
        solver = self.setup_solver(n_freqs=3, kdim=(7, 7))
        n_sources = 5
        
        # Initialize solver to get mesh grids
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        solver.src = source
        solver._pre_solve()
        
        n_harmonics_squared = solver.kdim[0] * solver.kdim[1]
        
        # Create batched identity matrices as would be needed in solver
        ident_mat_k = torch.eye(n_harmonics_squared, dtype=solver.tcomplex, device=solver.device)
        ident_mat_k2 = torch.eye(2 * n_harmonics_squared, dtype=solver.tcomplex, device=solver.device)
        
        # For batched operation, we can broadcast these
        # Shape for operations: (n_sources, n_freqs, size, size)
        
        # Test a typical solver operation pattern
        # Create some test diagonal matrices
        test_diag = torch.randn(n_sources, solver.n_freqs, n_harmonics_squared, 
                               dtype=solver.tcomplex, device=solver.device)
        test_mat_diag = torch.diag_embed(test_diag)
        
        # Typical operation: I - D where D is diagonal
        result = ident_mat_k[None, None, :, :] - test_mat_diag
        assert result.shape == (n_sources, solver.n_freqs, n_harmonics_squared, n_harmonics_squared)
        
        # Verify the operation
        for i in range(n_sources):
            for j in range(solver.n_freqs):
                expected = ident_mat_k - torch.diag(test_diag[i, j])
                torch.testing.assert_close(result[i, j], expected, rtol=1e-6, atol=1e-8)

    def test_gradient_flow_through_matrix_operations(self):
        """Test that gradients flow correctly through batched matrix operations."""
        n_sources = 2
        n_freqs = 2
        n_harmonics = 3
        device = "cpu"
        dtype = torch.float32  # Use float for gradient computation
        
        # Create matrices with gradients
        A = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, 
                       dtype=dtype, device=device, requires_grad=True)
        B = torch.randn(n_sources, n_freqs, n_harmonics, n_harmonics, 
                       dtype=dtype, device=device, requires_grad=True)
        
        # Perform operations
        C = A @ B
        D = C + torch.eye(n_harmonics, dtype=dtype, device=device)[None, None, :, :]
        
        # Compute loss
        loss = D.sum()
        loss.backward()
        
        # Check gradients exist
        assert A.grad is not None, "Gradient should flow to A"
        assert B.grad is not None, "Gradient should flow to B"
        
        # Gradients should have same shape as inputs
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])