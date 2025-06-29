import pytest
import torch
from unittest.mock import Mock

from torchrdit.utils import tensor_params_check, create_material, EigComplex


class TestUtilsCoverage:
    """Tests to improve coverage of utils.py missing lines."""

    def test_tensor_params_check_kwargs_error(self):
        """Test missing lines 71-72: kwargs tensor type error."""

        @tensor_params_check()
        def test_func(self, tensor_arg=None):
            return tensor_arg

        obj = Mock()
        with pytest.raises(TypeError, match="is not a torch.Tensor type"):
            test_func(obj, tensor_arg="not_a_tensor")

    def test_tensor_params_check_positional_error(self):
        """Test missing lines 64-65: positional tensor type error."""

        @tensor_params_check()
        def test_func(self, tensor_arg):
            return tensor_arg

        obj = Mock()
        with pytest.raises(TypeError, match="is not a torch.Tensor type"):
            test_func(obj, "not_a_tensor")

    def test_eig_complex_backward(self):
        """Test missing lines 151-175: EigComplex backward method."""
        # Create test tensors for backward pass
        matrix_d = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.complex128, requires_grad=True)
        matrix_u = torch.eye(2, dtype=torch.complex128)
        shape = torch.Size([2, 2])
        eps = torch.tensor(1e-6, dtype=torch.complex128)

        # Mock the context
        class MockContext:
            saved_tensors = (matrix_d, matrix_u, shape, eps)

        ctx = MockContext()
        grad_matrix_d = torch.ones_like(matrix_d)
        grad_matrix_u = torch.ones_like(matrix_u)

        # Test the backward method
        result = EigComplex.backward(ctx, grad_matrix_d, grad_matrix_u)
        assert result is not None
        assert len(result) == 2
        assert result[1] is None  # Second return should be None

    def test_create_material_dispersive_invalid_data(self):
        """Test missing lines for create_material with invalid dispersive data."""
        # Test with invalid data file path
        with pytest.raises(Exception):  # This might raise different exceptions depending on implementation
            create_material(
                name="test_material", dielectric_dispersion=True, user_dielectric_file="/nonexistent/path/file.txt"
            )

    def test_create_material_frequency_data(self):
        """Test missing lines for frequency data conversion."""
        # Create a temporary file with frequency data
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# freq(THz) eps_real eps_imag\n")
            f.write("300.0 2.25 0.0\n")
            f.write("400.0 2.26 0.01\n")
            temp_file = f.name

        try:
            material = create_material(
                name="test_dispersive",
                dielectric_dispersion=True,
                user_dielectric_file=temp_file,
                data_format="freq-eps",
                data_unit="thz",
            )
            assert material.name == "test_dispersive"
            assert material.isdispersive_er is True
        finally:
            os.unlink(temp_file)

    def test_init_smatrix_ndim_2(self):
        """Test missing lines 343-346: smatrix initialization with 2D input."""
        from torchrdit.utils import init_smatrix

        # Test with 2D shape (ndim < 3)
        shape = (5, 5)
        dtype = torch.complex128
        device = torch.device("cpu")
        result = init_smatrix(shape, dtype, device)
        assert "S11" in result
        assert "S12" in result
        assert "S21" in result
        assert "S22" in result
        # Check that shapes are correct for 2D case
        assert result["S11"].shape == (5, 5)
        assert result["S12"].shape == (5, 5)

    def test_redhstar_2d(self):
        """Test missing line 388: 2D S-matrix combination."""
        from torchrdit.utils import redhstar

        # Create 2D S-matrices (without batch dimension)
        smat_a = {
            "S11": torch.zeros((3, 3), dtype=torch.complex128),
            "S12": torch.eye(3, dtype=torch.complex128),
            "S21": torch.eye(3, dtype=torch.complex128),
            "S22": torch.zeros((3, 3), dtype=torch.complex128),
        }
        smat_b = {
            "S11": torch.zeros((3, 3), dtype=torch.complex128),
            "S12": torch.eye(3, dtype=torch.complex128),
            "S21": torch.eye(3, dtype=torch.complex128),
            "S22": torch.zeros((3, 3), dtype=torch.complex128),
        }
        result = redhstar(smat_a, smat_b)
        assert "S11" in result
        assert "S12" in result
        assert "S21" in result
        assert "S22" in result
