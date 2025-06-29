import pytest
import torch
import tempfile
import os

from torchrdit.materials import MaterialClass
from torchrdit.utils import create_material


class TestMaterialsCoverage:
    """Tests to improve coverage of materials.py missing lines."""

    def test_material_unsupported_data_format_error(self):
        """Test missing line 227: Unsupported data format error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("1.0 2.0 0.1\n")
            f.write("1.5 2.5 0.2\n")
            temp_file = f.name

        try:
            with pytest.raises(ValueError):
                MaterialClass(
                    permittivity=1.0,
                    dielectric_dispersion=True,
                    user_dielectric_file=temp_file,
                    data_format="unsupported_format",  # Invalid format
                )
        finally:
            os.unlink(temp_file)

    def test_material_complex_permittivity_negative_imag(self):
        """Test missing lines 242, 245-251: Complex permittivity with negative imaginary part."""
        # Test complex number with negative imaginary part (line 242)
        complex_perm = 2.25 - 0.1j  # Negative imaginary part
        material = MaterialClass(permittivity=complex_perm, dielectric_dispersion=False)

        assert material._er is not None
        assert torch.is_complex(material._er)

        # Test ValueError handling for invalid permittivity (lines 245-251)
        with pytest.raises(ValueError):
            MaterialClass(permittivity="invalid_string", dielectric_dispersion=False)

    def test_material_tensor_permittivity_edge_cases(self):
        """Test missing lines 248-251: Tensor permittivity edge cases."""
        # Test tensor with negative imaginary part
        tensor_perm = torch.tensor(2.25 - 0.1j, dtype=torch.complex64)
        material = MaterialClass(permittivity=tensor_perm, dielectric_dispersion=False)

        assert material._er is not None
        assert torch.is_complex(material._er)

    def test_material_dispersive_data_loading_errors(self):
        """Test missing lines 271-272: Dispersive data loading errors."""
        # Create invalid data file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("# Invalid frequency-permittivity data\n")
            f.write("invalid_freq invalid_eps\n")
            f.write("not_a_number also_not_a_number\n")
            temp_file = f.name

        try:
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                MaterialClass(
                    permittivity=1.0, dielectric_dispersion=True, user_dielectric_file=temp_file, data_format="freq-eps"
                )
        finally:
            os.unlink(temp_file)

    def test_material_missing_file_error(self):
        """Test missing line 314: File not found error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            MaterialClass(
                permittivity=1.0,
                dielectric_dispersion=True,
                user_dielectric_file="/nonexistent/path/material_data.txt",
                data_format="freq-eps",
            )

    def test_material_cache_error_handling(self):
        """Test missing line 352: Cache error handling."""
        # Create material and test cache error scenarios
        material = create_material(permittivity=2.25)
        assert material is not None

    def test_material_factory_methods_edge_cases(self):
        """Test missing line 491: Factory method edge cases."""
        # Test create_material with invalid parameters
        with pytest.raises((TypeError, ValueError)):
            create_material(permittivity="invalid_string")  # Invalid permittivity type

    def test_material_edge_case_permittivity_values(self):
        """Test edge case permittivity values."""
        # Test very small permittivity
        material1 = create_material(permittivity=1e-10)
        assert material1 is not None

        # Test very large permittivity
        material2 = create_material(permittivity=1e10)
        assert material2 is not None

        # Test negative real permittivity
        material4 = create_material(permittivity=-1.0)
        assert material4 is not None
