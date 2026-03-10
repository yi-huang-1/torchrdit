"""Test device fallback behavior in utility functions.

Tests that utility functions properly resolve devices before allocating tensors,
falling back to CPU when invalid devices are requested.
"""

import warnings
import pytest
import torch

from torchrdit.utils import init_smatrix, _create_blur_kernel, operator_blur, blur_filter


class TestInitSmatrixDeviceFallback:
    """Test device resolution in init_smatrix()."""

    def test_init_smatrix_cpu_device(self):
        """Test init_smatrix with valid CPU device."""
        shape = (2, 4, 4)
        dtype = torch.complex64
        smatrix = init_smatrix(shape, dtype, device="cpu")
        
        assert smatrix["S11"].device.type == "cpu"
        assert smatrix["S12"].device.type == "cpu"
        assert smatrix["S21"].device.type == "cpu"
        assert smatrix["S22"].device.type == "cpu"

    def test_init_smatrix_mps_fallback_to_cpu(self):
        """Test init_smatrix with MPS device falls back to CPU with warning."""
        shape = (2, 4, 4)
        dtype = torch.complex64
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smatrix = init_smatrix(shape, dtype, device="mps")
            
            # Should have issued a warning
            assert len(w) == 1
            assert "mps" in str(w[0].message).lower()
            
            # All tensors should be on CPU
            assert smatrix["S11"].device.type == "cpu"
            assert smatrix["S12"].device.type == "cpu"
            assert smatrix["S21"].device.type == "cpu"
            assert smatrix["S22"].device.type == "cpu"

    def test_init_smatrix_invalid_device_fallback(self):
        """Test init_smatrix with invalid device falls back to CPU with warning."""
        shape = (2, 4, 4)
        dtype = torch.complex64
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smatrix = init_smatrix(shape, dtype, device="invalid_device")
            
            # Should have issued a warning
            assert len(w) == 1
            assert "unsupported" in str(w[0].message).lower()
            
            # All tensors should be on CPU
            assert smatrix["S11"].device.type == "cpu"
            assert smatrix["S12"].device.type == "cpu"
            assert smatrix["S21"].device.type == "cpu"
            assert smatrix["S22"].device.type == "cpu"

    def test_init_smatrix_2d_shape(self):
        """Test init_smatrix with 2D shape and MPS fallback."""
        shape = (4, 4)
        dtype = torch.complex64
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            smatrix = init_smatrix(shape, dtype, device="mps")
            
            # Should have issued a warning
            assert len(w) == 1
            
            # All tensors should be on CPU
            assert smatrix["S11"].device.type == "cpu"
            assert smatrix["S12"].device.type == "cpu"
            assert smatrix["S21"].device.type == "cpu"
            assert smatrix["S22"].device.type == "cpu"


class TestCreateBlurKernelDeviceFallback:
    """Test device resolution in _create_blur_kernel()."""

    def test_create_blur_kernel_cpu_device(self):
        """Test _create_blur_kernel with valid CPU device."""
        kernel = _create_blur_kernel(radius=3, device=torch.device("cpu"))
        assert kernel.device.type == "cpu"

    def test_create_blur_kernel_mps_fallback_to_cpu(self):
        """Test _create_blur_kernel with MPS device falls back to CPU with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kernel = _create_blur_kernel(radius=3, device=torch.device("mps"))
            
            # Should have issued a warning
            assert len(w) == 1
            assert "mps" in str(w[0].message).lower()
            
            # Kernel should be on CPU
            assert kernel.device.type == "cpu"

    def test_create_blur_kernel_invalid_device_fallback(self):
        """Test _create_blur_kernel with invalid device falls back to CPU with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kernel = _create_blur_kernel(radius=3, device="invalid_device")
            
            # Should have issued a warning
            assert len(w) == 1
            assert "unsupported" in str(w[0].message).lower()
            
            # Kernel should be on CPU
            assert kernel.device.type == "cpu"


class TestOperatorBlurDeviceFallback:
    """Test device resolution in operator_blur()."""

    def test_operator_blur_cpu_device(self):
        """Test operator_blur with valid CPU device."""
        rho = torch.rand(1, 1, 32, 32)
        result = operator_blur(rho, radius=2, device=torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_operator_blur_mps_fallback_to_cpu(self):
        """Test operator_blur with MPS device falls back to CPU with warning."""
        rho = torch.rand(1, 1, 32, 32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = operator_blur(rho, radius=2, device=torch.device("mps"))
            
            # Should have issued a warning
            assert len(w) == 1
            assert "mps" in str(w[0].message).lower()
            
            # Result should be on CPU
            assert result.device.type == "cpu"

    def test_operator_blur_invalid_device_fallback(self):
        """Test operator_blur with invalid device falls back to CPU with warning."""
        rho = torch.rand(1, 1, 32, 32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = operator_blur(rho, radius=2, device="invalid_device")
            
            # Should have issued a warning
            assert len(w) == 1
            assert "unsupported" in str(w[0].message).lower()
            
            # Result should be on CPU
            assert result.device.type == "cpu"


class TestBlurFilterDeviceFallback:
    """Test device resolution in blur_filter()."""

    def test_blur_filter_cpu_device(self):
        """Test blur_filter with valid CPU device."""
        rho = torch.rand(1, 1, 32, 32)
        result = blur_filter(rho, radius=2, device=torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_blur_filter_mps_fallback_to_cpu(self):
        """Test blur_filter with MPS device falls back to CPU with warning."""
        rho = torch.rand(1, 1, 32, 32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = blur_filter(rho, radius=2, device=torch.device("mps"))
            
            # Should have issued a warning
            assert len(w) == 1
            assert "mps" in str(w[0].message).lower()
            
            # Result should be on CPU
            assert result.device.type == "cpu"

    def test_blur_filter_invalid_device_fallback(self):
        """Test blur_filter with invalid device falls back to CPU with warning."""
        rho = torch.rand(1, 1, 32, 32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = blur_filter(rho, radius=2, device="invalid_device")
            
            # Should have issued a warning
            assert len(w) == 1
            assert "unsupported" in str(w[0].message).lower()
            
            # Result should be on CPU
            assert result.device.type == "cpu"

    def test_blur_filter_string_device_mps_fallback(self):
        """Test blur_filter with string device 'mps' falls back to CPU with warning."""
        rho = torch.rand(1, 1, 32, 32)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = blur_filter(rho, radius=2, device="mps")
            
            # Should have issued a warning
            assert len(w) == 1
            assert "mps" in str(w[0].message).lower()
            
            # Result should be on CPU
            assert result.device.type == "cpu"
