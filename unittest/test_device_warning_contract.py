"""
Test suite for device fallback warning contract.

This module ensures that device fallback warnings follow a strict contract:
- MPS warnings must mention both "mps" and "complex"
- CUDA unavailable warnings must mention both "cuda" and "not available"
- All warnings must include the requested device name
"""

import pytest
import warnings
import torch

from torchrdit.device import resolve_device


class TestDeviceWarningContract:
    """Test the warning message contract for device fallback."""

    @pytest.mark.device_fallback
    def test_mps_warning_contains_mps_and_complex(self):
        """MPS fallback warning must contain both 'mps' and 'complex'."""
        with pytest.warns(UserWarning) as record:
            result = resolve_device("mps")
        
        assert len(record) == 1
        warning_msg = str(record[0].message)
        assert "mps" in warning_msg.lower(), f"Warning missing 'mps': {warning_msg}"
        assert "complex" in warning_msg.lower(), f"Warning missing 'complex': {warning_msg}"
        assert result.fell_back is True
        assert result.resolved_device == torch.device("cpu")

    @pytest.mark.device_fallback
    def test_mps_warning_contains_requested_device(self):
        """MPS fallback warning must include the requested device."""
        with pytest.warns(UserWarning) as record:
            result = resolve_device("mps")
        
        warning_msg = str(record[0].message)
        assert "mps" in warning_msg.lower()
        assert result.requested_device == "mps"

    @pytest.mark.device_fallback
    def test_cuda_unavailable_warning_contains_cuda_and_not_available(self):
        """CUDA unavailable warning must contain both 'cuda' and 'not available'."""
        # Only run this test if CUDA is actually unavailable
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; skipping unavailable test")
        
        with pytest.warns(UserWarning) as record:
            result = resolve_device("cuda")
        
        assert len(record) == 1
        warning_msg = str(record[0].message)
        assert "cuda" in warning_msg.lower(), f"Warning missing 'cuda': {warning_msg}"
        assert "not available" in warning_msg.lower() or "unavailable" in warning_msg.lower(), \
            f"Warning missing 'not available' or 'unavailable': {warning_msg}"
        assert result.fell_back is True
        assert result.resolved_device == torch.device("cpu")

    @pytest.mark.device_fallback
    def test_cuda_unavailable_warning_contains_requested_device(self):
        """CUDA unavailable warning must include the requested device."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; skipping unavailable test")
        
        with pytest.warns(UserWarning) as record:
            result = resolve_device("cuda")
        
        warning_msg = str(record[0].message)
        assert "cuda" in warning_msg.lower()
        assert result.requested_device == "cuda"

    @pytest.mark.device_fallback
    def test_unsupported_device_warning_structure(self):
        """Unsupported device warning must follow contract."""
        with pytest.warns(UserWarning) as record:
            result = resolve_device("unsupported_device")
        
        assert len(record) == 1
        warning_msg = str(record[0].message)
        # Should mention the device and fallback
        assert "unsupported_device" in warning_msg or "unsupported" in warning_msg.lower()
        assert result.fell_back is True
        assert result.resolved_device == torch.device("cpu")

    @pytest.mark.device_fallback
    def test_cpu_device_no_warning(self):
        """CPU device should not trigger any warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_device("cpu")
        
        # Filter out any unrelated warnings
        device_warnings = [warning for warning in w 
                          if "device" in str(warning.message).lower() 
                          or "cpu" in str(warning.message).lower()]
        assert len(device_warnings) == 0, f"Unexpected warning for CPU: {device_warnings}"
        assert result.fell_back is False
        assert result.reason is None

    @pytest.mark.device_fallback
    def test_warning_message_in_device_resolution(self):
        """DeviceResolution.reason must match the warning message."""
        with pytest.warns(UserWarning) as record:
            result = resolve_device("mps")
        
        warning_msg = str(record[0].message)
        assert result.reason == warning_msg, \
            f"Reason mismatch: {result.reason} != {warning_msg}"

    @pytest.mark.device_fallback
    def test_mps_with_index_warning(self):
        """MPS with index should still trigger complex warning."""
        with pytest.warns(UserWarning) as record:
            result = resolve_device("mps:0")
        
        assert len(record) == 1
        warning_msg = str(record[0].message)
        assert "mps" in warning_msg.lower()
        assert "complex" in warning_msg.lower()
        assert result.requested_device == "mps:0"

    @pytest.mark.device_fallback
    def test_cuda_device_index_exceeds_available(self):
        """CUDA device index exceeding available should warn."""
        # Get a device index that definitely doesn't exist
        if torch.cuda.is_available():
            invalid_index = torch.cuda.device_count() + 10
        else:
            invalid_index = 0
        
        with pytest.warns(UserWarning) as record:
            result = resolve_device(f"cuda:{invalid_index}")
        
        assert len(record) == 1
        warning_msg = str(record[0].message)
        assert "cuda" in warning_msg.lower()
        assert result.fell_back is True
        assert result.resolved_device == torch.device("cpu")

    @pytest.mark.device_fallback
    def test_warning_is_user_warning(self):
        """All device fallback warnings must be UserWarning category."""
        with pytest.warns(UserWarning) as record:
            resolve_device("mps")
        
        assert len(record) == 1
        assert issubclass(record[0].category, UserWarning)

    @pytest.mark.device_fallback
    def test_multiple_fallbacks_independent_warnings(self):
        """Multiple fallback calls should each produce independent warnings."""
        with pytest.warns(UserWarning) as record:
            resolve_device("mps")
            resolve_device("unsupported_device")
        
        assert len(record) == 2
        messages = [str(r.message) for r in record]
        assert any("mps" in msg.lower() for msg in messages)
        assert any("unsupported" in msg.lower() for msg in messages)
