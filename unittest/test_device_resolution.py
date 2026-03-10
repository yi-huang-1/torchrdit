"""TDD test suite for device resolution middleware — deterministic, no hardware deps."""

import unittest
import warnings
from unittest.mock import patch

import torch

from torchrdit.device import DeviceResolution, resolve_device


class TestDeviceResolution(unittest.TestCase):

    def test_cpu_device_passthrough(self):
        """Given 'cpu', when resolved, then no fallback occurs."""
        result = resolve_device("cpu")

        self.assertIsInstance(result, DeviceResolution)
        self.assertEqual(result.requested_device, "cpu")
        self.assertEqual(result.resolved_device, torch.device("cpu"))
        self.assertFalse(result.fell_back)
        self.assertIsNone(result.reason)

    def test_invalid_device_string_falls_back_to_cpu(self):
        """Given unrecognised device, when resolved, then falls back with warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_device("tpu")

        self.assertTrue(result.fell_back)
        self.assertEqual(result.resolved_device, torch.device("cpu"))
        self.assertIsNotNone(result.reason)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0)

    def test_unavailable_cuda_falls_back_to_cpu(self):
        """Given 'cuda' unavailable, when resolved, then falls back with warning."""
        with patch("torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = resolve_device("cuda")

        self.assertTrue(result.fell_back)
        self.assertEqual(result.resolved_device, torch.device("cpu"))
        self.assertIsNotNone(result.reason)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0)

    def test_invalid_cuda_index_falls_back_to_cpu(self):
        """Given cuda:99 with 1 device, when resolved, then falls back citing index."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = resolve_device("cuda:99")

        self.assertTrue(result.fell_back)
        self.assertEqual(result.resolved_device, torch.device("cpu"))
        self.assertIn("99", result.reason)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0)

    def test_mps_rejected_unconditionally_with_complex_dtype_reason(self):
        """Given 'mps', when resolved, then rejected with complex-dtype reason."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_device("mps")

        self.assertTrue(result.fell_back)
        self.assertEqual(result.resolved_device, torch.device("cpu"))
        self.assertIn("mps", result.reason.lower())
        self.assertIn("complex", result.reason.lower())
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0)

    def test_torch_device_input_normalizes_to_torch_device(self):
        """Given torch.device('cpu') object, when resolved, then same as string."""
        result = resolve_device(torch.device("cpu"))

        self.assertIsInstance(result, DeviceResolution)
        self.assertEqual(result.requested_device, "cpu")
        self.assertEqual(result.resolved_device, torch.device("cpu"))
        self.assertFalse(result.fell_back)
        self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main()
