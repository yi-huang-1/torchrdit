import unittest
from unittest.mock import patch
import numpy as np
import torch

from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.solver import get_solver_builder, RCWASolver, RDITSolver


class TestDeviceBuilderFallback(unittest.TestCase):
    """Test device resolution integration in SolverBuilder.build()."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lam0 = np.array([1.0])
        self.air = create_material(name="air", permittivity=1.0, permeability=1.0)
        self.silicon = create_material(name="silicon", permittivity=11.7, permeability=1.0)
    
    def test_builder_with_invalid_cuda_falls_back_to_cpu(self):
        """Test that builder falls back to CPU when CUDA is unavailable.
        
        When .with_device("cuda") is called but CUDA is not available,
        the build() method should resolve the device and fall back to CPU.
        """
        # Mock torch.cuda.is_available to return False
        with patch('torchrdit.device.torch.cuda.is_available', return_value=False):
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver = (builder
                          .with_algorithm(Algorithm.RCWA)
                          .with_wavelengths(1.55)
                          .with_k_dimensions([5, 5])
                          .with_materials([self.air, self.silicon])
                          .with_device('cuda')
                          .build())
            
            # Verify solver was created and device is CPU
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), 'cpu')
    
    def test_from_config_device_is_resolved_at_build_time(self):
        """Test that device from config is resolved at build time.
        
        When device is specified in config, it should be resolved
        during build(), not during from_config().
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cuda"
        }
        
        # Mock torch.cuda.is_available to return False
        with patch('torchrdit.device.torch.cuda.is_available', return_value=False):
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver = builder.from_config(config).build()
            
            # Verify solver was created and device is CPU (fallback)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), 'cpu')
    
    def test_with_device_overrides_from_config_device_before_resolution(self):
        """Test that with_device() override takes precedence over config device.
        
        When both from_config(device=X) and with_device(Y) are used,
        with_device() should override the config value before resolution.
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cuda"
        }
        
        # Mock torch.cuda.is_available to return False
        with patch('torchrdit.device.torch.cuda.is_available', return_value=False):
            builder = get_solver_builder()
            solver = (builder
                      .from_config(config)
                      .with_device('cpu')  # Override config device
                      .build())
            
            # Verify solver was created with CPU device (override wins)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), 'cpu')
    
    def test_rdit_solver_device_resolution(self):
        """Test device resolution works for RDIT solver as well."""
        with patch('torchrdit.device.torch.cuda.is_available', return_value=False):
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver = (builder
                          .with_algorithm(Algorithm.RDIT)
                          .with_wavelengths(1.55)
                          .with_k_dimensions([5, 5])
                          .with_materials([self.air, self.silicon])
                          .with_device('cuda')
                          .build())
            
            # Verify RDIT solver was created and device is CPU
            self.assertIsInstance(solver, RDITSolver)
            self.assertEqual(str(solver.device), 'cpu')
    
    def test_cpu_device_no_fallback(self):
        """Test that CPU device does not trigger fallback."""
        builder = get_solver_builder()
        solver = (builder
                  .with_algorithm(Algorithm.RCWA)
                  .with_wavelengths(1.55)
                  .with_k_dimensions([5, 5])
                  .with_materials([self.air, self.silicon])
                  .with_device('cpu')
                  .build())
        
        # Verify solver was created with CPU device
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(str(solver.device), 'cpu')


if __name__ == '__main__':
    unittest.main()
