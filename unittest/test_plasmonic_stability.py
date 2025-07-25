"""Test suite for plasmonic material stabilization handling.

This module tests the automatic stabilization of materials near plasmon resonance
conditions (ε ≈ -1) to prevent matrix singularities in electromagnetic simulations.
"""

import torch
import numpy as np
from torchrdit.materials import MaterialClass


class TestPlasmonicStabilization:
    """Test plasmonic material stabilization functionality."""
    
    def test_exact_plasmon_resonance_stabilization(self):
        """Test that exact ε = -1.0 + 0j gets stabilized with minimal losses."""
        # Create material at exact plasmon resonance
        material = MaterialClass(
            name="test_plasmon",
            permittivity=-1.0 + 0j,
            permeability=1.0
        )
        
        # Get permittivity (should be stabilized)
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # Check that losses were added (negative imaginary part)
        assert epsilon.imag < 0, "Material should have negative imaginary part (losses)"
        assert epsilon.imag <= -1e-5, f"Minimal losses should be at least -1e-5, got {epsilon.imag}"
        assert torch.abs(epsilon.real + 1.0) < 0.001, f"Real part should remain close to -1.0, got {epsilon.real}"
    
    def test_near_plasmon_resonance_stabilization(self):
        """Test materials near plasmon resonance get stabilized."""
        test_cases = [
            -0.99 + 0j,    # Just above resonance
            -1.01 + 0j,    # Just below resonance
            -0.995 + 0j,   # Very close to resonance
        ]
        
        for eps_value in test_cases:
            material = MaterialClass(
                name=f"test_near_plasmon_{eps_value.real}",
                permittivity=eps_value,
                permeability=1.0
            )
            
            wavelengths = np.array([1.55])
            epsilon = material.get_permittivity(wavelengths)
            
            # Materials within threshold should be stabilized
            if abs(eps_value.real + 1.0) < 0.01:  # Default threshold
                assert epsilon.imag <= -1e-5, \
                    f"Material at ε={eps_value} should have losses, got {epsilon.imag}"
    
    def test_already_lossy_material_unchanged(self):
        """Test that materials with sufficient losses are not modified."""
        # Material already has significant losses
        original_epsilon = -1.0 - 0.1j
        material = MaterialClass(
            name="lossy_metal",
            permittivity=original_epsilon,
            permeability=1.0
        )
        
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # Should remain unchanged
        assert torch.allclose(epsilon, torch.tensor(original_epsilon)), \
            f"Lossy material should not be modified: {original_epsilon} -> {epsilon}"
    
    def test_material_with_small_losses_enhanced(self):
        """Test materials with insufficient losses get enhanced."""
        # Material has some losses but less than minimum
        original_epsilon = -1.0 - 1e-6j
        material = MaterialClass(
            name="weakly_lossy",
            permittivity=original_epsilon,
            permeability=1.0
        )
        
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # Losses should be increased to minimum
        assert epsilon.imag <= -1e-5, \
            f"Weak losses should be increased to at least -1e-5, got {epsilon.imag}"
        assert torch.abs(epsilon.real + 1.0) < 0.001, \
            f"Real part should remain close to -1.0, got {epsilon.real}"
    
    def test_non_resonant_materials_unchanged(self):
        """Test that materials far from resonance are not modified."""
        test_cases = [
            1.0 + 0j,      # Dielectric
            -10.0 - 1j,    # Metal far from resonance
            2.25 + 0j,     # Silicon
            -130 - 10j,    # Gold at optical frequencies
        ]
        
        for eps_value in test_cases:
            material = MaterialClass(
                name=f"non_resonant_{eps_value.real}",
                permittivity=eps_value,
                permeability=1.0
            )
            
            wavelengths = np.array([1.55])
            epsilon = material.get_permittivity(wavelengths)
            
            # Should remain unchanged
            assert torch.allclose(epsilon, torch.tensor(eps_value), atol=1e-10), \
                f"Non-resonant material should not be modified: {eps_value} -> {epsilon}"
    
    def test_dispersive_material_stabilization(self):
        """Test that dispersive materials are stabilized at resonant wavelengths."""
        # This test verifies the stabilization logic works with dispersive materials
        # We'll simulate the behavior by directly testing the function
        # since creating actual dispersive material files is complex
        
        # Import the function
        from torchrdit.materials import handle_plasmonic_materials
        
        # Simulate dispersive material values that cross plasmon resonance
        test_wavelengths = [1.0, 1.5, 2.0]
        test_epsilons = [-2.0 + 0j, -1.0 + 0j, -0.5 + 0j]  # Crosses ε = -1
        
        # Test each point
        for wl, eps in zip(test_wavelengths, test_epsilons):
            eps_tensor = torch.tensor(eps, dtype=torch.complex64)
            stabilized = handle_plasmonic_materials(eps_tensor, wavelength=wl)
            
            # At resonance (ε = -1.0), should be stabilized
            if abs(eps.real + 1.0) < 0.01:
                assert stabilized.imag <= -1e-5, \
                    f"Dispersive material at λ={wl}, ε={eps} should be stabilized"
            else:
                # Non-resonant points should pass through unchanged
                assert torch.allclose(stabilized, eps_tensor), \
                    f"Non-resonant ε={eps} should not be modified"
    
    def test_tensor_input_stabilization(self):
        """Test stabilization works with tensor inputs."""
        # Create material with tensor permittivity
        epsilon_tensor = torch.tensor(-1.0 + 0j, dtype=torch.complex64)
        material = MaterialClass(
            name="tensor_plasmon",
            permittivity=epsilon_tensor,
            permeability=1.0
        )
        
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # Should be stabilized
        assert epsilon.imag <= -1e-5, \
            f"Tensor material should have losses, got {epsilon.imag}"
    
    def test_multiple_wavelength_stabilization(self):
        """Test stabilization works for multiple wavelengths."""
        material = MaterialClass(
            name="plasmon_multi",
            permittivity=-1.0 + 0j,
            permeability=1.0
        )
        
        # Multiple wavelengths
        wavelengths = np.array([1.31, 1.55, 1.65])
        epsilon = material.get_permittivity(wavelengths)
        
        # All wavelengths should have same stabilization
        assert epsilon.shape == torch.Size([]), "Non-dispersive should return scalar"
        assert epsilon.imag <= -1e-5, "Should have losses at all wavelengths"
    
    def test_gradient_preservation(self):
        """Test that stabilization preserves gradients for optimization."""
        # Create material with gradient tracking
        epsilon_real = torch.tensor(-1.0, requires_grad=True)
        epsilon_imag = torch.tensor(0.0, requires_grad=True)
        epsilon_complex = torch.complex(epsilon_real, epsilon_imag)
        
        material = MaterialClass(
            name="gradient_test",
            permittivity=epsilon_complex,
            permeability=1.0
        )
        
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # Should maintain gradient capability
        assert isinstance(epsilon, torch.Tensor), "Output should be torch.Tensor"
        # Note: actual gradient test would require the stabilization function
        # to be implemented with differentiable operations
    
    def test_threshold_configuration(self):
        """Test that the detection threshold can be configured."""
        # Test with custom threshold
        material = MaterialClass(
            name="custom_threshold",
            permittivity=-0.95 + 0j,  # Just outside default threshold
            permeability=1.0,
            stabilization_params={'threshold': 0.1}  # Wider threshold
        )
        
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # With wider threshold, this should be stabilized
        assert epsilon.imag <= -1e-5, \
            f"Material at ε=-0.95 should be stabilized with threshold=0.1"
    
    def test_minimum_loss_configuration(self):
        """Test that the minimum loss value can be configured."""
        # Test with custom minimum loss
        material = MaterialClass(
            name="custom_min_loss",
            permittivity=-1.0 + 0j,  # Exact resonance
            permeability=1.0,
            stabilization_params={'min_loss': 1e-4}  # Larger minimum loss
        )
        
        wavelengths = np.array([1.55])
        epsilon = material.get_permittivity(wavelengths)
        
        # Should have at least the custom minimum loss
        assert epsilon.imag <= -1e-4, \
            f"Material should have at least -1e-4 loss, got {epsilon.imag}"
        assert epsilon.imag >= -1e-3, \
            "Loss should be reasonable"


class TestPlasmonicStabilizationHelpers:
    """Test helper functions for plasmonic stabilization."""
    
    def test_handle_plasmonic_materials_function(self):
        """Test the standalone stabilization function."""
        # Import the function (will fail until implemented)
        from torchrdit.materials import handle_plasmonic_materials
        
        # Test exact resonance
        epsilon = torch.tensor(-1.0 + 0j)
        stabilized = handle_plasmonic_materials(epsilon)
        assert stabilized.imag <= -1e-5
        
        # Test with custom parameters
        epsilon = torch.tensor(-1.0 + 0j)
        stabilized = handle_plasmonic_materials(
            epsilon, 
            min_loss=1e-4,
            threshold=0.05
        )
        assert stabilized.imag <= -1e-4
    
    def test_stabilize_epsilon_batch(self):
        """Test batch stabilization function."""
        # Import the function (will fail until implemented)
        from torchrdit.materials import stabilize_epsilon_batch
        
        # Batch of epsilon values
        epsilon_batch = torch.tensor([
            -1.0 + 0j,      # Needs stabilization
            -0.99 + 0j,     # Needs stabilization
            -10.0 - 1j,     # Already stable
            1.0 + 0j,       # Dielectric, no change
        ])
        
        stabilized = stabilize_epsilon_batch(epsilon_batch)
        
        # Check first two are stabilized
        assert stabilized[0].imag <= -1e-5
        assert stabilized[1].imag <= -1e-5
        # Check last two unchanged
        assert torch.allclose(stabilized[2], epsilon_batch[2])
        assert torch.allclose(stabilized[3], epsilon_batch[3])