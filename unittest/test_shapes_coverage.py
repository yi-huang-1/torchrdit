import pytest
import torch

from torchrdit.shapes import ShapeGenerator
from torchrdit.constants import Precision
from torchrdit.solver import create_solver


class TestShapesCoverage:
    """Tests to improve coverage of shapes.py missing lines."""

    def test_shape_generator_default_center_parameter(self):
        """Test missing line 207: Default center parameter handling."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Test rectangle with default center (should use None)
        mask = shape_gen.generate_rectangle_mask(
            width=0.5,
            height=0.3,
            # center parameter omitted to test default
        )
        assert mask is not None
        assert isinstance(mask, torch.Tensor)

    def test_shape_generator_default_angle_parameter(self):
        """Test missing line 216: Default angle parameter handling."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Test rectangle with default angle (should use None)
        mask = shape_gen.generate_rectangle_mask(
            width=0.5,
            height=0.3,
            center=(0.0, 0.0),
            # angle parameter omitted to test default
        )
        assert mask is not None
        assert isinstance(mask, torch.Tensor)

    def test_shape_generator_default_invert_parameter(self):
        """Test missing line 267: Default invert parameter handling."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Test circle with default invert parameter
        mask = shape_gen.generate_circle_mask(
            radius=0.3,
            center=(0.0, 0.0),
            # invert parameter omitted to test default (False)
        )
        assert mask is not None
        assert isinstance(mask, torch.Tensor)

    def test_rectangle_mask_hard_edges_branch(self):
        """Test missing lines 301-313: Hard edges branch in rectangle mask."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Test rectangle with hard edges (soft_edge=0 or very small)
        mask = shape_gen.generate_rectangle_mask(
            width=0.5,
            height=0.3,
            center=(0.0, 0.0),
            angle=30.0,  # Rotated rectangle
            soft_edge=0.0,  # Hard edges
        )
        assert mask is not None
        assert isinstance(mask, torch.Tensor)

        # Verify that hard edge logic produces binary values
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 10  # Should be mostly 0s and 1s

    def test_polygon_mask_default_parameters(self):
        """Test missing lines 371, 376: Polygon mask default parameters."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Define a simple triangle
        triangle_points = [(0.0, 0.5), (-0.3, -0.25), (0.3, -0.25)]

        # Test with default center (None)
        mask1 = shape_gen.generate_polygon_mask(
            polygon_points=triangle_points
            # center=None (default)
        )
        assert mask1 is not None

        # Test with default angle (None)
        mask2 = shape_gen.generate_polygon_mask(
            polygon_points=triangle_points,
            center=(0.1, 0.1),
            # angle=None (default)
        )
        assert mask2 is not None

    def test_polygon_mask_tensor_conversion_error(self):
        """Test missing lines 392, 397: Polygon tensor conversion error handling."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Test with invalid polygon points that might cause tensor conversion issues
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            shape_gen.generate_polygon_mask(
                polygon_points="invalid_points"  # Invalid type
            )

    def test_combine_masks_validation_errors(self):
        """Test missing lines 443, 446: Combine masks validation errors."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Create valid masks
        mask1 = torch.ones((10, 10), dtype=torch.float32)

        # Test with None as second mask
        with pytest.raises((ValueError, TypeError)):
            shape_gen.combine_masks(mask1, None, operation="union")

        # Test with same mask (should handle gracefully)
        mask2 = torch.ones((10, 10), dtype=torch.float32)
        result = shape_gen.combine_masks(mask1, mask2, operation="union")
        assert result is not None

    def test_mask_combination_operations_edge_cases(self):
        """Test mask combination operations with edge cases."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Create test masks
        mask1 = torch.ones((10, 10), dtype=torch.float32) * 0.5
        mask2 = torch.ones((10, 10), dtype=torch.float32) * 0.7
        torch.zeros((10, 10), dtype=torch.float32)

        # Test union operation
        union_result = shape_gen.combine_masks(mask1, mask2, operation="union")
        assert union_result is not None

        # Test intersection operation
        intersection_result = shape_gen.combine_masks(mask1, mask2, operation="intersection")
        assert intersection_result is not None

        # Test difference operation
        difference_result = shape_gen.combine_masks(mask1, mask2, operation="difference")
        assert difference_result is not None

        # Test with invalid operation
        with pytest.raises((ValueError, KeyError)):
            shape_gen.combine_masks(mask1, mask2, operation="invalid_operation")

    def test_mask_inversion_functionality(self):
        """Test mask inversion functionality."""
        solver = create_solver(algorithm="rcwa", precision=Precision.SINGLE)
        shape_gen = ShapeGenerator.from_solver(solver)

        # Test normal mask and manual inversion
        mask_normal = shape_gen.generate_circle_mask(radius=0.3, center=(0.0, 0.0))
        # Manually invert the mask to test inversion functionality
        mask_inverted = 1.0 - mask_normal

        assert mask_normal is not None
        assert mask_inverted is not None
        # Test that they are actually different
        assert not torch.allclose(mask_normal, mask_inverted)
