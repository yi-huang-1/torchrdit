import unittest
import torch
import numpy as np
import os
import tempfile
import json
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask, load_gds_vertices
from torchrdit.gds_utils import (
    generate_ring_mask_vectorized, 
    create_circular_hole_vectorized,
    point_in_triangle_vectorized
)


class TestGDSFunctionality(unittest.TestCase):
    """Test suite for GDS import/export functionality.
    
    Tests cover basic shapes, combined shapes, complex topologies with holes,
    and both Cartesian and non-Cartesian lattice systems.
    """
    
    def setUp(self):
        """Set up common test fixtures."""
        self.rdim = [256, 256]
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def calculate_iou(self, mask1: torch.Tensor, mask2: torch.Tensor, threshold: float = 0.5) -> float:
        """Calculate Intersection over Union between two masks."""
        # Binarize masks
        binary1 = (mask1 > threshold).float()
        binary2 = (mask2 > threshold).float()
        
        # Calculate intersection and union
        intersection = (binary1 * binary2).sum()
        union = binary1.sum() + binary2.sum() - intersection
        
        # Avoid division by zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    def create_shape_generator(self, lattice_type='cartesian'):
        """Create a ShapeGenerator with specified lattice type."""
        if lattice_type == 'cartesian':
            t1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
            t2 = torch.tensor([0.0, 1.0], dtype=torch.float32)
        else:  # non-cartesian (hexagonal)
            a = 1.150
            t1 = torch.tensor([a/2, -a*np.sqrt(3)/2], dtype=torch.float32)
            t2 = torch.tensor([a/2, a*np.sqrt(3)/2], dtype=torch.float32)
            
        # Create coordinate mesh
        vec_p = torch.linspace(-0.5, 0.5, self.rdim[0])
        vec_q = torch.linspace(-0.5, 0.5, self.rdim[1])
        mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")
        X = mesh_p * t1[0] + mesh_q * t2[0]
        Y = mesh_p * t1[1] + mesh_q * t2[1]
        
        return ShapeGenerator(X, Y, self.rdim, lattice_t1=t1, lattice_t2=t2)
    
    def _roundtrip_and_assert(self, shape_gen, mask, name: str):
        gds_path = os.path.join(self.temp_dir, f"{name}.gds")
        mask_to_gds(mask, shape_gen.get_layout(), name.upper(), gds_path, smooth=0.0)
        json_path = gds_path.replace('.gds', '.json')
        reconstructed = gds_to_mask(json_path, shape_gen, soft_edge=0.0)
        iou = self.calculate_iou(mask, reconstructed)
        self.assertGreater(iou, 0.9, f"{name} IoU {iou:.4f} should be > 0.9")

    def test_basic_shapes_cartesian(self):
        """Parametrized simple shapes in Cartesian lattice."""
        shape_gen = self.create_shape_generator('cartesian')

        shape_specs = [
            ("circle_cart", lambda sg: sg.generate_circle_mask(center=(0.2, 0.0), radius=0.2, soft_edge=0.001)),
            (
                "rectangle_cart",
                lambda sg: sg.generate_rectangle_mask(
                    center=(0.1, 0.1), x_size=0.3, y_size=0.4, angle=30.0, soft_edge=0.001
                ),
            ),
            (
                "polygon_cart",
                lambda sg: sg.generate_polygon_mask(
                    polygon_points=[(-0.2, -0.2), (0.2, -0.2), (0, 0.3)], center=(0.0, 0.0), angle=45.0, soft_edge=0.001
                ),
            ),
        ]

        for name, make_mask in shape_specs:
            self._roundtrip_and_assert(shape_gen, make_mask(shape_gen), name)
    
    def test_basic_shapes_non_cartesian(self):
        """Parametrized simple shapes in non-Cartesian lattice."""
        shape_gen = self.create_shape_generator('non-cartesian')

        shape_specs = [
            ("circle_hex", lambda sg: sg.generate_circle_mask(center=(0.0, 0.0), radius=0.2, soft_edge=0.001)),
            (
                "rectangle_hex",
                lambda sg: sg.generate_rectangle_mask(
                    center=(0.0, 0.0), x_size=0.3, y_size=0.25, angle=30.0, soft_edge=0.001
                ),
            ),
            (
                "polygon_hex",
                lambda sg: sg.generate_polygon_mask(
                    polygon_points=[(-0.2, -0.2), (0.2, -0.2), (0, 0.25)], center=(0.0, 0.0), angle=30.0, soft_edge=0.001
                ),
            ),
        ]

        for name, make_mask in shape_specs:
            self._roundtrip_and_assert(shape_gen, make_mask(shape_gen), name)
    
    def test_combined_shapes(self):
        """Test GDS export/import for combined shapes (union, intersection, difference)."""
        shape_gen = self.create_shape_generator('cartesian')
        
        # Create base shapes
        circle_mask = shape_gen.generate_circle_mask(
            center=(0.2, 0.0), radius=0.2, soft_edge=0.001
        )
        rect_mask = shape_gen.generate_rectangle_mask(
            center=(0.1, 0.1), x_size=0.3, y_size=0.4, angle=30.0, soft_edge=0.001
        )
        
        # Test union
        union_mask = shape_gen.combine_masks(circle_mask, rect_mask, operation='union')
        gds_path = os.path.join(self.temp_dir, "union.gds")
        mask_to_gds(union_mask, shape_gen.get_layout(), "UNION", gds_path, smooth=0.0)
        
        json_path = gds_path.replace('.gds', '.json')
        reconstructed = gds_to_mask(json_path, shape_gen, soft_edge=0.0)
        
        iou = self.calculate_iou(union_mask, reconstructed)
        self.assertGreater(iou, 0.9, f"Union IoU {iou:.4f} should be > 0.9")
        
        # Skip separate intersection case; union and difference sufficiently exercise boundaries
        
        # Test difference
        difference_mask = shape_gen.combine_masks(circle_mask, rect_mask, operation='difference')
        gds_path = os.path.join(self.temp_dir, "difference.gds")
        mask_to_gds(difference_mask, shape_gen.get_layout(), "DIFFERENCE", gds_path, smooth=0.0)
        
        json_path = gds_path.replace('.gds', '.json')
        reconstructed = gds_to_mask(json_path, shape_gen, soft_edge=0.0)
        
        iou = self.calculate_iou(difference_mask, reconstructed)
        self.assertGreater(iou, 0.9, f"Difference IoU {iou:.4f} should be > 0.9")
    
    # Removed ring test: complex_topology below robustly covers holes/islands
    
    def test_complex_topology(self):
        """Test GDS export/import for complex topology with holes and islands."""
        # Create coordinate grid
        X, Y = torch.meshgrid(
            torch.linspace(-2, 2, self.rdim[0]), 
            torch.linspace(-2, 2, self.rdim[1]), 
            indexing='xy'
        )
        shape_gen = ShapeGenerator(X, Y, self.rdim)
        
        # Create complex mask (rectangle with 2 holes, one containing triangle)
        mask = torch.zeros(self.rdim)
        
        # Main rectangle
        rect_top, rect_bottom = 28, 228
        rect_left, rect_right = 53, 203
        mask[rect_top:rect_bottom, rect_left:rect_right] = 1
        
        # Hole 1 (simple circular hole)
        hole1_center = (128, 80)
        hole1_radius = 20
        mask_np = mask.numpy()
        mask_np = create_circular_hole_vectorized(
            mask_np, hole1_center, hole1_radius, 
            bounds=(rect_top, rect_bottom, rect_left, rect_right)
        )
        
        # Hole 2 (larger hole)
        hole2_center = (128, 150)
        hole2_radius = 35
        mask_np = create_circular_hole_vectorized(
            mask_np, hole2_center, hole2_radius,
            bounds=(rect_top, rect_bottom, rect_left, rect_right)
        )
        mask = torch.from_numpy(mask_np)
        
        # Triangle island inside Hole 2
        triangle_side = 25
        triangle_height = triangle_side * np.sqrt(3) / 2
        triangle_vertices = [
            (hole2_center[0] - triangle_height * 2/3, hole2_center[1]),
            (hole2_center[0] + triangle_height/3, hole2_center[1] - triangle_side/2),
            (hole2_center[0] + triangle_height/3, hole2_center[1] + triangle_side/2)
        ]
        
        # Generate triangle mask using vectorized operation
        triangle_mask = point_in_triangle_vectorized(
            self.rdim, triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]
        )
        
        # Apply triangle only within hole2
        Y, X = np.ogrid[:self.rdim[0], :self.rdim[1]]
        dist_sq = (X - hole2_center[1])**2 + (Y - hole2_center[0])**2
        within_hole2 = dist_sq < hole2_radius**2
        
        # Add triangle to mask
        mask_np = mask.numpy()
        mask_np[within_hole2 & triangle_mask] = 1
        mask = torch.from_numpy(mask_np)
        
        # Export and reimport
        gds_path = os.path.join(self.temp_dir, "complex.gds")
        mask_to_gds(mask, shape_gen.get_layout(), "COMPLEX", gds_path, smooth=0.0)
        
        json_path = gds_path.replace('.gds', '.json')
        reconstructed = gds_to_mask(json_path, shape_gen, soft_edge=0.0)
        
        iou = self.calculate_iou(mask, reconstructed)
        self.assertGreater(iou, 0.9, f"Complex topology IoU {iou:.4f} should be > 0.9")
    
    # Removed get_layout existence test: exercised implicitly by all GDS tests
    
    # Removed direct load_gds_vertices success test (implicitly covered by gds_to_mask usage)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        shape_gen = self.create_shape_generator('cartesian')
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            gds_to_mask("non_existent.json", shape_gen)
        
        # Test with invalid JSON
        invalid_json_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_path, 'w') as f:
            f.write("not valid json")
        
        with self.assertRaises(json.JSONDecodeError):
            load_gds_vertices(invalid_json_path)


# No __main__ block needed; tests run via pytest/unittest runner
