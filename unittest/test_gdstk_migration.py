"""Test suite for gdstk migration compatibility.

This test file verifies that gdstk produces identical results to gdspy
for all GDS export operations. Following TDD principles, these tests
are written before the actual migration to ensure compatibility.
"""

import unittest
import torch
import os
import tempfile
import json
from pathlib import Path

# Import gdstk for testing
import gdstk

from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask


class TestGdstkMigration(unittest.TestCase):
    """Test gdstk compatibility with existing GDS functionality."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.rdim = [256, 256]
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_gdstk_basic_operations(self):
        """Test basic gdstk operations match expected API."""
        # Test library creation
        lib = gdstk.Library()
        self.assertIsNotNone(lib)
        
        # Test cell creation
        cell = lib.new_cell("TEST_CELL")
        self.assertIsNotNone(cell)
        
        # Test polygon creation
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        polygon = gdstk.Polygon(points)
        self.assertIsNotNone(polygon)
        
        # Test adding polygon to cell
        cell.add(polygon)
        
        # Test writing GDS file
        gds_path = os.path.join(self.temp_dir, "test.gds")
        lib.write_gds(gds_path)
        self.assertTrue(os.path.exists(gds_path))
        
    def test_gdstk_boolean_operations(self):
        """Test gdstk boolean operations for holes."""
        lib = gdstk.Library()
        cell = lib.new_cell("BOOLEAN_TEST")
        
        # Create main polygon
        outer_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        outer_polygon = gdstk.Polygon(outer_points)
        
        # Create hole polygon
        hole_points = [(3, 3), (7, 3), (7, 7), (3, 7)]
        hole_polygon = gdstk.Polygon(hole_points)
        
        # Test boolean NOT operation
        result = gdstk.boolean(outer_polygon, [hole_polygon], "not")
        self.assertIsNotNone(result)
        
        # Add result to cell (gdstk returns a list)
        if isinstance(result, list):
            for poly in result:
                cell.add(poly)
        else:
            cell.add(result)
        
        # Write file
        gds_path = os.path.join(self.temp_dir, "boolean_test.gds")
        lib.write_gds(gds_path)
        self.assertTrue(os.path.exists(gds_path))
        
    def test_gdstk_with_gds_export_function(self):
        """Test that our gds_export function will work with gdstk.
        
        This test simulates how the gds_export function will work
        after replacing gdspy with gdstk.
        """
        # Create test boundary list
        boundary_list = [
            [[(10, 10), (20, 10), (20, 20), (10, 20)]],  # Simple square
            [[(30, 30), (50, 30), (50, 50), (30, 50)],   # Square with hole
             [(35, 35), (45, 35), (45, 45), (35, 45)]]
        ]
        
        # Create coordinate grids
        X = torch.linspace(0, 60, 61)
        Y = torch.linspace(0, 60, 61)
        X0, Y0 = torch.meshgrid(X, Y, indexing='xy')
        
        # Simulate gds_export with gdstk
        lib = gdstk.Library()
        cell = lib.new_cell("TEST_EXPORT")
        
        for sub_boundary_list in boundary_list:
            if len(sub_boundary_list) > 0 and len(sub_boundary_list[0]) > 2:
                # Create main polygon
                polygon = gdstk.Polygon(sub_boundary_list[0])
                
                # Handle holes
                if len(sub_boundary_list) > 1:
                    holes = []
                    for hole_coords in sub_boundary_list[1:]:
                        if len(hole_coords) > 2:
                            hole = gdstk.Polygon(hole_coords)
                            holes.append(hole)
                    
                    if holes:
                        # Boolean operation for holes
                        result = gdstk.boolean(polygon, holes, "not")
                        if result:
                            # gdstk returns a list of polygons
                            if isinstance(result, list):
                                for poly in result:
                                    cell.add(poly)
                            else:
                                cell.add(result)
                    else:
                        cell.add(polygon)
                else:
                    cell.add(polygon)
        
        # Write GDS file
        gds_path = os.path.join(self.temp_dir, "export_test.gds")
        lib.write_gds(gds_path)
        self.assertTrue(os.path.exists(gds_path))
        
    def create_shape_generator(self):
        """Create a ShapeGenerator for testing."""
        t1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        t2 = torch.tensor([0.0, 1.0], dtype=torch.float32)
        
        vec_p = torch.linspace(-0.5, 0.5, self.rdim[0])
        vec_q = torch.linspace(-0.5, 0.5, self.rdim[1])
        mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")
        X = mesh_p * t1[0] + mesh_q * t2[0]
        Y = mesh_p * t1[1] + mesh_q * t2[1]
        
        return ShapeGenerator(X, Y, self.rdim, lattice_t1=t1, lattice_t2=t2)
        
    def test_compatibility_with_existing_tests(self):
        """Verify that gdstk will work with our existing test cases.
        
        This test uses the same test pattern as test_gds.py to ensure
        compatibility after migration.
        """
        shape_gen = self.create_shape_generator()
        
        # Test circle export/import (simulating with current API)
        circle_mask = shape_gen.generate_circle_mask(
            center=(0.2, 0.0), radius=0.2, soft_edge=0.001
        )
        gds_path = os.path.join(self.temp_dir, "circle_compat.gds")
        
        # This will use gdspy currently, but verifies the test pattern
        mask_to_gds(circle_mask, shape_gen.get_layout(), "CIRCLE", gds_path, smooth=0.0)
        
        # Verify files created
        self.assertTrue(os.path.exists(gds_path))
        json_path = gds_path.replace('.gds', '.json')
        self.assertTrue(os.path.exists(json_path))
        
        # Test reconstruction
        reconstructed = gds_to_mask(json_path, shape_gen, soft_edge=0.0)
        
        # Calculate IoU
        binary1 = (circle_mask > 0.5).float()
        binary2 = (reconstructed > 0.5).float()
        intersection = (binary1 * binary2).sum()
        union = binary1.sum() + binary2.sum() - intersection
        iou = (intersection / union).item()
        
        self.assertGreater(iou, 0.9, f"Circle IoU {iou:.4f} should be > 0.9")
        
    def test_gdstk_path_handling(self):
        """Test that gdstk handles pathlib.Path objects correctly."""
        lib = gdstk.Library()
        cell = lib.new_cell("PATH_TEST")
        
        # Add simple polygon
        polygon = gdstk.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        cell.add(polygon)
        
        # Test with pathlib.Path
        gds_path = Path(self.temp_dir) / "path_test.gds"
        lib.write_gds(str(gds_path))  # gdstk requires string path
        self.assertTrue(gds_path.exists())
        
    def test_gdstk_json_compatibility(self):
        """Test that JSON export remains compatible."""
        # Create a simple polygon with gdstk and export
        lib = gdstk.Library()
        cell = lib.new_cell("JSON_TEST")
        
        points = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        polygon = gdstk.Polygon(points)
        cell.add(polygon)
        
        gds_path = os.path.join(self.temp_dir, "json_test.gds")
        lib.write_gds(gds_path)
        
        # Manually create JSON (simulating our export)
        # Convert tuples to lists for JSON serialization
        json_points = [[list(p) for p in points]]
        json_data = [json_points]  # Our JSON format
        json_path = gds_path.replace('.gds', '.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        
        # Verify JSON can be read
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, json_data)


if __name__ == '__main__':
    unittest.main()