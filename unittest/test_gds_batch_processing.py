"""Test suite for GDS batch processing functionality.

This test file follows TDD principles - tests are written before implementation.
The tests expect batch processing support in mask_to_gds function.
"""

import unittest
import torch
import numpy as np
import os
import tempfile
import json
from typing import List
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask, load_gds_vertices


class TestGDSBatchProcessing(unittest.TestCase):
    """Tests for batch processing in GDS module."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.rdim = [128, 128]
        self.temp_dir = tempfile.mkdtemp()
        
        # Create shape generator
        X, Y = torch.meshgrid(
            torch.linspace(-1, 1, self.rdim[0]),
            torch.linspace(-1, 1, self.rdim[1]),
            indexing='xy'
        )
        self.shape_gen = ShapeGenerator(X, Y, self.rdim)
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_batch_mask_export_list_input(self):
        """Test exporting a list of masks to multiple GDS files."""
        # Create multiple masks
        masks = []
        
        # Circle mask
        circle_mask = self.shape_gen.generate_circle_mask(
            center=(0.0, 0.0), radius=0.3, soft_edge=0.001
        )
        masks.append(circle_mask)
        
        # Rectangle mask
        rect_mask = self.shape_gen.generate_rectangle_mask(
            center=(0.2, 0.2), width=0.4, height=0.3, angle=45.0, soft_edge=0.001
        )
        masks.append(rect_mask)
        
        # Polygon mask
        poly_mask = self.shape_gen.generate_polygon_mask(
            polygon_points=[(-0.3, -0.3), (0.3, -0.3), (0, 0.4)],
            soft_edge=0.001
        )
        masks.append(poly_mask)
        
        # Export batch
        base_path = os.path.join(self.temp_dir, "batch_test")
        cell_name = "BATCH"
        
        # Should return list of file paths
        result = mask_to_gds(
            masks, 
            self.shape_gen.get_layout(),
            cell_name,
            base_path,
            smooth=0.0
        )
        
        # Verify return type
        self.assertIsInstance(result, list, "Batch export should return list of file paths")
        self.assertEqual(len(result), 3, "Should return 3 file paths for 3 masks")
        
        # Verify files were created
        for i, file_path in enumerate(result):
            self.assertTrue(os.path.exists(file_path), f"GDS file {i} should exist")
            
            # Check JSON file too
            json_path = file_path.replace('.gds', '.json')
            self.assertTrue(os.path.exists(json_path), f"JSON file {i} should exist")
            
            # Verify file naming convention
            expected_name = f"{base_path}_{i}.gds"
            self.assertEqual(file_path, expected_name, 
                           f"File {i} should follow naming convention")
    
    def test_batch_mask_export_tensor_batch(self):
        """Test exporting a batched tensor (3D) to multiple GDS files."""
        # Create batched tensor (batch_size, height, width)
        batch_size = 4
        masks = torch.zeros(batch_size, *self.rdim)
        
        # Fill with different patterns
        for i in range(batch_size):
            center_offset = i * 0.2 - 0.3
            masks[i] = self.shape_gen.generate_circle_mask(
                center=(center_offset, 0), radius=0.2, soft_edge=0.001
            )
        
        # Export batch
        base_path = os.path.join(self.temp_dir, "tensor_batch")
        
        # Should handle 3D tensor input
        result = mask_to_gds(
            masks,
            self.shape_gen.get_layout(),
            "TENSOR_BATCH",
            base_path,
            smooth=0.0
        )
        
        self.assertIsInstance(result, list, "Should return list for batched tensor")
        self.assertEqual(len(result), batch_size, f"Should return {batch_size} file paths")
        
        # Verify all files exist
        for i in range(batch_size):
            gds_path = result[i]
            self.assertTrue(os.path.exists(gds_path))
            
            # Verify reconstruction
            json_path = gds_path.replace('.gds', '.json')
            reconstructed = gds_to_mask(json_path, self.shape_gen, soft_edge=0.0)
            
            # Check shape matches
            self.assertEqual(reconstructed.shape, tuple(self.rdim))
    
    def test_single_mask_compatibility(self):
        """Test that single mask export still works (backward compatibility)."""
        # Single mask
        mask = self.shape_gen.generate_circle_mask(
            center=(0, 0), radius=0.5, soft_edge=0.001
        )
        
        # Export single mask
        gds_path = os.path.join(self.temp_dir, "single.gds")
        
        result = mask_to_gds(
            mask,
            self.shape_gen.get_layout(),
            "SINGLE",
            gds_path,
            smooth=0.0
        )
        
        # Should return single string path, not list
        self.assertIsInstance(result, str, "Single mask should return string path")
        self.assertEqual(result, gds_path, "Should return the specified path")
        self.assertTrue(os.path.exists(result))
    
    def test_empty_batch_handling(self):
        """Test handling of empty batch."""
        # Empty list
        masks = []
        
        base_path = os.path.join(self.temp_dir, "empty_batch")
        
        result = mask_to_gds(
            masks,
            self.shape_gen.get_layout(),
            "EMPTY",
            base_path,
            smooth=0.0
        )
        
        # Should return empty list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0, "Empty batch should return empty list")
    
    def test_mixed_types_in_batch(self):
        """Test batch with mixed tensor and numpy arrays."""
        masks = []
        
        # Torch tensor
        torch_mask = self.shape_gen.generate_circle_mask(
            center=(0, 0), radius=0.3, soft_edge=0.001
        )
        masks.append(torch_mask)
        
        # Numpy array
        numpy_mask = self.shape_gen.generate_rectangle_mask(
            center=(0, 0), width=0.4, height=0.3, soft_edge=0.001
        ).numpy()
        masks.append(numpy_mask)
        
        base_path = os.path.join(self.temp_dir, "mixed_batch")
        
        result = mask_to_gds(
            masks,
            self.shape_gen.get_layout(),
            "MIXED",
            base_path,
            smooth=0.0
        )
        
        self.assertEqual(len(result), 2)
        for path in result:
            self.assertTrue(os.path.exists(path))
    
    def test_batch_with_different_parameters(self):
        """Test batch export with per-mask parameters."""
        masks = []
        params = []
        
        # Different smoothing for each mask
        for i in range(3):
            mask = self.shape_gen.generate_circle_mask(
                center=(i * 0.3 - 0.3, 0), radius=0.2, soft_edge=0.001
            )
            masks.append(mask)
            params.append({'smooth': 0.001 * (i + 1)})
        
        base_path = os.path.join(self.temp_dir, "param_batch")
        
        # If per-mask parameters are supported
        try:
            result = mask_to_gds(
                masks,
                self.shape_gen.get_layout(),
                "PARAM_BATCH",
                base_path,
                batch_params=params  # Optional: per-mask parameters
            )
            
            self.assertEqual(len(result), 3)
        except TypeError:
            # If not supported, at least verify uniform parameters work
            result = mask_to_gds(
                masks,
                self.shape_gen.get_layout(),
                "PARAM_BATCH",
                base_path,
                smooth=0.001
            )
            
            self.assertEqual(len(result), 3)
    
    def test_large_batch_performance(self):
        """Test performance with larger batch sizes."""
        import time
        
        # Create batch of 10 masks
        batch_size = 10
        masks = []
        
        for i in range(batch_size):
            angle = i * 36  # 360 / 10
            mask = self.shape_gen.generate_rectangle_mask(
                center=(0, 0), width=0.3, height=0.4, angle=angle, soft_edge=0.001
            )
            masks.append(mask)
        
        base_path = os.path.join(self.temp_dir, "large_batch")
        
        # Time the batch export
        start = time.time()
        
        result = mask_to_gds(
            masks,
            self.shape_gen.get_layout(),
            "LARGE_BATCH",
            base_path,
            smooth=0.0
        )
        
        end = time.time()
        batch_time = end - start
        
        # Compare with sequential export
        start = time.time()
        
        for i, mask in enumerate(masks):
            mask_to_gds(
                mask,
                self.shape_gen.get_layout(),
                f"SEQ_{i}",
                os.path.join(self.temp_dir, f"seq_{i}.gds"),
                smooth=0.0
            )
            
        end = time.time()
        seq_time = end - start
        
        # Batch should not be significantly slower than sequential
        # (ideally would be faster with parallel processing)
        self.assertLess(batch_time, seq_time * 1.5,
                       f"Batch processing should be efficient: {batch_time:.2f}s vs {seq_time:.2f}s")
        
        # Verify all files created
        self.assertEqual(len(result), batch_size)
        for path in result:
            self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()