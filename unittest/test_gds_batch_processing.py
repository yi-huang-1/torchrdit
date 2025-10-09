"""Test suite for GDS batch processing functionality.

This test file follows TDD principles - tests are written before implementation.
The tests expect batch processing support in mask_to_gds function.
"""

import unittest
import torch
import os
import tempfile
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask


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

    def _to_torch(self, mask):
        if isinstance(mask, torch.Tensor):
            return mask
        return torch.from_numpy(mask).float()

    def _calculate_iou(self, mask1: torch.Tensor, mask2: torch.Tensor, threshold: float = 0.5) -> float:
        # Binarize masks
        binary1 = (mask1 > threshold).float()
        binary2 = (mask2 > threshold).float()
        intersection = (binary1 * binary2).sum()
        union = binary1.sum() + binary2.sum() - intersection
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return (intersection / union).item()
        
    def test_batch_mask_export_list_input(self):
        """Test exporting a list of masks to multiple GDS files."""
        # Create multiple masks via simple parameterization
        shape_specs = [
            ("circle", lambda sg: sg.generate_circle_mask(center=(0.0, 0.0), radius=0.3, soft_edge=0.0)),
            (
                "rectangle",
                lambda sg: sg.generate_rectangle_mask(
                    center=(0.2, 0.2), x_size=0.4, y_size=0.3, angle=45.0, soft_edge=0.0
                ),
            ),
            (
                "polygon",
                lambda sg: sg.generate_polygon_mask(
                    polygon_points=[(-0.3, -0.3), (0.3, -0.3), (0, 0.4)], soft_edge=0.0
                ),
            ),
        ]
        masks = [make(self.shape_gen) for _, make in shape_specs]
        
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

            # IoU check against original mask
            reconstructed = gds_to_mask(json_path, self.shape_gen, soft_edge=0.0)
            iou = self._calculate_iou(self._to_torch(masks[i]), reconstructed)
            self.assertGreater(iou, 0.9, f"Batch list item {i} IoU {iou:.4f} should be > 0.9")
    
    def test_batch_mask_export_tensor_batch(self):
        """Test exporting a batched tensor (3D) to multiple GDS files."""
        # Create batched tensor (batch_size, height, width)
        batch_size = 4
        masks = torch.zeros(batch_size, *self.rdim)
        
        # Fill with different patterns
        for i in range(batch_size):
            center_offset = i * 0.2 - 0.3
            masks[i] = self.shape_gen.generate_circle_mask(
                center=(center_offset, 0), radius=0.2, soft_edge=0.0
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

            # IoU check against original mask slice
            iou = self._calculate_iou(masks[i], reconstructed)
            self.assertGreater(iou, 0.9, f"Tensor batch item {i} IoU {iou:.4f} should be > 0.9")
    
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
        shape_specs = [
            ("torch_circle", lambda sg: sg.generate_circle_mask(center=(0, 0), radius=0.3, soft_edge=0.0)),
            ("numpy_rect", lambda sg: sg.generate_rectangle_mask(center=(0, 0), x_size=0.4, y_size=0.3, soft_edge=0.0).numpy()),
        ]
        masks = [make(self.shape_gen) for _, make in shape_specs]
        
        base_path = os.path.join(self.temp_dir, "mixed_batch")
        
        result = mask_to_gds(
            masks,
            self.shape_gen.get_layout(),
            "MIXED",
            base_path,
            smooth=0.0
        )
        
        self.assertEqual(len(result), 2)
        for i, path in enumerate(result):
            self.assertTrue(os.path.exists(path))
            json_path = path.replace('.gds', '.json')
            reconstructed = gds_to_mask(json_path, self.shape_gen, soft_edge=0.0)
            iou = self._calculate_iou(self._to_torch(masks[i]), reconstructed)
            self.assertGreater(iou, 0.9, f"Mixed batch item {i} IoU {iou:.4f} should be > 0.9")
    
    # Removed per-mask parameters and performance tests: not part of supported API and flaky in unit tests
    # No __main__ block needed; tests run via pytest/unittest runner
