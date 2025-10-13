"""Gdstk migration compatibility sanity checks.

These tests verify our public GDS export/import pipeline remains correct
across the planned gdspy→gdstk migration by round-tripping masks through
`mask_to_gds` and `gds_to_mask` with an IoU assertion.
"""

import unittest
import torch
import os
import tempfile
import json

from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask


class TestGdstkMigration(unittest.TestCase):
    """Ensure our GDS round-trip stays correct during migration."""

    def setUp(self):
        """Set up common test fixtures."""
        self.rdim = [256, 256]
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_shape_generator(self):
        """Create a ShapeGenerator for testing (Cartesian lattice)."""
        t1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        t2 = torch.tensor([0.0, 1.0], dtype=torch.float32)

        vec_p = torch.linspace(-0.5, 0.5, self.rdim[0])
        vec_q = torch.linspace(-0.5, 0.5, self.rdim[1])
        mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")
        X = mesh_p * t1[0] + mesh_q * t2[0]
        Y = mesh_p * t1[1] + mesh_q * t2[1]

        return ShapeGenerator(X, Y, self.rdim, lattice_t1=t1, lattice_t2=t2)

    def test_circle_roundtrip_iou(self):
        """Round-trip a circle mask and assert IoU > 0.9."""
        shape_gen = self.create_shape_generator()

        # Create circle mask
        circle_mask = shape_gen.generate_circle_mask(
            center=(0.2, 0.0), radius=0.2, soft_edge=0.001
        )
        gds_path = os.path.join(self.temp_dir, "circle_compat.gds")

        # Uses current backend (gdspy or gdstk) via our API
        mask_to_gds(circle_mask, shape_gen.get_layout(), "CIRCLE", gds_path, smooth=0.0)

        # Verify files created
        self.assertTrue(os.path.exists(gds_path))
        json_path = gds_path.replace('.gds', '.json')
        self.assertTrue(os.path.exists(json_path))

        # Reconstruct
        reconstructed = gds_to_mask(json_path, shape_gen, soft_edge=0.0)

        # Calculate IoU
        binary1 = (circle_mask > 0.5).float()
        binary2 = (reconstructed > 0.5).float()
        intersection = (binary1 * binary2).sum()
        union = binary1.sum() + binary2.sum() - intersection
        iou = (intersection / union).item()

        self.assertGreater(iou, 0.9, f"Circle IoU {iou:.4f} should be > 0.9")

