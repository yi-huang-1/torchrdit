import unittest
import torch
import numpy as np

from torchrdit.layers import LayerManager
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm
from torchrdit.shapes import ShapeGenerator


class TestLayerDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in layers.py.
    
    This test suite ensures that the examples provided in the docstrings of
    layers.py work as expected. It tests examples for:
    1. Layer and its subclasses
    2. LayerBuilder and its subclasses
    3. LayerDirector
    4. LayerManager
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create standard lattice grid
        self.rdim = [256, 256]
        self.kdim = [5, 5]
        
        # Create materials for tests
        self.air = create_material(name="air", permittivity=1.0)
        self.silicon = create_material(name="silicon", permittivity=11.7)
        self.sio2 = create_material(name="SiO2", permittivity=2.25)
        
        # Create a vector grid for layer manager tests
        self.vec_p = torch.linspace(-0.5, 0.5, self.rdim[0])
        self.vec_q = torch.linspace(-0.5, 0.5, self.rdim[1])
        self.lattice_t1 = torch.tensor([1.0, 0.0])
        self.lattice_t2 = torch.tensor([0.0, 1.0])
    
    def test_module_examples(self):
        """Test the module-level examples shown in the docstring."""
        # Example from module docstring - using direct imports instead of from torchrdit
        # Create a solver
        solver = create_solver(algorithm=Algorithm.RCWA)
        
        # Add a material to the solver
        silicon = create_material(name="silicon", permittivity=11.7)
        solver.add_materials([silicon])
        
        # Add a homogeneous layer
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
        
        # Add a patterned layer
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.3), 
                      is_homogeneous=False)
        
        # Create a shape generator
        shape_gen = ShapeGenerator.from_solver(solver)
        
        # Pattern the layer with a circle
        circle_mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.25)
        solver.update_er_with_mask(circle_mask, layer_index=1)
        
        # Verify the layers were created correctly
        self.assertEqual(len(solver.layers), 2)
        self.assertAlmostEqual(solver.layers[0].thickness, 0.2, places=6)
        self.assertEqual(solver.layers[0].material_name, "silicon")
        self.assertTrue(solver.layers[0].is_homogeneous)
        
        self.assertAlmostEqual(solver.layers[1].thickness, 0.3, places=6)
        self.assertEqual(solver.layers[1].material_name, "silicon")
        self.assertFalse(solver.layers[1].is_homogeneous)
        
        # Additional assertion: mask application updates ermat to a non-uniform distribution
        ermat = solver.layer_manager.layers[1].ermat
        self.assertIsInstance(ermat, torch.Tensor)
        self.assertTrue(torch.isfinite(ermat).all())
        # Non-uniform: min and max magnitudes should differ for complex tensors
        mags = torch.abs(ermat)
        self.assertNotEqual(torch.min(mags).item(), torch.max(mags).item())
    
    # Removed unit tests duplicating behavior already covered in test_layer.py
    # (properties, builders, director, manager operations, toeplitz). This file now
    # focuses on validating module-level examples and integrated usage through solver.


if __name__ == '__main__':
    unittest.main() 
