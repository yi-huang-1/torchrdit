import unittest
import torch
import numpy as np

from torchrdit.layers import (
    HomogeneousLayer, GratingLayer, 
    HomogeneousLayerBuilder, GratingLayerBuilder,
    LayerDirector, LayerManager
)
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
        self.assertEqual(solver.layers[0].thickness, 0.2)
        self.assertEqual(solver.layers[0].material_name, "silicon")
        self.assertTrue(solver.layers[0].is_homogeneous)
        
        self.assertEqual(solver.layers[1].thickness, 0.3)
        self.assertEqual(solver.layers[1].material_name, "silicon")
        self.assertFalse(solver.layers[1].is_homogeneous)
    
    def test_layer_properties(self):
        """Test the property getters and setters of Layer class."""
        # Create a homogeneous layer for testing
        layer = HomogeneousLayer(thickness=1.0, material_name="silicon", is_optimize=False)
        
        # Test thickness property
        self.assertEqual(layer.thickness, 1.0)
        layer.thickness = 2.0
        self.assertEqual(layer.thickness, 2.0)
        
        # Test material_name property
        self.assertEqual(layer.material_name, "silicon")
        layer.material_name = "SiO2"
        self.assertEqual(layer.material_name, "SiO2")
        
        # Test is_homogeneous property (read-only for HomogeneousLayer)
        self.assertTrue(layer.is_homogeneous)
        
        # Test is_dispersive property
        self.assertFalse(layer.is_dispersive)
        layer.is_dispersive = True
        self.assertTrue(layer.is_dispersive)
        
        # Test is_optimize property
        self.assertFalse(layer.is_optimize)
        layer.is_optimize = True
        self.assertTrue(layer.is_optimize)
        
        # Test is_solved property
        self.assertFalse(layer.is_solved)
        layer.is_solved = True
        self.assertTrue(layer.is_solved)
    
    def test_homogeneous_layer(self):
        """Test HomogeneousLayer initialization and properties."""
        # Create a basic homogeneous layer
        homogeneous_layer = HomogeneousLayer(
            thickness=0.5,
            material_name="silicon",
            is_optimize=True
        )
        
        # Verify properties
        self.assertEqual(homogeneous_layer.thickness, 0.5)
        self.assertEqual(homogeneous_layer.material_name, "silicon")
        self.assertTrue(homogeneous_layer.is_optimize)
        self.assertTrue(homogeneous_layer.is_homogeneous)
        self.assertFalse(homogeneous_layer.is_dispersive)
        self.assertFalse(homogeneous_layer.is_solved)
        
        # Test string representation
        expected_str = "HomogeneousLayer(thickness=0.5, material_name=silicon)"
        self.assertEqual(str(homogeneous_layer), expected_str)
    
    def test_grating_layer(self):
        """Test GratingLayer initialization and properties."""
        # Create a basic grating layer
        grating_layer = GratingLayer(
            thickness=0.3,
            material_name="SiO2",
            is_optimize=True
        )
        
        # Verify properties
        self.assertEqual(grating_layer.thickness, 0.3)
        self.assertEqual(grating_layer.material_name, "SiO2")
        self.assertTrue(grating_layer.is_optimize)
        self.assertFalse(grating_layer.is_homogeneous)
        self.assertFalse(grating_layer.is_dispersive)
        self.assertFalse(grating_layer.is_solved)
        
        # Test string representation
        expected_str = "GratingLayer(thickness=0.3, material_name=SiO2)"
        self.assertEqual(str(grating_layer), expected_str)
    
    def test_homogeneous_layer_builder(self):
        """Test HomogeneousLayerBuilder creation and usage."""
        # Create a builder and build a layer step by step
        builder = HomogeneousLayerBuilder()
        
        # Create a new layer
        builder.create_layer()
        
        # Configure the layer properties
        builder.update_thickness(0.7)
        builder.update_material_name("SiO2")
        builder.set_optimize(True)
        builder.set_dispersive(True)
        
        # Get the completed layer
        layer = builder.get_layer_instance()
        
        # Verify the layer properties
        self.assertEqual(layer.thickness, 0.7)
        self.assertEqual(layer.material_name, "SiO2")
        self.assertTrue(layer.is_optimize)
        self.assertTrue(layer.is_dispersive)
        self.assertTrue(layer.is_homogeneous)
    
    def test_grating_layer_builder(self):
        """Test GratingLayerBuilder creation and usage."""
        # Create a builder and build a layer step by step
        builder = GratingLayerBuilder()
        
        # Create a new layer
        builder.create_layer()
        
        # Configure the layer properties
        builder.update_thickness(0.4)
        builder.update_material_name("silicon")
        builder.set_optimize(False)
        builder.set_dispersive(False)
        
        # Get the completed layer
        layer = builder.get_layer_instance()
        
        # Verify the layer properties
        self.assertEqual(layer.thickness, 0.4)
        self.assertEqual(layer.material_name, "silicon")
        self.assertFalse(layer.is_optimize)
        self.assertFalse(layer.is_dispersive)
        self.assertFalse(layer.is_homogeneous)
    
    def test_layer_director(self):
        """Test LayerDirector for building different layer types."""
        # Create a layer director
        director = LayerDirector()
        
        # Build a homogeneous layer
        homogeneous = director.build_layer(
            layer_type="homogeneous",
            thickness=0.2,
            material_name="silicon",
            is_optimize=True,
            is_dispersive=True
        )
        
        # Verify the homogeneous layer
        self.assertIsInstance(homogeneous, HomogeneousLayer)
        self.assertEqual(homogeneous.thickness, 0.2)
        self.assertEqual(homogeneous.material_name, "silicon")
        self.assertTrue(homogeneous.is_optimize)
        self.assertTrue(homogeneous.is_dispersive)
        self.assertTrue(homogeneous.is_homogeneous)
        
        # Build a grating layer
        grating = director.build_layer(
            layer_type="grating",
            thickness=0.3,
            material_name="SiO2",
            is_optimize=False,
            is_dispersive=False
        )
        
        # Verify the grating layer
        self.assertIsInstance(grating, GratingLayer)
        self.assertEqual(grating.thickness, 0.3)
        self.assertEqual(grating.material_name, "SiO2")
        self.assertFalse(grating.is_optimize)
        self.assertFalse(grating.is_dispersive)
        self.assertFalse(grating.is_homogeneous)
    
    def test_layer_manager_initialization(self):
        """Test LayerManager initialization."""
        # Create a layer manager
        manager = LayerManager(
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            vec_p=self.vec_p,
            vec_q=self.vec_q
        )
        
        # Verify initial state
        self.assertEqual(len(manager.layers), 0)
        self.assertEqual(manager.ref_material_name, "air")
        self.assertEqual(manager.trn_material_name, "air")
        self.assertFalse(manager.is_ref_dispersive)
        self.assertFalse(manager.is_trn_dispersive)
        self.assertEqual(manager.nlayer, 0)
    
    def test_layer_manager_add_layer(self):
        """Test adding layers to LayerManager."""
        # Create a layer manager
        manager = LayerManager(
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            vec_p=self.vec_p,
            vec_q=self.vec_q
        )
        
        # Add a homogeneous layer
        manager.add_layer(
            layer_type="homogeneous",
            thickness=torch.tensor(0.2),
            material_name="silicon",
            is_optimize=True,
            is_dispersive=False
        )
        
        # Add a grating layer
        manager.add_layer(
            layer_type="grating",
            thickness=torch.tensor(0.3),
            material_name="SiO2",
            is_optimize=False,
            is_dispersive=True
        )
        
        # Verify the layers were added correctly
        self.assertEqual(manager.nlayer, 2)
        self.assertIsInstance(manager.layers[0], HomogeneousLayer)
        self.assertIsInstance(manager.layers[1], GratingLayer)
        
        self.assertEqual(manager.layers[0].thickness, 0.2)
        self.assertEqual(manager.layers[0].material_name, "silicon")
        self.assertTrue(manager.layers[0].is_optimize)
        self.assertFalse(manager.layers[0].is_dispersive)
        
        self.assertEqual(manager.layers[1].thickness, 0.3)
        self.assertEqual(manager.layers[1].material_name, "SiO2")
        self.assertFalse(manager.layers[1].is_optimize)
        self.assertTrue(manager.layers[1].is_dispersive)
    
    def test_layer_manager_update_thickness(self):
        """Test updating layer thickness in LayerManager."""
        # Create a layer manager with one layer
        manager = LayerManager(
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            vec_p=self.vec_p,
            vec_q=self.vec_q
        )
        
        # Add a layer
        manager.add_layer(
            layer_type="homogeneous",
            thickness=torch.tensor(0.2),
            material_name="silicon",
            is_optimize=False,
            is_dispersive=False
        )
        
        # Update the layer thickness
        manager.update_layer_thickness(
            layer_index=0,
            thickness=torch.tensor(0.5)
        )
        
        # Verify the thickness was updated
        self.assertEqual(manager.layers[0].thickness, 0.5)
    
    def test_layer_manager_replace_layers(self):
        """Test replacing layer types in LayerManager."""
        # Create a layer manager
        manager = LayerManager(
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            vec_p=self.vec_p,
            vec_q=self.vec_q
        )
        
        # Add layers of different types
        manager.add_layer(
            layer_type="homogeneous",
            thickness=torch.tensor(0.2),
            material_name="silicon",
            is_optimize=False,
            is_dispersive=False
        )
        
        manager.add_layer(
            layer_type="grating",
            thickness=torch.tensor(0.3),
            material_name="SiO2",
            is_optimize=True,
            is_dispersive=True
        )
        
        # Replace first layer (homogeneous) with a grating layer
        manager.replace_layer_to_grating(0)
        
        # Replace second layer (grating) with a homogeneous layer
        manager.replace_layer_to_homogeneous(1)
        
        # Verify the replacements
        self.assertIsInstance(manager.layers[0], GratingLayer)
        self.assertFalse(manager.layers[0].is_homogeneous)
        self.assertEqual(manager.layers[0].thickness, 0.2)
        self.assertEqual(manager.layers[0].material_name, "silicon")
        
        self.assertIsInstance(manager.layers[1], HomogeneousLayer)
        self.assertTrue(manager.layers[1].is_homogeneous)
        self.assertEqual(manager.layers[1].thickness, 0.3)
        self.assertEqual(manager.layers[1].material_name, "SiO2")
    
    def test_layer_manager_update_ref_trn_layers(self):
        """Test updating reference and transmission layers in LayerManager."""
        # Create a layer manager
        manager = LayerManager(
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            vec_p=self.vec_p,
            vec_q=self.vec_q
        )
        
        # Update reference layer
        manager.update_ref_layer(
            material_name="silicon",
            is_dispersive=True
        )
        
        # Update transmission layer
        manager.update_trn_layer(
            material_name="SiO2",
            is_dispersive=False
        )
        
        # Verify updates
        self.assertEqual(manager.ref_material_name, "silicon")
        self.assertTrue(manager.is_ref_dispersive)
        
        self.assertEqual(manager.trn_material_name, "SiO2")
        self.assertFalse(manager.is_trn_dispersive)
    
    def test_gen_toeplitz_matrix(self):
        """Test generating Toeplitz matrices for layers."""
        # Create a simple layer manager
        manager = LayerManager(
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            vec_p=self.vec_p,
            vec_q=self.vec_q
        )
        
        # Add a layer and set up a simple permittivity distribution
        manager.add_layer(
            layer_type="grating",
            thickness=torch.tensor(0.3),
            material_name="silicon",
            is_optimize=False,
            is_dispersive=False
        )
        
        # Create a simple permittivity distribution (checkerboard pattern)
        checker = torch.ones(self.rdim)
        for i in range(self.rdim[0]):
            for j in range(self.rdim[1]):
                if (i // 32 + j // 32) % 2 == 0:
                    checker[i, j] = 2.0
        
        # Assign the pattern to the layer
        manager.layers[0].ermat = checker
        
        # Generate the Toeplitz matrix
        manager.gen_toeplitz_matrix(
            layer_index=0,
            n_harmonic1=1,
            n_harmonic2=1,
            param='er',
            method='FFT'
        )
        
        # Verify the Toeplitz matrix was generated
        self.assertIsNotNone(manager.layers[0].kermat)
        
        # Basic shape test - the implementation actually returns a [1, 1] tensor for these parameters
        self.assertEqual(manager.layers[0].kermat.shape, torch.Size([1, 1]))


if __name__ == '__main__':
    unittest.main() 