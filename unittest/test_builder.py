import unittest
import numpy as np
from pathlib import Path

from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material
from torchrdit.solver import (
    RCWASolver, RDITSolver, get_solver_builder, create_solver_from_builder
)
from torchrdit.algorithm import RCWAAlgorithm, RDITAlgorithm


class TestSolverBuilder(unittest.TestCase):
    """Test the SolverBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lam0 = np.array([1.0])
        self.air = create_material(name="air", permittivity=1.0, permeability=1.0)
        self.silicon = create_material(name="silicon", permittivity=11.7, permeability=1.0)
    
    def test_basic_builder(self):
        """Test basic builder functionality."""
        # Get a builder and configure it
        builder = get_solver_builder()
        solver = (builder
                  .with_algorithm(Algorithm.RCWA)
                  .with_wavelengths(1.55)
                  .with_k_dimensions([5, 5])
                  .with_materials([self.air, self.silicon])
                  .with_device('cpu')
                  .build())
        
        # Check that the solver was created properly
        self.assertIsInstance(solver, RCWASolver)
        self.assertAlmostEqual(solver.lam0[0], 1.55, places=6)
        self.assertEqual(solver.kdim, [5, 5])
        self.assertIsInstance(solver.algorithm, RCWAAlgorithm)
    
    def test_trn_ref_material_builder(self):
        """Test builder with transmission and reflection materials."""
        # Create a glass material
        glass = create_material(name="glass", permittivity=2.25)
        
        # Get a builder and configure it with trn/ref materials
        builder = get_solver_builder()
        solver = (builder
                  .with_algorithm(Algorithm.RCWA)
                  .with_wavelengths(1.55)
                  .with_materials([self.air]) # Only add air initially
                  .with_trn_material(self.silicon) # Should auto-add silicon
                  .with_ref_material("air") # Reference to existing material
                  .build())
        
        # Check materials were applied to solver via layer_manager
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(solver.layer_manager.trn_material_name, "silicon")
        self.assertEqual(solver.layer_manager.ref_material_name, "air")
        
        # Also test with explicit material instance for ref_material
        builder = get_solver_builder()
        solver = (builder
                  .with_algorithm(Algorithm.RCWA)
                  .with_wavelengths(1.55)
                  .with_materials([self.air]) 
                  .with_trn_material("air")
                  .with_ref_material(glass) # Pass material instance directly
                  .build())
        
        # Check that trn/ref materials are reflected on solver
        self.assertEqual(solver.layer_manager.trn_material_name, "air")
        self.assertEqual(solver.layer_manager.ref_material_name, "glass")
    
    def test_rdit_solver_builder(self):
        """Test building an RDIT solver."""
        builder = get_solver_builder()
        solver = (builder
                  .with_algorithm(Algorithm.RDIT)
                  .with_wavelengths([1.3, 1.55])
                  .with_precision(Precision.DOUBLE)
                  .with_rdit_order(15)
                  .build())
        
        # Check that the solver was created properly
        self.assertIsInstance(solver, RDITSolver)
        self.assertEqual(len(solver.lam0), 2)
        self.assertAlmostEqual(solver.lam0[0], 1.3, places=6)
        self.assertAlmostEqual(solver.lam0[1], 1.55, places=6)
        # Check RDIT order was set correctly
        self.assertIsInstance(solver.algorithm, RDITAlgorithm)
        self.assertEqual(solver.algorithm._rdit_order, 15)
        # Check precision propagates to solver dtype
        import torch
        self.assertEqual(solver.tfloat, torch.float64)
    
    def test_config_validation(self):
        """Test that unknown configuration keys are properly detected."""
        # Create a valid configuration
        valid_config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "rdim": [512, 512],
            "kdim": [3, 3]
        }
        
        # Test that valid configuration works
        builder = get_solver_builder()
        solver = builder.from_config(valid_config).build()
        self.assertIsInstance(solver, RCWASolver)
        
        # Create an invalid configuration with unknown keys
        invalid_config = valid_config.copy()
        invalid_config["unknown_key"] = "some_value"
        invalid_config["another_wrong_key"] = 42
        
        # Test that invalid configuration raises ValueError
        builder = get_solver_builder()
        with self.assertRaises(ValueError) as context:
            builder.from_config(invalid_config)
        
        # Check that the error message contains the unknown keys
        error_message = str(context.exception)
        self.assertIn("unknown_key", error_message)
        self.assertIn("another_wrong_key", error_message)
        self.assertIn("Valid keys are:", error_message)
    
    def test_case_insensitive_config(self):
        """Test that configuration keys are case-insensitive."""
        # Create a configuration with mixed case keys
        mixed_case_config = {
            "Algorithm": "RCWA",
            "WaVeLeNgThS": [1.55],
            "RDIM": [512, 512],
            "kdim": [3, 3],
            "DevICE": "cpu"
        }
        
        # Test that the mixed case configuration works
        builder = get_solver_builder()
        solver = builder.from_config(mixed_case_config).build()
        
        # Check that the solver was created properly
        self.assertIsInstance(solver, RCWASolver)
        self.assertAlmostEqual(solver.lam0[0], 1.55, places=6)
        self.assertEqual(solver.rdim, [512, 512])
        self.assertEqual(solver.kdim, [3, 3])
        
        # Check that we still get error for unknown keys regardless of case
        invalid_config = mixed_case_config.copy()
        invalid_config["UnKnOwN_KeY"] = "some_value"
        
        builder = get_solver_builder()
        with self.assertRaises(ValueError):
            builder.from_config(invalid_config)
    
    def test_trn_ref_materials_config(self):
        """Test the transmission and reflection material configuration."""
        # Create materials dictionary for the configuration
        materials_dict = {
            "air": {"permittivity": 1.0},
            "silicon": {"permittivity": 11.7}
        }
        
        # Create a configuration with trn_material and ref_material
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "rdim": [512, 512],
            "kdim": [3, 3],
            "materials": materials_dict,
            "trn_material": "air",
            "ref_material": "silicon",
            "layers": [
                {"material": "air", "thickness": 1.0}
            ]
        }
        
        # Build the solver from this configuration
        builder = get_solver_builder()
        solver = builder.from_config(config).build()
        
        # Validate that the solver is correctly configured
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(solver.layer_manager.trn_material_name, "air")
        self.assertEqual(solver.layer_manager.ref_material_name, "silicon")
    
    def test_create_solver_from_builder(self):
        """Test the create_solver_from_builder function."""
        # Define a configuration function
        def configure_builder(builder):
            return (builder
                    .with_algorithm(Algorithm.RCWA)
                    .with_wavelengths(1.0)
                    .with_materials([self.air, self.silicon]))
        
        # Create a solver using the function
        solver = create_solver_from_builder(configure_builder)

        # Check that the solver was created correctly
        self.assertIsInstance(solver, RCWASolver)

    def test_builder_layer_slice_count(self):
        """Builder.add_layer should propagate slice_count into the solver."""
        builder = get_solver_builder()
        solver = (
            builder.with_algorithm(Algorithm.RCWA)
            .with_wavelengths(1.55)
            .with_real_dimensions([16, 16])
            .with_k_dimensions([3, 3])
            .with_materials([self.air, self.silicon])
            .add_layer(
                {
                    "material": "silicon",
                    "thickness": 0.2,
                    "is_homogeneous": False,
                    "slice_count": 4,
                }
            )
            .build()
        )

        layer = solver.layer_manager.layers[0]
        self.assertEqual(layer.material_name, "silicon")
        self.assertEqual(layer.slice_count, 4)

    def test_config_layer_slice_count_defaults(self):
        """from_config should accept per-layer slice_count and sanitize invalid values."""
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "rdim": [32, 32],
            "kdim": [3, 3],
            "materials": {
                "air": {"permittivity": 1.0},
                "silicon": {"permittivity": 11.7},
            },
            "layers": [
                {"material": "silicon", "thickness": 0.3, "slice_count": 6},
                {"material": "air", "thickness": 0.1, "slice_count": 0},
            ],
        }

        builder = get_solver_builder()
        solver = builder.from_config(config).build()

        self.assertEqual(solver.layer_manager.layers[0].slice_count, 6)
        # Invalid (0) should fall back to 1 via Cell3D.add_layer sanitization
        self.assertEqual(solver.layer_manager.layers[1].slice_count, 1)
    
    def test_add_materials_and_layers(self):
        """Test adding materials and layers to the builder."""
        builder = get_solver_builder()
        
        # Add materials individually
        builder.add_material(self.air)
        builder.add_material(self.silicon)
        
        # Add layers
        builder.add_layer({
            "type": "uniform",
            "material": "air",
            "thickness": 1.0
        })
        
        builder.add_layer({
            "type": "uniform",
            "material": "silicon",
            "thickness": 0.5
        })
        
        # Build the solver
        solver = builder.build()
        
        # Verify via public layer_manager API
        self.assertIsInstance(solver, (RCWASolver, RDITSolver))
        self.assertEqual(solver.layer_manager.nlayer, 2)
        self.assertEqual(solver.layer_manager.layers[0].material_name, "air")
        self.assertEqual(solver.layer_manager.layers[1].material_name, "silicon")


class TestBuilderDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in builder.py.
    
    This test suite ensures that the examples provided in the docstrings of
    builder.py work as expected. It tests examples for:
    1. _create_materials function
    2. _add_layers function
    3. flip_config function
    4. SolverBuilder class and its methods
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create basic materials for testing
        self.air = create_material(name="air", permittivity=1.0)
        self.silicon = create_material(name="silicon", permittivity=11.7)
        self.sio2 = create_material(name="sio2", permittivity=2.25)
    
    def test_create_materials_example(self):
        """Test the _create_materials function example."""
        from torchrdit.builder import _create_materials
        import numpy as np
        
        # Example from _create_materials docstring
        materials_spec = {
            "Si": {"permittivity": 12.0},
            "SiO2": {"permittivity": 2.25},
            "Au": {
                "dielectric_dispersion": True,
                "dielectric_file": "SiO2-e.txt",  # Using a test file that exists
                "data_format": "freq-eps",
                "data_unit": "thz"
            }
        }
        
        # Pass the unittest directory as base_path since that's where our test files are
        materials = _create_materials(materials_spec, Path(__file__).parent)
        
        # Verify the materials were created correctly
        self.assertEqual(len(materials), 3)
        
        # Need to provide wavelengths parameter for get_permittivity
        wavelengths = np.array([1.55])  # Default test wavelength
        self.assertAlmostEqual(materials["Si"].get_permittivity(wavelengths), 12.0, places=6)
        self.assertAlmostEqual(materials["SiO2"].get_permittivity(wavelengths), 2.25, places=6)
        # Simply verify Au material was created without checking specific attribute
        self.assertIn("Au", materials)
    
    def test_add_layers_example(self):
        """Test the _add_layers function example."""
        from torchrdit.builder import _add_layers
        from torchrdit.solver import RCWASolver
        
        # Create materials dictionary
        materials_dict = {
            "SiO2": self.sio2,
            "Si": self.silicon
        }
        
        # Create a solver
        solver = RCWASolver(
            lam0=np.array([1.55]),
            rdim=[32, 32],
            kdim=[3, 3]
        )
        
        # Example from _add_layers docstring
        layers = [
            {"material": "SiO2", "thickness": 0.2, "is_homogeneous": True},
            {"material": "Si", "thickness": 0.5, "is_homogeneous": False},
        ]
        
        # Add layers to the solver
        _add_layers(solver, layers, materials_dict)
        
        # Verify layers were added correctly
        self.assertEqual(solver.layer_manager.nlayer, 2)
        self.assertAlmostEqual(solver.layer_manager.layers[0].thickness, 0.2, places=6)
        self.assertAlmostEqual(solver.layer_manager.layers[1].thickness, 0.5, places=6)
        self.assertTrue(solver.layer_manager.layers[0].is_homogeneous)
        self.assertFalse(solver.layer_manager.layers[1].is_homogeneous)
    
    def test_flip_config_example(self):
        """Test the flip_config function example."""
        from torchrdit.builder import flip_config
        
        # Example from flip_config docstring
        original_config = {
            "layers": [{"material": "air"}, {"material": "Si"}, {"material": "SiO2"}],
            "trn_material": "air",
            "ref_material": "SiO2"
        }
        
        flipped = flip_config(original_config)
        
        # Verify the configuration was flipped correctly
        self.assertEqual(flipped["layers"], [{"material": "SiO2"}, {"material": "Si"}, {"material": "air"}])
        self.assertEqual(flipped["trn_material"], "SiO2")
        self.assertEqual(flipped["ref_material"], "air")
    
    
    
    
    
    
    
    
    
    
    
    
    
    def test_with_lattice_vectors_example(self):
        """Test the with_lattice_vectors method example using solver attributes."""
        from torchrdit.builder import SolverBuilder
        import torch

        t1 = torch.tensor([[1.5, 0.0]])
        t2 = torch.tensor([[0.0, 1.0]])

        solver = SolverBuilder().with_lattice_vectors(t1, t2).build()
        # solver stores squeezed lattice vectors
        self.assertTrue(torch.allclose(solver.lattice_t1, t1.squeeze()))
        self.assertTrue(torch.allclose(solver.lattice_t2, t2.squeeze()))
    
    def test_add_layer_example(self):
        """Test the add_layer method example."""
        from torchrdit.builder import SolverBuilder
        from torchrdit.utils import create_material
        
        # Create materials for the test
        silicon = create_material(name="Si", permittivity=12.0)
        sio2 = create_material(name="SiO2", permittivity=2.25)
        
        # Start with a builder and add materials
        builder = SolverBuilder().with_materials([silicon, sio2])
        
        # Example from add_layer docstring - Add a homogeneous silicon layer
        builder.add_layer({
            "material": "Si",
            "thickness": 0.5,
            "is_homogeneous": True
        })
        
        # Example from add_layer docstring - Add a non-homogeneous layer
        builder.add_layer({
            "material": "SiO2",
            "thickness": 0.2,
            "is_homogeneous": False
        })
        
        # Build the solver and verify layers were added
        solver = builder.build()
        
        self.assertEqual(solver.layer_manager.nlayer, 2)
        self.assertAlmostEqual(solver.layer_manager.layers[0].thickness, 0.5, places=6)
        self.assertAlmostEqual(solver.layer_manager.layers[1].thickness, 0.2, places=6)
        self.assertTrue(solver.layer_manager.layers[0].is_homogeneous)
        self.assertFalse(solver.layer_manager.layers[1].is_homogeneous)
    
    def test_from_config_example(self):
        """Test the from_config method example."""
        from torchrdit.builder import SolverBuilder
        
        # Create a configuration dictionary (simplified from example)
        config_dict = {
            "algorithm": "RDIT",
            "wavelengths": [1.55],
            "kdim": [5, 5],
            "materials": {
                "Air": {"permittivity": 1.0},
                "Si": {"permittivity": 12.0}
            },
            "layers": [
                {"material": "Air", "thickness": 0.5},
                {"material": "Si", "thickness": 0.2}
            ]
        }
        
        # Example from from_config docstring - Load from dictionary
        builder = SolverBuilder().from_config(config_dict)
        solver = builder.build()
        
        self.assertEqual(solver.algorithm.name, "R-DIT")
        self.assertAlmostEqual(solver.lam0[0], 1.55, places=6)
        self.assertEqual(solver.kdim, [5, 5])
        self.assertEqual(solver.layer_manager.nlayer, 2)
        
        # Example from from_config docstring - Load and flip
        builder = SolverBuilder().from_config(config_dict, flip=True)
        solver = builder.build()
        
        # In flipped configuration, layers are reversed
        self.assertEqual(solver.layer_manager.nlayer, 2)
        self.assertAlmostEqual(solver.layer_manager.layers[0].thickness, 0.2, places=6)  # Reversed
        self.assertAlmostEqual(solver.layer_manager.layers[1].thickness, 0.5, places=6)  # Reversed


if __name__ == '__main__':
    unittest.main() 
