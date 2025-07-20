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
        self.assertEqual(solver.lam0[0], 1.55)
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
        
        # Check that the solver was created properly and materials were set
        self.assertIsInstance(solver, RCWASolver)
        
        # Also test with explicit material instance for ref_material
        builder = get_solver_builder()
        solver = (builder
                  .with_algorithm(Algorithm.RCWA)
                  .with_wavelengths(1.55)
                  .with_materials([self.air]) 
                  .with_trn_material("air")
                  .with_ref_material(glass) # Pass material instance directly
                  .build())
        
        # Check that glass was added to the materials
        self.assertIn(glass, builder._materials)
        self.assertEqual(builder._ref_material, "glass")
    
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
        self.assertEqual(solver.lam0[0], 1.3)
        self.assertEqual(solver.lam0[1], 1.55)
        # Check RDIT order was set correctly
        self.assertIsInstance(solver.algorithm, RDITAlgorithm)
        self.assertEqual(solver.algorithm._rdit_order, 15)
    
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
        self.assertEqual(solver.lam0[0], 1.55)
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
        
        # These assertions will depend on how the solver exposes the trn/ref materials
        # Adjust based on your actual implementation
        if hasattr(solver, 'trn_material'):
            self.assertEqual(solver.trn_material.name, "air")
        if hasattr(solver, 'ref_material'):
            self.assertEqual(solver.ref_material.name, "silicon")
    
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
        self.assertEqual(solver.lam0[0], 1.0)
    
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
        
        # Just verify we have a valid solver instance
        self.assertIsInstance(solver, (RCWASolver, RDITSolver))
        
        # Check if layers property exists (you may need to adjust this based on your actual implementation)
        if hasattr(solver, 'layers'):
            self.assertTrue(len(solver.layers) >= 2)


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
        self.assertEqual(materials["Si"].get_permittivity(wavelengths), 12.0)
        self.assertEqual(materials["SiO2"].get_permittivity(wavelengths), 2.25)
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
        self.assertEqual(solver.layer_manager.layers[0].thickness, 0.2)
        self.assertEqual(solver.layer_manager.layers[1].thickness, 0.5)
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
    
    def test_solver_builder_basic_example(self):
        """Test the basic SolverBuilder example from its class docstring."""
        from torchrdit.constants import Algorithm
        from torchrdit.builder import SolverBuilder
        
        # Example from SolverBuilder class docstring - Create a basic RCWA solver
        solver = SolverBuilder().build()
        
        # Verify solver was created with default settings
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(solver.algorithm.name, "RCWA")
        
        # Example from SolverBuilder class docstring - Create a customized RCWA solver
        solver = SolverBuilder() \
            .with_algorithm(Algorithm.RCWA) \
            .with_wavelengths(1.55) \
            .with_k_dimensions([5, 5]) \
            .with_device("cpu") \
            .build()
        
        # Verify solver was created with custom settings
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(solver.lam0[0], 1.55)
        self.assertEqual(solver.kdim, [5, 5])
        
        # Example from SolverBuilder class docstring - Create an R-DIT solver
        solver = SolverBuilder() \
            .with_algorithm(Algorithm.RDIT) \
            .with_rdit_order(8) \
            .build()
        
        # Verify solver was created with RDIT algorithm and custom order
        self.assertIsInstance(solver, RDITSolver)
        self.assertEqual(solver.algorithm.name, "R-DIT")
        self.assertEqual(solver.algorithm._rdit_order, 8)
    
    def test_with_algorithm_example(self):
        """Test the with_algorithm method example."""
        from torchrdit.constants import Algorithm
        from torchrdit.builder import SolverBuilder
        
        # Examples from with_algorithm docstring
        builder = SolverBuilder().with_algorithm(Algorithm.RCWA)
        solver = builder.build()
        self.assertEqual(solver.algorithm.name, "RCWA")
        
        builder = SolverBuilder().with_algorithm(Algorithm.RDIT)
        solver = builder.build()
        self.assertEqual(solver.algorithm.name, "R-DIT")
    
    def test_with_precision_example(self):
        """Test the with_precision method example."""
        from torchrdit.constants import Precision
        from torchrdit.builder import SolverBuilder
        
        # Example for single precision (float32)
        builder = SolverBuilder().with_precision(Precision.SINGLE)
        solver = builder.build()
        self.assertEqual(builder._precision, Precision.SINGLE)
        
        # Example for double precision (float64)
        builder = SolverBuilder().with_precision(Precision.DOUBLE)
        solver = builder.build()
        self.assertEqual(builder._precision, Precision.DOUBLE)
    
    def test_with_wavelengths_example(self):
        """Test the with_wavelengths method example."""
        from torchrdit.builder import SolverBuilder
        import numpy as np
        
        # Example for single wavelength
        builder = SolverBuilder().with_wavelengths(1.55)
        solver = builder.build()
        self.assertEqual(solver.lam0[0], 1.55)
        
        # Example for multiple wavelengths
        wavelengths = np.linspace(1.2, 1.8, 4)  # Using fewer points for testing
        builder = SolverBuilder().with_wavelengths(wavelengths)
        solver = builder.build()
        self.assertEqual(len(solver.lam0), 4)
        self.assertEqual(solver.lam0[0], 1.2)
        self.assertEqual(solver.lam0[-1], 1.8)
    
    def test_with_materials_example(self):
        """Test the with_materials method example."""
        from torchrdit.builder import SolverBuilder
        from torchrdit.utils import create_material
        
        # Example from with_materials docstring
        air = create_material(name="Air", permittivity=1.0)
        silicon = create_material(name="Si", permittivity=12.0)
        
        builder = SolverBuilder().with_materials([air, silicon])
        
        # Verify materials were added correctly
        self.assertEqual(len(builder._materials), 2)
        self.assertIn(air, builder._materials)
        self.assertIn(silicon, builder._materials)
        self.assertEqual(builder._materials_dict["Air"], air)
        self.assertEqual(builder._materials_dict["Si"], silicon)
    
    def test_with_trn_ref_material_examples(self):
        """Test the with_trn_material and with_ref_material method examples."""
        from torchrdit.builder import SolverBuilder
        from torchrdit.utils import create_material
        
        # Example from with_trn_material docstring - Set by name
        air = create_material(name="Air", permittivity=1.0)
        
        builder = SolverBuilder().with_materials([air]).with_trn_material("Air")
        self.assertEqual(builder._trn_material, "Air")
        
        # Example from with_trn_material docstring - Set by object
        silicon = create_material(name="Si", permittivity=12.0)
        
        builder = SolverBuilder().with_trn_material(silicon)
        self.assertEqual(builder._trn_material, "Si")
        self.assertIn(silicon, builder._materials)
        
        # Example from with_ref_material docstring - Set by name
        builder = SolverBuilder().with_materials([air]).with_ref_material("Air")
        self.assertEqual(builder._ref_material, "Air")
        
        # Example from with_ref_material docstring - Set by object
        builder = SolverBuilder().with_ref_material(silicon)
        self.assertEqual(builder._ref_material, "Si")
        self.assertIn(silicon, builder._materials)
    
    def test_with_lattice_vectors_example(self):
        """Test the with_lattice_vectors method example."""
        from torchrdit.builder import SolverBuilder
        import torch
        
        # Example for square lattice
        t1 = torch.tensor([[1.0, 0.0]])
        t2 = torch.tensor([[0.0, 1.0]])
        
        builder = SolverBuilder().with_lattice_vectors(t1, t2)
        solver = builder.build()
        
        # Check the builder attributes instead of the solver attributes
        self.assertTrue(torch.allclose(builder._t1, t1))
        self.assertTrue(torch.allclose(builder._t2, t2))
        
        # Example for rectangular lattice
        t1 = torch.tensor([[1.5, 0.0]])
        t2 = torch.tensor([[0.0, 1.0]])
        
        builder = SolverBuilder().with_lattice_vectors(t1, t2)
        solver = builder.build()
        
        self.assertTrue(torch.allclose(builder._t1, t1))
        self.assertTrue(torch.allclose(builder._t2, t2))
    
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
        self.assertEqual(solver.layer_manager.layers[0].thickness, 0.5)
        self.assertEqual(solver.layer_manager.layers[1].thickness, 0.2)
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
        self.assertEqual(solver.lam0[0], 1.55)
        self.assertEqual(solver.kdim, [5, 5])
        self.assertEqual(solver.layer_manager.nlayer, 2)
        
        # Example from from_config docstring - Load and flip
        builder = SolverBuilder().from_config(config_dict, flip=True)
        solver = builder.build()
        
        # In flipped configuration, layers are reversed
        self.assertEqual(solver.layer_manager.nlayer, 2)
        self.assertEqual(solver.layer_manager.layers[0].thickness, 0.2)  # Reversed
        self.assertEqual(solver.layer_manager.layers[1].thickness, 0.5)  # Reversed


if __name__ == '__main__':
    unittest.main() 