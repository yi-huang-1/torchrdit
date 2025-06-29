"""Unit tests for the algorithm strategy pattern implementation in TorchRDIT.

This module contains tests that verify the proper functioning of the algorithm
strategy pattern used in the TorchRDIT electromagnetic solver. The tests ensure:

1. The solver correctly delegates calculations to the algorithm strategy
2. Algorithm switching works correctly
3. RCWA and R-DIT algorithms function properly
4. Algorithm parameters can be configured

The tests use a MockAlgorithm to verify the delegation behavior without
depending on the actual electromagnetic calculations.
"""

import unittest
import torch
import numpy as np
from torchrdit.constants import Precision
from torchrdit.solver import FourierBaseSolver, RCWASolver, RDITSolver
from torchrdit.algorithm import SolverAlgorithm, RCWAAlgorithm, RDITAlgorithm
from torchrdit.utils import create_material


class MockAlgorithm(SolverAlgorithm):
    """Mock algorithm for testing algorithm delegation."""
    
    def __init__(self):
        self.solve_calls = 0
        self.last_params = {}
        self._rdit_order = 5
    
    @property
    def name(self):
        """Return the name of the algorithm.
        
        This property provides a human-readable identifier for the mock algorithm.
        
        Returns:
            str: The name of the algorithm ('MOCK')
        """
        return "MOCK"
    
    def solve_nonhomo_layer(self, layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, kdim, k_0, **kwargs):
        """Mock implementation of the non-homogeneous layer solver.
        
        This method records the calls and parameters for verification in tests,
        and returns a simple predefined scattering matrix. It doesn't perform
        any actual electromagnetic calculations.
        
        Args:
            layer_thickness: Thickness of the layer
            p_mat_i: P matrix for the layer
            q_mat_i: Q matrix for the layer
            mat_w0: W0 matrix
            mat_v0: V0 matrix
            kdim: Dimensions in k-space
            k_0: Wave number
            **kwargs: Additional parameters
            
        Returns:
            A simple predefined scattering matrix
        """
        self.solve_calls += 1
        self.last_params = {
            'layer_thickness': layer_thickness,
            'p_mat_i': p_mat_i,
            'q_mat_i': q_mat_i,
            'mat_w0': mat_w0,
            'mat_v0': mat_v0,
            'kdim': kdim,
            'k_0': k_0,
            'kwargs': kwargs
        }
        return {'S11': torch.tensor(1), 'S12': torch.tensor(2), 'S21': torch.tensor(3), 'S22': torch.tensor(4)}
    
    def set_rdit_order(self, rdit_order):
        """Set the R-DIT algorithm order.
        
        This mock implementation simply stores the order parameter for later verification.
        
        Args:
            rdit_order: The order value to store
        """
        self._rdit_order = rdit_order


class TestStrategyPattern(unittest.TestCase):
    """Test cases for the Algorithm Strategy Pattern implementation in TorchRDIT.
    
    This test suite verifies that the Strategy pattern is correctly implemented
    for the electromagnetic solver algorithms. The Strategy pattern allows the
    solver to switch between different algorithms (RCWA, R-DIT) at runtime,
    making the code more flexible and maintainable.
    
    The tests focus on:
    1. Proper delegation of calculations from solver to algorithm
    2. Correct algorithm initialization in concrete solver classes
    3. Ability to swap algorithms at runtime
    4. Parameter passing between solver and algorithm
    5. Error cases when algorithms are not properly configured
    """
    
    def setUp(self):
        # Create a basic solver for testing
        self.lam0 = np.array([1.0])
        self.solver = FourierBaseSolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3],
            precision=Precision.SINGLE,
            device='cpu'
        )
        
        # Create some basic materials for testing
        self.air = create_material(name='air', permittivity=1.0, permeability=1.0)
        self.silicon = create_material(name='silicon', permittivity=11.7, permeability=1.0)
        
        # Prepare some test parameters for algorithm calls
        self.layer_thickness = torch.tensor(0.5, dtype=torch.float32)
        n_harmonics = 9  # 3x3
        
        # Create dummy test matrices
        self.p_mat = torch.eye(2*n_harmonics, dtype=torch.complex64)
        self.q_mat = torch.eye(2*n_harmonics, dtype=torch.complex64)
        self.w0_mat = torch.eye(2*n_harmonics, dtype=torch.complex64)
        self.v0_mat = torch.eye(2*n_harmonics, dtype=torch.complex64)
        
        # Mock algorithm for testing delegation
        self.mock_algorithm = MockAlgorithm()
    
    def test_algorithm_property(self):
        """Test getting and setting the algorithm on the base solver."""
        # Test default
        self.assertIsNone(self.solver.algorithm)
        
        # Test setting via property
        self.solver.algorithm = self.mock_algorithm
        self.assertEqual(self.solver.algorithm, self.mock_algorithm)
        
        # Test type checking
        with self.assertRaises(TypeError):
            self.solver.algorithm = "not_an_algorithm"
    
    def test_concrete_solvers_set_algorithms(self):
        """Test that concrete solvers set their algorithm strategies correctly.
        
        This test verifies that the concrete solver implementations (RCWASolver and
        RDITSolver) automatically initialize with the appropriate algorithm strategy.
        
        RCWASolver should use RCWAAlgorithm, and RDITSolver should use RDITAlgorithm.
        This ensures that each solver uses the computational approach it was designed for.
        """
        rcwa_solver = RCWASolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3]
        )
        
        rdit_solver = RDITSolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3]
        )
        
        # Verify correct algorithm types are set
        self.assertIsInstance(rcwa_solver.algorithm, RCWAAlgorithm)
        self.assertIsInstance(rdit_solver.algorithm, RDITAlgorithm)
        
        # Check algorithm names
        self.assertEqual(rcwa_solver.algorithm.name, "RCWA")
        self.assertEqual(rdit_solver.algorithm.name, "R-DIT")
    
    def test_algorithm_method_delegation(self):
        """Test that the solver correctly delegates to the algorithm instance."""
        # Set mock algorithm
        self.solver.algorithm = self.mock_algorithm
        
        # Call the methods that should delegate
        result = self.solver._solve_nonhomo_layer(
            layer_thickness=self.layer_thickness,
            p_mat_i=self.p_mat,
            q_mat_i=self.q_mat,
            mat_w0=self.w0_mat,
            mat_v0=self.v0_mat,
            kdim=[3, 3],
            k_0=torch.tensor([2.0*np.pi/self.lam0[0]], dtype=torch.float32)
        )
        
        # Verify delegation occurred
        self.assertEqual(self.mock_algorithm.solve_calls, 1)
        self.assertEqual(result['S11'], torch.tensor(1))
        self.assertEqual(result['S12'], torch.tensor(2))
        
    
    def test_algorithm_swapping(self):
        """Test that algorithms can be swapped at runtime.
        
        This test verifies a key benefit of the Strategy pattern: the ability to
        change algorithms dynamically at runtime. It demonstrates that:
        
        1. The RCWA solver can use the R-DIT algorithm
        2. The R-DIT solver can use the RCWA algorithm
        3. The swapped algorithms maintain their identity and behavior
        
        This flexibility allows users to compare different computational approaches
        with the same solver configuration, which can be useful for validating results
        or choosing the most efficient algorithm for specific scenarios.
        """
        # Create solvers
        rcwa_solver = RCWASolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3]
        )
        
        rdit_solver = RDITSolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3]
        )
        
        # Save original algorithms
        original_rcwa_algorithm = rcwa_solver.algorithm
        original_rdit_algorithm = rdit_solver.algorithm
        
        # Swap algorithms
        rcwa_solver.algorithm = original_rdit_algorithm
        rdit_solver.algorithm = original_rcwa_algorithm
        
        # Verify swapping was successful
        self.assertEqual(rcwa_solver.algorithm.name, "R-DIT")
        self.assertEqual(rdit_solver.algorithm.name, "RCWA")
        
        # Verify they still work by checking algorithm type
        self.assertIsInstance(rcwa_solver.algorithm, RDITAlgorithm)
        self.assertIsInstance(rdit_solver.algorithm, RCWAAlgorithm)
    
    def test_algorithm_error_without_strategy(self):
        """Test that calling solver methods with an invalid algorithm raises an error."""
        # Test that algorithm must be a SolverAlgorithm instance
        with self.assertRaises(TypeError):
            self.solver.algorithm = None
        
        # Test that algorithm must be a SolverAlgorithm instance
        with self.assertRaises(TypeError):
            self.solver.algorithm = "not_an_algorithm"
        
        # Create a mock solver without initializing its algorithm
        class SolverWithoutAlgorithm(FourierBaseSolver):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Force algorithm to be None through direct attribute access
                # This bypasses the property setter's type checking
                self._algorithm = None
        
        solver_without_algo = SolverWithoutAlgorithm(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3],
            precision=Precision.SINGLE,
            device='cpu'
        )
        
        # Verify that the algorithm is None
        self.assertIsNone(solver_without_algo._algorithm)
        
        # Calling methods should still raise an error when algorithm is None
        with self.assertRaises(ValueError):
            solver_without_algo._solve_nonhomo_layer(
                layer_thickness=self.layer_thickness,
                p_mat_i=self.p_mat,
                q_mat_i=self.q_mat,
                mat_w0=self.w0_mat,
                mat_v0=self.v0_mat,
                kdim=[3, 3],
                k_0=torch.tensor([2.0*np.pi/self.lam0[0]], dtype=torch.float32)
            )
    
    def test_rdit_order_setting(self):
        """Test that R-DIT order can be set on solvers.
        
        This test verifies that both RCWASolver and RDITSolver correctly pass
        the R-DIT order parameter to their respective algorithm strategies.
        The R-DIT order affects the numerical approximation used in the
        Rigorous Diffraction Interface Theory algorithm.
        """
        # Create RCWA solver but set R-DIT order
        rcwa_solver = RCWASolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3]
        )
        
        # Create R-DIT solver with default order
        rdit_solver = RDITSolver(
            lam0=self.lam0,
            lengthunit='um',
            rdim=[16, 16],
            kdim=[3, 3]
        )
        
        # Set order on both
        rcwa_solver.set_rdit_order(5)
        rdit_solver.set_rdit_order(8)
        
        # Check the order was properly delegated to algorithms
        self.assertEqual(rcwa_solver.algorithm._rdit_order, 5)
        self.assertEqual(rdit_solver.algorithm._rdit_order, 8)


class TestAlgorithmDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples.
    
    This test suite ensures that the examples provided in the docstrings of
    algorithm.py work as expected. It tests all the examples found in:
    1. SolverAlgorithm class docstrings
    2. RCWAAlgorithm class docstrings
    3. RDITAlgorithm class docstrings
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def test_algorithm_creation_with_builder(self):
        """Test the SolverAlgorithm class example of creating solvers using SolverBuilder."""
        # Example from SolverAlgorithm class docstring
        from torchrdit.builder import SolverBuilder 
        from torchrdit.constants import Algorithm
        
        # Get a builder instance
        builder = SolverBuilder()
        
        # Configure with RCWA algorithm
        solver_rcwa = builder.with_algorithm(Algorithm.RCWA).build()
        
        # Configure with R-DIT algorithm
        solver_rdit = builder.with_algorithm(Algorithm.RDIT).build()
        
        # Verify solvers were created with correct algorithms
        self.assertEqual(solver_rcwa.algorithm.name, "RCWA")
        self.assertEqual(solver_rdit.algorithm.name, "R-DIT")
    
    def test_rdit_algorithm_creation_with_builder(self):
        """Test the RDITAlgorithm class example of creating a solver with R-DIT algorithm."""
        # Example from RDITAlgorithm class docstring
        from torchrdit.builder import SolverBuilder
        from torchrdit.constants import Algorithm
        
        builder = SolverBuilder()
        
        # Configure with R-DIT algorithm and set order
        solver = builder.with_algorithm(Algorithm.RDIT) \
                      .with_rdit_order(8) \
                      .build()
        
        # Verify solver was created with R-DIT algorithm and correct order
        self.assertEqual(solver.algorithm.name, "R-DIT")
        self.assertEqual(solver.algorithm._rdit_order, 8)
    
    def test_set_rdit_order_example(self):
        """Test the set_rdit_order method example."""
        # Example from RDITAlgorithm.set_rdit_order docstring
        from torchrdit.builder import SolverBuilder
        from torchrdit.constants import Algorithm
        
        # Setting RDIT order through the builder
        solver = SolverBuilder() \
                      .with_algorithm(Algorithm.RDIT) \
                      .with_rdit_order(8) \
                      .build()
        
        # Verify the order was set correctly
        self.assertEqual(solver.algorithm._rdit_order, 8)
    
    def test_algorithm_enum_example(self):
        """Test the Algorithm enum examples."""
        # Example from Algorithm enum docstring in constants.py
        from torchrdit.constants import Algorithm
        from torchrdit.solver import create_solver
        
        # Create solver with RCWA algorithm
        rcwa_solver = create_solver(
            algorithm=Algorithm.RCWA,
            rdim=[32, 32],  # Smaller dimensions for faster tests
            kdim=[3, 3]
        )
        
        # Create solver with RDIT algorithm
        rdit_solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[32, 32],  # Smaller dimensions for faster tests
            kdim=[3, 3]
        )
        
        # Check which algorithm a solver uses
        self.assertEqual(rcwa_solver.algorithm.name, "RCWA")
        self.assertEqual(rdit_solver.algorithm.name, "R-DIT")


if __name__ == '__main__':
    unittest.main() 