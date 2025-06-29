import unittest
import io
import sys
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from torchrdit.observers import ConsoleProgressObserver, SolverObserver
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material

# Check if tqdm is available to test TqdmProgressObserver
try:
    from tqdm import tqdm
    from torchrdit.observers import TqdmProgressObserver
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class TestObserversDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in observers.py.
    
    This test suite ensures that the examples provided in the docstrings of
    observers.py work as expected. It tests examples for:
    1. Creating and using ConsoleProgressObserver
    2. Creating and using TqdmProgressObserver (when available)
    3. Observer update method handling for different event types
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Prepare a sample source for solver tests
        self.sample_source = {"theta": 0, "phi": 0, "pte": 1.0, "ptm": 0.0}
        
    def test_module_examples_console_observer(self):
        """Test the module-level examples for using ConsoleProgressObserver."""
        # Test example 1: Basic usage with a console observer
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            solver = create_solver(algorithm=Algorithm.RCWA)
            observer = ConsoleProgressObserver(verbose=True)
            solver.add_observer(observer)
            
            # Skip the actual solving as it's not needed for this test
            # Instead, manually trigger an event to verify the observer works
            observer.update("calculation_starting", {"mode": "RCWA", "n_freqs": 1, "n_layers": 3})
            
            # Check output contains expected text
            output = fake_stdout.getvalue()
            self.assertIn("Starting calculation in RCWA mode", output)
    
    @unittest.skipIf(not TQDM_AVAILABLE, "tqdm not available")
    def test_module_examples_tqdm_observer(self):
        """Test the module-level examples for using TqdmProgressObserver."""
        # Test example 2: Using a progress bar with tqdm
        solver = create_solver(algorithm=Algorithm.RCWA)
        observer = TqdmProgressObserver()
        solver.add_observer(observer)
        
        # Verify observer is properly added to solver
        self.assertIn(observer, solver._observers)
        
        # Manually test the observer's update method
        # First create a progress bar
        observer.update("processing_layers", {"total": 2})
        self.assertIsNotNone(observer.layer_pbar)
        
        # Update progress
        observer.update("layer_completed", {})
        
        # Complete calculation and close progress bar
        observer.update("calculation_completed", {})
        self.assertIsNone(observer.layer_pbar)
    
    def test_console_observer_class_examples(self):
        """Test the ConsoleProgressObserver class examples."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            # Create a solver
            solver = create_solver(algorithm=Algorithm.RCWA)
            
            # Add a verbose observer for detailed output
            verbose_observer = ConsoleProgressObserver(verbose=True)
            solver.add_observer(verbose_observer)
            
            # Or use a non-verbose observer for minimal output
            minimal_observer = ConsoleProgressObserver(verbose=False)
            solver.add_observer(minimal_observer)
            
            # Verify both observers are added to solver
            self.assertIn(verbose_observer, solver._observers)
            self.assertIn(minimal_observer, solver._observers)
            
            # Test with a verbose observer
            verbose_observer.update("calculation_starting", 
                                   {"mode": "RCWA", "n_freqs": 1, "n_layers": 3})
            verbose_observer.update("initializing_k_vectors", {})
            
            # Test with a non-verbose observer
            minimal_observer.update("calculation_starting", 
                                   {"mode": "RCWA", "n_freqs": 1, "n_layers": 3})
            minimal_observer.update("initializing_k_vectors", {})
            
            # Check output
            output = fake_stdout.getvalue()
            self.assertIn("Starting calculation in RCWA mode", output)
            self.assertIn("Initializing k-vectors", output)
            
            # The initializing message should appear once (from verbose observer)
            count = output.count("Initializing k-vectors")
            self.assertEqual(count, 1)
    
    def test_console_observer_init_examples(self):
        """Test the ConsoleProgressObserver.__init__ method examples."""
        # Create a verbose observer
        verbose_observer = ConsoleProgressObserver(verbose=True)
        self.assertTrue(verbose_observer.verbose)
        
        # Create a minimal observer
        minimal_observer = ConsoleProgressObserver(verbose=False)
        self.assertFalse(minimal_observer.verbose)
    
    def test_console_observer_update_examples(self):
        """Test the ConsoleProgressObserver.update method examples."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            # This method is typically called by the solver, not directly
            observer = ConsoleProgressObserver()
            
            # Example of how the solver would call this method
            observer.update(
                "calculation_starting", 
                {"mode": "RCWA", "n_freqs": 1, "n_layers": 3}
            )
            
            # Check output
            output = fake_stdout.getvalue()
            self.assertIn("Starting calculation in RCWA mode with 1 frequencies and 3 layers", output)
    
    @unittest.skipIf(not TQDM_AVAILABLE, "tqdm not available")
    def test_tqdm_observer_class_examples(self):
        """Test the TqdmProgressObserver class examples."""
        # Create a solver
        solver = create_solver(algorithm=Algorithm.RCWA)
        
        # Add a progress bar observer
        observer = TqdmProgressObserver()
        solver.add_observer(observer)
        
        # Verify observer is added to solver
        self.assertIn(observer, solver._observers)
        
        # Test the observer by simulating solver events
        observer.update("processing_layers", {"total": 3})
        self.assertIsNotNone(observer.layer_pbar)
        
        # Update progress
        observer.update("layer_completed", {})
        observer.update("layer_completed", {})
        observer.update("layer_completed", {})
        
        # Complete calculation
        observer.update("calculation_completed", {})
        self.assertIsNone(observer.layer_pbar)
    
    @unittest.skipIf(not TQDM_AVAILABLE, "tqdm not available")
    def test_tqdm_observer_init_examples(self):
        """Test the TqdmProgressObserver.__init__ method examples."""
        # Create a progress bar observer
        observer = TqdmProgressObserver()
        
        # Progress bars should be None initially
        self.assertIsNone(observer.layer_pbar)
    
    @unittest.skipIf(not TQDM_AVAILABLE, "tqdm not available")
    def test_tqdm_observer_update_examples(self):
        """Test the TqdmProgressObserver.update method examples."""
        # This method is typically called by the solver, not directly
        observer = TqdmProgressObserver()
        
        # Example of how the solver would call this method
        observer.update("processing_layers", {"total": 5})
        self.assertIsNotNone(observer.layer_pbar)
        self.assertEqual(observer.layer_pbar.total, 5)
        
        # Later when a layer is completed
        observer.update("layer_completed", {})
        
        # Finally when calculation is done
        observer.update("calculation_completed", {})
        self.assertIsNone(observer.layer_pbar)
    
    def test_console_observer_event_types(self):
        """Test ConsoleProgressObserver with all documented event types."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            observer = ConsoleProgressObserver(verbose=True)
            
            # Test all documented event types
            observer.update("calculation_starting", 
                           {"mode": "RCWA", "n_freqs": 2, "n_layers": 3})
            observer.update("initializing_k_vectors", {})
            observer.update("setting_up_matrices", {})
            observer.update("processing_layers", {"total": 3})
            observer.update("layer_started", {"current": 1, "total": 3, "progress": 33.3})
            observer.update("connecting_external_regions", {})
            observer.update("calculating_fields", {})
            observer.update("assembling_final_data", {})
            observer.update("calculation_completed", {"n_freqs": 2})
            
            # Check output for each event type
            output = fake_stdout.getvalue()
            
            self.assertIn("Starting calculation in RCWA mode", output)
            self.assertIn("Initializing k-vectors", output)
            self.assertIn("Setting up matrices", output)
            self.assertIn("Processing 3 layers", output)
            self.assertIn("Layer 1/3 (33.3%)", output)
            self.assertIn("Connecting external regions", output)
            self.assertIn("Calculating fields", output)
            self.assertIn("Assembling final data", output)
            self.assertIn("Calculation completed", output)
    
    def test_console_observer_non_verbose(self):
        """Test ConsoleProgressObserver with verbose=False."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            observer = ConsoleProgressObserver(verbose=False)
            
            # Test all documented event types
            observer.update("calculation_starting", 
                           {"mode": "RCWA", "n_freqs": 2, "n_layers": 3})
            observer.update("initializing_k_vectors", {})
            observer.update("setting_up_matrices", {})
            observer.update("processing_layers", {"total": 3})
            observer.update("layer_started", {"current": 1, "total": 3, "progress": 33.3})
            observer.update("connecting_external_regions", {})
            observer.update("calculating_fields", {})
            observer.update("assembling_final_data", {})
            observer.update("calculation_completed", {"n_freqs": 2})
            
            # Check output - only start and completion should be logged
            output = fake_stdout.getvalue()
            
            self.assertIn("Starting calculation in RCWA mode", output)
            self.assertNotIn("Initializing k-vectors", output)
            self.assertNotIn("Setting up matrices", output)
            self.assertNotIn("Processing 3 layers", output)
            self.assertNotIn("Layer 1/3", output)
            self.assertNotIn("Connecting external regions", output)
            self.assertNotIn("Calculating fields", output)
            self.assertNotIn("Assembling final data", output)
            self.assertIn("Calculation completed", output)
    
    def test_comprehensive_solver_setup(self):
        """Test a comprehensive solver setup with observers as shown in docstrings."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            # Create a solver and materials for a simple simulation
            solver = create_solver(algorithm=Algorithm.RCWA)
            
            # Create simple materials
            air = create_material(name="air", permittivity=1.0)
            silicon = create_material(name="silicon", permittivity=11.7)
            
            # Add materials to solver
            solver.add_materials([air, silicon])
            
            # Add a simple layer
            solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
            
            # Set input/output materials
            solver.update_ref_material("air")
            solver.update_trn_material("air")
            
            # Add a console observer as shown in the docstring example
            console_observer = ConsoleProgressObserver(verbose=True)
            solver.add_observer(console_observer)
            
            # If tqdm is available, also add a progress bar observer
            if TQDM_AVAILABLE:
                tqdm_observer = TqdmProgressObserver()
                solver.add_observer(tqdm_observer)
            
            # Set the wavelength using the lam0 property
            solver.lam0 = 1.55
            
            # Verify wavelength was set correctly
            self.assertEqual(solver.lam0[0], 1.55)
            
            # Without actually running the solver (which could be expensive),
            # validate the observer setup by checking that observers exist
            for observer in solver._observers:
                if isinstance(observer, ConsoleProgressObserver):
                    # Verify the observer is correctly configured
                    self.assertEqual(observer.verbose, True)
                    self.assertIsNone(observer.start_time)  # Start time is set when calculation begins
                
                if TQDM_AVAILABLE and isinstance(observer, TqdmProgressObserver):
                    # Verify the tqdm observer is correctly configured
                    self.assertIsNone(observer.layer_pbar)  # Progress bar is created when layers are processed


if __name__ == '__main__':
    unittest.main() 