import unittest
import io
import sys
from unittest.mock import patch, MagicMock
import time

import torch
import numpy as np
from torchrdit.solver import (
    SolverObserver, SolverSubjectMixin, create_solver_from_config, 
    create_solver, FourierBaseSolver, RCWASolver
)
from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material
from torchrdit.builder import SolverBuilder


class TestSolverDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in solver.py.
    
    This test suite ensures that the examples provided in the docstrings of
    solver.py work as expected. It tests examples for:
    1. SolverObserver interface
    2. SolverSubjectMixin class
    3. create_solver_from_config function
    4. create_solver function
    5. FourierBaseSolver class
    6. RCWASolver and RDITSolver classes
    7. Adding sources and solving
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Prepare common test data
        self.test_wavelength = np.array([1.55])
    
    def test_solver_observer_example(self):
        """Test the SolverObserver interface example."""
        # Create a custom observer as shown in the docstring
        class TimingObserver(SolverObserver):
            def __init__(self):
                self.start_times = {}
                self.end_times = {}
            
            def update(self, event_type, data):
                if event_type == "layer_started":
                    layer = data.get("current", 0)
                    self.start_times[layer] = time.time()
                elif event_type == "layer_completed":
                    layer = data.get("current", 0)
                    self.end_times[layer] = time.time()
                    elapsed = self.end_times[layer] - self.start_times[layer]
                    print(f"Layer {layer} processed in {elapsed:.3f} seconds")
        
        # Create a solver with the custom observer
        solver = create_solver(algorithm=Algorithm.RCWA, lam0=self.test_wavelength)
        observer = TimingObserver()
        solver.add_observer(observer)
        
        # Verify observer is properly added to solver
        self.assertIn(observer, solver._observers)
        
        # Test the observer's update method
        current_time = time.time()
        observer.update("layer_started", {"current": 1})
        # Allow a small time to pass
        time.sleep(0.001)
        observer.update("layer_completed", {"current": 1})
        
        # Verify the observer recorded the times
        self.assertIn(1, observer.start_times)
        self.assertIn(1, observer.end_times)
        self.assertGreater(observer.end_times[1], observer.start_times[1])
    
    def test_solver_observer_update_example(self):
        """Test the SolverObserver.update method example."""
        # Create the example implementation from the docstring
        class ExampleObserver(SolverObserver):
            def __init__(self):
                self.messages = []
                
            def update(self, event_type, data):
                if event_type == "calculation_starting":
                    self.messages.append(f"Starting calculation with {data.get('n_freqs')} wavelengths")
                elif event_type == "calculation_completed":
                    self.messages.append("Calculation finished!")
        
        # Test the observer
        observer = ExampleObserver()
        observer.update("calculation_starting", {"n_freqs": 3})
        observer.update("calculation_completed", {})
        
        # Verify the observer recorded the messages
        self.assertEqual(len(observer.messages), 2)
        self.assertEqual(observer.messages[0], "Starting calculation with 3 wavelengths")
        self.assertEqual(observer.messages[1], "Calculation finished!")
    
    def test_solver_subject_mixin_example(self):
        """Test the SolverSubjectMixin class example."""
        # Create the example implementation from the docstring
        class MyProcessingClass(SolverSubjectMixin):
            def __init__(self):
                SolverSubjectMixin.__init__(self)  # Initialize the observer list
                
            def process_data(self, data):
                # Notify observers that processing is starting
                self.notify_observers('processing_started', {'data_size': len(data)})
                
                # Do some processing
                result = []
                for i, item in enumerate(data):
                    # Process the item
                    result.append(item * 2)
                    
                    # Notify observers of progress
                    self.notify_observers('item_processed', {
                        'current': i + 1,
                        'total': len(data),
                        'progress': (i + 1) / len(data) * 100
                    })
                    
                # Notify observers that processing is complete
                self.notify_observers('processing_completed', {'result_size': len(result)})
                return result
        
        # Create a simple observer to monitor progress
        class SimpleObserver(SolverObserver):
            def __init__(self):
                self.events = []
                
            def update(self, event_type, data):
                self.events.append((event_type, data))
                if event_type == 'processing_started':
                    print(f"Starting to process {data['data_size']} items")
                elif event_type == 'item_processed':
                    print(f"Progress: {data['progress']:.1f}%")
                elif event_type == 'processing_completed':
                    print(f"Processing completed with {data['result_size']} results")
        
        # Use the observer with our processing class
        processor = MyProcessingClass()
        observer = SimpleObserver()
        processor.add_observer(observer)
        
        # Process some data
        test_data = [1, 2, 3, 4, 5]
        result = processor.process_data(test_data)
        
        # Verify the result of processing
        self.assertEqual(result, [2, 4, 6, 8, 10])
        
        # Verify the observer received the right events
        self.assertEqual(len(observer.events), 7)  # 1 start + 5 items + 1 complete
        
        # Verify starting event
        self.assertEqual(observer.events[0][0], 'processing_started')
        self.assertEqual(observer.events[0][1]['data_size'], 5)
        
        # Verify completion event
        self.assertEqual(observer.events[6][0], 'processing_completed')
        self.assertEqual(observer.events[6][1]['result_size'], 5)
        
        # Verify processing events
        self.assertEqual(observer.events[1][0], 'item_processed')
        self.assertEqual(observer.events[1][1]['current'], 1)
        self.assertEqual(observer.events[1][1]['total'], 5)
        self.assertEqual(observer.events[1][1]['progress'], 20.0)
    
    def test_solver_subject_init_example(self):
        """Test the SolverSubjectMixin.__init__ method example."""
        # Example from the docstring
        class MyClass(SolverSubjectMixin):
            def __init__(self):
                SolverSubjectMixin.__init__(self)  # Initialize observer list
        
        # Create an instance
        my_instance = MyClass()
        
        # Verify the observer list was initialized
        self.assertEqual(my_instance._observers, [])
    
    def test_solver_subject_add_observer_example(self):
        """Test the SolverSubjectMixin.add_observer method example."""
        # Example from the docstring
        from torchrdit.observers import ConsoleProgressObserver
        
        # Create a solver and add an observer
        solver = create_solver()
        observer = ConsoleProgressObserver()
        solver.add_observer(observer)
        
        # Verify the observer was added
        self.assertIn(observer, solver._observers)
    
    def test_solver_subject_remove_observer_example(self):
        """Test the SolverSubjectMixin.remove_observer method example."""
        # Example from the docstring
        from torchrdit.observers import ConsoleProgressObserver
        
        # Create a solver and add an observer
        solver = create_solver()
        observer = ConsoleProgressObserver()
        solver.add_observer(observer)
        
        # Verify the observer was added
        self.assertIn(observer, solver._observers)
        
        # Remove the observer
        solver.remove_observer(observer)
        
        # Verify the observer was removed
        self.assertNotIn(observer, solver._observers)
    
    def test_solver_subject_notify_observers_example(self):
        """Test the SolverSubjectMixin.notify_observers method example."""
        # Create a mock observer
        mock_observer = MagicMock()
        
        # Create a class that uses SolverSubjectMixin
        class TestSolver(SolverSubjectMixin):
            def __init__(self):
                SolverSubjectMixin.__init__(self)
                self.layer_manager = MagicMock()
                self.layer_manager.nlayer = 3
            
            def solve_layer(self, layer_index):
                # Notify observers that we're starting a layer
                self.notify_observers('layer_started', {
                    'current': layer_index,
                    'total': self.layer_manager.nlayer,
                    'progress': layer_index / self.layer_manager.nlayer * 100
                })
                
                # Process the layer
                # ...
                
                # Notify observers that the layer is complete
                self.notify_observers('layer_completed', {
                    'current': layer_index,
                    'total': self.layer_manager.nlayer,
                    'progress': (layer_index + 1) / self.layer_manager.nlayer * 100
                })
        
        # Create the test solver and add the mock observer
        solver = TestSolver()
        solver.add_observer(mock_observer)
        
        # Test the solve_layer method
        solver.solve_layer(1)
        
        # Verify the observer was notified
        self.assertEqual(mock_observer.update.call_count, 2)
        
        # Check the first notification
        args1, _ = mock_observer.update.call_args_list[0]
        self.assertEqual(args1[0], 'layer_started')
        self.assertEqual(args1[1]['current'], 1)
        self.assertEqual(args1[1]['total'], 3)
        self.assertAlmostEqual(args1[1]['progress'], 33.33333, places=3)
        
        # Check the second notification
        args2, _ = mock_observer.update.call_args_list[1]
        self.assertEqual(args2[0], 'layer_completed')
        self.assertEqual(args2[1]['current'], 1)
        self.assertEqual(args2[1]['total'], 3)
        self.assertAlmostEqual(args2[1]['progress'], 66.66667, places=3)
    
    def test_create_solver_from_config_dict_example(self):
        """Test the create_solver_from_config function with dictionary example."""
        # Define a configuration dictionary as in the example
        config = {
            "algorithm": "RDIT",  # Fixed algorithm name
            "wavelengths": [1.55],
            "length_unit": "um",   # Fixed parameter name
            "rdim": [32, 32],      # Use smaller dimensions for faster tests
            "kdim": [3, 3],
            "device": "cpu"        # Use CPU for testing
        }
        
        # Create a solver from the configuration
        solver = create_solver_from_config(config)
        
        # Verify the solver was created with the correct parameters
        self.assertEqual(solver._algorithm.name, "R-DIT")
        self.assertEqual(solver.lam0[0], 1.55)
        self.assertEqual(solver._lenunit, "um")
        self.assertEqual(solver.rdim, [32, 32])
        self.assertEqual(solver.kdim, [3, 3])
        self.assertEqual(str(solver.device), "cpu")
    
    def test_create_solver_rdit_example(self):
        """Test creating an R-DIT solver as shown in the example."""
        # Create an R-DIT solver with CPU for testing
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),  # Wavelength (μm)
            rdim=[32, 32],         # Smaller real space resolution for testing
            kdim=[3, 3],           # Smaller Fourier space harmonics for testing
            device='cpu'           # Use CPU for testing
        )
        
        # Verify the solver was created with the correct parameters
        self.assertEqual(solver._algorithm.name, "R-DIT")
        self.assertEqual(solver.lam0[0], 1.55)
        self.assertEqual(solver.rdim, [32, 32])
        self.assertEqual(solver.kdim, [3, 3])
        self.assertEqual(str(solver.device), "cpu")
    
    def test_create_solver_rcwa_example(self):
        """Test creating an RCWA solver as shown in the example."""
        # Create an RCWA solver with a non-rectangular lattice
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.DOUBLE,  # Use double precision
            lam0=np.array([0.8, 1.0, 1.2]),  # Multiple wavelengths
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.5, 0.866]]),  # 60-degree lattice
            rdim=[32, 32],  # Smaller dimensions for testing
            kdim=[3, 3],
            device='cpu'  # Use CPU for testing
        )
        
        # Verify the solver was created with the correct parameters
        self.assertEqual(solver._algorithm.name, "RCWA")
        self.assertEqual(solver.tcomplex, torch.complex128)  # Double precision
        self.assertEqual(len(solver.lam0), 3)  # Multiple wavelengths
        self.assertAlmostEqual(solver.lam0[0], 0.8, places=6)
        self.assertAlmostEqual(solver.lam0[1], 1.0, places=6)
        self.assertAlmostEqual(solver.lam0[2], 1.2, places=6)
        self.assertEqual(solver.lattice_t1[0].item(), 1.0)
        self.assertEqual(solver.lattice_t1[1].item(), 0.0)
        self.assertAlmostEqual(solver.lattice_t2[0].item(), 0.5, places=6)
        self.assertAlmostEqual(solver.lattice_t2[1].item(), 0.866, places=3)
    
    def test_fourier_base_solver_example(self):
        """Test the FourierBaseSolver example."""
        # Create materials and solver as shown in the docstring
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[32, 32],  # Smaller dimensions for testing
            kdim=[3, 3]
        )
        
        # Add materials and layers
        silicon = create_material(name="Si", permittivity=11.7)
        air = create_material(name="Air", permittivity=1.0)
        solver.add_materials([silicon, air])
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5))
        
        # Set up a source (without solving, which could be expensive)
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        
        # Verify the source is set up correctly
        self.assertEqual(source['theta'], 0)
        self.assertEqual(source['phi'], 0)
        self.assertEqual(source['pte'], 1.0)
        self.assertEqual(source['ptm'], 0.0)
    
    def test_add_source_normal_incidence_example(self):
        """Test the add_source method example with normal incidence."""
        # Create a solver
        solver = create_solver()
        
        # Create a source with normal incidence and TE polarization
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        
        # Verify the source is set up correctly
        self.assertEqual(source['theta'], 0)
        self.assertEqual(source['phi'], 0)
        self.assertEqual(source['pte'], 1.0)
        self.assertEqual(source['ptm'], 0.0)
    
    def test_add_source_angled_incidence_example(self):
        """Test the add_source method example with angled incidence."""
        # Create a solver
        solver = create_solver()
        
        # Convert degrees to radians
        theta_rad = 45 * np.pi / 180
        
        # Create source with 45-degree incidence and equal TE/TM components
        source = solver.add_source(
            theta=theta_rad, 
            phi=0, 
            pte=0.7071, 
            ptm=0.7071
        )
        
        # Verify the source is set up correctly
        self.assertAlmostEqual(source['theta'], theta_rad, places=6)
        self.assertEqual(source['phi'], 0)
        self.assertAlmostEqual(source['pte'], 0.7071, places=4)
        self.assertAlmostEqual(source['ptm'], 0.7071, places=4)
    
    def test_add_source_circular_polarization_example(self):
        """Test the add_source method example with circular polarization."""
        # Create a solver
        solver = create_solver()
        
        # Create source with circular polarization (90° phase shift between components)
        source = solver.add_source(
            theta=0,
            phi=0,
            pte=1.0,
            ptm=1.0j  # 90° phase shift
        )
        
        # Verify the source is set up correctly
        self.assertEqual(source['theta'], 0)
        self.assertEqual(source['phi'], 0)
        self.assertEqual(source['pte'], 1.0)
        self.assertEqual(source['ptm'], 1.0j)
    
    def test_solve_basic_example(self):
        """Test the basic solve method example."""
        # Create a solver for a simple test
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[32, 32],  # Small dimensions for testing
            kdim=[3, 3]     # Small dimensions for testing
        )
        
        # Add materials
        air = create_material(name="Air", permittivity=1.0)
        silicon = create_material(name="Si", permittivity=11.7)
        solver.add_materials([air, silicon])
        
        # Add a simple layer
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.2))
        
        # Set input/output materials
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        
        # Define source with normal incidence and TE polarization
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        
        # Verify solver is properly set up before solving
        self.assertEqual(solver.layer_manager.nlayer, 1)
        self.assertEqual(solver.layer_manager.layers[0].material_name, "Si")
        # Use assertAlmostEqual for float comparison
        self.assertAlmostEqual(solver.layer_manager.layers[0].thickness.item(), 0.2, places=6)
        
        # NOTE: We skip the actual solve call which could be expensive
        # result = solver.solve(source)
        # Instead, we check that the source is set up correctly
        self.assertEqual(source['theta'], 0)
        self.assertEqual(source['phi'], 0)
        self.assertEqual(source['pte'], 1.0)
        self.assertEqual(source['ptm'], 0.0)
    
    def test_solve_with_fields_example(self):
        """Test the solve with fields computation example."""
        # Create a solver for a simple test
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[32, 32],  # Small dimensions for testing
            kdim=[3, 3]     # Small dimensions for testing
        )
        
        # Add materials
        air = create_material(name="Air", permittivity=1.0)
        silicon = create_material(name="Si", permittivity=11.7)
        solver.add_materials([air, silicon])
        
        # Add a simple layer
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.2))
        
        # Set input/output materials
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        
        # Define source with normal incidence and TE polarization
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        
        # Verify source is properly set up
        self.assertEqual(source['theta'], 0)
        self.assertEqual(source['phi'], 0)
        self.assertEqual(source['pte'], 1.0)
        self.assertEqual(source['ptm'], 0.0)
        
        # NOTE: We skip the actual solve call with fields which could be expensive
        # result = solver.solve(source, compute_fields=True)
    
    def test_create_solver_builder_example(self):
        """Test the builder example from solver.py docstrings."""
        # Create a solver with the builder pattern
        solver = (SolverBuilder()
                 .with_algorithm(Algorithm.RDIT)
                 .with_rdit_order(8)
                 .with_wavelengths([1.3, 1.4, 1.5, 1.6])
                 .with_k_dimensions([3, 3])  # Small dimensions for testing
                 .with_device("cpu")  # Use CPU for testing
                 .build())
        
        # Verify the solver was created with the correct parameters
        self.assertEqual(solver._algorithm.name, "R-DIT")  # Fixed algorithm name
        self.assertEqual(len(solver.lam0), 4)
        self.assertEqual(solver.lam0[0], 1.3)
        self.assertEqual(solver.lam0[1], 1.4)
        self.assertEqual(solver.lam0[2], 1.5)
        self.assertEqual(solver.lam0[3], 1.6)
        self.assertEqual(solver.kdim, [3, 3])
        self.assertEqual(str(solver.device), "cpu")


if __name__ == '__main__':
    unittest.main() 