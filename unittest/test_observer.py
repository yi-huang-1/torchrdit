"""Unit tests for the observer pattern in the solver."""

import unittest
import torch
import numpy as np
from collections import defaultdict

from torchrdit.solver import SolverObserver, create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material


class EventCountingObserver(SolverObserver):
    """Observer that counts how many times each event has been called."""
    
    def __init__(self):
        self.event_counts = defaultdict(int)
        self.event_data = {}
        
    def update(self, event_type, data):
        self.event_counts[event_type] += 1
        self.event_data[event_type] = data
        
    def get_count(self, event_type):
        return self.event_counts.get(event_type, 0)
        
    def get_data(self, event_type):
        return self.event_data.get(event_type, None)


class TestObserverPattern(unittest.TestCase):
    """Test cases for the observer pattern in the solver."""
    
    # Constants for test configuration
    um = 1
    nm = 1e-3 * um
    
    lam0 = np.array([0.5, 0.6, 0.7])  # Three wavelengths in um
    rdim = [64, 64]  # Small spatial dimensions for speed
    kdim = [3, 3]    # Small k-space for speed
    
    # Material properties
    er_air = 1.0
    er_silicon = 12.0
    er_silica = 2.25
    
    # Layer thickness
    thickness = 0.5  # In um
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a small solver for testing
        self.solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=self.lam0,
            rdim=self.rdim,
            kdim=self.kdim
        )
        
        # Create and add materials
        air = create_material(name="air", permittivity=self.er_air, permeability=1.0)
        silicon = create_material(name="silicon", permittivity=self.er_silicon, permeability=1.0)
        silica = create_material(name="silica", permittivity=self.er_silica, permeability=1.0)
        self.solver.add_materials([air, silicon, silica])
        
        # Create a test structure
        self.solver.update_ref_material("air")  # Set ambient material
        self.solver.add_layer(  # Add silicon layer
            material_name="silicon",
            thickness=torch.tensor(self.thickness),
            is_homogeneous=True
        )
        self.solver.update_trn_material("silica")  # Set substrate material
        
        # Add a simple pattern to the silicon layer (half/half pattern)
        mask = torch.zeros(self.rdim)
        mask[:, :self.rdim[1]//2] = 1.0
        self.solver.update_er_with_mask(mask, 0)
        
        # Create the source
        self.source = self.solver.add_source(theta=0, phi=0, pte=1, ptm=0)
        
        # Create observer for tracking events
        self.observer = EventCountingObserver()
        self.solver.add_observer(self.observer)
    
    def test_batch_mode_events(self):
        """Test that events are properly triggered in batch frequency mode."""
        # Solve in batch mode
        self.solver.solve(self.source, is_solve_batch=True)
        
        # Verify calculation started and completed events
        self.assertEqual(self.observer.get_count("calculation_starting"), 1)
        self.assertEqual(self.observer.get_count("calculation_completed"), 1)
        
        # Verify information in calculation_starting event
        start_data = self.observer.get_data("calculation_starting")
        self.assertEqual(start_data["mode"], "solving_structure")
        self.assertEqual(start_data["n_freqs"], len(self.lam0))
        self.assertEqual(start_data["n_layers"], 1)
        
        # Verify layer processing events
        self.assertEqual(self.observer.get_count("processing_layers"), 1)
        self.assertEqual(self.observer.get_count("layer_started"), 1)
        self.assertEqual(self.observer.get_count("layer_completed"), 1)
        
        # Verify no frequency events were triggered in batch mode
        self.assertEqual(self.observer.get_count("processing_frequencies"), 0)
        self.assertEqual(self.observer.get_count("frequency_started"), 0)
        self.assertEqual(self.observer.get_count("frequency_completed"), 0)
    
    def test_multiple_observers(self):
        """Test that multiple observers all receive notifications."""
        # Create additional observers
        observer2 = EventCountingObserver()
        observer3 = EventCountingObserver()
        
        # Add observers to the solver
        self.solver.add_observer(observer2)
        self.solver.add_observer(observer3)
        
        # Solve in batch mode
        self.solver.solve(self.source, is_solve_batch=True)
        
        # Verify all observers received the same events
        self.assertEqual(
            self.observer.get_count("calculation_starting"),
            observer2.get_count("calculation_starting")
        )
        self.assertEqual(
            self.observer.get_count("calculation_completed"),
            observer3.get_count("calculation_completed")
        )
        self.assertEqual(
            observer2.get_count("layer_started"),
            observer3.get_count("layer_started")
        )
    
    def test_observer_removal(self):
        """Test that removing an observer stops it from receiving notifications."""
        # Create and add a second observer
        observer2 = EventCountingObserver()
        self.solver.add_observer(observer2)
        
        # Remove the second observer
        self.solver.remove_observer(observer2)
        
        # Solve in batch mode
        self.solver.solve(self.source, is_solve_batch=True)
        
        # Verify the first observer received events but the second didn't
        self.assertGreater(self.observer.get_count("calculation_starting"), 0)
        self.assertEqual(observer2.get_count("calculation_starting"), 0)


if __name__ == '__main__':
    unittest.main() 