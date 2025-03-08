"""Implementations of observer classes for tracking solver progress."""

import time
from typing import Optional, Dict, Any

from .solver import SolverObserver

try:
    from tqdm import tqdm  # Optional dependency
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ConsoleProgressObserver(SolverObserver):
    """Observer that prints solver progress to the console."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the console observer.
        
        Args:
            verbose: Whether to print detailed progress messages (default: True)
        """
        self.verbose = verbose
        self.start_time = None
        
    def update(self, event_type: str, data: dict) -> None:
        """Handle notifications from the solver.
        
        Args:
            event_type: The type of event that occurred
            data: Additional data related to the event
        """
        if event_type == "calculation_starting":
            self.start_time = time.time()
            mode = data.get("mode", "unknown")
            n_freqs = data.get("n_freqs", 0)
            n_layers = data.get("n_layers", 0)
            print(f"Starting calculation in {mode} mode with {n_freqs} frequencies and {n_layers} layers...")
            
        elif event_type == "initializing_k_vectors" and self.verbose:
            print("Initializing k-vectors...")
            
        elif event_type == "setting_up_matrices" and self.verbose:
            print("Setting up matrices...")
            
        elif event_type == "processing_layers" and self.verbose:
            total = data.get("total", 0)
            print(f"Processing {total} layers...")
            
        elif event_type == "layer_started" and self.verbose:
            current = data.get("current", 0)
            total = data.get("total", 0)
            progress = data.get("progress", 0)
            print(f"Layer {current}/{total} ({progress:.1f}%)...")
                
        elif event_type == "connecting_external_regions" and self.verbose:
            print("Connecting external regions...")
                
        elif event_type == "calculating_fields" and self.verbose:
            print("Calculating fields...")
                
        elif event_type == "assembling_final_data" and self.verbose:
            print("Assembling final data...")
                
        elif event_type == "calculation_completed":
            elapsed_time = time.time() - self.start_time
            n_freqs = data.get("n_freqs", 0)
            print(f"Calculation completed in {elapsed_time:.2f} seconds for {n_freqs} frequencies.")


if TQDM_AVAILABLE:
    class TqdmProgressObserver(SolverObserver):
        """Observer that displays solver progress using tqdm progress bars."""
        
        def __init__(self):
            """Initialize the tqdm progress observer."""
            self.layer_pbar = None
            
        def update(self, event_type: str, data: dict) -> None:
            """Called when the solver notifies of an event.
            
            Args:
                event_type: The type of event that occurred
                data: Additional data related to the event
            """
            if event_type == "processing_layers":
                total = data.get("total", 0)
                self.layer_pbar = tqdm(total=total, desc="Layers", position=0, leave=False)
                
            elif event_type == "layer_completed":
                if self.layer_pbar is not None:
                    self.layer_pbar.update(1)
                    
            elif event_type == "calculation_completed":
                if self.layer_pbar is not None:
                    self.layer_pbar.close()
                    self.layer_pbar = None 