"""Observer implementations for tracking and reporting solver progress.

This module provides concrete implementations of the SolverObserver interface
for monitoring the progress of electromagnetic simulations. These observers
follow the Observer design pattern, allowing them to receive notifications
from solvers without tight coupling.

The module includes:
- ConsoleProgressObserver: Prints progress information to the console
- TqdmProgressObserver: Displays a progress bar using the tqdm library

These observers can be attached to solvers to provide real-time feedback
during potentially long-running simulations, making it easier to monitor
the progress and estimated completion time of calculations.

Examples:
```python
from torchrdit.solver import create_solver
from torchrdit.observers import ConsoleProgressObserver
from torchrdit.constants import Algorithm

# Create a solver and add a console observer
solver = create_solver(algorithm=Algorithm.RCWA)
observer = ConsoleProgressObserver(verbose=True)
solver.add_observer(observer)

# Run the solver - progress will be reported to the console
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source)
```

Using a progress bar with tqdm:
```python
from torchrdit.solver import create_solver
from torchrdit.observers import TqdmProgressObserver

# Create a solver and add a progress bar observer
solver = create_solver(algorithm=Algorithm.RCWA)
observer = TqdmProgressObserver()
solver.add_observer(observer)

# Run the solver - progress will be displayed with a progress bar
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source)
```

Keywords:
    observer pattern, progress tracking, simulation monitoring, console output,
    progress bar, solver feedback, long-running calculation, tqdm
"""
import time
from typing import Optional, Dict, Any

from .solver import SolverObserver

try:
    from tqdm import tqdm  # Optional dependency
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ConsoleProgressObserver(SolverObserver):
    """Observer that prints solver progress to the console.
    
    This class implements the SolverObserver interface to provide real-time
    feedback on solver progress through console output. It tracks the start
    time of the solving process and reports various events such as layer
    processing, iteration completion, and overall solve completion.
    
    When verbose mode is enabled, detailed information about each step of
    the solving process is printed. Otherwise, only major milestones are
    reported.
    
    Attributes:
        verbose (bool): Whether to print detailed progress messages.
        start_time (float): The timestamp when calculation started.
    
    Examples:
    ```python
    from torchrdit.solver import create_solver
    from torchrdit.observers import ConsoleProgressObserver
    
    # Create a solver
    solver = create_solver()
    
    # Add a verbose observer for detailed output
    verbose_observer = ConsoleProgressObserver(verbose=True)
    solver.add_observer(verbose_observer)
    
    # Or use a non-verbose observer for minimal output
    minimal_observer = ConsoleProgressObserver(verbose=False)
    solver.add_observer(minimal_observer)
    
    # Run the solver - progress will be reported to the console
    source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
    result = solver.solve(source)
    ```
    
    Note:
        This observer is useful for tracking long-running simulations and
        for debugging purposes, as it provides visibility into the internal
        stages of the solver's execution.
    
    Keywords:
        console output, progress reporting, solver monitoring, text output,
        verbose logging, simulation tracking, calculation time, observer pattern
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the console observer.
        
        Args:
            verbose (bool): Whether to print detailed progress messages.
                     When True, all steps of the solving process are reported.
                     When False, only major milestones are reported.
                     Default is True.
        
        Examples:
        ```python
        from torchrdit.observers import ConsoleProgressObserver
        from torchrdit.solver import create_solver
        solver = create_solver()
        # Create a verbose observer
        verbose_observer = ConsoleProgressObserver(verbose=True)
        solver.add_observer(verbose_observer)
        
        # Create a minimal observer
        minimal_observer = ConsoleProgressObserver(verbose=False)
        solver.add_observer(minimal_observer)
        ```
        
        Keywords:
            initialization, verbosity setting, observer creation
        """
        self.verbose = verbose
        self.start_time = None
        
    def update(self, event_type: str, data: dict) -> None:
        """Handle notifications from the solver.
        
        This method is called by the solver when significant events occur
        during the solving process. It interprets the event type and 
        associated data to print appropriate progress messages to the console.
        
        Args:
            event_type (str): The type of event that occurred. Common event types include:
                      - "calculation_starting": Triggered when the solve process begins
                      - "initializing_k_vectors": When setting up k-vectors
                      - "setting_up_matrices": When preparing matrices for calculation
                      - "processing_layers": When starting to process material layers
                      - "layer_started": When beginning work on a specific layer
                      - "connecting_external_regions": When connecting external regions
                      - "calculating_fields": When calculating fields
                      - "assembling_final_data": When preparing the final result
                      - "calculation_completed": When the solve process is finished
            data (dict): Additional data related to the event. The contents vary
                  by event type but may include:
                  - "mode": Solver mode
                  - "n_freqs": Number of frequencies being solved
                  - "n_layers": Number of layers in the structure
                  - "current": Current layer index
                  - "total": Total number of items (e.g., layers)
                  - "progress": Percentage complete
        
        Examples:
        ```python
        from torchrdit.observers import ConsoleProgressObserver
        from torchrdit.solver import create_solver
        solver = create_solver()
        observer = ConsoleProgressObserver()
        solver.add_observer(observer)
        # Example of how the solver would call this method
        observer.update(
            "calculation_starting", 
            {"mode": "RCWA", "n_freqs": 1, "n_layers": 3}
        )
        ```
        
        Note:
            This method is part of the Observer pattern implementation and
            is called automatically by the solver when events occur.
        
        Keywords:
            event handling, progress notification, console output, observer update,
            event types, calculation progress
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
        """Observer that displays solver progress using tqdm progress bars.
        
        This class implements the SolverObserver interface to provide visual
        feedback on solver progress through tqdm progress bars. It creates
        progress bars for layer processing and other iterative operations,
        making it easy to track the progress of long-running simulations.
        
        Attributes:
            layer_pbar (tqdm): Progress bar for tracking layer processing.
        
        Note:
            This observer requires the tqdm package to be installed. If tqdm
            is not available, this class will not be defined.
            
        Examples:
        ```python
        from torchrdit.solver import create_solver
        from torchrdit.observers import TqdmProgressObserver
        
        # Create a solver
        solver = create_solver()
        
        # Add a progress bar observer
        observer = TqdmProgressObserver()
        solver.add_observer(observer)
        
        # Run the solver - progress will be displayed with a progress bar
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        result = solver.solve(source) # SolverResult object
        ```
            
        Keywords:
            progress bar, tqdm, visual feedback, solver monitoring, progress tracking,
            layer processing, user interface, interactive display, observer pattern
        """
        
        def __init__(self):
            """Initialize the tqdm progress observer.
            
            Creates a new progress bar observer without displaying any progress bars.
            Progress bars are created when appropriate events are received.
            
            Examples:
            ```python
            from torchrdit.observers import TqdmProgressObserver
            # Create a progress bar observer
            observer = TqdmProgressObserver()
            # Progress bars will be created when events are received
            ```
            
            Keywords:
                initialization, progress bar, observer creation
            """
            self.layer_pbar = None
            
        def update(self, event_type: str, data: dict) -> None:
            """Called when the solver notifies of an event.
            
            This method is called by the solver when significant events occur
            during the solving process. It creates, updates, or closes tqdm
            progress bars based on the event type and associated data.
            
            Args:
                event_type (str): The type of event that occurred. Main event types handled:
                          - "processing_layers": Creates a progress bar for layer processing
                          - "layer_completed": Updates the layer progress bar
                          - "calculation_completed": Closes all progress bars
                data (dict): Additional data related to the event, which may include:
                      - "total": Total number of items (e.g., layers)
            
            Examples:
            ```python
            from torchrdit.observers import TqdmProgressObserver
            # This method is typically called by the solver, not directly
            observer = TqdmProgressObserver()
            # Example of how the solver would call this method
            observer.update("processing_layers", {"total": 5})
            # Later when a layer is completed
            observer.update("layer_completed", {})
            # Finally when calculation is done
            observer.update("calculation_completed", {})
            ```
            
            Note:
                This method is part of the Observer pattern implementation and
                is called automatically by the solver when events occur.
            
            Keywords:
                event handling, progress bar update, tqdm, observer update,
                progress tracking, event types
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