# Observer Module

## Overview
The `torchrdit.observers` module provides implementations of the Observer pattern for tracking and reporting progress during electromagnetic simulations.

## Key Components
The module includes several observer classes for monitoring simulation progress:

- **ConsoleProgressObserver**: Prints progress information to the console
- **TqdmProgressObserver**: Displays a progress bar using the tqdm library (if available)

## Usage Examples

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

## API Reference

Below is the complete API reference for the observers module, automatically generated from the source code.

# Table of Contents

* [torchrdit.observers](#torchrdit.observers)
  * [ConsoleProgressObserver](#torchrdit.observers.ConsoleProgressObserver)
    * [\_\_init\_\_](#torchrdit.observers.ConsoleProgressObserver.__init__)
    * [update](#torchrdit.observers.ConsoleProgressObserver.update)

<a id="torchrdit.observers"></a>

# torchrdit.observers

Observer implementations for tracking and reporting solver progress.

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

**Examples**:

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

<a id="torchrdit.observers.ConsoleProgressObserver"></a>

## ConsoleProgressObserver Objects

```python
class ConsoleProgressObserver(SolverObserver)
```

Observer that prints solver progress to the console.

This class implements the SolverObserver interface to provide real-time
feedback on solver progress through console output. It tracks the start
time of the solving process and reports various events such as layer
processing, iteration completion, and overall solve completion.

When verbose mode is enabled, detailed information about each step of
the solving process is printed. Otherwise, only major milestones are
reported.

**Attributes**:

- `verbose` _bool_ - Whether to print detailed progress messages.
- `start_time` _float_ - The timestamp when calculation started.
  

**Examples**:

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
  

**Notes**:

  This observer is useful for tracking long-running simulations and
  for debugging purposes, as it provides visibility into the internal
  stages of the solver's execution.
  
  Keywords:
  console output, progress reporting, solver monitoring, text output,
  verbose logging, simulation tracking, calculation time, observer pattern

<a id="torchrdit.observers.ConsoleProgressObserver.__init__"></a>

#### \_\_init\_\_

```python
def __init__(verbose: bool = True)
```

Initialize the console observer.

**Arguments**:

- `verbose` _bool_ - Whether to print detailed progress messages.
  When True, all steps of the solving process are reported.
  When False, only major milestones are reported.
  Default is True.
  

**Examples**:

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

<a id="torchrdit.observers.ConsoleProgressObserver.update"></a>

#### update

```python
def update(event_type: str, data: dict) -> None
```

Handle notifications from the solver.

This method is called by the solver when significant events occur
during the solving process. It interprets the event type and
associated data to print appropriate progress messages to the console.

**Arguments**:

- `event_type` _str_ - The type of event that occurred. Common event types include:
  - "calculation_starting": Triggered when the solve process begins
  - "initializing_k_vectors": When setting up k-vectors
  - "setting_up_matrices": When preparing matrices for calculation
  - "processing_layers": When starting to process material layers
  - "layer_started": When beginning work on a specific layer
  - "connecting_external_regions": When connecting external regions
  - "calculating_fields": When calculating fields
  - "assembling_final_data": When preparing the final result
  - "calculation_completed": When the solve process is finished
- `data` _dict_ - Additional data related to the event. The contents vary
  by event type but may include:
  - "mode": Solver mode
  - "n_freqs": Number of frequencies being solved
  - "n_layers": Number of layers in the structure
  - "current": Current layer index
  - "total": Total number of items (e.g., layers)
  - "progress": Percentage complete
  

**Examples**:

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
  

**Notes**:

  This method is part of the Observer pattern implementation and
  is called automatically by the solver when events occur.
  
  Keywords:
  event handling, progress notification, console output, observer update,
  event types, calculation progress

