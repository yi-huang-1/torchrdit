"""Module for batched electromagnetic simulation results.

This module provides data structures for organizing and analyzing results from
batched electromagnetic simulations where multiple sources are processed simultaneously.
It extends the single-source SolverResults to handle multiple incident conditions
efficiently.

Classes:
    BatchedFieldComponents: Container for batched field components.
    BatchedSolverResults: Main results container for batched simulations.

Examples:
Basic usage with batched sources:

```python
# Solve with multiple sources
sources = [
    solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0),
    solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
    solver.add_source(theta=45*deg, phi=0, pte=0.7, ptm=0.3)
]
batched_results = solver.solve(sources)

# Access bulk results
all_transmission = batched_results.transmission  # Shape: (3, n_freqs)

# Access individual results
result_0 = batched_results[0]  # Returns SolverResults for first source

# Find optimal source
best_idx = batched_results.find_optimal_source('max_transmission')
```
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Iterator, Optional
import torch
from .results import SolverResults, FieldComponents, ScatteringMatrix, WaveVectors


@dataclass
class BatchedFieldComponents:
    """Field components for batched electromagnetic simulations.

    Contains the spatial distribution of electromagnetic field components
    for multiple sources simultaneously.

    Attributes:
        x (torch.Tensor): X-component of the electromagnetic field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        y (torch.Tensor): Y-component of the electromagnetic field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        z (torch.Tensor): Z-component of the electromagnetic field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
    """

    x: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    y: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    z: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])

    def __getitem__(self, idx: int) -> FieldComponents:
        """Extract field components for a single source."""
        return FieldComponents(x=self.x[idx], y=self.y[idx], z=self.z[idx])


@dataclass
class BatchedSolverResults:
    """Results container for batched electromagnetic simulations.

    Comprehensive container for results from simulations with multiple sources,
    providing efficient storage and convenient access to both bulk and individual
    source results.

    Attributes:
        reflection (torch.Tensor): Total reflection efficiency for each source/wavelength.
            Shape: (n_sources, n_freqs)
        transmission (torch.Tensor): Total transmission efficiency for each source/wavelength.
            Shape: (n_sources, n_freqs)
        loss (torch.Tensor): Total loss for each source/wavelength.
            Shape: (n_sources, n_freqs)
        reflection_diffraction (torch.Tensor): Reflection efficiencies for each order.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        transmission_diffraction (torch.Tensor): Transmission efficiencies for each order.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        Erx (torch.Tensor): X-component of reflected E-field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        Ery (torch.Tensor): Y-component of reflected E-field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        Erz (torch.Tensor): Z-component of reflected E-field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        Etx (torch.Tensor): X-component of transmitted E-field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        Ety (torch.Tensor): Y-component of transmitted E-field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        Etz (torch.Tensor): Z-component of transmitted E-field.
            Shape: (n_sources, n_freqs, kdim[0], kdim[1])
        n_sources (int): Number of sources in the batch.
        source_parameters (List[Dict]): Original source dictionaries.
        structure_matrix (Optional[ScatteringMatrix]): Scattering matrix (source-independent).
        wave_vectors (Optional[WaveVectors]): Wave vectors (may be source-dependent).

    Examples:
    ```python
    # Access bulk results
    all_trans = batched_results.transmission  # (n_sources, n_freqs)

    # Get results for specific source
    source_2 = batched_results[2]  # Returns SolverResults

    # Iterate over all sources
    for i, result in enumerate(batched_results):
        print(f"Source {i}: T={result.transmission[0]:.3f}")

    # Find best performing source
    best_idx = batched_results.find_optimal_source('max_transmission')
    ```
    """

    # Overall efficiencies
    reflection: torch.Tensor  # (n_sources, n_freqs)
    transmission: torch.Tensor  # (n_sources, n_freqs)
    loss: torch.Tensor  # (n_sources, n_freqs)

    # Diffraction efficiencies
    reflection_diffraction: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    transmission_diffraction: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])

    # Field components - using individual tensors for backward compatibility
    Erx: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    Ery: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    Erz: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    Etx: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    Ety: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])
    Etz: torch.Tensor  # (n_sources, n_freqs, kdim[0], kdim[1])

    # Metadata
    n_sources: int
    source_parameters: List[Dict]

    # Optional shared data
    structure_matrix: Optional[ScatteringMatrix] = None
    wave_vectors: Optional[WaveVectors] = None

    def __len__(self) -> int:
        """Return number of sources in the batch."""
        return self.n_sources

    def __getitem__(self, idx: Union[int, slice]) -> Union[SolverResults, "BatchedSolverResults"]:
        """Get results for specific source(s).

        Args:
            idx: Integer index or slice for source selection.

        Returns:
            SolverResults for single index, BatchedSolverResults for slice.
        """
        if isinstance(idx, int):
            # Handle negative indexing
            if idx < 0:
                idx = self.n_sources + idx
            if idx < 0 or idx >= self.n_sources:
                raise IndexError(f"Source index {idx} out of range for {self.n_sources} sources")

            # Extract single source results
            return SolverResults(
                reflection=self.reflection[idx],
                transmission=self.transmission[idx],
                reflection_diffraction=self.reflection_diffraction[idx],
                transmission_diffraction=self.transmission_diffraction[idx],
                reflection_field=FieldComponents(x=self.Erx[idx], y=self.Ery[idx], z=self.Erz[idx]),
                transmission_field=FieldComponents(x=self.Etx[idx], y=self.Ety[idx], z=self.Etz[idx]),
                structure_matrix=self.structure_matrix,
                wave_vectors=self.wave_vectors,
                raw_data={},  # Could populate with source-specific data
            )

        elif isinstance(idx, slice):
            # Handle slicing
            indices = range(*idx.indices(self.n_sources))
            if len(indices) == 0:
                raise ValueError("Empty slice")

            return BatchedSolverResults(
                reflection=self.reflection[idx],
                transmission=self.transmission[idx],
                loss=self.loss[idx],
                reflection_diffraction=self.reflection_diffraction[idx],
                transmission_diffraction=self.transmission_diffraction[idx],
                Erx=self.Erx[idx],
                Ery=self.Ery[idx],
                Erz=self.Erz[idx],
                Etx=self.Etx[idx],
                Ety=self.Ety[idx],
                Etz=self.Etz[idx],
                n_sources=len(indices),
                source_parameters=[self.source_parameters[i] for i in indices],
                structure_matrix=self.structure_matrix,
                wave_vectors=self.wave_vectors,
            )

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self) -> Iterator[SolverResults]:
        """Iterate over individual source results."""
        for i in range(self.n_sources):
            yield self[i]

    def get_source_result(self, idx: int) -> SolverResults:
        """Get results for a specific source.

        Args:
            idx: Source index.

        Returns:
            SolverResults for the specified source.
        """
        return self[idx]

    @property
    def as_list(self) -> List[SolverResults]:
        """Get all results as a list of SolverResults objects."""
        return list(self)

    def find_optimal_source(self, metric: str = "max_transmission", frequency_idx: Optional[int] = None) -> int:
        """Find the source index that optimizes the specified metric.

        Args:
            metric: Optimization criterion. Options:
                - 'max_transmission': Maximum transmission
                - 'min_reflection': Minimum reflection
                - 'max_efficiency': Maximum total efficiency (T+R)
            frequency_idx: Specific frequency index to optimize for.
                If None, uses average over all frequencies.

        Returns:
            Index of the optimal source.

        Examples:
        ```python
        # Find source with highest transmission
        best_idx = results.find_optimal_source('max_transmission')
        best_result = results[best_idx]

        # Find source with lowest reflection at specific frequency
        best_idx = results.find_optimal_source('min_reflection', frequency_idx=0)
        ```
        """
        if frequency_idx is None:
            # Use average over frequencies
            if metric == "max_transmission":
                values = self.transmission.mean(dim=1)
                return torch.argmax(values).item()
            elif metric == "min_reflection":
                values = self.reflection.mean(dim=1)
                return torch.argmin(values).item()
            elif metric == "max_efficiency":
                values = (self.transmission + self.reflection).mean(dim=1)
                return torch.argmax(values).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            # Use specific frequency
            if metric == "max_transmission":
                return torch.argmax(self.transmission[:, frequency_idx]).item()
            elif metric == "min_reflection":
                return torch.argmin(self.reflection[:, frequency_idx]).item()
            elif metric == "max_efficiency":
                values = self.transmission[:, frequency_idx] + self.reflection[:, frequency_idx]
                return torch.argmax(values).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

    def get_parameter_sweep_data(
        self, parameter: str, metric: str, frequency_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract parameter sweep data for plotting.

        Args:
            parameter: Source parameter name ('theta', 'phi', 'pte', 'ptm').
            metric: Result metric ('transmission', 'reflection', 'loss').
            frequency_idx: Frequency index to extract data for.

        Returns:
            Tuple of (parameter_values, metric_values) tensors.

        Examples:
        ```python
        # Get data for angle sweep
        angles, trans = results.get_parameter_sweep_data('theta', 'transmission')

        # Plot the sweep
        plt.plot(angles * 180/np.pi, trans)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Transmission')
        ```
        """
        # Extract parameter values
        param_values = torch.tensor([src[parameter] for src in self.source_parameters])

        # Extract metric values
        if metric == "transmission":
            metric_values = self.transmission[:, frequency_idx]
        elif metric == "reflection":
            metric_values = self.reflection[:, frequency_idx]
        elif metric == "loss":
            metric_values = self.loss[:, frequency_idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return param_values, metric_values
