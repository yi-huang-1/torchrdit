# TorchRDIT Documentation

Welcome to the `TorchRDIT` documentation. `TorchRDIT` is an advanced software package designed for the inverse design of meta-optics devices, utilizing an eigendecomposition-free implementation of Rigorous Diffraction Interface Theory (R-DIT). It provides a GPU-accelerated and fully differentiable framework powered by PyTorch, enabling the efficient optimization of photonic structures.

## Key Features

- **Differentiable**: Built on PyTorch for seamless integration with deep learning and optimization
- **High-Performance**: Efficient implementations of RCWA and RDIT algorithms
- **Flexible**: Support for complex geometries, dispersive materials, and various layer structures
- **Extensible**: Easy to integrate into machine learning and inverse design workflows

## Documentation

### User Guide

- [Getting Started](Getting-Started) - Installation and basic usage
- [Examples](Examples) - Detailed examples showing how to use TorchRDIT

### API Reference

- [API Overview](API-Overview)
- [Algorithm Module](Algorithm) - Implementation of electromagnetic solvers
- [Builder Module](Builder) - Fluent API for creating simulations
- [Cell Module](Cell) - Cell geometry definitions
- [Constants Module](Constants) - Physical constants and enumerations
- [Layers Module](Layers) - Layer definitions and operations
- [Materials Module](Materials) - Material property definitions
- [Material Proxy Module](MaterialProxy) - Material data handling and unit conversion
- [Observers Module](Observers) - Progress tracking and reporting
- [Results Module](Results) - Structured data containers for simulation results
- [Shapes Module](Shapes) - Shape generation for photonic structures
- [Solver Module](Solver) - Core solver functionality
- [Utils](Utils) - Utility functions
- [Visualization](Visualization) - Tools for visualizing results

## Examples

TorchRDIT comes with several example files in the `examples/` directory

For more detailed explanations of each example, see the [Examples](Examples) page.
