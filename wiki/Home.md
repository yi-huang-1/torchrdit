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

## New in v0.1.24: GDS Export/Import

TorchRDIT now includes industry-standard GDS file format support:
- Export photonic masks to GDS files for fabrication
- Import masks from GDS vertex data
- Support for complex topologies with holes
- Batch processing for multiple designs
- See the [GDS](GDS) page for details

## New in v0.1.22: Source Batching

TorchRDIT now supports efficient batched processing of multiple sources:
- Process multiple incident angles simultaneously
- 3-6x performance improvement for parameter sweeps
- Full gradient support for multi-condition optimization
- See the [Source Batching](SourceBatching) page for details

## Examples

TorchRDIT comes with several example files in the `examples/` directory

For more detailed explanations of each example, see the [Examples](Examples) page.
