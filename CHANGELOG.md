# Changelog

All notable changes to TorchRDIT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.26] - 2025-07-25

### Added

- **Plasmonic Material Stabilization**: Automatic stabilization for materials near plasmon resonance
  - Prevents matrix singularities at ε ≈ -1.0
  - Configurable parameters via `stabilization_params` dictionary
  - Custom `min_loss` and `threshold` settings for specific applications
  - Full backward compatibility with sensible defaults
  - Physical damping based on industry-standard approach

- **Extreme Grazing Incidence Protection**: Numerical stability for extreme angles
  - Automatic protection for kinc_z underflow at θ > 89.5°
  - Configurable `min_kinc_z` threshold for fine-tuning
  - Prevents numerical breakdown in single precision
  - Maintains energy conservation for protected cases

- **Differentiable Stability Operations**: Gradient-preserving alternatives
  - `softplus_protect_kz()` for complex kz protection
  - `softplus_clamp_min()` for real value clamping
  - Smooth transitions prevent gradient discontinuities
  - Full differentiability for optimization workflows

- **Comprehensive Test Suites**:
  - `test_plasmonic_stability.py`: 13 tests for material stabilization
  - `test_extreme_grazing_incidence.py`: 4 tests for angle protection
  - `test_differentiable_stability.py`: 11 tests for gradient flow
  - All tests pass without warnings or skips

### Changed

- Material class now accepts `stabilization_params` for customization
- Helper functions export properly with `__all__` list in materials.py
- Tensor operations use PyTorch native functions (NumPy 2.0 compatibility)

### Fixed

- Matrix singularity at plasmon resonance (ε = -1.0) now prevented
- Numerical underflow at extreme grazing incidence (single precision)
- NumPy 2.0 deprecation warnings in tensor conversions
- All previously skipped tests now pass

## [0.1.25] - 2025-01-24

### Changed

- **Migrated from gdspy to gdstk**: Replaced legacy gdspy library with modern gdstk for improved platform compatibility
  - gdstk provides better build support across different platforms
  - Maintains full backward compatibility with existing GDS functionality
  - All GDS tests pass without modification
  - IoU metrics remain > 0.9 for all test shapes
  - No API changes visible to end users

### Removed

- gdspy dependency (replaced with gdstk v0.9.60)

## [0.1.24] - 2025-01-24

### Added

- **GDS Export/Import Functionality**: Industry-standard GDSII format support for fabrication workflows
  - `mask_to_gds()`: Export binary masks to GDS files with JSON vertex data
  - `gds_to_mask()`: Import masks from GDS JSON files  
  - `load_gds_vertices()`: Load polygon vertices from JSON files
  - Support for complex topologies including holes and disconnected regions
  - Batch processing for multiple masks
  - Smart coordinate extrapolation for shapes extending beyond boundaries
  - Both Cartesian and non-Cartesian lattice system support
  - High-fidelity reconstruction (IoU > 0.9)
  - Modern pathlib support for all file operations
- Material API optimizations with Akima1DInterpolator for improved interpolation stability
- LRU caching for dispersive material queries with configurable cache size
- Comprehensive test suite for material interpolation
- GDS export example demonstrating complex topologies

### Changed

- Material property interpolation now uses Akima splines by default with scipy fallback
- Improved memory efficiency for repeated material property queries
- Added scikit-image dependency for GDS contour extraction

## [0.1.23] - 2025-01-21

### Added

- **Fully Tensorized Source Batching**: Complete GPU-parallel processing across sources
  - Unified `_pre_solve` and `_pre_solve_batched` methods
  - Unified `_initialize_k_vectors` for single and batched sources
  - Unified `_process_layer` and `_process_layer_batched` methods
  - Unified `solve` and `_solve_batched` methods

### Changed

- Eliminated ~500 lines of duplicate code through unification
- Improved performance with full PyTorch broadcasting
- Fixed dispersive homogeneous layer tensor indexing bug

### Fixed

- 0-dimensional tensor handling in dispersive layers
- `redhstar` now supports 4D tensors for batched sources

## [0.1.22] - 2025-01-20

### Added
- **Source Batching**: Process multiple incident angles and polarizations simultaneously
  - New `BatchedSolverResults` class for handling results from multiple sources
  - `solver.solve()` now accepts a list of sources for batch processing
  - Helper methods: `find_optimal_source()` and `get_parameter_sweep_data()`
  - Full gradient support for multi-source optimization workflows
  - Comprehensive tutorial notebook: `examples/source_batching_tutorial.ipynb`

### Changed
- `solver.solve()` method enhanced to handle both single sources and source lists
- Performance improvements: 3-6x speedup for multiple source simulations
- Memory optimization for large parameter sweeps

### Fixed
- Improved memory management for GPU operations
- Enhanced validation for source parameters

### Documentation
- Added comprehensive source batching tutorial notebook
- Updated API documentation for new classes and methods

## [0.1.21] - 2024-12-15

### Added
- Migration to uv package manager
- Context7 MCP support for enhanced documentation
- Basic benchmark framework

### Changed
- Code quality improvements with ruff linter
- Updated dependencies to latest versions

### Fixed
- Various bug fixes and stability improvements

## [0.1.20] - 2024-11-30

### Added
- Performance optimizations in R-DIT algorithm
- Enhanced GPU memory management
- Support for Python 3.13

### Changed
- Optimized algorithm.py for better performance
- Improved error handling in solver operations

## [0.1.19] - 2024-11-15

### Fixed
- Fixed bugs when using 'air' as input materials
- Fixed shapes APIs not accepting non-tensor angle arguments

### Documentation
- Updated README with modified examples
- Generated updated wiki pages

## [0.1.18] - 2024-11-01

### Added
- New APIs for results handling with enhanced functionality
- Shapes API for differentiable geometric operations
- Improved docstring documentation

### Changed
- Enhanced results processing capabilities
- Better integration with optimization workflows

## [0.1.17] - 2024-10-15

### Added
- Support for more complex material distributions
- Enhanced visualization capabilities
- Improved observer pattern implementation

### Fixed
- Memory leaks in long-running simulations
- Edge cases in material property calculations

## [0.1.16] - 2024-10-01

### Added
- Support for anisotropic materials
- Enhanced builder pattern for solver construction
- Improved error messages and debugging information

### Changed
- Optimized memory usage for large simulations
- Better handling of edge cases in layer processing

## [0.1.15] - 2024-09-15

### Added
- Support for non-uniform grids
- Enhanced material library with more built-in materials
- Improved gradient computation stability

### Fixed
- Numerical stability issues in extreme parameter regimes
- GPU memory management for very large simulations

## [0.1.14] - 2024-09-01

### Added
- Support for oblique incident angles
- Enhanced polarization handling
- Improved convergence monitoring

### Changed
- Optimized Fourier space calculations
- Better memory management for GPU operations

## [0.1.13] - 2024-08-15

### Added
- Support for dispersive materials with frequency-dependent properties
- Material data loading from external files
- Enhanced unit test coverage

### Fixed
- Issues with complex material property interpolation
- Edge cases in multi-layer stack calculations

## [0.1.12] - 2024-08-01

### Added
- R-DIT solver implementation with eigendecomposition-free algorithm
- 16.2x speedup over traditional RCWA
- Full automatic differentiation support

### Changed
- Major performance improvements in core algorithms
- Enhanced GPU utilization

## [0.1.11] - 2024-07-15

### Added
- Multi-wavelength simulation capabilities
- Batch processing for parameter sweeps
- Enhanced visualization tools

### Fixed
- Memory efficiency issues for large-scale simulations
- Gradient computation accuracy

## [0.1.10] - 2024-07-01

### Added
- Support for arbitrary unit cell geometries
- Enhanced shape generation utilities
- Improved documentation and examples

### Changed
- Refactored core solver architecture
- Better separation of concerns in module design

---

For older versions, please refer to the [release history](https://github.com/yi-huang-1/torchrdit/releases) on GitHub.
