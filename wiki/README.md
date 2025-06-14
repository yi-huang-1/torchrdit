# TorchRDIT Documentation

This directory contains automatically generated documentation for the TorchRDIT project.

## How to Update the Documentation

The documentation is generated automatically using pydoc-markdown. To update it:

1. Ensure your docstrings in the code are up-to-date
2. Run the documentation generator script: `python generate_docs.py`
3. The updated documentation will be created in this directory

## Documentation Structure

- **API-Overview.md**: Overview of the entire API
- **Algorithm.md**: Documentation for the algorithm module
- **Builder.md**: Documentation for the builder module
- **Cell.md**: Documentation for the cell module
- **Constants.md**: Documentation for physical constants and enumerations
- **Layers.md**: Documentation for the layers module
- **Materials.md**: Documentation for the materials module
- **MaterialProxy.md**: Documentation for the material data proxy and unit conversion
- **Observers.md**: Documentation for the observers module
- **Results.md**: Documentation for the results module
- **Shapes.md**: Documentation for the shapes module
- **Solver.md**: Documentation for the solver module
- **Utils.md**: Documentation for utility functions
- **Visualization.md**: Documentation for visualization tools
- **Getting-Started.md**: Guide for getting started with TorchRDIT
- **Examples.md**: Code examples showing how to use TorchRDIT
- **_Sidebar.md**: Navigation sidebar for the wiki

## How to Write Good Docstrings

For your documentation to be most effective, follow these guidelines for docstrings:

```python
def calculate_field(mesh, material_properties, frequency):
    """
    Calculate electromagnetic fields on a given mesh.
    
    Parameters
    ----------
    mesh : Mesh
        The computational mesh on which to calculate fields.
    material_properties : dict
        Dictionary mapping mesh elements to material properties:
        - 'epsilon': relative permittivity
        - 'mu': relative permeability
        - 'sigma': conductivity
    frequency : float
        Operating frequency in Hz.
        
    Returns
    -------
    ndarray
        Complex E-field values at mesh nodes.
        
    Notes
    -----
    This function solves Maxwell's equations using the FDTD method.
    The governing equation is:
    
    ∇ x (∇ x E) - k₀²εᵣE = 0
    
    where k₀ is the wavenumber and εᵣ is the relative permittivity.
    
    Examples
    --------
    ```python
    mesh = create_rectangular_mesh(0.1, 10, 10, 10)
    props = {'epsilon': 1.0, 'mu': 1.0, 'sigma': 0.0}
    E = calculate_field(mesh, props, 1e9)
    ```
    """
```

For classes, document the class purpose and all important methods.

## GitHub Wiki Integration

This documentation is designed to be automatically deployed to the GitHub wiki.
A GitHub Actions workflow in `.github/workflows/wiki.yml` handles this automatically
when changes are pushed to the main branch.

## Important Note

DO NOT modify these files directly. All files in this directory are automatically generated.
If you need to make changes, update the docstrings in the source code or modify the generator script.
