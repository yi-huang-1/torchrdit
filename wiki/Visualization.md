# Table of Contents

* [torchrdit.viz](#torchrdit.viz)
  * [plot2d](#torchrdit.viz.plot2d)
  * [plot\_layer](#torchrdit.viz.plot_layer)
  * [display\_fitted\_permittivity](#torchrdit.viz.display_fitted_permittivity)
  * [plot\_cross\_section](#torchrdit.viz.plot_cross_section)

<a id="torchrdit.viz"></a>

# torchrdit.viz

Visualization module for TorchRDIT electromagnetic simulations.

This module provides functions for visualizing simulation results, material properties,
and field distributions in TorchRDIT. It includes tools for:

1. Plotting 2D field distributions (electric fields, magnetic fields)
2. Visualizing material property distributions in layers
3. Displaying dispersive material properties
4. Analyzing fitted dispersive material models
5. Cross-section visualization of layer stack structures

The visualization functions in this module work with matplotlib to create
publication-quality figures for analyzing simulation results and understanding
the behavior of electromagnetic fields in complex structures.

Key functions:
- plot2d: General-purpose function for plotting 2D data with customizable options
- plot_layer: Visualize the permittivity distribution in a specific layer
- plot_cross_section: Visualize cross-section of layer stack structure (XZ/YZ planes)
- display_fitted_permittivity: Plot fitted dispersive material properties in a simulation
- display_dispersive_profile: Visualize raw dispersive material data

All functions return matplotlib objects that can be further customized as needed.

Keywords:
    visualization, plotting, matplotlib, field distribution, permittivity,
    material properties, dispersive materials, electromagnetic simulation,
    cross-section, layer stack

<a id="torchrdit.viz.plot2d"></a>

#### plot2d

```python
def plot2d(data: Any,
           layout: Any,
           func="abs",
           outline=None,
           fig_ax=None,
           cbar=True,
           cmap="magma",
           outline_alpha=0.5,
           title="",
           labels=("x", "y"))
```

Create a 2D visualization of electromagnetic field data or material distributions.

This function generates a 2D plot of field data or material properties (like
permittivity) using matplotlib's pcolormesh. It handles both real and complex data,
applying the specified function (abs, real, imag) for visualization. The function
can also overlay an outline on the plot to highlight structural boundaries.

**Arguments**:

- `data` - 2D data to plot (torch.Tensor or numpy.ndarray). Can be complex-valued.
  For field data, this is typically the electric or magnetic field.
  For material data, this is typically the permittivity distribution.
- `layout` - Tuple of mesh grid coordinates (X, Y) defining the spatial coordinates
  for each point in the data. Should be obtained from sim.get_layout()
  or similar function.
- `func` - Function to apply to the data for visualization if the data is complex.
  Options are 'abs' (magnitude), 'real' (real part), 'imag' (imaginary part),
  'log10' (log scale), or other numpy/torch functions.
  Default is 'abs'.
- `outline` - Optional binary mask (0s and 1s) to overlay as an outline on the plot.
  This is useful for highlighting boundaries between different materials.
  Default is None (no outline).
- `fig_ax` - Matplotlib figure and axis to use for plotting. If None, a new
  figure and axis will be created.
  Default is None.
- `cbar` - Whether to include a colorbar with the plot.
  Default is True.
- `cmap` - Matplotlib colormap to use for the data visualization.
  Default is 'magma'.
- `outline_alpha` - Transparency of the outline overlay (0-1).
  Default is 0.5.
- `title` - Title for the plot.
  Default is '' (no title).
- `labels` - Tuple of (x-axis label, y-axis label).
  Default is ('x', 'y').
  

**Returns**:

- `matplotlib.axes.Axes` - The matplotlib axes object containing the plot,
  allowing for further customization if needed.
  

**Examples**:

```python
from torchrdit.solver import create_solver
from torchrdit.viz import plot2d
import matplotlib.pyplot as plt

# Create a solver
solver = create_solver()
# Plot the real part of the electric field distribution
x_grid, y_grid = solver.get_layout()
results = solver.solve(solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0))
e_field = results['tx']  # Transmission field x-component

fig, ax = plt.subplots(figsize=(8, 6))
plot2d(data=e_field,
        layout=(x_grid, y_grid),
        func='real',
        cmap='RdBu_r',
        title='Transmission Field (X) Distribution',
        labels=('x (μm)', 'y (μm)'))
plt.show()
```
  
  Keywords:
  visualization, plotting, field distribution, color map, electromagnetic,
  pcolormesh, complex data, contour, outline, colorbar, 2D plot

<a id="torchrdit.viz.plot_layer"></a>

#### plot\_layer

```python
def plot_layer(sim,
               layer_index: int = 0,
               frequency_index: int = 0,
               fig_ax=None,
               func="real",
               cmap="BuGn",
               labels=("x", "y"),
               title="")
```

Visualize the permittivity distribution of a specific layer in the simulation.

This function creates a 2D plot of the permittivity (ε) distribution within a
specified layer of the simulation structure. For dispersive materials, the
permittivity at a specific frequency is plotted.

**Arguments**:

- `sim` - A simulation object (typically a FourierBaseSolver instance) that
  contains the layer structure to be visualized.
- `layer_index` - Index of the layer to plot (0-based indexing).
  Default is 0 (first layer).
- `frequency_index` - Index of the frequency to use for dispersive materials.
  Only relevant for frequency-dependent (dispersive) materials.
  Default is 0 (first frequency).
- `fig_ax` - Matplotlib figure and axis to use for plotting. If None, a new
  figure and axis will be created. Default is None.
- `func` - Function to apply to the permittivity data before plotting.
  Options include 'real', 'imag', 'abs', 'log10', etc.
  Default is 'real' (plot real part of permittivity).
- `cmap` - Matplotlib colormap to use for the plot.
  Default is 'BuGn'.
- `labels` - Tuple of (x-axis label, y-axis label).
  Default is ('x', 'y').
- `title` - Title for the plot. If empty, a default title with layer information
  will be used. Default is ''.
  

**Returns**:

- `matplotlib.axes.Axes` - The matplotlib axes object containing the plot,
  allowing for further customization if needed.
  

**Examples**:

```python
# Create a simulation with multiple layers
solver = create_solver(algorithm=Algorithm.RDIT)
solver.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
solver.add_layer(material_name='sio2', thickness=torch.tensor(0.3))

# Plot the permittivity distribution of the second layer
fig, ax = plt.subplots(figsize=(8, 6))
plot_layer(solver, layer_index=1, fig_ax=ax,
            title='SiO2 Layer', labels=('x (μm)', 'y (μm)'))
plt.show()
```
  
  Keywords:
  permittivity, layer visualization, material distribution, RCWA, R-DIT,
  electromagnetic simulation, layer structure, dispersive materials

<a id="torchrdit.viz.display_fitted_permittivity"></a>

#### display\_fitted\_permittivity

```python
def display_fitted_permittivity(sim, fig_ax=None)
```

Visualize fitted permittivity for dispersive materials in the simulation.

This function plots the real and imaginary parts of the permittivity for
dispersive materials in the simulation, showing both the original data points
and the fitted curves used in the simulation. This is useful for verifying
the quality of the polynomial fit to the material dispersion data.

The plot includes:
- Original data points for the real part (ε')
- Fitted curve for the real part at simulation wavelengths
- Original data points for the imaginary part (ε")
- Fitted curve for the imaginary part at simulation wavelengths

**Arguments**:

- `sim` - A simulation object (typically a FourierBaseSolver instance) that
  contains the dispersive materials to be visualized.
- `fig_ax` - Matplotlib axis to use for plotting. If None, a new figure and
  axis will be created. Default is None.
  

**Returns**:

- `tuple` - A tuple containing (primary_axis, secondary_axis) where:
  - primary_axis: The main matplotlib axis (for real part)
  - secondary_axis: The secondary y-axis (for imaginary part)
  

**Notes**:

  This function automatically detects all dispersive materials in the simulation
  and plots their fitted permittivity data. If no dispersive materials are found,
  it prints a message indicating this.
  

**Examples**:

```python
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.viz import display_fitted_permittivity
import torch
import matplotlib.pyplot as plt

# Create a simulation with a dispersive material (e.g., gold)
solver = create_solver(
    algorithm=Algorithm.RDIT,
    lam0=[1.55],
    lengthunit='um',
    rdim=[512, 512],
    kdim=[5, 5],
    device='cpu'
)
gold = create_material(name='gold', dielectric_dispersion=True,
                        user_dielectric_file='gold_data.txt')
solver.add_materials([gold])
solver.add_layer(material_name='gold', thickness=torch.tensor(0.1))

# Visualize the fitted permittivity
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax1, ax2 = display_fitted_permittivity(solver, fig_ax=ax)
ax1.set_xlabel('Wavelength (μm)')
ax1.set_ylabel('Real Part (ε')')
ax2.set_ylabel('Imaginary Part (ε")')
plt.title('Gold Permittivity vs Wavelength')
plt.show()
```
  
  Keywords:
  dispersive material, permittivity, wavelength dependence, polynomial fitting,
  material characterization, complex permittivity, optical properties,
  Drude-Lorentz model, material dispersion

<a id="torchrdit.viz.plot_cross_section"></a>

#### plot\_cross\_section

```python
def plot_cross_section(sim,
                       plane: str = "xz",
                       slice_position: float = 0.0,
                       frequency_index: int = 0,
                       fig_ax=None,
                       material_colormap: Optional[Dict[str, str]] = None,
                       labels: Optional[Tuple[str, str]] = None,
                       title: str = "",
                       layer_boundaries: bool = True,
                       z_scale: float = 1.0,
                       show_semi_infinite: bool = True,
                       show_legend: bool = True)
```

Visualize cross-section of layer stack structure.

Creates a 2D cross-sectional view of the layer stack showing material
distributions with different colors. Supports both XZ-plane (y=0) and
YZ-plane (x=0) visualizations. Includes reflection and transmission
semi-infinite regions for complete electromagnetic domain visualization.

This function helps users understand the structure of their layered
electromagnetic simulation domains by showing different materials with
distinct colors and transparent air/vacuum regions. It integrates with
the existing TorchRDIT solver workflow and coordinate systems.

**Arguments**:

- `sim` - Solver instance with configured layers and materials.
  Must have layer_manager with layers and get_layout() method.
- `plane` - Cross-section plane ('xz' for XZ-plane or 'yz' for YZ-plane).
  Default is 'xz'.
- `slice_position` - Position along the slicing axis (y-coordinate for XZ plane,
  x-coordinate for YZ plane). Default is 0.0.
- `frequency_index` - Index for selecting frequency in dispersive materials.
  Only relevant for frequency-dependent materials. Default is 0.
- `fig_ax` - Matplotlib axis object to use for plotting. If None, a new
  figure and axis will be created. Default is None.
- `material_colormap` - Dictionary mapping material names to color strings.
  If None, uses default distinguishable colors. Default is None.
- `labels` - Tuple of (horizontal_axis_label, vertical_axis_label). If None,
  automatically determined from plane. Default is None.
- `title` - Title for the plot. If empty, a descriptive title is generated.
  Default is "".
- `layer_boundaries` - Whether to show layer interface lines as vertical
  boundaries. Default is True.
- `z_scale` - Scaling factor for Z-axis display to adjust aspect ratio.
  Default is 1.0.
- `show_semi_infinite` - Whether to include reflection and transmission
  semi-infinite regions (each 10% of effective layer thickness).
  Default is True.
- `show_legend` - Whether to display material legend with colors and permittivity
  values. Legend positioned outside plot area. Default is True.
  

**Returns**:

- `matplotlib.axes.Axes` - The matplotlib axes object containing the plot,
  allowing for further customization if needed.
  

**Raises**:

- `ValueError` - If plane is not 'xz' or 'yz'.
- `AttributeError` - If sim doesn't have required layer_manager or get_layout method.
  

**Examples**:

```python
from torchrdit.solver import create_solver
from torchrdit.viz import plot_cross_section
from torchrdit.utils import create_material
import torch
import matplotlib.pyplot as plt

# Create a solver with layered structure
solver = create_solver(algorithm=Algorithm.RCWA, lam0=[1.55], rdim=[256, 256])

# Add materials
silicon = create_material(name="silicon", permittivity=11.7)
sio2 = create_material(name="sio2", permittivity=2.25)
solver.add_materials([silicon, sio2])

# Add layers
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
solver.add_layer(material_name="sio2", thickness=torch.tensor(0.3))

# Show XZ cross-section at y=0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_cross_section(solver, plane='xz', fig_ax=ax1, title='XZ Cross-Section')
plot_cross_section(solver, plane='yz', fig_ax=ax2, title='YZ Cross-Section')
plt.tight_layout()
plt.show()

# Custom material colors and slice position
custom_colors = {'silicon': 'blue', 'sio2': 'red'}
plot_cross_section(solver, plane='xz', slice_position=0.1,
                  material_colormap=custom_colors,
                  title='XZ Cross-Section at y=0.1μm')
plt.show()
```
  
  Keywords:
  layer stack, cross-section, visualization, XZ plane, YZ plane, materials,
  electromagnetic simulation, structure visualization, layer boundaries,
  permittivity distribution, photonic structure

