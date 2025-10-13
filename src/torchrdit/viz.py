"""Visualization module for TorchRDIT electromagnetic simulations.

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
"""

from typing import Any, Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot2d(
    data: Any,
    layout: Any,
    func="abs",
    outline=None,
    fig_ax=None,
    cbar=True,
    cmap="magma",
    outline_alpha=0.5,
    title="",
    labels=("x", "y"),
):
    """Create a 2D visualization of electromagnetic field data or material distributions.

    This function generates a 2D plot of field data or material properties (like
    permittivity) using matplotlib's pcolormesh. It handles both real and complex data,
    applying the specified function (abs, real, imag) for visualization. The function
    can also overlay an outline on the plot to highlight structural boundaries.

    Args:
        data: 2D data to plot (torch.Tensor or numpy.ndarray). Can be complex-valued.
             For field data, this is typically the electric or magnetic field.
             For material data, this is typically the permittivity distribution.
        layout: Tuple of mesh grid coordinates (X, Y) defining the spatial coordinates
               for each point in the data. Should be obtained from sim.get_layout()
               or similar function.
        func: Function to apply to the data for visualization if the data is complex.
             Options are 'abs' (magnitude), 'real' (real part), 'imag' (imaginary part),
             'log10' (log scale), or other numpy/torch functions.
             Default is 'abs'.
        outline: Optional binary mask (0s and 1s) to overlay as an outline on the plot.
                This is useful for highlighting boundaries between different materials.
                Default is None (no outline).
        fig_ax: Matplotlib figure and axis to use for plotting. If None, a new
               figure and axis will be created.
               Default is None.
        cbar: Whether to include a colorbar with the plot.
             Default is True.
        cmap: Matplotlib colormap to use for the data visualization.
             Default is 'magma'.
        outline_alpha: Transparency of the outline overlay (0-1).
                      Default is 0.5.
        title: Title for the plot.
              Default is '' (no title).
        labels: Tuple of (x-axis label, y-axis label).
               Default is ('x', 'y').

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object containing the plot,
                            allowing for further customization if needed.

    Examples:
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
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().resolve_conj().cpu().numpy()

    # Unpack layout data
    # shape, res, sdim = layout
    mesh_xo, mesh_yo = layout

    if isinstance(mesh_xo, torch.Tensor):
        mesh_xo = mesh_xo.detach().cpu().numpy()
    if isinstance(mesh_yo, torch.Tensor):
        mesh_yo = mesh_yo.detach().cpu().numpy()

    if func == "abs":
        np_func = np.abs
    elif func == "real":
        np_func = np.real
    elif func == "imag":
        np_func = np.imag
    else:
        func = "abs"
        np_func = np.abs

    if fig_ax is None:
        _, fig_ax = plt.subplots(1, 1)

    pcolor_obj = fig_ax.pcolor(
        mesh_xo, mesh_yo, np_func(data), vmax=np_func(data).max(), vmin=np_func(data).min(), cmap=cmap
    )

    if outline is not None:
        if isinstance(outline, torch.Tensor):
            outline = outline.detach().cpu().numpy()
        fig_ax.contour(np_func(outline).T, 0, colors="k", alpha=outline_alpha)

    fig_ax.set_ylabel(labels[1])
    fig_ax.set_xlabel(labels[0])
    fig_ax.set_title(title)
    fig_ax.set_aspect("equal")

    if cbar:
        plt.colorbar(pcolor_obj, ax=fig_ax)

    return fig_ax


def plot_layer(
    sim,
    layer_index: int = 0,
    frequency_index: int = 0,
    fig_ax=None,
    func="real",
    cmap="BuGn",
    labels=("x", "y"),
    title="",
):
    """Visualize the permittivity distribution of a specific layer in the simulation.

    This function creates a 2D plot of the permittivity (ε) distribution within a
    specified layer of the simulation structure. For dispersive materials, the
    permittivity at a specific frequency is plotted.

    Args:
        sim: A simulation object (typically a FourierBaseSolver instance) that
             contains the layer structure to be visualized.
        layer_index: Index of the layer to plot (0-based indexing).
                    Default is 0 (first layer).
        frequency_index: Index of the frequency to use for dispersive materials.
                       Only relevant for frequency-dependent (dispersive) materials.
                       Default is 0 (first frequency).
        fig_ax: Matplotlib figure and axis to use for plotting. If None, a new
               figure and axis will be created. Default is None.
        func: Function to apply to the permittivity data before plotting.
             Options include 'real', 'imag', 'abs', 'log10', etc.
             Default is 'real' (plot real part of permittivity).
        cmap: Matplotlib colormap to use for the plot.
             Default is 'BuGn'.
        labels: Tuple of (x-axis label, y-axis label).
               Default is ('x', 'y').
        title: Title for the plot. If empty, a default title with layer information
              will be used. Default is ''.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object containing the plot,
                            allowing for further customization if needed.

    Examples:
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
    """
    ret = None
    if sim.layer_manager.layers[layer_index].is_dispersive is False:
        ret = plot2d(
            data=sim.layer_manager.layers[layer_index].ermat[:, :],
            layout=sim.get_layout(),
            func=func,
            fig_ax=fig_ax,
            cmap=cmap,
            labels=labels,
            title=title,
        )
    else:
        ret = plot2d(
            data=sim.layer_manager.layers[layer_index].ermat[frequency_index, :, :],
            layout=sim.get_layout(),
            func=func,
            fig_ax=fig_ax,
            cmap=cmap,
            labels=labels,
            title=title,
        )

    return ret


def display_fitted_permittivity(sim, fig_ax=None):
    """Visualize fitted permittivity for dispersive materials in the simulation.

    This function plots the real and imaginary parts of the permittivity for
    dispersive materials in the simulation, showing both the original data points
    and the fitted curves used in the simulation. This is useful for verifying
    the quality of the polynomial fit to the material dispersion data.

    The plot includes:
    - Original data points for the real part (ε')
    - Fitted curve for the real part at simulation wavelengths
    - Original data points for the imaginary part (ε")
    - Fitted curve for the imaginary part at simulation wavelengths

    Args:
        sim: A simulation object (typically a FourierBaseSolver instance) that
             contains the dispersive materials to be visualized.
        fig_ax: Matplotlib axis to use for plotting. If None, a new figure and
               axis will be created. Default is None.

    Returns:
        tuple: A tuple containing (primary_axis, secondary_axis) where:
              - primary_axis: The main matplotlib axis (for real part)
              - secondary_axis: The secondary y-axis (for imaginary part)

    Note:
        This function automatically detects all dispersive materials in the simulation
        and plots their fitted permittivity data. If no dispersive materials are found,
        it prints a message indicating this.

    Examples:
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
    ax1.set_ylabel('Real Part (ε\')')
    ax2.set_ylabel('Imaginary Part (ε")')
    plt.title('Gold Permittivity vs Wavelength')
    plt.show()
    ```

    Keywords:
        dispersive material, permittivity, wavelength dependence, polynomial fitting,
        material characterization, complex permittivity, optical properties,
        Drude-Lorentz model, material dispersion
    """

    if True not in [sim.layer_manager.layers[ii].is_dispersive for ii in range(sim.layer_manager.nlayer)]:
        print("No dispersive material loaded.")
        return None

    if fig_ax is None:
        fig, fig_ax = plt.subplots(2, 1, figsize=(10, 12))

    ax1, ax2 = fig_ax

    for imat in range(sim.layer_manager.nlayer):
        if sim.layer_manager.layers[imat].is_dispersive is True:
            matname = sim.layer_manager.layers[imat].material_name
            wls = sim._matlib[matname].fitted_data["wavelengths"]
            data_eps1 = sim._matlib[matname].fitted_data["data_eps1"]
            data_eps2 = sim._matlib[matname].fitted_data["data_eps2"]
            pe1 = sim._matlib[matname].fitted_data["fitted_eps1"]
            pe2 = sim._matlib[matname].fitted_data["fitted_eps2"]

            ax1.plot(wls, data_eps1, "r.", label=f"data e' ({matname})")
            ax1.plot(sim._lam0, pe1(sim._lam0), "c^-", label=f"fitted e' ({matname})")

            ax2.plot(wls, data_eps2, "g*", label=f'data e" ({matname})')
            ax2.plot(sim._lam0, pe2(sim._lam0), "m.-", label=f'fitted e" ({matname})')

    ax1.set_title("Permittivity vs Wavelength")
    ax1.set_xlabel(f"Wavelength [{sim._lenunit}]")
    ax1.set_ylabel("Eps' [Real Part]")
    ax1.legend(loc="best")

    ax2.set_xlabel(f"Wavelength [{sim._lenunit}]")
    ax2.set_ylabel('Eps" [Imaginary Part]')
    ax2.legend(loc="best")

    return ax1, ax2


def plot_cross_section(
    sim,
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
    show_legend: bool = True,
):
    """Visualize cross-section of layer stack structure.

    Creates a 2D cross-sectional view of the layer stack showing material
    distributions with different colors. Supports both XZ-plane (y=0) and
    YZ-plane (x=0) visualizations. Includes reflection and transmission
    semi-infinite regions for complete electromagnetic domain visualization.

    This function helps users understand the structure of their layered
    electromagnetic simulation domains by showing different materials with
    distinct colors and transparent air/vacuum regions. It integrates with
    the existing TorchRDIT solver workflow and coordinate systems.

    Args:
        sim: Solver instance with configured layers and materials.
             Must have layer_manager with layers and get_layout() method.
        plane: Cross-section plane ('xz' for XZ-plane or 'yz' for YZ-plane).
              Default is 'xz'.
        slice_position: Position along the slicing axis (y-coordinate for XZ plane,
                       x-coordinate for YZ plane). Default is 0.0.
        frequency_index: Index for selecting frequency in dispersive materials.
                        Only relevant for frequency-dependent materials. Default is 0.
        fig_ax: Matplotlib axis object to use for plotting. If None, a new
               figure and axis will be created. Default is None.
        material_colormap: Dictionary mapping material names to color strings.
                          If None, uses default distinguishable colors. Default is None.
        labels: Tuple of (horizontal_axis_label, vertical_axis_label). If None,
               automatically determined from plane. Default is None.
        title: Title for the plot. If empty, a descriptive title is generated.
              Default is "".
        layer_boundaries: Whether to show layer interface lines as vertical
                         boundaries. Default is True.
        z_scale: Scaling factor for Z-axis display to adjust aspect ratio.
                Default is 1.0.
        show_semi_infinite: Whether to include reflection and transmission
                           semi-infinite regions (each 10% of effective layer thickness).
                           Default is True.
        show_legend: Whether to display material legend with colors and permittivity
                    values. Legend positioned outside plot area. Default is True.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object containing the plot,
                            allowing for further customization if needed.

    Raises:
        ValueError: If plane is not 'xz' or 'yz'.
        AttributeError: If sim doesn't have required layer_manager or get_layout method.

    Examples:
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
    """

    # Validate plane parameter
    if plane.lower() not in ["xz", "yz"]:
        raise ValueError(f"Invalid plane '{plane}'. Must be 'xz' or 'yz'.")

    # Check if solver has required attributes
    if not hasattr(sim, "layer_manager"):
        raise AttributeError("Solver must have layer_manager attribute.")
    if not hasattr(sim, "get_layout"):
        raise AttributeError("Solver must have get_layout() method.")

    # Get spatial layout
    X, Y = sim.get_layout()

    # Calculate layer positions and materials
    layer_info = _get_layer_structure_info(sim, z_scale, show_semi_infinite)

    # Create material colormap if not provided
    if material_colormap is None:
        material_colormap = _create_material_colormap_by_name(sim)

    # Create figure if needed
    if fig_ax is None:
        fig, fig_ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot layers using rectangle-based approach
    _draw_layer_rectangles(fig_ax, sim, layer_info, material_colormap, plane, slice_position, frequency_index)

    # Add dashed line separation between all layers
    if layer_boundaries:
        _add_comprehensive_layer_boundaries(fig_ax, layer_info)

    # Add floating text boxes with layer IDs
    _add_layer_id_labels(fig_ax, layer_info)

    # Add material legend
    if show_legend:
        _add_material_legend_by_name(fig_ax, material_colormap, sim)

    # Get length unit from solver
    length_unit = getattr(sim, "_lenunit", "um")
    unit_display = "μm" if length_unit.lower() == "um" else length_unit

    # Set axis limits and ticks based on spatial layout and layer structure
    if plane.lower() == "xz":
        x_min, x_max = X.min().item(), X.max().item()
        fig_ax.set_xlim(x_min, x_max)
        coord_label = "y"
        # Generate and set smart ticks for X-axis
        x_ticks = _generate_axis_ticks(x_min, x_max)
        fig_ax.set_xticks(x_ticks)
    else:  # yz plane
        x_min, x_max = Y.min().item(), Y.max().item()
        fig_ax.set_xlim(x_min, x_max)
        coord_label = "x"
        # Generate and set smart ticks for Y-axis
        y_ticks = _generate_axis_ticks(x_min, x_max)
        fig_ax.set_xticks(y_ticks)

    # Set Z limits and ticks based on layer structure (inverted so z=0 is at top)
    if layer_info:
        z_min = min(layer["z_start"] for layer in layer_info)
        z_max = max(layer["z_end"] for layer in layer_info)
        fig_ax.set_ylim(z_max, z_min)  # Inverted: z=0 at top, positive z downward

        # Generate and set smart ticks for Z-axis based on layer boundaries
        z_ticks = _generate_axis_ticks(z_min, z_max, layer_info)
        fig_ax.set_yticks(z_ticks)

    # Set labels with dynamic units
    if labels is None:
        if plane.lower() == "xz":
            labels = (f"X ({unit_display})", f"Z ({unit_display})")
        else:  # yz plane
            labels = (f"Y ({unit_display})", f"Z ({unit_display})")

    fig_ax.set_xlabel(labels[0])
    fig_ax.set_ylabel(labels[1])

    # Set title with dynamic unit
    if not title:
        plane_name = plane.upper()
        title = f"{plane_name} Cross-Section at {coord_label}={slice_position:.3f}{unit_display}"

    fig_ax.set_title(title)
    fig_ax.set_aspect("equal")

    return fig_ax


def _create_material_colormap_by_name(sim) -> Dict[str, str]:
    """Create color mapping based on material names.

    Args:
        sim: Solver instance

    Returns:
        Dictionary mapping material names to colors
    """
    # Pre-defined colors for common materials
    common_material_colors = {
        "Si": "#2196F3",  # Silicon - blue
        "Silicon": "#2196F3",  # Silicon - blue
        "si": "#2196F3",  # Silicon - blue (lowercase)
        "silicon": "#2196F3",  # Silicon - blue (lowercase)
        "SiO2": "#4CAF50",  # SiO2 - green
        "Silica": "#4CAF50",  # Silica - green
        "sio2": "#4CAF50",  # SiO2 - green (lowercase)
        "silica": "#4CAF50",  # Silica - green (lowercase)
        "Air": "white",  # Air - white/transparent
        "Vacuum": "white",  # Vacuum - white/transparent
        "air": "white",  # Air - white/transparent (lowercase)
        "vacuum": "white",  # Vacuum - white/transparent (lowercase)
        "GaAs": "#FF5722",  # GaAs - orange
        "InP": "#9C27B0",  # InP - purple
        "AlGaAs": "#607D8B",  # AlGaAs - blue gray
    }

    # Additional colors for other materials
    material_colors = [
        "#FF9800",  # Orange
        "#E91E63",  # Pink
        "#795548",  # Brown
        "#9E9E9E",  # Gray
        "#FFC107",  # Amber
        "#673AB7",  # Deep purple
        "#3F51B5",  # Indigo
        "#009688",  # Teal
        "#CDDC39",  # Lime
        "#8BC34A",  # Light green
        "#03A9F4",  # Light blue
        "#17becf",  # Cyan
    ]

    # Build material name to color mapping
    material_colormap = {}

    # Collect all unique material names
    material_names = set()

    # Add reflection and transmission materials (semi-infinite regions)
    if hasattr(sim, "layer_manager") and hasattr(sim.layer_manager, "ref_material_name"):
        material_names.add(sim.layer_manager.ref_material_name)

    if hasattr(sim, "layer_manager") and hasattr(sim.layer_manager, "trn_material_name"):
        material_names.add(sim.layer_manager.trn_material_name)

    # Add layer materials
    for layer in sim.layer_manager.layers:
        if hasattr(layer, "material_name") and layer.material_name:
            material_names.add(layer.material_name)
        elif hasattr(layer, "background_material") and layer.background_material:
            material_names.add(layer.background_material)

    # Assign colors to materials
    color_index = 0
    for material_name in sorted(material_names):
        if material_name in common_material_colors:
            material_colormap[material_name] = common_material_colors[material_name]
        else:
            material_colormap[material_name] = material_colors[color_index % len(material_colors)]
            color_index += 1

    # Add special region identifiers
    material_colormap["_reflection"] = "#2E7D32"  # Dark green for reflection region
    material_colormap["_transmission"] = "#E0E0E0"  # Light gray for transmission region

    return material_colormap


def _get_layer_structure_info(sim, z_scale: float = 1.0, include_semi_infinite: bool = True):
    """Get layer structure information for rectangle-based visualization.

    Args:
        sim: Solver instance
        z_scale: Z-axis scaling factor
        include_semi_infinite: Whether to include reflection/transmission regions

    Returns:
        List of layer info dictionaries with positions, materials, and types
    """
    layer_info = []

    # Calculate effective layer thickness first
    total_thickness = 0.0
    for layer in sim.layer_manager.layers:
        if hasattr(layer, "thickness") and layer.thickness is not None:
            if isinstance(layer.thickness, torch.Tensor):
                thickness_val = layer.thickness.item()
            else:
                thickness_val = float(layer.thickness)
            total_thickness += thickness_val
        else:
            total_thickness += 0.1  # Default thickness

    # Semi-infinite region thickness (10% of total effective thickness each)
    semi_infinite_thickness = 0.1 * total_thickness if include_semi_infinite else 0.0

    # Start positioning - z=0 is at the interface between reflection and first layer
    if include_semi_infinite:
        # Reflection layer extends from negative z to z=0
        reflection_material = sim.layer_manager.ref_material_name if hasattr(sim, "layer_manager") else "air"
        layer_info.append(
            {
                "layer_id": "R",
                "z_start": -semi_infinite_thickness * z_scale,
                "z_end": 0.0,
                "material_name": reflection_material,
                "layer_type": "reflection",
                "is_patterned": False,
            }
        )
        current_z = 0.0  # First finite layer starts at z=0
    else:
        current_z = 0.0  # If no semi-infinite, still start at z=0

    # Add finite layers
    for i, layer in enumerate(sim.layer_manager.layers):
        # Get layer thickness
        if hasattr(layer, "thickness") and layer.thickness is not None:
            if isinstance(layer.thickness, torch.Tensor):
                thickness_val = layer.thickness.item()
            else:
                thickness_val = float(layer.thickness)
        else:
            thickness_val = 0.1  # Default thickness

        # Get material name
        if hasattr(layer, "material_name") and layer.material_name:
            material_name = layer.material_name
        elif hasattr(layer, "background_material") and layer.background_material:
            material_name = layer.background_material
        else:
            material_name = f"Material_{i}"

        # Check if layer is patterned
        is_patterned = hasattr(layer, "ermat") and layer.ermat is not None and len(layer.ermat.shape) >= 2

        layer_info.append(
            {
                "layer_id": i,
                "z_start": current_z,
                "z_end": current_z + thickness_val * z_scale,
                "material_name": material_name,
                "layer_type": "finite",
                "is_patterned": is_patterned,
                "layer_obj": layer,
            }
        )

        current_z += thickness_val * z_scale

    # Add transmission semi-infinite layer at the bottom
    if include_semi_infinite:
        transmission_material = sim.layer_manager.trn_material_name if hasattr(sim, "layer_manager") else "air"
        layer_info.append(
            {
                "layer_id": "T",
                "z_start": current_z,
                "z_end": current_z + semi_infinite_thickness * z_scale,
                "material_name": transmission_material,
                "layer_type": "transmission",
                "is_patterned": False,
            }
        )

    return layer_info


def _draw_layer_rectangles(
    fig_ax, sim, layer_info, material_colormap, plane: str, slice_position: float, frequency_index: int = 0
):
    """Draw layer rectangles using material-based coloring.

    Args:
        fig_ax: Matplotlib axis object
        sim: Solver instance
        layer_info: Layer structure information
        material_colormap: Material name to color mapping
        plane: 'xz' or 'yz'
        slice_position: Position along slicing axis
        frequency_index: Index for dispersive materials
    """
    import matplotlib.patches as patches

    # Get spatial layout
    X, Y = sim.get_layout()

    # Calculate spatial extents
    if plane.lower() == "xz":
        coord_min = X.min().item()
        coord_max = X.max().item()
    else:  # yz plane
        coord_min = Y.min().item()
        coord_max = Y.max().item()

    # Draw each layer as a rectangle
    for layer in layer_info:
        material_name = layer["material_name"]
        layer_type = layer["layer_type"]

        # Get material color
        if material_name in material_colormap:
            color = material_colormap[material_name]
        elif layer_type == "reflection" and "_reflection" in material_colormap:
            color = material_colormap["_reflection"]
        elif layer_type == "transmission" and "_transmission" in material_colormap:
            color = material_colormap["_transmission"]
        else:
            color = "#CCCCCC"  # Default gray

        # Skip transparent materials
        if color == "none":
            continue

        # Draw layer rectangle
        if layer["is_patterned"] and layer_type == "finite":
            # For patterned layers, draw background first then overlay pattern
            _draw_patterned_layer_rectangle(
                fig_ax, layer, coord_min, coord_max, material_colormap, plane, slice_position, frequency_index, sim
            )
        else:
            # Draw solid color rectangle for homogeneous layers
            rect = patches.Rectangle(
                (coord_min, layer["z_start"]),
                coord_max - coord_min,
                layer["z_end"] - layer["z_start"],
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
                linewidth=0.5,
            )
            fig_ax.add_patch(rect)


def _draw_patterned_layer_rectangle(
    fig_ax, layer, coord_min, coord_max, material_colormap, plane, slice_position, frequency_index, sim=None
):
    """Draw rectangle for patterned layer with background and pattern overlay.

    Args:
        fig_ax: Matplotlib axis object
        layer: Layer information dictionary
        coord_min, coord_max: Spatial coordinate limits
        material_colormap: Material name to color mapping
        plane: 'xz' or 'yz'
        slice_position: Position along slicing axis
        frequency_index: Index for dispersive materials
        sim: Solver instance for coordinate system information
    """
    import matplotlib.patches as patches

    # Get the actual layer object for pattern extraction
    layer_obj = layer["layer_obj"]

    # Get background material from layer object
    if hasattr(layer_obj, "bg_material") and layer_obj.bg_material:
        bg_material = layer_obj.bg_material
    else:
        bg_material = "air"  # Default fallback

    # Get background material color
    if bg_material in material_colormap:
        bg_color = material_colormap[bg_material]
    else:
        # If not in colormap, use white for air/vacuum, otherwise gray
        air_materials = ["air", "vacuum", "", "Air", "Vacuum"]
        if bg_material in air_materials:
            bg_color = "white"
        else:
            bg_color = "#CCCCCC"  # Default gray for unknown materials

    # Draw background rectangle
    if bg_color != "none":
        bg_rect = patches.Rectangle(
            (coord_min, layer["z_start"]),
            coord_max - coord_min,
            layer["z_end"] - layer["z_start"],
            facecolor=bg_color,
            edgecolor="black",
            alpha=0.7,
            linewidth=0.5,
        )
        fig_ax.add_patch(bg_rect)

    # Now extract and visualize the actual pattern overlay
    if hasattr(layer_obj, "ermat") and layer_obj.ermat is not None:
        try:
            _draw_pattern_overlay(
                fig_ax,
                layer_obj,
                layer,
                coord_min,
                coord_max,
                material_colormap,
                plane,
                slice_position,
                frequency_index,
                sim,
            )
        except Exception as e:
            print(f"Warning: Could not render pattern overlay: {e}")
            # Fallback to pattern indicator text
            _add_pattern_indicator_text(fig_ax, layer, coord_min, coord_max)
    else:
        # No pattern data available, just add indicator text
        _add_pattern_indicator_text(fig_ax, layer, coord_min, coord_max)


def _add_comprehensive_layer_boundaries(fig_ax, layer_info):
    """Add dashed lines only between reflection/transmission layers and finite layers.

    Args:
        fig_ax: Matplotlib axis object
        layer_info: Layer structure information
    """
    # Get current axis limits to draw full-width lines
    _ = fig_ax.get_xlim()

    # Add horizontal dashed lines only at special interfaces
    for i, layer in enumerate(layer_info[:-1]):  # Skip last layer
        z_boundary = layer["z_end"]
        next_layer = layer_info[i + 1]

        # Only draw lines at reflection-finite or finite-transmission interfaces
        if layer["layer_type"] == "reflection" or next_layer["layer_type"] == "transmission":
            # Special interfaces (reflection-finite or finite-transmission)
            linestyle = "-."
            alpha = 0.8
            linewidth = 1.5
            if layer["layer_type"] == "reflection":
                color = "red"
                label = "Reflection Interface" if i == 0 else None
            else:
                color = "blue"
                label = "Transmission Interface" if i == len(layer_info) - 2 else None
            
            fig_ax.axhline(y=z_boundary, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, label=label)


def _add_layer_id_labels(fig_ax, layer_info):
    """Add floating text boxes with layer IDs.

    Args:
        fig_ax: Matplotlib axis object
        layer_info: Layer structure information
    """
    # Use relative position from left edge (2% from left)
    x_pos = 0.02  # Relative position in axes coordinates

    # Count finite layers to decide whether to show individual layer labels
    finite_layer_count = sum(1 for layer in layer_info if layer["layer_type"] == "finite")

    # Add layer ID labels
    for layer in layer_info:
        mid_z = (layer["z_start"] + layer["z_end"]) / 2
        layer_id = layer["layer_id"]

        # Format layer ID text
        if layer["layer_type"] == "reflection":
            text = "Reflection Layer"
        elif layer["layer_type"] == "transmission":
            text = "Transmission Layer"
        elif finite_layer_count <= 5:
            # Only show individual layer labels if there are 5 or fewer finite layers
            text = f"Layer: {layer_id}"
        else:
            # Skip individual layer labels when there are more than 5 finite layers
            continue

        # Add floating text (no border) using mixed transform
        # x in axes coordinates, y in data coordinates
        fig_ax.text(
            x_pos,
            mid_z,
            text,
            ha="left",
            va="center",
            fontsize=10,
            weight="regular",
            transform=fig_ax.get_yaxis_transform(),
            zorder=10,
        )


def _add_material_legend_by_name(fig_ax, material_colormap, sim):
    """Add material legend based on material names.

    Args:
        fig_ax: Matplotlib axis object
        material_colormap: Material name to color mapping
        sim: Solver instance
    """
    import matplotlib.patches as patches

    # Create legend elements
    legend_elements = []

    # Add materials used in the structure
    used_materials = set()

    # Collect materials from layers
    for layer in sim.layer_manager.layers:
        # Add foreground material (primary layer material)
        if hasattr(layer, "material_name") and layer.material_name:
            used_materials.add(layer.material_name)

        # Add background material for patterned layers
        if hasattr(layer, "bg_material") and layer.bg_material:
            used_materials.add(layer.bg_material)
        elif hasattr(layer, "background_material") and layer.background_material:
            used_materials.add(layer.background_material)

    # Add semi-infinite region materials
    if hasattr(sim, "layer_manager"):
        used_materials.add(sim.layer_manager.ref_material_name)
        used_materials.add(sim.layer_manager.trn_material_name)

    # Create legend patches
    for material_name in sorted(used_materials):
        if material_name in material_colormap:
            color = material_colormap[material_name]
            # Always include air/vacuum in legend even if transparent elsewhere
            air_materials = ["air", "vacuum", "", "Air", "Vacuum"]
            if color != "none" or material_name in air_materials:
                # Use white for air materials if they were marked transparent
                display_color = "white" if material_name in air_materials and color == "none" else color
                patch = patches.Rectangle(
                    (0, 0), 1, 1, facecolor=display_color, alpha=0.7, edgecolor="black", linewidth=0.5
                )
                legend_elements.append((patch, material_name))

    # Skip adding special region keys since actual materials are already included above

    # Create legend
    if legend_elements:
        patches_list, labels_list = zip(*legend_elements)
        fig_ax.legend(
            patches_list,
            labels_list,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="Materials",
            frameon=True,
            fancybox=True,
            shadow=True,
        )


def _generate_axis_ticks(coord_min, coord_max, layer_info=None):
    """Generate smart tick positions for cross-section plots.

    Args:
        coord_min: Minimum coordinate value
        coord_max: Maximum coordinate value
        layer_info: Optional layer information for Z-axis ticks

    Returns:
        List of tick positions
    """
    ticks = []

    if layer_info is None:
        # For horizontal axes (X/Y), show edge ticks and some intermediate ones
        ticks.extend([coord_min, coord_max])

        # Add some intermediate ticks based on range
        range_val = coord_max - coord_min
        if range_val > 0:
            # Add 2-4 intermediate ticks depending on range
            if range_val <= 2:
                # Small range: add middle tick
                mid_val = (coord_min + coord_max) / 2
                ticks.append(mid_val)
            elif range_val <= 5:
                # Medium range: add 2 intermediate ticks
                quarter = range_val / 4
                ticks.extend([coord_min + quarter, coord_min + 3 * quarter])
            else:
                # Large range: add 3 intermediate ticks
                quarter = range_val / 4
                ticks.extend([coord_min + quarter, coord_min + 2 * quarter, coord_min + 3 * quarter])
    else:
        # For Z-axis, limit the number of ticks to avoid clutter
        # Collect all unique z positions
        z_positions = []
        for layer in layer_info:
            z_positions.extend([layer["z_start"], layer["z_end"]])
        
        # Remove duplicates and sort
        all_ticks = sorted(list(set(z_positions)))
        
        # If we have too many ticks, thin them out
        max_ticks = 7  # Maximum number of z-axis ticks
        
        if len(all_ticks) <= max_ticks:
            ticks = all_ticks
        else:
            # Keep start and end ticks, then select evenly spaced intermediate ticks
            ticks = [all_ticks[0], all_ticks[-1]]  # Start and end
            
            # Add evenly spaced intermediate ticks
            n_intermediate = max_ticks - 2  # Subtract start and end ticks
            if n_intermediate > 0:
                step = len(all_ticks) // (n_intermediate + 1)
                for i in range(1, n_intermediate + 1):
                    idx = i * step
                    if idx < len(all_ticks) - 1:  # Don't duplicate the end tick
                        ticks.append(all_ticks[idx])
            
            ticks = sorted(list(set(ticks)))

    return ticks


def _add_pattern_indicator_text(fig_ax, layer, coord_min, coord_max):
    """Add pattern indicator text as fallback visualization.

    Args:
        fig_ax: Matplotlib axis object
        layer: Layer information dictionary
        coord_min, coord_max: Spatial coordinate limits
    """
    mid_z = (layer["z_start"] + layer["z_end"]) / 2
    mid_coord = (coord_min + coord_max) / 2
    fig_ax.text(
        mid_coord,
        mid_z,
        "PATTERNED",
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        rotation=0,
    )


def _extract_pattern_cross_section(layer_obj, plane, slice_position, sim=None, frequency_index=0):
    """Extract 1D cross-section from the pattern mask at specified slice position.

    Args:
        layer_obj: Layer object with mask_format
        plane: 'xz' or 'yz'
        slice_position: Position along slicing axis
        sim: Solver instance for coordinate system information (required for non-Cartesian lattices)
        frequency_index: Index for dispersive materials

    Returns:
        Tuple of (cross_section_1d, coord_axis) or (None, None) if no mask
    """
    import numpy as np
    import torch

    if not hasattr(layer_obj, "mask_format") or layer_obj.mask_format is None:
        return None, None

    mask_data = layer_obj.mask_format
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()

    # Handle dispersive case (3D mask)
    if len(mask_data.shape) > 2:
        mask_data = mask_data[frequency_index, :, :] if frequency_index < mask_data.shape[0] else mask_data[0, :, :]

    # Take real part if complex
    if np.iscomplexobj(mask_data):
        mask_data = np.real(mask_data)

    # Check if we have solver info and if lattice is non-Cartesian
    is_cartesian = True
    if sim is not None and hasattr(sim, "get_cell_type"):
        from torchrdit.cell import CellType

        try:
            cell_type = sim.get_cell_type()
            is_cartesian = cell_type == CellType.Cartesian
        except Exception:
            is_cartesian = True  # Fallback to simple slicing if cell type check fails

    if is_cartesian:
        # For Cartesian lattices, use fast direct slicing (existing behavior)
        if plane.lower() == "xz":
            slice_idx = mask_data.shape[1] // 2  # Middle X index (column)
            cross_section = mask_data[:, slice_idx]  # Extract column (Y direction)
            coord_axis = np.linspace(-0.5, 0.5, len(cross_section))
        else:  # yz plane
            slice_idx = mask_data.shape[0] // 2  # Middle Y index (row)
            cross_section = mask_data[slice_idx, :]  # Extract row (X direction)
            coord_axis = np.linspace(-0.5, 0.5, len(cross_section))
    else:
        # For non-Cartesian lattices, use proper coordinate interpolation
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError:
            # Fallback to simple slicing if scipy is not available
            if plane.lower() == "xz":
                slice_idx = mask_data.shape[1] // 2
                cross_section = mask_data[:, slice_idx]
                coord_axis = np.linspace(-0.5, 0.5, len(cross_section))
            else:
                slice_idx = mask_data.shape[0] // 2
                cross_section = mask_data[slice_idx, :]
                coord_axis = np.linspace(-0.5, 0.5, len(cross_section))
            return cross_section, coord_axis

        # Get real-space coordinates from solver
        X, Y = sim.get_layout()
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()

        # Create interpolator for the mask data
        # mask_data is indexed as (rows=Y, cols=X) in grid coordinates
        p_vals = np.arange(mask_data.shape[0])  # Y/row indices
        q_vals = np.arange(mask_data.shape[1])  # X/col indices
        interpolator = RegularGridInterpolator(
            (p_vals, q_vals), mask_data, method="linear", bounds_error=False, fill_value=0.0
        )

        # Define the cross-section line in real space
        if plane.lower() == "xz":
            # XZ plane at y = slice_position
            x_range = np.linspace(X.min(), X.max(), 200)
            y_target = slice_position

            # Find corresponding (p, q) coordinates for points on this line
            cross_section = []
            real_x_coords = []

            for x_val in x_range:
                # Find closest real-space point to (x_val, y_target)
                distances = (X - x_val) ** 2 + (Y - y_target) ** 2
                min_idx = np.unravel_index(np.argmin(distances), X.shape)
                p_coord, q_coord = min_idx

                # Interpolate mask value at this (p, q) coordinate
                mask_val = interpolator((p_coord, q_coord))
                cross_section.append(mask_val)
                real_x_coords.append(X[min_idx])

            cross_section = np.array(cross_section)
            # Normalize coordinates to [-0.5, 0.5] range for compatibility
            coord_axis = (np.array(real_x_coords) - X.mean()) / (X.max() - X.min())

        else:  # yz plane
            # YZ plane at x = slice_position
            y_range = np.linspace(Y.min(), Y.max(), 200)
            x_target = slice_position

            # Find corresponding (p, q) coordinates for points on this line
            cross_section = []
            real_y_coords = []

            for y_val in y_range:
                # Find closest real-space point to (x_target, y_val)
                distances = (X - x_target) ** 2 + (Y - y_val) ** 2
                min_idx = np.unravel_index(np.argmin(distances), X.shape)
                p_coord, q_coord = min_idx

                # Interpolate mask value at this (p, q) coordinate
                mask_val = interpolator((p_coord, q_coord))
                cross_section.append(mask_val)
                real_y_coords.append(Y[min_idx])

            cross_section = np.array(cross_section)
            # Normalize coordinates to [-0.5, 0.5] range for compatibility
            coord_axis = (np.array(real_y_coords) - Y.mean()) / (Y.max() - Y.min())

    return cross_section, coord_axis


def _draw_pattern_overlay(
    fig_ax,
    layer_obj,
    layer_info,
    coord_min,
    coord_max,
    material_colormap,
    plane,
    slice_position,
    frequency_index,
    sim=None,
):
    """Draw the actual pattern overlay on top of background.

    Args:
        fig_ax: Matplotlib axis object
        layer_obj: Actual layer object with ermat data
        layer_info: Layer information dictionary
        coord_min, coord_max: Spatial coordinate limits
        material_colormap: Material name to color mapping
        plane: 'xz' or 'yz'
        slice_position: Position along slicing axis
        frequency_index: Index for dispersive materials
        sim: Solver instance for coordinate system information
    """
    import matplotlib.patches as patches
    import numpy as np

    # Get the foreground material from the layer's material_name (the material assigned to the layer)
    fg_material = layer_obj.material_name

    # Get foreground color
    if fg_material in material_colormap:
        fg_color = material_colormap[fg_material]
    else:
        fg_color = "#2196F3"  # Default to blue for silicon

    # Extract the actual cross-section from the mask at the slice position
    cross_section, coord_axis = _extract_pattern_cross_section(layer_obj, plane, slice_position, sim, frequency_index)

    if cross_section is not None:
        # Scale coordinates to match the actual coordinate range
        coord_scaled = coord_min + (coord_axis + 0.5) * (coord_max - coord_min)

        # Identify contiguous regions where mask > 0.5 (pattern material)
        pattern_regions = cross_section > 0.5

        # Find continuous segments
        diff_pattern = np.diff(np.concatenate(([False], pattern_regions, [False])).astype(int))
        starts = np.where(diff_pattern == 1)[0]  # Start of pattern regions
        ends = np.where(diff_pattern == -1)[0]  # End of pattern regions

        # Draw rectangles for each pattern segment
        for start_idx, end_idx in zip(starts, ends):
            if end_idx > start_idx:  # Valid segment
                rect_start = coord_scaled[start_idx]
                rect_end = coord_scaled[min(end_idx - 1, len(coord_scaled) - 1)]
                rect_width = rect_end - rect_start

                if rect_width > 0:  # Valid width
                    pattern_rect = patches.Rectangle(
                        (rect_start, layer_info["z_start"]),
                        rect_width,
                        layer_info["z_end"] - layer_info["z_start"],
                        facecolor=fg_color,
                        edgecolor="black",
                        alpha=0.8,
                        linewidth=0.5,
                    )
                    fig_ax.add_patch(pattern_rect)

        # Pattern information text removed per user request - no percentage fill text box
