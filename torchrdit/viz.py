"""Visualization module for TorchRDIT electromagnetic simulations.

This module provides functions for visualizing simulation results, material properties,
and field distributions in TorchRDIT. It includes tools for:

1. Plotting 2D field distributions (electric fields, magnetic fields)
2. Visualizing material property distributions in layers
3. Displaying dispersive material properties
4. Analyzing fitted dispersive material models

The visualization functions in this module work with matplotlib to create
publication-quality figures for analyzing simulation results and understanding
the behavior of electromagnetic fields in complex structures.

Key functions:
- plot2d: General-purpose function for plotting 2D data with customizable options
- plot_layer: Visualize the permittivity distribution in a specific layer
- display_fitted_permittivity: Plot fitted dispersive material properties in a simulation
- display_dispersive_profile: Visualize raw dispersive material data

All functions return matplotlib objects that can be further customized as needed.
"""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot2d(data: Any,
           layout: Any,
           func='abs',
           outline = None,
           fig_ax=None,
           cbar=True,
           cmap='magma',
           outline_alpha=0.5,
           title='',
           labels=('x', 'y')):
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
    
    Example:
        ```python
        # Plot the real part of the electric field distribution
        x_grid, y_grid = solver.get_layout()
        e_field = results['fields']['Ex']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plot2d(data=e_field, 
               layout=(x_grid, y_grid),
               func='real',
               cmap='RdBu_r',
               title='Electric Field (Ex) Distribution',
               labels=('x (μm)', 'y (μm)'))
        plt.show()
        ```
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

    if func == 'abs':
        np_func = np.abs
    elif func == 'real':
        np_func = np.real
    elif func == 'imag':
        np_func = np.imag
    else:
        func = 'abs'
        np_func = np.abs

    if fig_ax is None:
        _, fig_ax = plt.subplots(1, 1)

    pcolor_obj = fig_ax.pcolor(mesh_xo,
                            mesh_yo,
                            np_func(data),
                            vmax=np_func(data).max(),
                            vmin=np_func(data).min(),
                            cmap=cmap)

    if outline is not None:
        if isinstance(outline, torch.Tensor):
            outline = outline.detach().cpu().numpy()
        fig_ax.contour(np_func(outline).T, 0, colors='k', alpha=outline_alpha)

    fig_ax.set_ylabel(labels[1])
    fig_ax.set_xlabel(labels[0])
    fig_ax.set_title(title)

    if cbar:
        plt.colorbar(pcolor_obj, ax=fig_ax)

    return fig_ax


def plot_layer(sim,
               layer_index: int = 0,
               frequency_index: int = 0,
               fig_ax = None,
               func = 'real',
               cmap = 'BuGn',
               labels=('x', 'y'),
               title=''):
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
                            
    Example:
        ```python
        # Create a simulation with multiple layers
        solver = create_solver(algorithm=Algorithm.RDIT)
        solver.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
        solver.add_layer(material_name='sio2', thickness=torch.tensor(0.3))
        
        # Plot the permittivity distribution of the second layer
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_layer(solver, layer_index=1, fig_ax=ax, title='SiO2 Layer')
        plt.show()
        ```
    """
    ret = None
    if sim.layer_manager.layers[layer_index].is_dispersive is False:
        ret = plot2d(data=sim.layer_manager.layers[layer_index].ermat[:, :],
                      layout = sim.get_layout(),
                      func=func, fig_ax=fig_ax, cmap=cmap, labels=labels, title=title)
    else:
        ret = plot2d(data=sim.layer_manager.layers[layer_index].ermat[frequency_index, :, :],
                      layout = sim.get_layout(),
                      func=func, fig_ax=fig_ax, cmap=cmap, labels=labels, title=title)

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
        
    Example:
        ```python
        # Create a simulation with a dispersive material (e.g., gold)
        solver = create_solver(algorithm=Algorithm.RDIT)
        gold = create_material(name='gold', dielectric_dispersion=True, 
                             user_dielectric_file='gold_data.txt')
        solver.add_materials([gold])
        solver.add_layer(material_name='gold', thickness=torch.tensor(0.1))
        
        # Visualize the fitted permittivity
        fig, ax = plt.subplots(figsize=(10, 6))
        ax1, ax2 = display_fitted_permittivity(solver, ax)
        ax1.set_xlabel('Wavelength (μm)')
        ax1.set_ylabel('Real Part (ε\')')
        ax2.set_ylabel('Imaginary Part (ε")')
        plt.title('Gold Permittivity vs Wavelength')
        plt.show()
        ```
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
            wls = sim._matlib[matname].fitted_data['wavelengths']
            data_eps1 = sim._matlib[matname].fitted_data['data_eps1']
            data_eps2 = sim._matlib[matname].fitted_data['data_eps2']
            pe1 = sim._matlib[matname].fitted_data['fitted_crv1']
            pe2 = sim._matlib[matname].fitted_data['fitted_crv2']

            ax1.plot(wls, data_eps1, 'r.', label=f'data e\' ({matname})')
            ax1.plot(sim._lam0, pe1(sim._lam0), 'c^-', label=f'fitted e\' ({matname})')

            ax2.plot(wls, data_eps2, 'g*', label=f'data e\" ({matname})')
            ax2.plot(sim._lam0, pe2(sim._lam0), 'm.-', label=f'fitted e\" ({matname})')

    ax1.set_title('Permittivity vs Wavelength')
    ax1.set_xlabel(f"Wavelength [{sim._lenunit}]")
    ax1.set_ylabel('Eps\' [Real Part]')
    ax1.legend(loc='best')

    ax2.set_xlabel(f"Wavelength [{sim._lenunit}]")
    ax2.set_ylabel('Eps\" [Imaginary Part]')
    ax2.legend(loc='best')

    return ax1, ax2


def display_dispersive_profile(material,
                               lengthunit: str = 'um',  # length unit used in the solver
                               ):
    """Visualize the raw dispersive material data without fitting.
    
    This function plots the raw dispersive material data (real and imaginary parts
    of permittivity) as loaded from the material data file, before any polynomial
    fitting is applied. This is useful for inspecting the original material data
    and understanding its wavelength dependence.
    
    Unlike display_fitted_permittivity, which shows both the original data and the
    fitted curves used in simulations, this function shows only the original data
    extracted from the material data file.
    
    Args:
        material: A MaterialClass instance with dispersive properties.
                This should be a material created with dielectric_dispersion=True.
        lengthunit: Unit of length for the wavelength axis.
                  Common values include 'um' (micrometers), 'nm' (nanometers).
                  Default is 'um'.
    
    Returns:
        tuple: A tuple containing (figure, primary_axis, secondary_axis) where:
              - figure: The matplotlib Figure object
              - primary_axis: The main matplotlib axis (for real part)
              - secondary_axis: The secondary y-axis (for imaginary part)
              
    Note:
        This function requires that the material has loaded dispersive data.
        If no data is available, it will return None.
        
    Example:
        ```python
        # Create a dispersive material
        from torchrdit.utils import create_material
        
        gold = create_material(
            name='gold',
            dielectric_dispersion=True,
            user_dielectric_file='gold_data.txt',
            data_format='wl-nk',
            data_unit='um'
        )
        
        # Visualize the raw dispersion data
        fig, ax1, ax2 = display_dispersive_profile(gold, lengthunit='um')
        ax1.set_xlabel('Wavelength (μm)')
        ax1.set_ylabel('Real Part (ε\')')
        ax2.set_ylabel('Imaginary Part (ε")')
        plt.title('Gold Permittivity Data')
        plt.show()
        ```
    """

    ret = None

    disp_er_sorted = material._extract_laoded_data(lengthunit=lengthunit)
    if disp_er_sorted is not None:
        fig, fig_ax = plt.subplots()
        ln1 = fig_ax.plot(disp_er_sorted[:, 0],
                      disp_er_sorted[:, 1], 'r-', label='e\'')
        # ax.legend(loc='best')
        fig_ax2 = fig_ax.twinx()
        ln2 = fig_ax2.plot(disp_er_sorted[:, 0],
                       disp_er_sorted[:, 2], 'g--', label='e\"')
        # ax2.legend(loc='best')
        fig_ax.set_title(f"Permittivity [{material._name}]")
        fig_ax.set_xlabel(f"Wavelength [{lengthunit}]")
        fig_ax.set_ylabel('Eps\' [Real Part]')
        fig_ax2.set_ylabel('Eps\" [Imag Part]')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        fig_ax.legend(lns, labs, loc='best')

        ret = fig,fig_ax

    return ret

