"""Helper functions for plotting and visualization."""

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
    """Function plots 2D structure or field data.

    This function plot a 2D matrix using matplotlib.

    Args:
        data (Union[torch.Tensor, np.ndarray]): 2D data to plot.
        layout (Union[torch.Tensor, np.ndarray]): Mesh grid data including X and Y.
        func (str, optional): Functions for the complex data.('abs', 'real' or 'imag'). Defaults to 'abs'.
        outline (Union[torch.Tensor, np.ndarray], optional): Outline data. Defaults to None.
        fig_ax (_type_, optional): Handler of matplotlib. Defaults to None.
        cbar (bool, optional): If enable colorbar. Defaults to True.
        cmap (str, optional): Choose colormap for colorbar. Defaults to 'magma'.
        outline_alpha (float, optional): Alpha value of outline if applicable. Defaults to 0.5.
        title (str, optional): Title of the plot. Defaults to ''.
        labels (tuple, optional): Lables of x and y axes. Defaults to ('x', 'y').

    Returns:
        matplotlib.axes
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().resolve_conj().numpy()

    # Unpack layout data
    # shape, res, sdim = layout
    mesh_xo, mesh_yo = layout

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
            outline = outline.detach().numpy()
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
    """plot_layer.

    Plot the pattern of the specified layer.

    Args:
        layer_index (int): layer_index
        batch_index (int): batch_index
        frequency_index (int): frequency_index
        fig_ax:
        func:
        cmap:
        labels:
        title:
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


def display_fitted_permittivity(sim, fig_ax):
    """display_fitted_permittivity.

    Displays information about the dispersive permittivity.
    """
    # if True not in self._layerstruct._is_dispers:
    if True not in [sim.layer_manager.layers[ii].is_dispersive for ii in range(sim.layer_manager.nlayer)]:
        print("No dispersive material loaded.")
    else:
        for imat in range(sim.layer_manager.nlayer):
            if sim.layer_manager.layers[imat].is_dispersive is True:
                matname = sim.layer_manager.layers[imat].material_name
                wls = sim._matlib[matname].fitted_data['wavelengths']
                data_eps1 = sim._matlib[matname].fitted_data['data_eps1']
                data_eps2 = sim._matlib[matname].fitted_data['data_eps2']
                pe1 = sim._matlib[matname].fitted_data['fitted_crv1']
                pe2 = sim._matlib[matname].fitted_data['fitted_crv2']
                if fig_ax is None:
                    _, fig_ax = plt.subplots()
                ln1data = fig_ax.plot(wls,
                                  data_eps1, 'r.', label='data e\'')
                ln1fit = fig_ax.plot(sim._lam0,
                                 pe1(sim._lam0), 'c^-', label='fitted e\'')
                fig_ax2 = fig_ax.twinx()
                ln2data = fig_ax2.plot(wls,
                                   data_eps2, 'g*', label='data e\"')
                ln2fit = fig_ax2.plot(sim._lam0,
                                  pe2(sim._lam0), 'm.-', label='fitted e\"')
                # ax2.legend(loc='best')
                fig_ax.set_title(f"Permittivity [{matname}]")
                fig_ax.set_xlabel(f"Wavelength [{sim._lenunit}]")
                fig_ax.set_ylabel('Eps\' [Real Part]')
                fig_ax2.set_ylabel('Eps\" [Imag Part]')
                lns = ln1data + ln1fit + ln2data + ln2fit
                labs = [l.get_label() for l in lns]
                fig_ax.legend(lns, labs, loc='best')


def display_dispersive_profile(material,
                               lengthunit: str = 'um',  # length unit used in the solver
                               ):
    """display_dispersive_profile.

    Plot the original and fitted dispersive profile.

    Args:
        lengthunit (str): lengthunit
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

