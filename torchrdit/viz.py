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
