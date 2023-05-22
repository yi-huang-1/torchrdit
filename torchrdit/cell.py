""" This file defines classes for models to be simulated. """
from typing import Callable, Tuple, Union, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils import gen_toeplitz2d_dispersive, gen_toeplitz2d_normal, tensor_params_check, create_material
from .materials import MaterialClass
from .constants import lengthunit_dict
from .logger import Logger
from .layers import LayerManager
from .viz import plot2d

# Function Type
FuncType = Callable[..., Any]

class Cell3D():
    """ Base class of unit cell """

    n_kermat_gen = 0
    n_kurmat_gen = 0

    def __init__(self,
                 batch_size: int,  # batch size of data set
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: Union[list, None] = None,  # dimensions in real space: (H, W)
                 kdim: Union[list, None] = None,  # dimensions in k space: (kH, kW)
                 materiallist: Union[list, None] = None,  # list of materials
                 t1: Union[list, None] = None,  # lattice vector in real space
                 t2: Union[list, None] = None,  # lattice vector in real space
                 device: Union[str, torch.device] = 'cpu') -> None:

        self.device = device

        # Define floating points
        self.tcomplex = torch.complex64
        self.tfloat = torch.float32
        self.tint = torch.int32
        self.nfloat = np.float32

        # Create a logger
        self.solver_logger = Logger()

        self.n_batches = batch_size

        if rdim is None:
            self.rdim = [512, 512]
        else:
            self.rdim = rdim

        if kdim is None:
            self.kdim = [3,3]
        else:
            self.kdim = kdim

        if t1 is None:
            self.lattice_t1 = self._add_lattice_vectors([1.0, 0.0])
        else:
            self.lattice_t1 = self._add_lattice_vectors(t1)

        if t2 is None:
            self.lattice_t2 = self._add_lattice_vectors([0.0, 1.0])
        else:
            self.lattice_t2 = self._add_lattice_vectors(t2)

        # Add materials to material_lib
        self._matlib = {}
        if materiallist is not None:
            self.add_materials(material_list=materiallist)

        # Add default material
        _mat_air = create_material(name='air')
        self.add_materials(material_list=[_mat_air])

        self.layer_manager = LayerManager(n_batches=self.n_batches)
        self.update_trn_material(trn_material='air')
        self.update_ref_material(ref_material='air')


        # build device
        vec_p = torch.linspace(-0.5, 0.5, rdim[0], dtype= self.tfloat, device=self.device)
        vec_q = torch.linspace(-0.5, 0.5, rdim[1], dtype= self.tfloat, device=self.device)

        [mesh_q, mesh_p] = torch.meshgrid(vec_q, vec_p, indexing='xy')

        mesh_q = mesh_q.unsqueeze(0).expand(self.n_batches, -1, -1)
        mesh_p = mesh_p.unsqueeze(0).expand(self.n_batches, -1, -1)

        self.XO = mesh_p * self.lattice_t1[:, 0].unsqueeze(-1).unsqueeze(-1) +\
            mesh_q * self.lattice_t2[:, 0].unsqueeze(-1).unsqueeze(-1)
        self.YO = mesh_p * self.lattice_t1[:, 1].unsqueeze(-1).unsqueeze(-1) +\
            mesh_q * self.lattice_t2[:, 1].unsqueeze(-1).unsqueeze(-1)


        # scaling factor of lengths
        self._lenunit = lengthunit.lower()
        self._len_scale = lengthunit_dict[self._lenunit]

    def add_materials(self, material_list: Union[list, None] = None):
        """add_materials.

        Args:
            material_list (list): material_list
        """
        if isinstance(material_list, list):
            for imat in material_list:
                if isinstance(imat, MaterialClass):
                    self._matlib[imat.name] = imat
                else:
                    verr_str = "The element of the argument should be the [MaterialClass] type."
                    self.solver_logger.error(f"ValueError: {verr_str}")
                    raise ValueError(verr_str)
        else:
            verr_str = "Input argument should be a list."
            self.solver_logger.error(f"ValueError: {verr_str}")
            raise ValueError(verr_str)

    def add_layer(self,
                  material_name: str,
                  thickness: torch.Tensor,
                  is_homogeneous: bool = True,
                  is_optimize: bool = False):
        """add_layer.

        Add a new layer to the layer_manager instance.

        Args:
            material_name (str): material_name
            thickness (torch.Tensor): thickness
            is_homogeneous (bool): is_homogeneous
            is_optimize (bool): is_optimize
        """

        if material_name in self._matlib:
            if is_homogeneous:
                self.layer_manager.add_layer(layer_type='homogenous',
                                             thickness=thickness,
                                             material_name=material_name,
                                             is_optimize=is_optimize,
                                             is_dispersive=self._matlib[material_name].isdispersive_er)
            else:
                self.layer_manager.add_layer(layer_type='grating',
                                             thickness=thickness,
                                             material_name=material_name,
                                             is_optimize=is_optimize,
                                             is_dispersive=self._matlib[material_name].isdispersive_er)

        else:
            str_rterr = f"No materials named [{material_name} exists in the material lib.]"
            self.solver_logger.error(str_rterr)
            raise RuntimeError(str_rterr)

    @property
    def layers(self):
        """layers.

        Returns all created layers in the layer_manager instance.
        """
        return self.layer_manager.layers

    def plot_layer(self,
                   layer_index: int = 0,
                   batch_index: int = 0,
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
        if self.layer_manager.layers[layer_index].is_dispersive is False:
            ret = plot2d(data=self.layer_manager.layers[layer_index].ermat[batch_index, :, :],
                          layout = self.get_layout(batch_index=batch_index),
                          func=func, fig_ax=fig_ax, cmap=cmap, labels=labels, title=title)
        else:
            ret = plot2d(data=self.layer_manager.layers[layer_index].ermat[batch_index, frequency_index, :, :],
                          layout = self.get_layout(batch_index=batch_index),
                          func=func, fig_ax=fig_ax, cmap=cmap, labels=labels, title=title)

        return ret

    @tensor_params_check()
    def _add_lattice_vectors(self, lattice_vec: torch.Tensor) -> torch.Tensor:
        """_add_lattice_vectors.

        Checks the input parameters and adds to the lattice vectors.

        Args:
            lattice_vec (torch.Tensor): lattice_vec

        Returns:
            torch.Tensor:
        """
        if lattice_vec.dtype != self.tfloat:
            terr_str = f"The element of the argument should be the [{self.tfloat}] type."
            self.solver_logger.error(f"TypeError: {terr_str}")
            raise TypeError(terr_str)
        return lattice_vec.to(self.device)

    def update_trn_material(self, trn_material: str) -> None:
        """update_trn_material.

        Updates the material of the transmission layer.

        Args:
            trn_material (str): trn_material

        Returns:
            None:
        """
        if trn_material in self._matlib:
            self.layer_manager.update_trn_layer(material_name=trn_material,
                                                is_dispersive=self._matlib[trn_material].isdispersive_er)
        else:
            str_rterr = f"No materials named [{trn_material} exists in the material lib.]"
            self.solver_logger.error(str_rterr)
            raise RuntimeError(str_rterr)

    def update_ref_material(self, ref_material: str):
        """update_ref_material.

        Updates the material of the reflection layer.

        Args:
            ref_material (str): ref_material
        """
        if ref_material in self._matlib:
            self.layer_manager.update_ref_layer(material_name=ref_material,
                                                is_dispersive=self._matlib[ref_material].isdispersive_er)
        else:
            str_rterr = f"No materials named [{ref_material} exists in the material lib.]"
            self.solver_logger.error(str_rterr)
            raise RuntimeError(str_rterr)

    def get_layer_structure(self):
        """get_layer_structure.

        Prints information of all layers.
        """
        # reflection layer
        print("------------------------------------")
        print("layer # Reflection")
        # print(f"\tmaterial name: {self._layerstruct._ref_mat_name}")
        print(f"\tmaterial name: {self.layer_manager.ref_material_name}")
        print(f"\tpermittivity: {self.er1}")
        print(f"\tpermeability: {self.ur1}")

        # structure layers
        print("------------------------------------")
        # for ilay in range(self._layerstruct._nlayer):
        for ilay in range(self.layer_manager.nlayer):
            print(f"layer # {ilay}")
            print(f"\tmaterial name: {self.layer_manager.layers[ilay].material_name}")
            print(f"\tthinkness = {self.layer_manager.layers[ilay].thickness}")
            print(f"\tdispersive: {self.layer_manager.layers[ilay].is_dispersive}")
            print(f"\thomogeneous: {self.layer_manager.layers[ilay].is_homogeneous}")
            print(f"\tto be optimized: {self.layer_manager.layers[ilay].is_optimize}")
            print("------------------------------------")

        # Transmission layers
        print("layer # Transmission")
        print(f"\tmaterial name: {self.layer_manager.trn_material_name}")
        print(f"\tpermittivity: {self.er2}")
        print(f"\tpermeability: {self.ur2}")
        print("------------------------------------")

    def display_fitted_permittivity(self):
        """display_fitted_permittivity.

        Displays information about the dispersive permittivity.
        """
        # if True not in self._layerstruct._is_dispers:
        if True not in [self.layer_manager.layers[ii].is_dispersive for ii in range(self.layer_manager.nlayer)]:
            print("No dispersive material loaded.")
        else:
            for imat in range(self.layer_manager.nlayer):
                if self.layer_manager.layers[imat].is_dispersive is True:
                    matname = self.layer_manager.layers[imat].material_name
                    wls = self._matlib[matname].fitted_data['wavelengths']
                    data_eps1 = self._matlib[matname].fitted_data['data_eps1']
                    data_eps2 = self._matlib[matname].fitted_data['data_eps2']
                    pe1 = self._matlib[matname].fitted_data['fitted_crv1']
                    pe2 = self._matlib[matname].fitted_data['fitted_crv2']
                    _, fig_ax = plt.subplots()
                    ln1data = fig_ax.plot(wls,
                                      data_eps1, 'r.', label='data e\'')
                    ln1fit = fig_ax.plot(self._lam0,
                                     pe1(self._lam0), 'c^-', label='fitted e\'')
                    fig_ax2 = fig_ax.twinx()
                    ln2data = fig_ax2.plot(wls,
                                       data_eps2, 'g*', label='data e\"')
                    ln2fit = fig_ax2.plot(self._lam0,
                                      pe2(self._lam0), 'm.-', label='fitted e\"')
                    # ax2.legend(loc='best')
                    fig_ax.set_title(f"Permittivity [{matname}]")
                    fig_ax.set_xlabel(f"Wavelength [{self._lenunit}]")
                    fig_ax.set_ylabel('Eps\' [Real Part]')
                    fig_ax2.set_ylabel('Eps\" [Imag Part]')
                    lns = ln1data + ln1fit + ln2data + ln2fit
                    labs = [l.get_label() for l in lns]
                    fig_ax.legend(lns, labs, loc='best')

    def get_layout(self, batch_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """get_layout.

        Returns the layout matrices of the cell.

        Args:
            batch_index (int): batch_index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        return self.XO[batch_index, :, :], self.YO[batch_index, :, :]

    def gen_toeplitz_matrix(self, layer_index: int, param: str = 'er', is_dispersive: bool = False) -> torch.Tensor:
        """gen_toeplitz_matrix.

        Generates the Toeplitz matrix.

        Args:
            layer_index (int): layer_index
            param (str): param
            is_dispersive (bool): is_dispersive

        Returns:
            torch.Tensor:
        """

        # The dimensions of ermat/urmat is (n_batches, n_layers, H, W)
        # The dimensions of kermat/kurmat is (n_batches, n_layers, kH, kW)

        # numbers of spatial harmonics
        # kHW = self.kdim[0] * self.kdim[1]
        if param == 'er':
            if is_dispersive is True:
                kmatrix = gen_toeplitz2d_dispersive(
                    self.layer_manager.layers[layer_index].ermat,
                    self.kdim[0], self.kdim[1])
            else:
                kmatrix = gen_toeplitz2d_normal(
                    self.layer_manager.layers[layer_index].ermat,
                    self.kdim[0], self.kdim[1])
            self.n_kermat_gen += 1
        elif param == 'ur':
            kmatrix = gen_toeplitz2d_normal(
                self.layer_manager.layers[layer_index].urmat, self.kdim[0], self.kdim[1])
            self.n_kurmat_gen += 1

        return kmatrix

    @property
    def lam0(self):
        """lam0.
        attribute.
        """
        return self._lam0

    @lam0.setter
    def lam0(self, value):
        if isinstance(value, float):
            self._lam0 = np.array([value])
        elif isinstance(value, np.ndarray):
            self._lam0 = value
        else:
            raise ValueError(f"Wrong input type {type(value)}")

    @property
    def er1(self):
        """er1.
        attribute.
        """
        return torch.tensor(self._matlib[self.layer_manager.ref_material_name].er,
                            dtype=self.tcomplex, device=self.device)

    @property
    def er2(self):
        """er2.
        attribute.
        """
        return torch.tensor(self._matlib[self.layer_manager.trn_material_name].er,
                            dtype=self.tcomplex, device=self.device)

    @property
    def ur1(self):
        """ur1.
        attribute.
        """
        return torch.tensor(self._matlib[self.layer_manager.ref_material_name].ur,
                            dtype=self.tcomplex, device=self.device)

    @property
    def ur2(self):
        """ur2.
        attribute.
        """
        return torch.tensor(self._matlib[self.layer_manager.trn_material_name].ur,
                            dtype=self.tcomplex, device=self.device)
