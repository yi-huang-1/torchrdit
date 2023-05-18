import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Callable, Tuple, Union, Any

from .utils import gen_toeplitz2d_dispersive, gen_toeplitz2d_normal, tensor_params_check
from .materials import materials
from .constants import *
from .logger import Logger
from .layers import LayerManager
from .viz import plot2d

# Function Type
FuncType = Callable[..., Any]

class cell3d():
    """ Base class of unit cell """

    n_kermat_gen = 0
    n_kurmat_gen = 0

    def __init__(self,
                 batch_size: int,  # batch size of data set
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3, 3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: list = [1.0, 0.0],  # lattice vector in real space
                 t2: list = [0.0, 1.0],  # lattice vector in real space
                 device: Union[str, torch.device] = 'cpu') -> None:

        self.device = device

        # Define floating points
        self.tcomplex = torch.complex128
        self.tfloat = torch.float64
        self.tint = torch.int64
        self.nfloat = np.float64

        # Create a logger
        self.solver_logger = Logger()
        
        self.n_batches = batch_size

        self.rdim = rdim
        self.kdim = kdim

        self.t1 = self._add_lattice_vectors(t1)
        self.t2 = self._add_lattice_vectors(t2) 

        # Add materials to material_lib
        self._matlib = dict()
        self.add_materials(material_list=materiallist)

        # Add default material
        _mat_air = materials(name='air')
        self.add_materials(material_list=[_mat_air])

        self.layer_manager = LayerManager(n_batches=self.n_batches)
        self.update_trn_material(trn_material='air')
        self.update_ref_material(ref_material='air')


        # build device
        p = torch.linspace(-0.5, 0.5, rdim[0], dtype= self.tfloat, device=self.device)
        q = torch.linspace(-0.5, 0.5, rdim[1], dtype= self.tfloat, device=self.device)

        [Q, P] = torch.meshgrid(q, p, indexing='xy')

        Q = Q.unsqueeze(0).expand(self.n_batches, -1, -1)
        P = P.unsqueeze(0).expand(self.n_batches, -1, -1)

        self.XO = P * self.t1[:, 0].unsqueeze(-1).unsqueeze(-1) + Q * self.t2[:, 0].unsqueeze(-1).unsqueeze(-1)
        self.YO = P * self.t1[:, 1].unsqueeze(-1).unsqueeze(-1) + Q * self.t2[:, 1].unsqueeze(-1).unsqueeze(-1)


        # scaling factor of lengths
        self._lenunit = lengthunit.lower()
        self._len_scale = lengthunit_dict[self._lenunit]

    def add_materials(self, material_list: list = []):
        """add_materials.

        Args:
            material_list (list): material_list
        """
        if type(material_list) == list:
            for imat in material_list:
                if type(imat) == materials:
                    self._matlib[imat.name] = imat
                else:
                    verr_str = "The element of the argument should be the [materials] type."
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
        if self.layer_manager.layers[layer_index].is_dispersive == False:
            return plot2d(data=self.layer_manager.layers[layer_index].ermat[batch_index, :, :],
                          layout = self.get_layout(batch_index=batch_index),
                          func=func, fig_ax=fig_ax, cmap=cmap, labels=labels, title=title)
        else:
            return plot2d(data=self.layer_manager.layers[layer_index].ermat[batch_index, frequency_index, :, :],
                          layout = self.get_layout(batch_index=batch_index),
                          func=func, fig_ax=fig_ax, cmap=cmap, labels=labels, title=title)

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
        print(f"layer # Reflection")
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
        print(f"layer # Transmission")
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
            print(f"No dispersive material loaded.")
        else:
            for imat in range(self.layer_manager.nlayer):
                if self.layer_manager.layers[imat].is_dispersive == True:
                    matname = self.layer_manager.layers[imat].material_name 
                    wls = self._matlib[matname]._fitted_data['wavelengths']
                    data_eps1 = self._matlib[matname]._fitted_data['data_eps1']
                    data_eps2 = self._matlib[matname]._fitted_data['data_eps2']
                    pe1 = self._matlib[matname]._fitted_data['fitted_crv1']
                    pe2 = self._matlib[matname]._fitted_data['fitted_crv2']
                    _, ax = plt.subplots()
                    ln1data = ax.plot(wls,
                                      data_eps1, 'r.', label='data e\'')
                    ln1fit = ax.plot(self._lam0,
                                     pe1(self._lam0), 'c^-', label='fitted e\'')
                    ax2 = ax.twinx()
                    ln2data = ax2.plot(wls,
                                       data_eps2, 'g*', label='data e\"')
                    ln2fit = ax2.plot(self._lam0,
                                      pe2(self._lam0), 'm.-', label='fitted e\"')
                    # ax2.legend(loc='best')
                    ax.set_title(f"Permittivity [{matname}]")
                    ax.set_xlabel(f"Wavelength [{self._lenunit}]")
                    ax.set_ylabel('Eps\' [Real Part]')
                    ax2.set_ylabel('Eps\" [Imag Part]')
                    lns = ln1data + ln1fit + ln2data + ln2fit
                    labs = [l.get_label() for l in lns]
                    ax.legend(lns, labs, loc='best')

    def get_layout(self, batch_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """get_layout.

        Returns the layout matrices of the cell.

        Args:
            batch_index (int): batch_index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        return self.XO[batch_index, :, :], self.YO[batch_index, :, :]

    def gen_Toeplitz_matrix(self, layer_index: int, param: str = 'er', is_dispersive: bool = False) -> torch.Tensor:
        """gen_Toeplitz_matrix.

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
            if is_dispersive == True:
                kmatrix = gen_toeplitz2d_dispersive(
                    self.layer_manager.layers[layer_index].ermat, self.kdim[0], self.kdim[1], tcomplex=self.tcomplex, tint=self.tint)
            else:
                kmatrix = gen_toeplitz2d_normal(
                    self.layer_manager.layers[layer_index].ermat, self.kdim[0], self.kdim[1], tcomplex=self.tcomplex, tint=self.tint)
            self.n_kermat_gen += 1
        elif param == 'ur':
            kmatrix = gen_toeplitz2d_normal(
                self.layer_manager.layers[layer_index].urmat, self.kdim[0], self.kdim[1], tcomplex=self.tcomplex, tint=self.tint)
            self.n_kurmat_gen += 1

        return kmatrix

    @property
    def lam0(self):
        return self._lam0

    @lam0.setter
    def lam0(self, value):
        if type(value) == self.nfloat or type(value) == float:
            self._lam0 = np.array([value])
        elif type(value) == np.ndarray:
            self._lam0 = value

    @property
    def er1(self):
        # return torch.tensor(self._layerstruct._er1, dtype=torch.complex128, device=self.device)
        return torch.tensor(self._matlib[self.layer_manager.ref_material_name].er,
                            dtype=self.tcomplex, device=self.device)

    @property
    def er2(self):
        # return torch.tensor(self._layerstruct._er2, dtype=torch.complex128, device=self.device)
        return torch.tensor(self._matlib[self.layer_manager.trn_material_name].er,
                            dtype=self.tcomplex, device=self.device)

    @property
    def ur1(self):
        # return torch.tensor(self._layerstruct._ur1, dtype=torch.complex128, device=self.device)
        return torch.tensor(self._matlib[self.layer_manager.ref_material_name].ur,
                            dtype=self.tcomplex, device=self.device)

    @property
    def ur2(self):
        # return torch.tensor(self._layerstruct._ur2, dtype=torch.complex128, device=self.device)
        return torch.tensor(self._matlib[self.layer_manager.trn_material_name].ur,
                            dtype=self.tcomplex, device=self.device)

