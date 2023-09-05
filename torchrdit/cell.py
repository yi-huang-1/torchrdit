""" This file defines classes for models to be simulated. """
from typing import Callable, Tuple, Union, Any

import numpy as np
import torch

from .utils import tensor_params_check, create_material
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

    # Define floating points
    tcomplex = torch.complex64
    tfloat = torch.float32
    tint = torch.int32
    nfloat = np.float32

    def __init__(self,
                 batch_size: int,  # batch size of data set
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 device: Union[str, torch.device] = 'cpu') -> None:

        self.device = device

        # Create a logger
        self.solver_logger = Logger()

        self.n_batches = batch_size

        if isinstance(rdim, list) is False or len(rdim) != 2:
            raise ValueError(f"Invalid input rdim [{rdim}]")
        self.rdim = rdim

        if kdim is None:
            self.kdim = [3,3]
        else:
            self.kdim = kdim

        if t1 is None:
            self.lattice_t1 = self._add_lattice_vectors(torch.tensor([[1.0, 0.0]]))
        else:
            self.lattice_t1 = self._add_lattice_vectors(t1)

        if t2 is None:
            self.lattice_t2 = self._add_lattice_vectors(torch.tensor([[0.0, 1.0]]))
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

    def add_materials(self, material_list: list = []):
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

        self._init_dispersive_materials()

    def add_layer(self,
                  material_name: Any,
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
        if isinstance(material_name, MaterialClass):
            if material_name.name not in self._matlib:
                self.add_materials([material_name])
            material_name = material_name.name

        if isinstance(material_name, str) and material_name in self._matlib:
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

    def _init_dispersive_materials(self) -> None:
        """_init_dispersive_materials.
        
        Initialize the dispersive profile of materials.

        Args:

        Returns:
            None:
        """
        for imat in self._matlib.values():
            if imat.isdispersive_er is True:
                # if type(imat.er) == type(None):
                # if isinstance(imat.er, NoneType):
                if imat.er is None:
                    imat.load_dispersive_er(
                        self._lam0, self._lenunit)

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
            try:
                lattice_vec = lattice_vec.to(self.tfloat)
            except:
                terr_str = f"The element of the argument should be the [{self.tfloat}] type."
                self.solver_logger.error(f"TypeError: {terr_str}")
                raise TypeError(terr_str)
        return lattice_vec.to(self.device)

    def update_trn_material(self, trn_material: Any) -> None:
        """update_trn_material.

        Updates the material of the transmission layer.

        Args:
            trn_material (str): trn_material

        Returns:
            None:
        """
        if isinstance(trn_material, MaterialClass):
            if trn_material.name not in self._matlib:
                self.add_materials([trn_material])
            trn_material = trn_material.name

        if isinstance(trn_material, str) and trn_material in self._matlib:
            self.layer_manager.update_trn_layer(material_name=trn_material,
                                                is_dispersive=self._matlib[trn_material].isdispersive_er)
        else:
            str_rterr = f"No materials named [{trn_material} exists in the material lib.]"
            self.solver_logger.error(str_rterr)
            raise RuntimeError(str_rterr)

    def update_ref_material(self, ref_material: Any) -> None:
        """update_ref_material.

        Updates the material of the reflection layer.

        Args:
            ref_material (str): ref_material
        """
        if isinstance(ref_material, MaterialClass):
            if ref_material.name not in self._matlib:
                self.add_materials([ref_material])
            ref_material = ref_material.name

        if isinstance(ref_material, str) and ref_material in self._matlib:
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

        print("layer # Transmission")
        print(f"\tmaterial name: {self.layer_manager.trn_material_name}")
        print(f"\tpermittivity: {self.er2}")
        print(f"\tpermeability: {self.ur2}")
        print("------------------------------------")

    def get_layout(self, batch_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """get_layout.

        Returns the layout matrices of the cell.

        Args:
            batch_index (int): batch_index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        return self.XO[batch_index, :, :], self.YO[batch_index, :, :]

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

