""" This file defines all classes about layers. """
from abc import ABCMeta, abstractmethod
from .utils import tensor_params_check

from torch.fft import fft2, fftshift
import torch
import numpy as np

class Layer(metaclass=ABCMeta):
    """Layer.
    This is the base class of Layer objects. This class is abstract.
    """

    @abstractmethod
    def __init__(self, thickness: float = 0.0, material_name: str = '', is_optimize: bool = False, **kwargs) -> None:
        """
        Initialize the Layer instance.

        Args:
            thickness (float): Thickness of the layer.
            material_name (str): Name of the material.
            is_optimize (bool): Flag indicating if the layer is optimized.
            **kwargs: Additional keyword arguments.
        """
        self._thickness = thickness
        self._material_name = material_name
        self._is_homogeneous = True
        self._is_optimize = is_optimize
        self._is_dispersive = False
        self._is_solved = False

        self.ermat = None
        self.urmat = None
        self.kermat = None
        self.kurmat = None
        self.mask_format = None

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the layer."""
        pass

    @property
    def thickness(self) -> float:
        """Return the thickness of the layer."""
        return self._thickness

    @thickness.setter
    def thickness(self, thickness: float):
        """Update the thickness of the layer.

        Args:
            thickness (float): Thickness of the layer.
        """
        self._thickness = thickness

    @property
    def material_name(self) -> str:
        """Return the name of the material."""
        return self._material_name

    @material_name.setter
    def material_name(self, material_name: str):
        """Update the name of the material.

        Args:
            material_name (str): Name of the material.
        """
        self._material_name = material_name

    @property
    def is_homogeneous(self) -> bool:
        """Return the homogeneity flag of the layer."""
        return self._is_homogeneous

    @property
    def is_dispersive(self) -> bool:
        """Return the dispersive flag of the layer."""
        return self._is_dispersive

    @is_dispersive.setter
    def is_dispersive(self, is_dispersive: bool):
        """Set the dispersive flag of the layer.

        Args:
            is_dispersive (bool): Dispersive flag.
        """
        self._is_dispersive = is_dispersive

    @property
    def is_optimize(self) -> bool:
        """Return the optimization flag of the layer."""
        return self._is_optimize

    @is_optimize.setter
    def is_optimize(self, is_optimize: bool):
        """Set the optimization flag of the layer.

        Args:
            is_optimize (bool): Optimization flag.
        """
        self._is_optimize = is_optimize

    @property
    def is_solved(self) -> bool:
        """Return the solved flag of the layer."""
        return self._is_solved

    @is_solved.setter
    def is_solved(self, is_solved: bool):
        """Set the solved flag of the layer.

        Args:
            is_solved (bool): Solved flag.
        """
        self._is_solved = is_solved

class LayerBuilder(metaclass=ABCMeta):
    """LayerBuilder.
    This is the base class of the layer builder. This class is abstract.
    """

    @abstractmethod
    def __init__(self) -> None:
        self.layer = None

    @abstractmethod
    def create_layer(self):
        """create_layer.
        abstractmethod of creating layer.
        """
        pass

    def update_thickness(self, thickness):
        """update_thickness.
        abstractmethod of updating layer thickness.

        Args:
            thickness:
        """
        self.layer.thickness = thickness

    def update_material_name(self, material_name):
        """update_material_name.
        abstractmethod of updating material type.

        Args:
            material_name:
        """
        self.layer.material_name = material_name

    def set_optimize(self, is_optimize):
        """set_optimize.
        abstractmethod of setting layer to be optimized.

        Args:
            is_optimize:
        """
        self.layer.is_optimize = is_optimize

    # read-only property
    def get_layer_instance(self):
        """get_layer_instance.
        abstractmethod of getting layer object.
        """
        return self.layer

    def set_dispersive(self, is_dispersive):
        """set_dispersive.
        abstractmethod of set meterial dispersive.

        Args:
            is_dispersive:
        """
        self.layer.is_dispersive = is_dispersive

class HomogeneousLayer(Layer):
    """HomogeneousLayer.
    This class represents a homogeneous layer.
    """

    def __init__(self, thickness: float = 0.0, material_name: str = '', is_optimize: bool = False, **kwargs) -> None:
        """
        Initialize the HomogeneousLayer instance.

        Args:
            thickness (float): Thickness of the layer.
            material_name (str): Name of the material.
            is_optimize (bool): Flag indicating if the layer is optimized.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(thickness, material_name, is_optimize, **kwargs)

    def __str__(self) -> str:
        """Return a string representation of the homogeneous layer."""
        return f"HomogeneousLayer(thickness={self._thickness}, material_name={self._material_name})"

class GratingLayer(Layer):
    """GratingLayer.
    This class represents a grating layer.
    """

    def __init__(self, thickness: float = 0.0, material_name: str = '', is_optimize: bool = False, **kwargs) -> None:
        """
        Initialize the GratingLayer instance.

        Args:
            thickness (float): Thickness of the layer.
            material_name (str): Name of the material.
            is_optimize (bool): Flag indicating if the layer is optimized.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(thickness, material_name, is_optimize, **kwargs)
        self._is_homogeneous = False

    def __str__(self) -> str:
        """Return a string representation of the grating layer."""
        return f"GratingLayer(thickness={self._thickness}, material_name={self._material_name})"

class HomogeneousLayerBuilder(LayerBuilder):
    """HomogeneousLayerBuilder.
    This class defines the builder for HomogeneousLayer.
    """

    def __init__(self) -> None:
        super().__init__()

    def create_layer(self):
        self.layer = HomogeneousLayer()

class GratingLayerBuilder(HomogeneousLayerBuilder):
    """GratingLayerBuilder.
    This class defines the builder for GratingLayer.
    """
    def __init__(self) -> None:
        super().__init__()

    def create_layer(self):
        self.layer = GratingLayer()

class LayerDirector:
    """LayerDirector.
    This class defines the director class for layer builders.
    """

    def __init__(self) -> None:
        pass

    def build_layer(self, layer_type, thickness, material_name, is_optimize = False, is_dispersive = False) -> Layer:
        """build_layer.
        This function runs the actual building process for different types of layers.

        Args:
            layer_type:
            thickness:
            material_name:
            is_optimize:
            is_dispersive:

        Returns:
            Layer:
        """
        if layer_type == 'homogeneous':
            layer_builder = HomogeneousLayerBuilder()
        elif layer_type == 'grating':
            layer_builder = GratingLayerBuilder()
        else:
            layer_builder = HomogeneousLayerBuilder()

        # layer_builder.create_layer(thickness=thickness, material_name=material_name, is_optimize=is_optimize)
        layer_builder.create_layer()
        layer_builder.update_thickness(thickness=thickness)
        layer_builder.update_material_name(material_name=material_name)
        layer_builder.set_optimize(is_optimize=is_optimize)
        layer_builder.set_dispersive(is_dispersive=is_dispersive)

        return layer_builder.get_layer_instance()

class LayerManager:
    """LayerManager.

    A class to manage layer instances.
    """

    def __init__(self,
                 lattice_t1,
                 lattice_t2,
                 vec_p,
                 vec_q) -> None:
        self.layers = []
        self.layer_director = LayerDirector()
        
        # semi-infinite layer
        self._ref_material_name = 'air'
        self._trn_material_name = 'air'

        self._is_ref_dispers = False
        self._is_trn_dispers = False

        self.lattice_t1 = lattice_t1
        self.lattice_t2 = lattice_t2
        self.vec_p = vec_p
        self.vec_q = vec_q

    def gen_toeplitz_matrix(self, layer_index: int,
                            n_harmonic1: int,
                            n_harmonic2: int,
                            param: str = 'er',
                            method :str = 'FFT'):
        # The dimensions of ermat/urmat is (n_layers, H, W)
        # The dimensions of kermat/kurmat is (n_layers, kH, kW)

        if param == 'er':
            if self.layers[layer_index].ermat is None:
                raise ValueError(f"Permittivity distribution of specified layer {layer_index} not found!")

            self.layers[layer_index].kermat = self._gen_toeplitz2d(
                self.layers[layer_index].ermat, n_harmonic1, n_harmonic2, method)
        elif param == 'ur':
            if self.layers[layer_index].urmat is None:
                raise ValueError(f"Permeability distribution of specified layer {layer_index} not found!")

            self.layers[layer_index].kurmat = self._gen_toeplitz2d(
                self.layers[layer_index].urmat, n_harmonic1, n_harmonic2, method)


    def _gen_toeplitz2d(self, input_matrix: torch.Tensor,
                        nharmonic_1: int = 1,
                        nharmonic_2: int = 1,
                        method :str = 'FFT') -> torch.Tensor:
        """This function constructs Toeplitz matrices from
        a real-space 2D grid.

        Args:
            input_matrix (torch.Tensor): Grid in real space: (H, W)
            nharmonic_1 (int): Number of harmonics for the T1 direction.
            nharmonic_2 (int): Number of harmonics for the T2 direction.
            method (str): method of computing Toeplitz matrix. ['FFT': for all cell type; 'Analytical': only for Cartesian]
        Returns:
            toep_matrix (torch.Tensor): Toeplitz matrix: (n_harmonics, n_harmonics)
        """

        if input_matrix.ndim == 2:
            n_height, n_width = input_matrix.size()
        elif input_matrix.ndim == 3:
            _, n_height, n_width = input_matrix.size()
        else:
            raise ValueError(f"Unexpected dimensions of input_matrix [{input_matrix.shape}]")

        # Computes indices of spatial harmonics
        nharmonic_1 = int(nharmonic_1)
        nharmonic_2 = int(nharmonic_2)
        n_harmonics = nharmonic_1 * nharmonic_2

        rows = torch.arange(n_harmonics, dtype = torch.int64, device = input_matrix.device)
        cols = torch.arange(n_harmonics, dtype = torch.int64, device = input_matrix.device)

        rows, cols = torch.meshgrid(rows, cols, indexing = "ij")

        row_s = torch.div(rows, nharmonic_2, rounding_mode = 'floor')
        prow_t = rows - row_s * nharmonic_2
        col_s = torch.div(cols, nharmonic_2, rounding_mode = 'floor')
        pcol_t = cols - col_s * nharmonic_2
        qf_t = row_s - col_s
        pf_t = prow_t - pcol_t

        if method == 'FFT':
            # compute fourier coefficents of input_matrix for only the last two dimensions
            finput_matrix = fftshift(fftshift(fft2(input_matrix), dim=-1), dim=-2)\
                / (n_height * n_width)

            # compute zero-order harmonic
            p_0 = torch.floor(torch.tensor(n_height / 2)).to(torch.int64)
            q_0 = torch.floor(torch.tensor(n_width / 2)).to(torch.int64)

            if finput_matrix.ndim == 2:
                return finput_matrix[p_0 - pf_t, q_0 - qf_t]
            elif finput_matrix.ndim == 3:
                return finput_matrix[:, p_0 - pf_t, q_0 - qf_t]
            else:
                return None
        elif method == 'Analytical':
            """
            This method is deveploped based on: https://github.com/2iw31Zhv/diffsmat_py
            @author: Ziwei Zhu
            """
            ms = torch.arange(-nharmonic_1+1, nharmonic_1, device = input_matrix.device)
            ns = torch.arange(-nharmonic_2+1, nharmonic_2, device = input_matrix.device)

            lx = (self.lattice_t1[0] + self.lattice_t2[0])
            ly = (self.lattice_t1[1] + self.lattice_t2[1])

            dx = lx / n_height
            dy = ly / n_width

            kx = 2 * np.pi / lx * ms
            ky = 2 * np.pi / ly * ns

            xs = self.vec_p * lx
            ys = self.vec_q * ly
            
            basis_kx_conj = torch.exp(1j * kx[:, None] * xs[None, :]) * torch.special.sinc(ms * dx / lx)[:, None]
            basis_ky_conj = torch.exp(1j * ky[None, :] * ys[:, None]) * torch.special.sinc(ns * dy / ly)[None, :]

            finput_matrix =  dx * dy * basis_kx_conj @ input_matrix.transpose(-2, -1) @ basis_ky_conj / (lx * ly)

            if finput_matrix.ndim == 2:
                return finput_matrix[nharmonic_1 + pf_t - 1, nharmonic_2 + qf_t - 1]
            elif finput_matrix.ndim == 3:
                return finput_matrix[:, nharmonic_1 + pf_t - 1, nharmonic_2 + qf_t - 1]
            else:
                return None


    @tensor_params_check(check_start_index=2, check_stop_index=2)
    def add_layer(self, layer_type, thickness, material_name, is_optimize = False, is_dispersive = False):
        """add_layer.
        This function adds new layer object to the layer manager.

        Args:
            layer_type:
            thickness:
            material_name:
            is_optimize:
            is_dispersive:
        """
        new_layer = self.layer_director.build_layer(layer_type=layer_type,
                                                    thickness=thickness,
                                                    material_name=material_name,
                                                    is_optimize=is_optimize,
                                                    is_dispersive=is_dispersive)
        self.layers.append(new_layer)

    def replace_layer_to_homogeneous(self, layer_index):
        """replace_layer_to_homogeneous.
        This fucntion change the type of the layer to homogenous.

        Args:
            layer_index:
        """
        new_layer = self.layer_director.build_layer(layer_type='homogenous',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    def replace_layer_to_grating(self, layer_index):
        """replace_layer_to_grating.
        This fucntion change the type of the layer to grating.

        Args:
            layer_index:
        """
        new_layer = self.layer_director.build_layer(layer_type='grating',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    @tensor_params_check(check_start_index=2, check_stop_index=2)
    def update_layer_thickness(self, layer_index, thickness):
        """update_layer_thickness.
        This function updates the thickness of the layer.

        Args:
            layer_index:
            thickness:
        """
        self.layers[layer_index].thickness = thickness

    def update_trn_layer(self, material_name: str, is_dispersive: bool):
        """update_trn_layer.
        This function updates the transmission layer.

        Args:
            material_name (str): material_name
            is_dispersive (bool): is_dispersive
        """
        self._trn_material_name = material_name
        self._is_trn_dispers = is_dispersive

    def update_ref_layer(self, material_name: str, is_dispersive: bool):
        """update_ref_layer.
        This function updates the reflection layer.

        Args:
            material_name (str): material_name
            is_dispersive (bool): is_dispersive
        """
        self._ref_material_name = material_name
        self._is_ref_dispers = is_dispersive

    @property
    def ref_material_name(self) -> str:
        """ref_material_name.
        returns the material name of the reflection layer.

        Args:

        Returns:
            str:
        """
        return self._ref_material_name

    @property
    def trn_material_name(self) -> str:
        """trn_material_name.
        returns the material name of the transmission layer.

        Args:

        Returns:
            str:
        """
        return self._trn_material_name

    @property
    def is_ref_dispersive(self) -> bool:
        """is_ref_dispersive.
        returns whether reflection layer material is dispersive.

        Args:

        Returns:
            bool:
        """
        return self._is_ref_dispers

    @property
    def is_trn_dispersive(self) -> bool:
        """is_trn_dispersive.
        returns whether transmission layer material is dispersive.

        Args:

        Returns:
            bool:
        """
        return self._is_trn_dispers

    @property
    def nlayer(self) -> int:
        """nlayer.
        return number of layers in the model.

        Args:

        Returns:
            int:
        """
        return len(self.layers)
