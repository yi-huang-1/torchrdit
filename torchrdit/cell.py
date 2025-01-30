""" This file defines classes for models to be simulated. """
from typing import Callable, Tuple, Union, List, Any

import numpy as np
import torch

from .utils import tensor_params_check, create_material
from .materials import MaterialClass
from .constants import lengthunit_dict
from .logger import Logger
from .layers import LayerManager

from skimage.draw import disk, rectangle, polygon
from skimage.measure import grid_points_in_poly
from matplotlib.path import Path

# Function Type
FuncType = Callable[..., Any]

class CellType:
    Cartesian = 'Cartesian'
    Other = 'Other'

class ShapeGenerator:
    """ Class to generate binary shape masks """
    def __init__(self, XO: torch.Tensor, YO: torch.Tensor, rdim: Tuple[int, int]):
        assert isinstance(XO, torch.Tensor) and isinstance(YO, torch.Tensor), "XO and YO must be torch.Tensor"
        self.rdim = rdim
        
        self.X_real = XO
        self.Y_real = YO
    
    def generate_circle_mask(self, center=None, radius=0.1):
        """
        Generate a binary mask for a circle based on real coordinates.

        Parameters:
            center (tuple): Real coordinates of the circle center (x, y).
            radius (float): Radius of the circle in real units.

        Returns:
            torch.Tensor: Binary mask for the circle.
        """
        mask = np.zeros(self.rdim, dtype=np.uint8)
        if center is None:
            center = (self.X_real.mean().item(), self.Y_real.mean().item())

        # Mask condition based on real coordinates
        distance = np.sqrt((self.X_real.numpy() - center[0])**2 + (self.Y_real.numpy() - center[1])**2)
        mask[distance <= radius] = 1

        return torch.tensor(mask, dtype=torch.uint8)

    
    def generate_rectangle_mask(self, bottom_left=None, top_right=None):
        """
        Generate a binary mask for a rectangle based on real coordinates.
        
        Parameters:
            bottom_left (tuple): Real coordinates of the bottom-left corner (x, y).
            top_right (tuple): Real coordinates of the top-right corner (x, y).

        Returns:
            torch.Tensor: Binary mask for the rectangle.
        """
        mask = np.zeros(self.rdim, dtype=np.uint8)
        if bottom_left is None:
            bottom_left = (self.X_real.min().item(), self.Y_real.min().item())
        if top_right is None:
            top_right = (self.X_real.max().item(), self.Y_real.max().item())

        # Mask condition based on real coordinates
        mask[
            (self.X_real >= bottom_left[0]) & (self.X_real <= top_right[0]) &
            (self.Y_real >= bottom_left[1]) & (self.Y_real <= top_right[1])
        ] = 1

        return torch.tensor(mask, dtype=torch.uint8)
    

    def generate_polygon_mask(self, polygon_points, invert=False):
        """
        Generate a binary mask for a polygon based on real coordinates.

        Parameters:
            polygon_points (list of tuples): List of real coordinates of the polygon vertices [(x1, y1), (x2, y2), ...].
            invert (bool): If True, inverts the mask to fill the outside region.

        Returns:
            torch.Tensor: Binary mask for the polygon.
        """
        mask = np.zeros(self.rdim, dtype=np.uint8)

        # Create a path object for the polygon in real space
        poly_path = Path(polygon_points)

        # Flatten the grid points into (x, y) pairs
        points = np.vstack((self.X_real.ravel(), self.Y_real.ravel())).T

        # Use the Path.contains_points method for efficient point-in-polygon testing
        inside = poly_path.contains_points(points)

        # Reshape the result back to the grid dimensions
        mask = inside.reshape(self.rdim)

        if invert:
            mask = 1 - mask  # Invert the mask

        return torch.tensor(mask, dtype=torch.uint8)
    
    def combine_masks(self, mask1, mask2, operation="union"):
        """
        Combines two binary masks using a specified boolean operation.

        Parameters:
            mask1 (torch.Tensor): First binary mask.
            mask2 (torch.Tensor): Second binary mask.
            operation (str): Boolean operation ('union', 'intersection', 'difference', 'subtract').

        Returns:
            torch.Tensor: The combined mask.
        """
        if operation == "union":
            return mask1 | mask2
        elif operation == "intersection":
            return mask1 & mask2
        elif operation == "difference":
            return mask1 ^ mask2
        elif operation == "subtract":
            return mask1 & ~mask2
        else:
            raise ValueError("Invalid operation. Choose from 'union', 'intersection', 'difference', or 'subtract'.")

    def _point_in_polygon(self, point, polygon):
        """
        Determines if a point is inside a polygon using a vectorized ray-casting algorithm.

        Parameters:
            point (tuple): Point coordinates (x, y).
            polygon (np.ndarray or torch.Tensor): Array of polygon vertices [(x1, y1), (x2, y2), ...].

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        # Ensure inputs are numpy arrays
        if isinstance(polygon, torch.Tensor):
            polygon = polygon.numpy()
        x, y = point
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(y, torch.Tensor):
            y = y.item()

        # Vectorized ray-casting
        x1, y1 = polygon[:-1, 0], polygon[:-1, 1]
        x2, y2 = polygon[1:, 0], polygon[1:, 1]

        # Check if the point is within the vertical range of each edge
        within_y_bounds = (y > np.minimum(y1, y2)) & (y <= np.maximum(y1, y2))

        # Avoid horizontal edges where no intersection can occur
        non_horizontal = y1 != y2

        # Calculate the intersection of the horizontal ray with polygon edges
        xinters = np.where(non_horizontal, (y - y1) * (x2 - x1) / (y2 - y1) + x1, np.inf)

        # Check if the x-coordinate of the point is less than the intersection
        intersects = (x <= xinters) & within_y_bounds

        # Determine if the number of intersections is odd
        return np.sum(intersects) % 2 == 1

class Cell3D():
    """ Base class of unit cell """

    n_kermat_gen = 0
    n_kurmat_gen = 0

    # Define floating points
    tcomplex = torch.complex128
    tfloat = torch.float64
    tint = torch.int64
    nfloat = np.float64

    cell_type = CellType.Cartesian

    def __init__(self,
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: List[int] = [512, 512],  # dimensions in real space: (H, W)
                 kdim: List[int] = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: List[MaterialClass] = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 device: Union[str, torch.device] = 'cpu') -> None:

        self.device = device

        # Create a logger
        self.solver_logger = Logger()

        if isinstance(rdim, list) is False or len(rdim) != 2:
            raise ValueError(f"Invalid input rdim [{rdim}]")
        self.rdim = rdim

        if isinstance(kdim, list) is False or len(kdim) != 2:
            raise ValueError(f"Invalid input kdim [{kdim}]")
        self.kdim = kdim

        if t1 is None:
            raise ValueError(f"Invalid input t1 [{t1}]")
        else:
            self.lattice_t1 = self._add_lattice_vectors(t1)

        if t2 is None:
            raise ValueError(f"Invalid input t2 [{t2}]")
        else:
            self.lattice_t2 = self._add_lattice_vectors(t2)
            
        self.cell_type = self.get_cell_type()

        # Add materials to material_lib
        self._matlib = {}
        if materiallist is not None:
            self.add_materials(material_list=materiallist)

        # Add default material
        _mat_air = create_material(name='air')
        self.add_materials(material_list=[_mat_air])

        # build device
        self.vec_p = torch.linspace(-0.5, 0.5, rdim[0], dtype= self.tfloat, device=self.device)
        self.vec_q = torch.linspace(-0.5, 0.5, rdim[1], dtype= self.tfloat, device=self.device)

        [mesh_q, mesh_p] = torch.meshgrid(self.vec_q, self.vec_p, indexing='xy')

        self.XO = mesh_p * self.lattice_t1[0] +\
            mesh_q * self.lattice_t2[0]
        self.YO = mesh_p * self.lattice_t1[1] +\
            mesh_q * self.lattice_t2[1]
        
        # Initialize shape generator
        self.shapes = ShapeGenerator(self.XO, self.YO, tuple(self.rdim))

        # scaling factor of lengths
        self._lenunit = lengthunit.lower()
        self._len_scale = lengthunit_dict[self._lenunit]

        self.layer_manager = LayerManager(lattice_t1=self.lattice_t1,
                                          lattice_t2=self.lattice_t2,
                                          vec_p=self.vec_p,
                                          vec_q=self.vec_q)
        self.update_trn_material(trn_material='air')
        self.update_ref_material(ref_material='air')

    def get_circle_mask(self, center=(0, 0), radius=20):
        return self.shapes.generate_circle_mask(center=center, radius=radius)
    
    def get_rectangle_mask(self, bottom_left=(50, 50), top_right=(100, 100)):
        return self.shapes.generate_rectangle_mask(bottom_left=bottom_left, top_right=top_right)
    
    def get_polygon_mask(self, polygon_points, invert=False):
        return self.shapes.generate_polygon_mask(polygon_points, invert)
    
    def combine_masks(self, mask1, mask2, operation="union"):
        return self.shapes.combine_masks(mask1, mask2, operation)


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
                if imat.er is None:
                    imat.load_dispersive_er(
                        self._lam0, self._lenunit)

    @property
    def layers(self):
        """layers.

        Returns all created layers in the layer_manager instance.
        """
        return self.layer_manager.layers


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
        return lattice_vec.squeeze().to(self.device)

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
        # cell type
        print("------------------------------------")
        print(f"Cell Type: {self.cell_type}")
        # reflection layer
        print("------------------------------------")
        print("layer # Reflection")
        print(f"\tmaterial name: {self.layer_manager.ref_material_name}")
        print(f"\tpermittivity: {self.er1}")
        print(f"\tpermeability: {self.ur1}")

        # structure layers
        print("------------------------------------")
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

    def get_layout(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """get_layout.

        Returns the layout matrices of the cell.

        Args:

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        return self.XO, self.YO

    @property
    def er1(self):
        """er1.
        attribute.
        """
        return self._matlib[self.layer_manager.ref_material_name].er.to(self.tcomplex).detach().clone().to(self.device)

    @property
    def er2(self):
        """er2.
        attribute.
        """
        return self._matlib[self.layer_manager.trn_material_name].er.to(self.tcomplex).detach().clone().to(self.device)


    @property
    def ur1(self):
        """ur1.
        attribute.
        """
        return self._matlib[self.layer_manager.ref_material_name].ur.to(self.tcomplex).detach().clone().to(self.device)

    @property
    def ur2(self):
        """ur2.
        attribute.
        """
        return self._matlib[self.layer_manager.trn_material_name].ur.to(self.tcomplex).detach().clone().to(self.device)

    @property
    def lengthunit(self):
        """lengthunit.
        attribute.
        """
        return self._lenunit

    def get_cell_type(self):
        if self.lattice_t1[0] == 0 and self.lattice_t1[1] != 0 and self.lattice_t2[0] != 0 and self.lattice_t2[1] == 0:
            return CellType.Cartesian
        elif self.lattice_t1[0] != 0 and self.lattice_t1[1] == 0 and self.lattice_t2[0] == 0 and self.lattice_t2[1] != 0:
            return CellType.Cartesian
        else:
            return CellType.Other


