""" This file defines classes for models to be simulated. """
from typing import Callable, Tuple, Union, List, Any

import numpy as np
import torch

from .utils import tensor_params_check, create_material
from .materials import MaterialClass
from .constants import lengthunit_dict
from .layers import LayerManager

from matplotlib.path import Path

# Function Type
FuncType = Callable[..., Any]

class CellType:
    """Enumeration of supported cell geometry types in TorchRDIT.
    
    This class defines the types of coordinate systems that can be used for
    the unit cell in electromagnetic simulations:
    
    - Cartesian: Standard Cartesian coordinate system with rectangular grid
    - Other: Alternative coordinate systems (e.g., for non-rectangular lattices)
    
    These cell types affect how coordinates are interpreted and how the
    Fourier transforms are computed in the simulation.
    """
    Cartesian = 'Cartesian'
    Other = 'Other'

class ShapeGenerator:
    """Class to generate binary shape masks for photonic structures.
    
    This class provides methods to create binary masks representing various geometric 
    shapes (circles, rectangles, polygons) that can be used to define material 
    distributions in photonic structures. These masks are used in the 
    electromagnetic solver to specify regions with different material properties.
    
    The coordinates for these shapes are defined in a real-space coordinate system,
    which is mapped to the computational grid defined by rdim.
    """
    def __init__(self, XO: torch.Tensor, YO: torch.Tensor, rdim: Tuple[int, int]):
        """Initialize a shape generator with coordinate grids.
        
        Args:
            XO: Tensor containing the x-coordinates of each point in the grid.
               This should be a 2D tensor with the same shape as the computational grid.
            YO: Tensor containing the y-coordinates of each point in the grid.
               This should be a 2D tensor with the same shape as the computational grid.
            rdim: Dimensions of the real-space grid as (height, width).
               This determines the resolution of the generated masks.
               
        Note:
            The XO and YO tensors typically come from a meshgrid operation and
            represent the real-space coordinates corresponding to each grid point.
        """
        assert isinstance(XO, torch.Tensor) and isinstance(YO, torch.Tensor), "XO and YO must be torch.Tensor"
        self.rdim = rdim
        
        self.X_real = XO
        self.Y_real = YO
    
    def generate_circle_mask(self, center=None, radius=0.1):
        """Generate a binary mask for a circle.
        
        Creates a binary mask (0s and 1s) where points inside the specified circle
        are set to 1 and points outside are set to 0. The circle is defined by its
        center and radius in the real coordinate system.
        
        Args:
            center: Tuple of (x, y) real coordinates of the circle center.
                   If None, the center of the computational domain is used.
            radius: Radius of the circle in real units.
                   Default is 0.1.
                   
        Returns:
            torch.Tensor: Binary mask with 1s inside the circle and 0s outside.
                       Has shape self.rdim and device matching self.X_real.
                       
        Example:
            ```python
            # Create a circle in the center with radius 0.2
            generator = ShapeGenerator(X_grid, Y_grid, (512, 512))
            circle_mask = generator.generate_circle_mask(center=(0, 0), radius=0.2)
            ```
        """
        mask = torch.zeros(self.rdim, dtype=torch.uint8, device=self.X_real.device)
        if center is None:
            center = (self.X_real.mean().item(), self.Y_real.mean().item())

        # Mask condition based on real coordinates
        distance = torch.sqrt((self.X_real - center[0])**2 + (self.Y_real - center[1])**2)
        mask[distance <= radius] = 1

        return mask

    
    def generate_rectangle_mask(self, bottom_left=None, top_right=None):
        """Generate a binary mask for a rectangle.
        
        Creates a binary mask (0s and 1s) where points inside the specified rectangle
        are set to 1 and points outside are set to 0. The rectangle is defined by its
        bottom-left and top-right corners in the real coordinate system.
        
        Args:
            bottom_left: Tuple of (x, y) real coordinates of the bottom-left corner.
                       If None, the minimum coordinates in the domain are used.
            top_right: Tuple of (x, y) real coordinates of the top-right corner.
                     If None, the maximum coordinates in the domain are used.
                     
        Returns:
            torch.Tensor: Binary mask with 1s inside the rectangle and 0s outside.
                       Has shape self.rdim and dtype torch.uint8.
                       
        Example:
            ```python
            # Create a rectangle from (-0.5, -0.5) to (0.5, 0.5)
            generator = ShapeGenerator(X_grid, Y_grid, (512, 512))
            rect_mask = generator.generate_rectangle_mask(
                bottom_left=(-0.5, -0.5), 
                top_right=(0.5, 0.5)
            )
            ```
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
        """Generate a binary mask for an arbitrary polygon.
        
        Creates a binary mask (0s and 1s) where points inside the specified polygon
        are set to 1 and points outside are set to 0. The polygon is defined by its
        vertices in the real coordinate system.
        
        This method uses matplotlib's Path for efficient point-in-polygon testing,
        making it suitable for complex polygons with many vertices.
        
        Args:
            polygon_points: List of (x, y) tuples representing the vertices of the polygon
                         in real coordinates. The vertices should be ordered to form a 
                         valid polygon (either clockwise or counterclockwise).
            invert: If True, inverts the mask so that points outside the polygon are 1
                  and points inside are 0. Default is False.
                     
        Returns:
            torch.Tensor: Binary mask with 1s inside the polygon and 0s outside
                       (or the inverse if invert=True). Has shape self.rdim and 
                       dtype torch.uint8.
                       
        Example:
            ```python
            # Create a triangle mask
            generator = ShapeGenerator(X_grid, Y_grid, (512, 512))
            triangle_points = [(0, 0), (0.5, 0.5), (0, 0.5)]
            triangle_mask = generator.generate_polygon_mask(triangle_points)
            ```
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
        """Combine two binary masks using a specified boolean operation.
        
        This method allows for combining multiple shape masks to create more complex
        geometries. The available operations are:
        
        - 'union': Returns points that are in either mask (OR operation)
        - 'intersection': Returns points that are in both masks (AND operation)
        - 'difference': Returns points that are in either mask but not both (XOR operation)
        - 'subtract': Returns points that are in mask1 but not in mask2
        
        Args:
            mask1: First binary mask (torch.Tensor of dtype torch.uint8)
            mask2: Second binary mask (torch.Tensor of dtype torch.uint8)
            operation: String specifying the boolean operation to perform.
                     Must be one of 'union', 'intersection', 'difference', or 'subtract'.
                     Default is 'union'.
                     
        Returns:
            torch.Tensor: The combined binary mask with the same shape and dtype as the inputs.
            
        Raises:
            ValueError: If an invalid operation is specified.
            
        Example:
            ```python
            # Create two circle masks and combine them
            generator = ShapeGenerator(X_grid, Y_grid, (512, 512))
            circle1 = generator.generate_circle_mask(center=(-0.2, 0), radius=0.3)
            circle2 = generator.generate_circle_mask(center=(0.2, 0), radius=0.3)
            
            # Create a mask resembling a figure-8
            figure8 = generator.combine_masks(circle1, circle2, operation="union")
            
            # Create a lens shape (only where the circles overlap)
            lens = generator.combine_masks(circle1, circle2, operation="intersection")
            ```
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
        """Determine if a point is inside a polygon using ray-casting.
        
        This is a helper method that implements the ray-casting algorithm to determine
        whether a given point is inside a polygon. A ray is cast from the point in
        any direction, and the number of intersections with the polygon edges is counted.
        If the count is odd, the point is inside; if even, the point is outside.
        
        Note:
            This is a private method used internally by the polygon mask generation.
            For most cases, the generate_polygon_mask method should be used instead,
            as it is more efficient for generating masks for entire grids.
        
        Args:
            point: Tuple of (x, y) coordinates of the point to test
            polygon: List or array of (x, y) coordinates of the polygon vertices
                   The vertices should form a closed polygon
                   
        Returns:
            bool: True if the point is inside the polygon, False otherwise
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
    """Base class for defining 3D unit cells in electromagnetic simulations.
    
    The Cell3D class provides the foundational structure for defining and managing
    the geometric and material properties of a unit cell in TorchRDIT. It handles:
    
    - Setting up the computational grid in real and Fourier space
    - Managing material properties and layers
    - Creating and manipulating binary shape masks for complex geometries
    - Tracking reference and transmission materials
    - Providing coordinate transformation utilities
    
    This class serves as a base for the FourierBaseSolver class and provides all the
    necessary functionality to define the structure being simulated, including
    material layers, shapes, and boundary conditions.
    
    Material properties can be specified either as static values or as wavelength-dependent
    (dispersive) models, supporting both simple and complex electromagnetic simulations.
    
    Attributes:
        cell_type: Type of coordinate system (Cartesian or Other)
        tcomplex, tfloat, tint, nfloat: Data type definitions for consistency
        layer_manager: Manager for handling the layers in the structure
    """

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
        """Initialize a Cell3D object with geometric and material properties.
        
        This constructor sets up the computational grid in both real and Fourier space,
        initializes materials, and creates a coordinate system based on the
        provided lattice vectors. It also initializes the layer manager for handling
        material layers in the structure.
        
        Args:
            lengthunit: Unit of length for all dimensions in the simulation.
                      Common values: 'um' (micrometers), 'nm' (nanometers).
                      Default is 'um'.
            rdim: Dimensions of the real-space grid as [height, width].
                This determines the spatial resolution of the simulation.
                Default is [512, 512].
            kdim: Dimensions in Fourier space as [kheight, kwidth].
                This determines the number of Fourier harmonics used in the simulation.
                Default is [3, 3].
            materiallist: List of material objects to be used in the simulation.
                       Each material should be an instance of MaterialClass.
                       Default is an empty list.
            t1: First lattice vector defining the unit cell.
                Default is [[1.0, 0.0]] (unit vector in x-direction).
            t2: Second lattice vector defining the unit cell.
                Default is [[0.0, 1.0]] (unit vector in y-direction).
            device: The device to run computations on ('cpu' or 'cuda').
                  Default is 'cpu'.
                  
        Raises:
            ValueError: If any of the input parameters have invalid values or formats.
                      
        Note:
            - The default configuration creates a square unit cell with dimensions 1×1.
            - Air is automatically added as a default material if not included in materiallist.
            - The real space coordinates range from -0.5 to 0.5 (in cell units) and
              are then transformed using the lattice vectors.
        """
        self.device = device

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
        """Generate a binary mask for a circle.
        
        Convenience method that wraps ShapeGenerator.generate_circle_mask.
        See that method for detailed documentation.
        
        Args:
            center: Tuple of (x, y) coordinates for the circle center.
                  Default is (0, 0).
            radius: Radius of the circle in real units.
                  Default is 20.
                  
        Returns:
            torch.Tensor: Binary mask with 1s inside the circle and 0s outside.
        """
        return self.shapes.generate_circle_mask(center=center, radius=radius)
    
    def get_rectangle_mask(self, bottom_left=(50, 50), top_right=(100, 100)):
        """Generate a binary mask for a rectangle.
        
        Convenience method that wraps ShapeGenerator.generate_rectangle_mask.
        See that method for detailed documentation.
        
        Args:
            bottom_left: Tuple of (x, y) coordinates for the bottom-left corner.
                       Default is (50, 50).
            top_right: Tuple of (x, y) coordinates for the top-right corner.
                     Default is (100, 100).
                     
        Returns:
            torch.Tensor: Binary mask with 1s inside the rectangle and 0s outside.
        """
        return self.shapes.generate_rectangle_mask(bottom_left=bottom_left, top_right=top_right)
    
    def get_polygon_mask(self, polygon_points, invert=False):
        """Generate a binary mask for a polygon.
        
        Convenience method that wraps ShapeGenerator.generate_polygon_mask.
        See that method for detailed documentation.
        
        Args:
            polygon_points: List of (x, y) tuples representing the vertices of the polygon.
            invert: If True, inverts the mask (1s outside, 0s inside).
                  Default is False.
                  
        Returns:
            torch.Tensor: Binary mask with 1s inside the polygon and 0s outside
                       (or the inverse if invert=True).
        """
        return self.shapes.generate_polygon_mask(polygon_points, invert)
    
    def combine_masks(self, mask1, mask2, operation="union"):
        """Combine two binary masks using a boolean operation.
        
        Convenience method that wraps ShapeGenerator.combine_masks.
        See that method for detailed documentation.
        
        Args:
            mask1: First binary mask.
            mask2: Second binary mask.
            operation: String specifying the boolean operation ('union', 'intersection',
                     'difference', or 'subtract'). Default is 'union'.
                     
        Returns:
            torch.Tensor: The combined binary mask.
        """
        return self.shapes.combine_masks(mask1, mask2, operation)

    def add_materials(self, material_list: list = []):
        """Add materials to the solver's material library.
        
        This method adds one or more materials to the solver's internal material
        library, making them available for use in layers. Materials can be either
        non-dispersive (constant properties) or dispersive (wavelength-dependent).
        
        Materials must be instances of the MaterialClass, which can be created
        using the create_material utility function.
        
        Args:
            material_list: List of MaterialClass instances to add to the material library.
                        Each material should have a unique name that will be used to
                        reference it when creating layers.
                        
        Example:
            ```python
            # Create and add materials
            silicon = create_material(name='silicon', permittivity=11.7)
            sio2 = create_material(name='sio2', permittivity=2.25)
            
            # Add materials to the solver
            solver.add_materials([silicon, sio2])
            
            # Now these materials can be used in layers
            solver.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
            ```
            
        Note:
            The 'air' material (permittivity=1.0) is automatically added to all
            solvers by default and does not need to be explicitly added.
        """
        if isinstance(material_list, list):
            for imat in material_list:
                if isinstance(imat, MaterialClass):
                    self._matlib[imat.name] = imat
                else:
                    verr_str = "The element of the argument should be the [MaterialClass] type."
                    raise ValueError(verr_str)
        else:
            verr_str = "Input argument should be a list."
            raise ValueError(verr_str)

        self._init_dispersive_materials()

    def add_layer(self,
                  material_name: Any,
                  thickness: torch.Tensor,
                  is_homogeneous: bool = True,
                  is_optimize: bool = False):
        """Add a new material layer to the structure.
        
        This method adds a layer with specified material and thickness to the
        simulation structure. Layers are stacked in the order they are added,
        with the first layer added being at the bottom of the stack (closest to
        the reference region) and subsequent layers building upward.
        
        The layer can be either homogeneous (uniform material throughout) or
        non-homogeneous (patterned, with material distribution defined by a mask).
        For non-homogeneous layers, you need to call update_er_with_mask() after
        adding the layer to define the material distribution.
        
        Args:
            material_name: Name of the material for this layer, or a MaterialClass instance.
                        Must be a material that exists in the material library.
            thickness: Thickness of the layer as a torch.Tensor.
                     The units are determined by the lengthunit parameter of the solver.
            is_homogeneous: Whether the layer has uniform material properties (True) or
                          is patterned with a spatial distribution (False).
                          Default is True.
            is_optimize: Whether this layer's parameters (e.g., thickness) should be
                       included in optimization. Set to True if you plan to optimize
                       this layer's properties. Default is False.
                       
        Raises:
            RuntimeError: If the specified material does not exist in the material library.
            
        Example:
            ```python
            # Add a homogeneous silicon layer with thickness 0.2 μm
            solver.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
            
            # Add a patterned layer with silicon as the base material
            solver.add_layer(material_name='silicon', thickness=torch.tensor(0.1), 
                           is_homogeneous=False)
            
            # Define the pattern using a mask (e.g., a circle of silicon in air)
            mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
            solver.update_er_with_mask(mask=mask, layer_index=1, bg_material='air')
            ```
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


