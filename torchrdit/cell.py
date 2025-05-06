"""Module for defining base class (Cell3D) of solver classes in electromagnetic simulations.

This module provides classes for creating and manipulating 3D unit cells for use in electromagnetic simulations with TorchRDIT. The key 
components include:

- CellType: Enumeration of supported cell geometry types
- Cell3D: Base class for defining 3D unit cells with material properties

These classes form the foundation for defining the geometric and material properties
of structures to be simulated with RCWA or R-DIT algorithms.

Examples:
```python
# Basic usage: The Cell3D class is not intended to be used directly. Instead, it serves as a
# parent class for solver implementations in solver.py (RCWASolver and RDITSolver).
```

Keywords:
    unit cell, geometry, shape generation, mask, material, layer, photonics
"""
from typing import Callable, Tuple, Union, List, Any

import numpy as np
import torch

from .utils import tensor_params_check, create_material
from .materials import MaterialClass
from .constants import lengthunit_dict
from .layers import LayerManager

# Function Type
FuncType = Callable[..., Any]

class CellType:
    """Enumeration of supported cell geometry types in TorchRDIT.
    
    This class defines the types of coordinate systems that can be used for
    the unit cell in electromagnetic simulations. The cell type affects how
    coordinates are interpreted and how Fourier transforms are computed.
    
    Attributes:
        Cartesian: Standard Cartesian coordinate system with rectangular grid.
            Used for rectangular lattices aligned with coordinate axes.
        Other: Alternative coordinate systems used for non-rectangular lattices
            or rotated structures.
    
    Examples:
    ```python
    # Check the cell type
    import torch
    from torchrdit.cell import Cell3D
    cell = Cell3D()
    print(cell.cell_type)
    # Cartesian
    # Cell type is automatically determined from lattice vectors
    cell = Cell3D(t1=torch.tensor([[1.0, 0.5]]), 
                  t2=torch.tensor([[0.0, 1.0]]))
    print(cell.cell_type)
    # Other
    ```
    
    Keywords:
        coordinate system, Cartesian, unit cell, lattice type, grid
    """
    Cartesian = 'Cartesian'
    Other = 'Other'

class Cell3D():
    """Base parent class for defining and structuring cells in electromagnetic solvers.
    
    The Cell3D class provides the foundational structure for defining and managing
    the geometric and material properties of a unit cell in TorchRDIT. It serves
    as the core building block for creating photonic structures to be simulated
    with electromagnetic solvers.
    
    Note:
        Cell3D is not intended to be used directly. Instead, it serves as a
        parent class for solver implementations in solver.py (RCWASolver and RDITSolver).
        Most users should create solver instances using the create_solver() function
        rather than instantiating Cell3D directly.
    
    Key capabilities:
    - Setting up computational grids in real and Fourier space
    - Managing material properties and multilayer stacks
    - Creating and manipulating binary shape masks for complex geometries
    - Controlling reference and transmission materials
    - Providing coordinate transformation utilities
    
    This class serves as a base for solver classes (FourierBaseSolver, RCWASolver, RDITSolver)
    and provides all the necessary functionality to define the structure being simulated,
    including material layers, shape patterns, and boundary conditions.
    
    Attributes:
        cell_type (CellType): Type of coordinate system (Cartesian or Other)
        tcomplex (torch.dtype): Complex data type used for calculations
        tfloat (torch.dtype): Float data type used for calculations
        tint (torch.dtype): Integer data type used for calculations
        nfloat (np.dtype): NumPy float data type for compatibility
        rdim (List[int]): Dimensions in real space [height, width]
        kdim (List[int]): Dimensions in k-space [kheight, kwidth]
        layer_manager (LayerManager): Manager for handling material layers
        
    Examples:
    ```python
    # Create a solver (recommended way) instead of using Cell3D directly
    import torch
    import numpy as np
    from torchrdit.solver import create_solver
    from torchrdit.utils import create_material
    from torchrdit.constants import Algorithm
    
    # Create an RCWA solver
    solver = create_solver(
        algorithm=Algorithm.RCWA,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5]
    )
    
    # Add materials
    silicon = create_material(name="silicon", permittivity=11.7)
    sio2 = create_material(name="sio2", permittivity=2.25)
    solver.add_materials([silicon, sio2])
    
    # Add layers
    solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
    solver.add_layer(material_name="sio2", thickness=torch.tensor(0.1))
    
    # Create a patterned layer
    solver.add_layer(material_name="silicon", thickness=torch.tensor(0.3),
                   is_homogeneous=False)
    circle_mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
    ```
    
    Keywords:
        base class, parent class, solver foundation, electromagnetic simulation, 
        photonics, layers, materials, computational grid, Fourier transform, 
        shape generation, inheritance
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
        
        Note:
            Users should generally not instantiate Cell3D directly, but rather
            create a solver instance using the create_solver() function.
        
        Args:
            lengthunit (str): Unit of length for all dimensions in the simulation.
                      Common values: 'um' (micrometers), 'nm' (nanometers).
                      Default is 'um'.
            rdim (List[int]): Dimensions of the real-space grid as [height, width].
                This determines the spatial resolution of the simulation.
                Default is [512, 512].
            kdim (List[int]): Dimensions in Fourier space as [kheight, kwidth].
                This determines the number of Fourier harmonics used in the simulation.
                Default is [3, 3].
            materiallist (List[MaterialClass]): List of material objects to be used in the simulation.
                       Each material should be an instance of MaterialClass.
                       Default is an empty list.
            t1 (torch.Tensor): First lattice vector defining the unit cell.
                Default is [[1.0, 0.0]] (unit vector in x-direction).
            t2 (torch.Tensor): Second lattice vector defining the unit cell.
                Default is [[0.0, 1.0]] (unit vector in y-direction).
            device (Union[str, torch.device]): The device to run computations on ('cpu' or 'cuda').
                  Default is 'cpu'.
                  
        Raises:
            ValueError: If any of the input parameters have invalid values or formats.
                      
        Examples:
        ```python
        # Instead of creating Cell3D directly, create a solver:
        from torchrdit.solver import create_solver
        from torchrdit.constants import Algorithm
        import torch
        # Create an RDIT solver
        rdit_solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[1024, 1024],
            kdim=[7, 7]
        )
        
        # Create an RCWA solver with non-rectangular lattice
        rcwa_solver = create_solver(
            algorithm=Algorithm.RCWA,
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.5, 0.866]]),  # 30-degree lattice
            rdim=[512, 512],
            kdim=[5, 5]
        )
        
        # Create a solver with GPU acceleration
        gpu_solver = create_solver(
            algorithm=Algorithm.RCWA,
            device="cuda"
        )
        ```
        
        Keywords:
            initialization, base class, parent class, computational grid, lattice vectors, 
            spatial resolution, Fourier harmonics
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

        # scaling factor of lengths
        self._lenunit = lengthunit.lower()
        self._len_scale = lengthunit_dict[self._lenunit]

        self.layer_manager = LayerManager(lattice_t1=self.lattice_t1,
                                          lattice_t2=self.lattice_t2,
                                          vec_p=self.vec_p,
                                          vec_q=self.vec_q)
        self.update_trn_material(trn_material='air')
        self.update_ref_material(ref_material='air')

    def get_shape_generator_params(self):
        """Get the parameters for the shape generator.
        
        Returns:
            dict: Dictionary containing the parameters for the shape generator.

        Examples:
        ```python
        from torchrdit.solver import create_solver
        from torchrdit.constants import Algorithm
        import torch
        from torchrdit.shapes import ShapeGenerator
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[1024, 1024],
            kdim=[7, 7]
        )
        params = solver.get_shape_generator_params()
        shape_gen = ShapeGenerator(**params)
        ```
        """
        return {"XO": self.XO, "YO": self.YO, "rdim": tuple(self.rdim),
                "lattice_t1": self.lattice_t1, "lattice_t2": self.lattice_t2,
                "tcomplex": self.tcomplex, "tfloat": self.tfloat, "tint": self.tint,
                "nfloat": self.nfloat}


    def add_materials(self, material_list: list = []):
        """Add materials to the solver's material library.
        
        This method adds one or more materials to the cell's internal material
        library, making them available for use in layers. Materials can be either
        non-dispersive (constant properties) or dispersive (wavelength-dependent).
        
        Args:
            material_list (list): List of MaterialClass instances to add to the material library.
                        Each material should have a unique name that will be used to
                        reference it when creating layers.
        
        Raises:
            ValueError: If any element in the list is not a MaterialClass instance or
                      if the input is not a list.
                        
        Examples:
        ```python
        from torchrdit.solver import create_solver
        from torchrdit.constants import Algorithm
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[1024, 1024],
            kdim=[7, 7]
        )
        # Create and add materials
        from torchrdit.utils import create_material
        silicon = create_material(name='silicon', permittivity=11.7)
        sio2 = create_material(name='sio2', permittivity=2.25)
        # Add multiple materials at once
        solver.add_materials([silicon, sio2])
        # Add a single material
        gold = create_material(name='gold', permittivity=complex(-10.0, 1.5))
        solver.add_materials([gold])
        ```
            
        Note:
            The 'air' material (permittivity=1.0) is automatically added to all
            Cell3D instances by default and does not need to be explicitly added.
        
        Keywords:
            materials, permittivity, dielectric properties, material library,
            optical materials, dispersive materials
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
        
        Args:
            material_name (Union[str, MaterialClass]): Name of the material for this layer,
                        or a MaterialClass instance. Must be a material that exists in
                        the material library or a new material to be added.
            thickness (torch.Tensor): Thickness of the layer as a torch.Tensor.
                     The units are determined by the lengthunit parameter of the solver.
            is_homogeneous (bool): Whether the layer has uniform material properties (True) or
                          is patterned with a spatial distribution (False).
                          Default is True.
            is_optimize (bool): Whether this layer's parameters (e.g., thickness) should be
                       included in optimization. Set to True if you plan to optimize
                       this layer's properties. Default is False.
                       
        Raises:
            RuntimeError: If the specified material does not exist in the material library.
            
        Examples:
        ```python
        # Create a cell and add materials
        from torchrdit.cell import Cell3D
        from torchrdit.utils import create_material
        import torch
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        sio2 = create_material(name='sio2', permittivity=2.25)
        cell.add_materials([silicon, sio2])
        
        # Add a homogeneous silicon layer with thickness 0.2 μm
        cell.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
        
        # Add a layer using a material object directly
        air = create_material(name='air2', permittivity=1.0)
        cell.add_layer(material_name=air, thickness=torch.tensor(0.1))
        
        # Add a patterned (non-homogeneous) layer
        cell.add_layer(
            material_name='silicon',
            thickness=torch.tensor(0.3),
            is_homogeneous=False
        )
        
        # Add a layer that will be optimized
        cell.add_layer(
            material_name='sio2',
            thickness=torch.tensor(0.15),
            is_optimize=True
        )
        ```
        
        Keywords:
            layer, material, thickness, homogeneous, patterned, photonic structure,
            multilayer, stack, optimization
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
        """Initialize the dispersive profile of materials.
        
        This internal method initializes wavelength-dependent (dispersive) material
        properties for all materials in the material library that have dispersive
        behavior. It loads the appropriate values for the current wavelength settings.
        
        Note:
            This method is called automatically when materials are added and should
            not typically be called directly by users.
        
        Keywords:
            dispersive materials, wavelength-dependent, initialization, internal method
        """
        for imat in self._matlib.values():
            if imat.isdispersive_er is True:
                if imat.er is None:
                    imat.load_dispersive_er(
                        self._lam0, self._lenunit)

    @property
    def layers(self):
        """Get all created layers in the layer structure.
        
        Returns:
            List: All created layers in the layer_manager instance.
            
        Examples:
        ```python
        from torchrdit.cell import Cell3D
        import torch
        # Access all layers
        cell = Cell3D()
        cell.add_layer(material_name='air', thickness=torch.tensor(0.2))
        cell.add_layer(material_name='air', thickness=torch.tensor(0.3))
        print(f"Number of layers: {len(cell.layers)}")
        # Access properties of specific layers
        for i, layer in enumerate(cell.layers):
            print(f"Layer {i}: {layer.material_name}, thickness={layer.thickness}")
        ```
        
        Keywords:
            layers, layer access, layer properties, layer stack
        """
        return self.layer_manager.layers


    @tensor_params_check()
    def _add_lattice_vectors(self, lattice_vec: torch.Tensor) -> torch.Tensor:
        """Process and validate lattice vectors.
        
        This internal method checks the input lattice vector parameters and
        converts them to the appropriate data type and device.
        
        Args:
            lattice_vec (torch.Tensor): Lattice vector to process
        
        Returns:
            torch.Tensor: Processed lattice vector with correct dtype and device
            
        Raises:
            TypeError: If the vector cannot be converted to the correct dtype
            
        Keywords:
            lattice vector, validation, internal method
        """
        if lattice_vec.dtype != self.tfloat:
            try:
                lattice_vec = lattice_vec.to(self.tfloat)
            except:
                terr_str = f"The element of the argument should be the [{self.tfloat}] type."
                raise TypeError(terr_str)
        return lattice_vec.squeeze().to(self.device)

    def update_trn_material(self, trn_material: Any) -> None:
        """Update the transmission layer material.
        
        This method sets or changes the material used for the transmission layer,
        which is the semi-infinite region above the layered structure. This material
        affects the boundary conditions and the calculation of transmission coefficients.
        
        Args:
            trn_material (Union[str, MaterialClass]): Name of the material to use for
                        the transmission layer, or a MaterialClass instance.
                        Must be a material that exists in the material library
                        or a new material to be added.
        
        Raises:
            RuntimeError: If the specified material does not exist in the material library
                        and is not a MaterialClass instance.
                        
        Examples:
        ```python
        from torchrdit.cell import Cell3D
        from torchrdit.utils import create_material
        # Set transmission material by name
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        cell.add_materials([silicon])
        cell.update_trn_material(trn_material='silicon')
        
        # Set transmission material by providing a material object
        water = create_material(name='water', permittivity=1.77)
        cell.update_trn_material(trn_material=water)
        ```
        
        Keywords:
            transmission layer, output medium, boundary condition, semi-infinite region
        """
        if isinstance(trn_material, MaterialClass):
            if trn_material.name not in self._matlib:
                self.add_materials([trn_material])
            trn_material = trn_material.name

        if isinstance(trn_material, str) and trn_material in self._matlib:
            self.layer_manager.update_trn_layer(material_name=trn_material,
                                                is_dispersive=self._matlib[trn_material].isdispersive_er)
        elif trn_material.lower() == 'air': 
            pass
        else:
            str_rterr = f"No materials named [{trn_material} exists in the material lib.]"
            raise RuntimeError(str_rterr)

    def update_ref_material(self, ref_material: Any) -> None:
        """Update the reflection layer material.
        
        This method sets or changes the material used for the reflection layer,
        which is the semi-infinite region below the layered structure. This material
        affects the boundary conditions and the calculation of reflection coefficients.
        
        Args:
            ref_material (Union[str, MaterialClass]): Name of the material to use for
                        the reflection layer, or a MaterialClass instance.
                        Must be a material that exists in the material library
                        or a new material to be added.
        
        Raises:
            RuntimeError: If the specified material does not exist in the material library
                        and is not a MaterialClass instance.
                        
        Examples:
        ```python
        from torchrdit.cell import Cell3D
        from torchrdit.utils import create_material
        # Set reflection material by name
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        cell.add_materials([silicon])
        cell.update_ref_material(ref_material='silicon')
        
        # Set reflection material by providing a material object
        metal = create_material(name='silver', permittivity=complex(-15.0, 1.0))
        cell.update_ref_material(ref_material=metal)
        ```
        
        Keywords:
            reflection layer, input medium, boundary condition, semi-infinite region
        """
        if isinstance(ref_material, MaterialClass):
            if ref_material.name not in self._matlib:
                self.add_materials([ref_material])
            ref_material = ref_material.name

        if isinstance(ref_material, str) and ref_material in self._matlib:
            self.layer_manager.update_ref_layer(material_name=ref_material,
                                                is_dispersive=self._matlib[ref_material].isdispersive_er)
        elif ref_material.lower() == 'air': 
            pass
        else:
            str_rterr = f"No materials named [{ref_material} exists in the material lib.]"
            raise RuntimeError(str_rterr)

    def get_layer_structure(self):
        """Print information about all layers in the structure.
        
        This method displays detailed information about the current layer structure,
        including the reflection layer, all intermediate layers, and the transmission
        layer. For each layer, it shows:
        - Material name
        - Thickness (for intermediate layers)
        - Permittivity and permeability
        - Whether the layer is dispersive, homogeneous, or to be optimized
        
        Examples:
        ```python
        from torchrdit.cell import Cell3D
        from torchrdit.utils import create_material
        import torch
        # Create a cell with multiple layers and display information
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        sio2 = create_material(name='sio2', permittivity=2.25)
        cell.add_materials([silicon, sio2])
        cell.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
        cell.add_layer(material_name='sio2', thickness=torch.tensor(0.1))
        cell.get_layer_structure()
        ```
        
        Keywords:
            layer structure, information display, debugging, layer properties
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
        """Get the coordinate grid tensors of the cell.
        
        This method returns the x and y coordinate tensors that define the real-space
        grid of the unit cell. These tensors can be used for visualization or for
        creating custom shape masks.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing (X, Y) coordinate tensors,
                                            each with shape (rdim[0], rdim[1]).
                                            
        Examples:
        ```python
        # Get coordinate grids and use them for visualization
        import matplotlib.pyplot as plt
        import torch
        from torchrdit.cell import Cell3D
        cell = Cell3D()
        X, Y = cell.get_layout()
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), torch.ones_like(X).cpu().numpy())
        plt.axis('equal')
        plt.title('Unit Cell Coordinate Grid')
        plt.colorbar(label='Grid Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        ```
        
        Keywords:
            coordinate grid, layout, visualization, real-space coordinates
        """
        return self.XO, self.YO

    @property
    def er1(self):
        """Get the permittivity of the reflection layer.
        
        Returns:
            torch.Tensor: Complex tensor containing the permittivity (εᵣ) of the 
                       reflection layer material.
                       
        Keywords:
            permittivity, reflection layer, material property, dielectric constant
        """
        return self._matlib[self.layer_manager.ref_material_name].er.to(self.tcomplex).detach().clone().to(self.device)

    @property
    def er2(self):
        """Get the permittivity of the transmission layer.
        
        Returns:
            torch.Tensor: Complex tensor containing the permittivity (εᵣ) of the 
                       transmission layer material.
                       
        Keywords:
            permittivity, transmission layer, material property, dielectric constant
        """
        return self._matlib[self.layer_manager.trn_material_name].er.to(self.tcomplex).detach().clone().to(self.device)


    @property
    def ur1(self):
        """Get the permeability of the reflection layer.
        
        Returns:
            torch.Tensor: Complex tensor containing the permeability (μᵣ) of the 
                       reflection layer material.
                       
        Keywords:
            permeability, reflection layer, material property, magnetic property
        """
        return self._matlib[self.layer_manager.ref_material_name].ur.to(self.tcomplex).detach().clone().to(self.device)

    @property
    def ur2(self):
        """Get the permeability of the transmission layer.
        
        Returns:
            torch.Tensor: Complex tensor containing the permeability (μᵣ) of the 
                       transmission layer material.
                       
        Keywords:
            permeability, transmission layer, material property, magnetic property
        """
        return self._matlib[self.layer_manager.trn_material_name].ur.to(self.tcomplex).detach().clone().to(self.device)

    @property
    def lengthunit(self):
        """Get the length unit used in the simulation.
        
        Returns:
            str: The unit of length used in the simulation (e.g., 'um', 'nm').
            
        Keywords:
            length unit, dimension, measurement unit
        """
        return self._lenunit

    def get_cell_type(self):
        """Determine the type of cell based on lattice vectors.
        
        This method analyzes the lattice vectors and determines whether the cell
        is a standard Cartesian cell (with vectors aligned to the coordinate axes)
        or a more general cell type.
        
        Returns:
            CellType: The type of cell (CellType.Cartesian or CellType.Other).
            
        Examples:
        ```python
        # Create cells with different lattice vectors and check their types
        import torch
        from torchrdit.cell import Cell3D
        cell1 = Cell3D(t1=torch.tensor([[1.0, 0.0]]), t2=torch.tensor([[0.0, 1.0]]))
        print(cell1.get_cell_type())  # Cartesian
        
        cell2 = Cell3D(t1=torch.tensor([[1.0, 0.2]]), t2=torch.tensor([[0.0, 1.0]]))
        print(cell2.get_cell_type())  # Other
        ```
        
        Keywords:
            cell type, lattice vectors, coordinate system, Cartesian
        """
        if self.lattice_t1[0] == 0 and self.lattice_t1[1] != 0 and self.lattice_t2[0] != 0 and self.lattice_t2[1] == 0:
            return CellType.Cartesian
        elif self.lattice_t1[0] != 0 and self.lattice_t1[1] == 0 and self.lattice_t2[0] == 0 and self.lattice_t2[1] != 0:
            return CellType.Cartesian
        else:
            return CellType.Other


