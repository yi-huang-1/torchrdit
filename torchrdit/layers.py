"""Module for defining and managing material layers in TorchRDIT electromagnetic simulations.

This module provides classes to define, create, and manage the material layers that
make up the structure being simulated in TorchRDIT. It implements a builder pattern
to create different types of layers (homogeneous and patterned/grating) and a manager
class to handle collections of layers.

The layer system is a core component of the TorchRDIT electromagnetic solver, 
representing the physical structure of the device being simulated. Each layer 
has properties such as thickness, material composition, and can be either 
homogeneous (uniform material) or patterned (spatially varying material distribution).

Classes:
    Layer: Abstract base class for all layer types.
    HomogeneousLayer: Layer with uniform material properties.
    GratingLayer: Layer with spatially varying material properties.
    LayerBuilder: Abstract base class for layer builders.
    HomogeneousLayerBuilder: Builder for homogeneous layers.
    GratingLayerBuilder: Builder for grating/patterned layers.
    LayerDirector: Coordinates the layer building process.
    LayerManager: Manages collections of layers and layer operations.

Note:
    Users typically don't interact with these classes directly, but rather through
    the solver classes defined in solver.py. The solver provides more user-friendly
    methods for adding and manipulating layers.

Examples:
```python
# Users typically add layers through solver interfaces:
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
import torch
# Create a solver
solver = create_solver(algorithm=Algorithm.RCWA)
# Add a material to the solver
silicon = create_material(name="silicon", permittivity=11.7)
solver.add_materials([silicon])
# Add a homogeneous layer
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
# Add a patterned layer
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.3), 
               is_homogeneous=False)
# Pattern the layer with a circle
circle_mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
solver.update_er_with_mask(circle_mask, layer_index=1)
```
    
Keywords:
    layers, material layers, layer management, homogeneous layers, grating layers,
    patterned layers, layer stack, photonic structure, simulation structure,
    builder pattern, layer properties, electromagnetic simulation
"""
from abc import ABCMeta, abstractmethod
from .utils import tensor_params_check

from torch.fft import fft2, fftshift
import torch
import numpy as np

class Layer(metaclass=ABCMeta):
    """Abstract base class for all layer types in TorchRDIT electromagnetic simulations.
    
    This class defines the interface for all layer types in the simulation. It is an
    abstract base class that cannot be instantiated directly. Instead, concrete
    layer classes (HomogeneousLayer, GratingLayer) should be used, typically through
    the solver's add_layer method rather than direct instantiation.
    
    Each Layer represents a physical layer in the simulated structure with properties
    such as thickness, material composition, and homogeneity. Layers are used to model
    the electromagnetic response of different regions in the simulated structure.
    
    Note:
        This class is not intended for direct use by end users. Users should create
        and manipulate layers through the solver interface (create_solver, add_layer, etc.)
        rather than instantiating Layer subclasses directly.
    
    Attributes:
        thickness (float): Thickness of the layer in the simulation's length units.
        material_name (str): Name of the material used in this layer.
        is_homogeneous (bool): Whether the layer has uniform material properties.
        is_dispersive (bool): Whether the layer's material has frequency-dependent properties.
        is_optimize (bool): Whether the layer's parameters are subject to optimization.
        is_solved (bool): Whether the electromagnetic response of this layer has been solved.
        ermat (torch.Tensor, optional): Permittivity distribution matrix.
        urmat (torch.Tensor, optional): Permeability distribution matrix.
        kermat (torch.Tensor, optional): Fourier-transformed permittivity matrix.
        kurmat (torch.Tensor, optional): Fourier-transformed permeability matrix.
    
    Keywords:
        abstract class, layer interface, layer properties, material layer,
        electromagnetic simulation, photonic structure
    """

    @abstractmethod
    def __init__(self, thickness: float = 0.0, material_name: str = '', is_optimize: bool = False, **kwargs) -> None:
        """Initialize a Layer instance with given properties.
        
        This abstract method initializes basic properties common to all layer types.
        It must be implemented by all concrete Layer subclasses.

        Args:
            thickness (float): Thickness of the layer in the simulation's length units.
                       Default is 0.0.
            material_name (str): Name of the material used for this layer.
                          The material must exist in the solver's material library.
                          Default is an empty string.
            is_optimize (bool): Flag indicating if the layer's parameters 
                        (e.g., thickness) should be optimized during parameter sweeps.
                        Default is False.
            **kwargs: Additional keyword arguments specific to subclasses.
        
        Note:
            This method is not intended to be called directly by users. Layers
            should be created through the solver interface.
        
        Keywords:
            initialization, layer creation, layer properties, material assignment
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
        """Get the thickness of the layer.
        
        Returns:
            float: The thickness of the layer in the simulation's length units.
        
        Keywords:
            thickness, layer property, geometric parameter
        """
        return self._thickness

    @thickness.setter
    def thickness(self, thickness: float):
        """Set the thickness of the layer.

        Args:
            thickness (float): The new thickness value for the layer 
                       in the simulation's length units.
        
        Keywords:
            thickness, layer property, update parameter
        """
        self._thickness = thickness

    @property
    def material_name(self) -> str:
        """Get the name of the material assigned to this layer.
        
        Returns:
            str: The name of the material used in this layer.
        
        Keywords:
            material, layer property, material assignment
        """
        return self._material_name

    @material_name.setter
    def material_name(self, material_name: str):
        """Set the material name for this layer.

        Args:
            material_name (str): The name of the material to assign to this layer.
                          The material must exist in the solver's material library.
        
        Keywords:
            material, layer property, update material
        """
        self._material_name = material_name

    @property
    def is_homogeneous(self) -> bool:
        """Check if the layer has uniform material properties.
        
        Returns:
            bool: True if the layer has uniform material properties,
                 False if it has spatially varying properties.
        
        Keywords:
            homogeneous, uniform, layer property, material distribution
        """
        return self._is_homogeneous

    @property
    def is_dispersive(self) -> bool:
        """Check if the layer's material has frequency-dependent properties.
        
        Returns:
            bool: True if the material properties depend on frequency/wavelength,
                 False otherwise.
        
        Keywords:
            dispersive, frequency-dependent, wavelength-dependent, material property
        """
        return self._is_dispersive

    @is_dispersive.setter
    def is_dispersive(self, is_dispersive: bool):
        """Set the dispersive flag for the layer's material.

        Args:
            is_dispersive (bool): True if the material properties should be
                          treated as frequency/wavelength-dependent, False otherwise.
        
        Keywords:
            dispersive, frequency-dependent, wavelength-dependent, material property
        """
        self._is_dispersive = is_dispersive

    @property
    def is_optimize(self) -> bool:
        """Check if the layer's parameters are subject to optimization.
        
        Returns:
            bool: True if the layer's parameters should be included in optimization
                 procedures, False otherwise.
        
        Keywords:
            optimization, parameter sweep, inverse design
        """
        return self._is_optimize

    @is_optimize.setter
    def is_optimize(self, is_optimize: bool):
        """Set the optimization flag for the layer.

        Args:
            is_optimize (bool): True if the layer's parameters should be included
                        in optimization procedures, False otherwise.
        
        Keywords:
            optimization, parameter sweep, inverse design
        """
        self._is_optimize = is_optimize

    @property
    def is_solved(self) -> bool:
        """Check if the electromagnetic response of this layer has been solved.
        
        Returns:
            bool: True if the layer's electromagnetic response has been calculated,
                 False otherwise.
        
        Keywords:
            solved, calculated, electromagnetic response
        """
        return self._is_solved

    @is_solved.setter
    def is_solved(self, is_solved: bool):
        """Set the solved flag for the layer.

        Args:
            is_solved (bool): True if the layer's electromagnetic response 
                      has been calculated, False otherwise.
        
        Keywords:
            solved, calculated, electromagnetic response
        """
        self._is_solved = is_solved

class LayerBuilder(metaclass=ABCMeta):
    """Abstract base class for builders that create layer instances.
    
    This class implements the builder pattern for creating different types of layers
    in TorchRDIT. It provides the interface for concrete builders that create
    specific types of layers (homogeneous or grating/patterned).
    
    The builder pattern separates the construction of complex objects from their
    representation, allowing the same construction process to create different
    representations.
    
    Note:
        This class is primarily used internally by TorchRDIT and is not intended
        for direct use by end users. Users should create layers through the 
        solver interface.
    
    Attributes:
        layer (Layer, optional): The layer instance being constructed.
        
    Keywords:
        builder pattern, layer creation, design pattern, factory, abstract class
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the LayerBuilder instance.
        
        Sets the layer attribute to None, to be populated by create_layer.
        """
        self.layer = None

    @abstractmethod
    def create_layer(self):
        """Create a new layer instance of the appropriate type.
        
        This abstract method must be implemented by subclasses to instantiate
        the specific type of layer they build.
        
        Keywords:
            layer creation, instantiation, builder pattern
        """
        pass

    def update_thickness(self, thickness):
        """Update the thickness of the layer being built.

        Args:
            thickness (float): The thickness value to set for the layer.
            
        Keywords:
            thickness, layer property, parameter update
        """
        self.layer.thickness = thickness

    def update_material_name(self, material_name):
        """Set the material for the layer being built.

        Args:
            material_name (str): The name of the material to use for the layer.
                          Must be a material that exists in the solver's material library.
                          
        Keywords:
            material, layer property, material assignment
        """
        self.layer.material_name = material_name

    def set_optimize(self, is_optimize):
        """Set whether the layer's parameters should be optimized.

        Args:
            is_optimize (bool): True if the layer should be included in 
                        optimization procedures, False otherwise.
                        
        Keywords:
            optimization, parameter sweep, design optimization
        """
        self.layer.is_optimize = is_optimize

    # read-only property
    def get_layer_instance(self):
        """Get the constructed layer instance.
        
        Returns:
            Layer: The fully configured layer instance.
            
        Keywords:
            layer instance, builder result, object creation
        """
        return self.layer

    def set_dispersive(self, is_dispersive):
        """Set whether the layer's material has frequency-dependent properties.

        Args:
            is_dispersive (bool): True if the material should be treated as
                          having frequency/wavelength-dependent properties, False otherwise.
                          
        Keywords:
            dispersive, frequency-dependent, wavelength-dependent, material property
        """
        self.layer.is_dispersive = is_dispersive

class HomogeneousLayer(Layer):
    """Layer with uniform material properties throughout its volume.
    
    This class represents a layer with homogeneous (spatially uniform) material
    properties. The permittivity and permeability are constant throughout the
    entire layer, making it suitable for representing bulk materials, thin films,
    or other uniform regions.
    
    Homogeneous layers are computationally efficient as they don't require
    spatial discretization of material properties.
    
    Note:
        Users typically do not instantiate this class directly, but rather
        create homogeneous layers through the solver's add_layer method with
        is_homogeneous=True (the default).
    
    Attributes:
        Inherits all attributes from the Layer base class.
        
    Examples:
    ```python
    # Create a homogeneous layer through the solver interface
    from torchrdit.solver import create_solver
    from torchrdit.utils import create_material
    import torch
    solver = create_solver()
    silicon = create_material(name="silicon", permittivity=11.7)
    solver.add_materials([silicon])
    # Add a homogeneous layer with thickness 0.2 μm
    solver.add_layer(
        material_name="silicon", 
        thickness=torch.tensor(0.2)
    )
    ```
    
    Keywords:
        homogeneous layer, uniform material, bulk material, thin film, constant properties
    """

    def __init__(self, thickness: float = 0.0, material_name: str = '', is_optimize: bool = False, **kwargs) -> None:
        """Initialize a HomogeneousLayer instance.

        Args:
            thickness (float): Thickness of the layer in the simulation's length units.
                       Default is 0.0.
            material_name (str): Name of the material to use for this layer.
                          Must be a material that exists in the solver's material library.
                          Default is an empty string.
            is_optimize (bool): Whether the layer's parameters should be included in
                        optimization procedures. Default is False.
            **kwargs: Additional keyword arguments passed to the parent class.
        
        Keywords:
            initialize homogeneous layer, create uniform layer
        """
        super().__init__(thickness, material_name, is_optimize, **kwargs)

    def __str__(self) -> str:
        """Get a string representation of the homogeneous layer.
        
        Returns:
            str: A string describing the homogeneous layer instance.
        """
        return f"HomogeneousLayer(thickness={self._thickness}, material_name={self._material_name})"

class GratingLayer(Layer):
    """Layer with spatially varying material properties.
    
    This class represents a layer with inhomogeneous (spatially varying) material
    properties, such as photonic crystals, metamaterials, diffraction gratings,
    or other patterned structures. The permittivity and permeability can vary
    throughout the layer based on binary masks or other patterns.
    
    Grating layers require spatial discretization of material properties and are
    generally more computationally intensive than homogeneous layers.
    
    Note:
        Users typically do not instantiate this class directly, but rather
        create grating layers through the solver's add_layer method with
        is_homogeneous=False.
    
    Attributes:
        Inherits all attributes from the Layer base class.
        The attribute is_homogeneous is always False for grating layers.
        
    Examples:
    ```python
    # Create a patterned (grating) layer through the solver interface
    from torchrdit.solver import create_solver
    from torchrdit.utils import create_material
    import torch
    solver = create_solver()
    silicon = create_material(name="silicon", permittivity=11.7)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([silicon, air])
    
    # Add a patterned layer with thickness 0.5 μm
    solver.add_layer(
        material_name="silicon", 
        thickness=torch.tensor(0.5),
        is_homogeneous=False
    )
    
    # Create a circular pattern in the layer
    circle_mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
    solver.update_er_with_mask(
        mask=circle_mask, 
        layer_index=0, 
        bg_material="air"
    )
    ```
    
    Keywords:
        grating layer, patterned layer, inhomogeneous, photonic crystal, metamaterial,
        spatial variation, material distribution, photonic device
    """

    def __init__(self, thickness: float = 0.0, material_name: str = '', is_optimize: bool = False, **kwargs) -> None:
        """Initialize a GratingLayer instance.

        Args:
            thickness (float): Thickness of the layer in the simulation's length units.
                       Default is 0.0.
            material_name (str): Name of the material to use as the foreground material.
                          Must be a material that exists in the solver's material library.
                          Default is an empty string.
            is_optimize (bool): Whether the layer's parameters should be included in
                        optimization procedures. Default is False.
            **kwargs: Additional keyword arguments passed to the parent class.
        
        Keywords:
            initialize grating layer, create patterned layer
        """
        super().__init__(thickness, material_name, is_optimize, **kwargs)
        self._is_homogeneous = False

    def __str__(self) -> str:
        """Get a string representation of the grating layer.
        
        Returns:
            str: A string describing the grating layer instance.
        """
        return f"GratingLayer(thickness={self._thickness}, material_name={self._material_name})"

class HomogeneousLayerBuilder(LayerBuilder):
    """Builder for creating HomogeneousLayer instances.
    
    This class is a concrete implementation of the LayerBuilder abstract class
    that creates and configures HomogeneousLayer instances. It follows the 
    builder pattern to separate the construction details from the representation.
    
    HomogeneousLayerBuilder creates layers with uniform material properties
    throughout their volume, suitable for bulk materials, thin films, or other
    uniform regions.
    
    Note:
        This class is used internally by the LayerDirector and is not typically
        accessed directly by users. Users should create layers through the solver
        interface.
    
    Keywords:
        builder pattern, homogeneous layer, uniform material, layer creation
    """

    def __init__(self) -> None:
        """Initialize a HomogeneousLayerBuilder instance.
        
        Initializes the builder with no layer instance created yet.
        """
        super().__init__()

    def create_layer(self):
        """Create a new empty HomogeneousLayer instance.
        
        This method instantiates a new HomogeneousLayer with default values.
        The properties of the layer are set through subsequent method calls
        to update_thickness, update_material_name, etc.
        
        Keywords:
            create layer, instantiate layer, homogeneous layer
        """
        self.layer = HomogeneousLayer()

class GratingLayerBuilder(HomogeneousLayerBuilder):
    """Builder for creating GratingLayer instances.
    
    This class is a concrete implementation of the LayerBuilder abstract class
    that creates and configures GratingLayer instances. It follows the 
    builder pattern to separate the construction details from the representation.
    
    GratingLayerBuilder creates layers with spatially varying material properties,
    suitable for photonic crystals, metamaterials, diffraction gratings, or other
    patterned structures.
    
    Note:
        This class is used internally by the LayerDirector and is not typically
        accessed directly by users. Users should create layers through the solver
        interface with is_homogeneous=False.
    
    Keywords:
        builder pattern, grating layer, patterned layer, inhomogeneous material,
        layer creation, photonic crystal, metamaterial
    """
    def __init__(self) -> None:
        """Initialize a GratingLayerBuilder instance.
        
        Initializes the builder with no layer instance created yet.
        """
        super().__init__()

    def create_layer(self):
        """Create a new empty GratingLayer instance.
        
        This method instantiates a new GratingLayer with default values.
        The properties of the layer are set through subsequent method calls
        to update_thickness, update_material_name, etc.
        
        Keywords:
            create layer, instantiate layer, grating layer, patterned layer
        """
        self.layer = GratingLayer()

class LayerDirector:
    """Director class that coordinates the layer building process.
    
    This class implements the director component of the builder pattern,
    coordinating the sequence of steps needed to create fully configured
    layer instances using the appropriate builder based on the layer type.
    
    The director abstracts the layer creation process, allowing the client
    code (typically LayerManager) to create layers without knowing the 
    details of how they are constructed.
    
    Note:
        This class is used internally by TorchRDIT and is not intended for
        direct use by end users.
        
    Keywords:
        builder pattern, layer creation, director, instantiation coordinator
    """

    def __init__(self) -> None:
        """Initialize the LayerDirector instance.
        
        Creates a new LayerDirector with no configuration required.
        """
        pass

    def build_layer(self, layer_type, thickness, material_name, is_optimize = False, is_dispersive = False) -> Layer:
        """Build a fully configured layer of the specified type.
        
        This method coordinates the layer building process, selecting the appropriate
        builder based on the layer_type, creating a new layer instance, and configuring
        it with the provided parameters.

        Args:
            layer_type (str): Type of layer to create. Must be one of:
                       'homogeneous' for a layer with uniform material properties
                       'grating' for a layer with spatially varying material properties
            thickness (float): Thickness of the layer in the simulation's length units.
            material_name (str): Name of the material for the layer. Must be a material
                          that exists in the solver's material library.
            is_optimize (bool): Whether the layer's parameters should be included in
                        optimization procedures. Default is False.
            is_dispersive (bool): Whether the material has frequency-dependent properties.
                           Default is False.

        Returns:
            Layer: A fully configured layer instance of the specified type.
            
        Keywords:
            build layer, create layer, layer construction, layer configuration
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
    """Manager for organizing and manipulating layers in an electromagnetic simulation.
    
    The LayerManager class is responsible for maintaining a collection of layers
    that form the structure being simulated in TorchRDIT. It provides methods for
    adding, updating, and configuring layers, as well as handling the material
    properties of the semi-infinite regions above (transmission) and below
    (reflection) the layered structure.
    
    This class also provides utilities for converting between real-space material
    distributions and their Fourier-space representations using Toeplitz matrices,
    which are crucial for the RCWA and RDIT algorithms.
    
    Note:
        This class is not typically accessed directly by users but is rather used 
        internally by solver classes. Users interact with layers through the solver
        interface.
    
    Attributes:
        layers (List[Layer]): List of Layer instances that make up the structure.
        layer_director (LayerDirector): Director that coordinates layer creation.
        _ref_material_name (str): Name of the material for the reflection region.
        _trn_material_name (str): Name of the material for the transmission region.
        _is_ref_dispers (bool): Whether the reflection material is dispersive.
        _is_trn_dispers (bool): Whether the transmission material is dispersive.
        lattice_t1 (torch.Tensor): First lattice vector defining the unit cell.
        lattice_t2 (torch.Tensor): Second lattice vector defining the unit cell.
        vec_p (torch.Tensor): Vector of p-coordinates in the unit cell.
        vec_q (torch.Tensor): Vector of q-coordinates in the unit cell.
    
    Examples:
    ```python
    # Users typically interact with layers through the solver interface:
    from torchrdit.solver import create_solver
    from torchrdit.utils import create_material
    import torch
    # Create solver with RCWA algorithm
    solver = create_solver()
    # Add materials
    silicon = create_material(name="silicon", permittivity=11.7)
    sio2 = create_material(name="sio2", permittivity=2.25)
    solver.add_materials([silicon, sio2])
    # Add layers
    solver.add_layer(material_name="sio2", thickness=torch.tensor(0.1))
    solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
    # Set bottom (reflection) region to silicon
    solver.update_ref_material("silicon")
    # Set top (transmission) region to sio2
    solver.update_trn_material("sio2")
    ```
    
    Keywords:
        layer management, layer stack, structure definition, layer organization,
        layer manipulation, reflection region, transmission region, toeplitz matrix
    """

    def __init__(self,
                 lattice_t1,
                 lattice_t2,
                 vec_p,
                 vec_q) -> None:
        """Initialize a LayerManager instance.
        
        Args:
            lattice_t1 (torch.Tensor): First lattice vector defining the unit cell.
            lattice_t2 (torch.Tensor): Second lattice vector defining the unit cell.
            vec_p (torch.Tensor): Vector of p-coordinates in the unit cell grid.
            vec_q (torch.Tensor): Vector of q-coordinates in the unit cell grid.
            
        Keywords:
            initialization, layer manager creation
        """
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
        """Generate Toeplitz matrix for the specified layer and parameter.
        
        This method converts the real-space material distribution of a layer
        to its Fourier-space representation using a Toeplitz matrix. This is a
        critical operation for the RCWA and RDIT algorithms that operate in
        Fourier space.
        
        Args:
            layer_index (int): Index of the layer in the layers list.
            n_harmonic1 (int): Number of harmonics for the first dimension.
            n_harmonic2 (int): Number of harmonics for the second dimension.
            param (str): Parameter to convert, either 'er' for permittivity or
                   'ur' for permeability. Default is 'er'.
            method (str): Method for computing the Toeplitz matrix.
                    'FFT': Uses Fast Fourier Transform, works for all cell types.
                    'Analytical': Uses analytical formulation, only for Cartesian cells.
                    Default is 'FFT'.
        
        Raises:
            ValueError: If the specified layer does not have the required material
                      property distribution set.
                      
        Keywords:
            toeplitz matrix, fourier transform, RCWA, spatial harmonics,
            material distribution, permittivity, permeability
        """
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
        """Construct Toeplitz matrices from a real-space 2D grid.
        
        This internal method implements the algorithms for converting between
        real-space material distributions and their Fourier-space representations
        as Toeplitz matrices.

        Args:
            input_matrix (torch.Tensor): Grid in real space with shape (H, W) or (batch, H, W).
            nharmonic_1 (int): Number of harmonics for the first dimension (T1 direction).
                       Default is 1.
            nharmonic_2 (int): Number of harmonics for the second dimension (T2 direction).
                       Default is 1.
            method (str): Method for computing the Toeplitz matrix.
                    'FFT': Uses Fast Fourier Transform, works for all cell types.
                    'Analytical': Uses analytical formulation, only for Cartesian cells.
                    Default is 'FFT'.
        
        Returns:
            torch.Tensor: Toeplitz matrix with shape (n_harmonics, n_harmonics) or
                       (batch, n_harmonics, n_harmonics), where 
                       n_harmonics = nharmonic_1 * nharmonic_2.
        
        Raises:
            ValueError: If input_matrix has unexpected dimensions.
            
        Keywords:
            toeplitz matrix, fourier transform, convolution matrix, RCWA,
            spatial harmonics, internal method
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
        """Add a new layer to the layer structure.
        
        This method creates a new layer of the specified type with the given properties
        and adds it to the layer structure. Layers are stacked in the order they are added,
        with the first layer added being at the bottom of the stack (closest to the
        reflection region) and subsequent layers building upward.

        Args:
            layer_type (str): Type of layer to create. Must be one of:
                       'homogeneous' for a layer with uniform material properties
                       'grating' for a layer with spatially varying material properties
            thickness (torch.Tensor): Thickness of the layer in the simulation's length units.
            material_name (str): Name of the material for the layer. Must be a material
                          that exists in the solver's material library.
            is_optimize (bool): Whether the layer's parameters should be included in
                        optimization procedures. Default is False.
            is_dispersive (bool): Whether the material has frequency-dependent properties.
                           Default is False.
                           
        Note:
            Users typically do not call this method directly but rather use the
            solver's add_layer method, which provides a more user-friendly interface.
            
        Keywords:
            add layer, create layer, layer stack, material layer, layer configuration
        """
        new_layer = self.layer_director.build_layer(layer_type=layer_type,
                                                    thickness=thickness,
                                                    material_name=material_name,
                                                    is_optimize=is_optimize,
                                                    is_dispersive=is_dispersive)
        self.layers.append(new_layer)

    def replace_layer_to_homogeneous(self, layer_index):
        """Convert a layer to a homogeneous layer.
        
        This method replaces the layer at the specified index with a new homogeneous
        layer that has the same thickness, material name, and other properties.

        Args:
            layer_index (int): Index of the layer to convert in the layers list.
            
        Keywords:
            convert layer, homogeneous layer, layer modification
        """
        new_layer = self.layer_director.build_layer(layer_type='homogenous',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    def replace_layer_to_grating(self, layer_index):
        """Convert a layer to a grating (patterned) layer.
        
        This method replaces the layer at the specified index with a new grating
        layer that has the same thickness, material name, and other properties.

        Args:
            layer_index (int): Index of the layer to convert in the layers list.
            
        Keywords:
            convert layer, grating layer, patterned layer, layer modification
        """
        new_layer = self.layer_director.build_layer(layer_type='grating',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    @tensor_params_check(check_start_index=2, check_stop_index=2)
    def update_layer_thickness(self, layer_index, thickness):
        """Update the thickness of a layer.
        
        This method changes the thickness of the layer at the specified index.

        Args:
            layer_index (int): Index of the layer to update in the layers list.
            thickness (torch.Tensor): New thickness value for the layer in the
                     simulation's length units.
                     
        Keywords:
            update thickness, layer modification, geometric parameter
        """
        self.layers[layer_index].thickness = thickness

    def update_trn_layer(self, material_name: str, is_dispersive: bool):
        """Update the transmission region material.
        
        This method sets the material for the semi-infinite region above the
        layer stack (transmission region).

        Args:
            material_name (str): Name of the material to use for the transmission region.
                          Must be a material that exists in the solver's material library.
            is_dispersive (bool): Whether the material has frequency-dependent properties.
            
        Keywords:
            transmission region, output medium, top medium, semi-infinite region
        """
        self._trn_material_name = material_name
        self._is_trn_dispers = is_dispersive

    def update_ref_layer(self, material_name: str, is_dispersive: bool):
        """Update the reflection region material.
        
        This method sets the material for the semi-infinite region below the
        layer stack (reflection region).

        Args:
            material_name (str): Name of the material to use for the reflection region.
                          Must be a material that exists in the solver's material library.
            is_dispersive (bool): Whether the material has frequency-dependent properties.
            
        Keywords:
            reflection region, input medium, bottom medium, semi-infinite region
        """
        self._ref_material_name = material_name
        self._is_ref_dispers = is_dispersive

    @property
    def ref_material_name(self) -> str:
        """Get the name of the material used in the reflection region.

        Returns:
            str: Name of the material for the semi-infinite region below the layer stack.
            
        Keywords:
            reflection material, bottom material, input medium
        """
        return self._ref_material_name

    @property
    def trn_material_name(self) -> str:
        """Get the name of the material used in the transmission region.

        Returns:
            str: Name of the material for the semi-infinite region above the layer stack.
            
        Keywords:
            transmission material, top material, output medium
        """
        return self._trn_material_name

    @property
    def is_ref_dispersive(self) -> bool:
        """Check if the reflection region material has frequency-dependent properties.

        Returns:
            bool: True if the reflection material is dispersive, False otherwise.
            
        Keywords:
            dispersive material, frequency-dependent, wavelength-dependent
        """
        return self._is_ref_dispers

    @property
    def is_trn_dispersive(self) -> bool:
        """Check if the transmission region material has frequency-dependent properties.

        Returns:
            bool: True if the transmission material is dispersive, False otherwise.
            
        Keywords:
            dispersive material, frequency-dependent, wavelength-dependent
        """
        return self._is_trn_dispers

    @property
    def nlayer(self) -> int:
        """Get the number of layers in the structure.

        Returns:
            int: Number of material layers in the layer stack.
            
        Keywords:
            layer count, stack size, structure depth
        """
        return len(self.layers)
