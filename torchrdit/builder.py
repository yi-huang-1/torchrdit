from typing import List, Dict, Union, Any, TYPE_CHECKING
import torch
import numpy as np
from .constants import Algorithm, Precision
from .utils import create_material
from .algorithm import SolverAlgorithm

# Import types for type checking only, not at runtime
if TYPE_CHECKING:
    from .solver import RCWASolver, RDITSolver

# Function that was previously TorchrditConfig._create_materials
def _create_materials(materials_dict: Dict[str, Any], base_path: str) -> Dict[str, Any]:
    """
    Create material objects from dictionary.
    
    Args:
        materials_dict (Dict[str, Any]): Dictionary of material specifications
        base_path (str): Base path for relative file references
        
    Returns:
        Dict[str, Any]: Dictionary of created material objects
    """
    materials = {}
    for name, props in materials_dict.items():
        if props.get("dielectric_dispersion", False):
            materials[name] = create_material(
                name=name,
                dielectric_dispersion=True,
                user_dielectric_file=os.path.join(base_path, props["dielectric_file"]),
                data_format=props.get("data_format", "freq-eps"),
                data_unit=props.get("data_unit", "thz")
            )
        else:
            materials[name] = create_material(
                name=name,
                permittivity=props["permittivity"]
            )
    return materials

# Function that was previously TorchrditConfig._add_layers
def _add_layers(solver: Union["RCWASolver", "RDITSolver"], layers_list: List[Dict[str, Any]], 
               materials: Dict[str, Any]) -> None:
    """
    Add layers to the solver.
    
    Args:
        solver (Union["RCWASolver", "RDITSolver"]): The solver to add layers to
        layers_list (List[Dict[str, Any]]): List of layer configurations
        materials (Dict[str, Any]): Dictionary of material objects
    """
    for layer in layers_list:
        material_name = layer["material"]
        if isinstance(layer["thickness"], torch.Tensor):
            thickness = layer["thickness"]
        else:
            thickness = torch.tensor(layer["thickness"], dtype=torch.float32)
        is_homogeneous = layer.get("is_homogeneous", True)
        is_optimize = layer.get("is_optimize", False)

        solver.add_layer(
            material_name=materials[material_name],
            thickness=thickness,
            is_homogeneous=is_homogeneous,
            is_optimize=is_optimize
        )


# Function that was previously TorchrditConfig.flip_config
def flip_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a flipped version of the configuration (reversed layers).
    
    Args:
        config (Dict[str, Any]): Original configuration
        
    Returns:
        Dict[str, Any]: Flipped configuration
    """
    flipped_config = config.copy()
    flipped_config["layers"] = list(reversed(config["layers"]))

    if "trn_material" in config and "ref_material" in config:
        flipped_config["trn_material"], flipped_config["ref_material"] = (
            config["ref_material"], config["trn_material"]
        )
    elif "trn_material" in config:
        flipped_config["ref_material"] = config["trn_material"]
        del flipped_config["trn_material"]
    elif "ref_material" in config:
        flipped_config["trn_material"] = config["ref_material"]
        del flipped_config["ref_material"]

    return flipped_config

class SolverBuilder:
    """Builder for creating and configuring solver instances."""
    
    def __init__(self):
        """Initialize the builder with default values."""
        # Default values
        self._algorithm_type = Algorithm.RCWA
        self._algorithm_instance = None
        self._precision = Precision.SINGLE
        self._lam0 = np.array([1.0])
        self._lengthunit = 'um'
        self._rdim = [512, 512]
        self._kdim = [3, 3]
        self._materials = []
        self._materials_dict = {}
        self._t1 = torch.tensor([[1.0, 0.0]])
        self._t2 = torch.tensor([[0.0, 1.0]])
        self._is_use_FFF = True
        self._device = 'cpu'
        self._layers = []
        self._rdit_order = None
        self._config_path = None
        
    def with_algorithm(self, algorithm_type: Algorithm) -> 'SolverBuilder':
        """Set the algorithm type."""
        self._algorithm_type = algorithm_type
        return self
    
    def with_algorithm_instance(self, algorithm: SolverAlgorithm) -> 'SolverBuilder':
        """Set a specific algorithm instance directly."""
        self._algorithm_instance = algorithm
        return self
    
    def with_precision(self, precision: Precision) -> 'SolverBuilder':
        """Set the precision."""
        self._precision = precision
        return self
    
    def with_wavelengths(self, lam0: Union[float, np.ndarray]) -> 'SolverBuilder':
        """Set the wavelengths to be solved."""
        if isinstance(lam0, float) or isinstance(lam0, int):
            self._lam0 = np.array([float(lam0)])
        else:
            self._lam0 = np.array(lam0)
        return self
    
    def with_length_unit(self, unit: str) -> 'SolverBuilder':
        """Set the length unit."""
        self._lengthunit = unit
        return self
    
    def with_real_dimensions(self, rdim: List[int]) -> 'SolverBuilder':
        """Set the dimensions in real space."""
        self._rdim = rdim
        return self
    
    def with_k_dimensions(self, kdim: List[int]) -> 'SolverBuilder':
        """Set the dimensions in k space."""
        self._kdim = kdim
        return self
    
    def with_materials(self, materials: List[Any]) -> 'SolverBuilder':
        """Set the list of materials."""
        self._materials = materials
        self._materials_dict = {mat.name: mat for mat in materials if hasattr(mat, 'name')}
        return self
    
    def add_material(self, material: Any) -> 'SolverBuilder':
        """Add a single material to the list."""
        self._materials.append(material)
        if hasattr(material, 'name'):
            self._materials_dict[material.name] = material
        return self
    
    def with_lattice_vectors(self, t1: torch.Tensor, t2: torch.Tensor) -> 'SolverBuilder':
        """Set the lattice vectors."""
        self._t1 = t1
        self._t2 = t2
        return self
    
    def with_fff(self, use_fff: bool) -> 'SolverBuilder':
        """Set whether to use Fast Fourier Factorization."""
        self._is_use_FFF = use_fff
        return self
    
    def with_device(self, device: Union[str, torch.device]) -> 'SolverBuilder':
        """Set the computation device."""
        self._device = device
        return self
    
    def with_rdit_order(self, order: int) -> 'SolverBuilder':
        """Set the RDIT order (only used for RDIT algorithm)."""
        self._rdit_order = order
        return self
    
    def add_layer(self, layer_config: Dict[str, Any]) -> 'SolverBuilder':
        """Add a layer configuration to be added after solver creation."""
        self._layers.append(layer_config)
        return self
    
    def from_config(self, config: Union[str, Dict[str, Any]], flip: bool = False) -> 'SolverBuilder':
        """Configure the builder from a config file or dictionary."""
        import json
        
        # Store original config path if it's a string
        if isinstance(config, str):
            self._config_path = config
            with open(config, 'r') as f:
                config = json.load(f)
        
        # Apply flip if needed
        if flip:
            config = flip_config(config)
        
        # Extract parameters from config
        if 'algorithm' in config:
            self._algorithm_type = Algorithm[config['algorithm']]
        if 'precision' in config:
            self._precision = Precision[config['precision']]
        if 'wavelengths' in config:
            self._lam0 = np.array(config['wavelengths'])
        if 'length_unit' in config:
            self._lengthunit = config['length_unit']
        if 'rdim' in config:
            self._rdim = config['rdim']
        if 'kdim' in config:
            self._kdim = config['kdim']
        if 'lattice_vectors' in config:
            t1_data = config['lattice_vectors']['t1']
            t2_data = config['lattice_vectors']['t2']
            self._t1 = torch.tensor(t1_data)
            self._t2 = torch.tensor(t2_data)
        if 'use_fff' in config:
            self._is_use_FFF = config['use_fff']
        if 'device' in config:
            self._device = config['device']
        if 'rdit_order' in config:
            self._rdit_order = config['rdit_order']
        
        # Process materials and layers
        base_path = config.get('base_path', '')
        if 'materials' in config:
            materials_dict = _create_materials(config['materials'], base_path)
            self._materials = list(materials_dict.values())
            self._materials_dict = materials_dict
        
        if 'layers' in config:
            self._layers = config['layers']
        
        return self
    
    def build(self):
        """Build and return the configured solver.
        
        Returns:
            Union[RCWASolver, RDITSolver]: The configured solver instance
        """
        # Lazy import to avoid circular dependencies
        from .solver import RCWASolver, RDITSolver
        
        # Create the appropriate solver type based on algorithm
        if self._algorithm_type == Algorithm.RCWA:
            solver = RCWASolver(
                lam0=self._lam0,
                lengthunit=self._lengthunit,
                rdim=self._rdim,
                kdim=self._kdim,
                materiallist=self._materials,
                t1=self._t1,
                t2=self._t2,
                is_use_FFF=self._is_use_FFF,
                precision=self._precision,
                device=self._device
            )
            
            # If custom algorithm instance provided, override the default
            if self._algorithm_instance:
                solver.algorithm = self._algorithm_instance
                
        else:  # RDIT
            solver = RDITSolver(
                lam0=self._lam0,
                lengthunit=self._lengthunit,
                rdim=self._rdim,
                kdim=self._kdim,
                materiallist=self._materials,
                t1=self._t1,
                t2=self._t2,
                is_use_FFF=self._is_use_FFF,
                precision=self._precision,
                device=self._device
            )
            
            # If custom algorithm instance provided, override the default
            if self._algorithm_instance:
                solver.algorithm = self._algorithm_instance
            
            # Set RDIT order if specified
            if self._rdit_order is not None:
                solver.set_rdit_order(self._rdit_order)
        
        # Add layers if specified
        if self._layers and self._materials_dict:
            _add_layers(solver, self._layers, self._materials_dict)
        
        return solver