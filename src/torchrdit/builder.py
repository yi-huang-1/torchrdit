from typing import List, Dict, Union, Any, Optional, TYPE_CHECKING
from pathlib import Path
import torch
import numpy as np
from .constants import Algorithm, Precision
from .materials import MaterialClass
from .utils import create_material
from .algorithm import SolverAlgorithm
import os
from .path_utils import infer_caller_dir, resolve_data_path

# Import types for type checking only, not at runtime
if TYPE_CHECKING:
    from .solver import RCWASolver, RDITSolver


# Function that was previously TorchrditConfig._create_materials
def _create_materials(
    materials_dict: Dict[str, Any],
    config_dir: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """Create material objects from dictionary specification.

    This internal utility function converts a dictionary of material specifications
    into actual material objects that can be used by the solver.

    Args:
        materials_dict (Dict[str, Any]): Dictionary of material specifications.
            Each key is a material name, and each value is a dictionary containing
            the material properties.
        config_dir (str | os.PathLike | None): Directory context for resolving relative
            material data files (e.g. dispersive ``dielectric_file``). Resolution order:
            ``config_dir`` (typically the JSON config file directory), then caller script
            directory, then current working directory.

    Returns:
        Dict[str, Any]: Dictionary of created material objects with material names as keys.

	    Examples:
	    ```python
	    materials_spec = {
        "Si": {"permittivity": 12.0},
        "SiO2": {"permittivity": 2.25},
        "Au": {
            "dielectric_dispersion": True,
            "dielectric_file": "materials/gold.txt",
            "data_format": "freq-eps",
            "data_unit": "thz"
        },
        "SiC_inmem": {
            "dielectric_dispersion": True,
            "dispersion": {
                "wavelengths_um": [1.0, 1.5, 2.0],
                "n": [2.6, 2.55, 2.5],
                "k": [0.1, 0.08, 0.05],
            },
        },
	    }
	    materials = _create_materials(materials_spec, config_dir="./data")
	    ```
	
	    Notes:
	        - In-memory dispersion samples under ``"dispersion"`` accept Python sequences,
	          NumPy arrays, or torch tensors; values are converted to CPU NumPy internally
	          (no autograd gradients) to build an interpolation table.

	    Keywords:
	        materials, permittivity, dielectric dispersion, material creation
	    """
    materials: Dict[str, Any] = {}
    for name, props in materials_dict.items():
        if not isinstance(name, str):
            raise ValueError(f"Material name must be a string, got {type(name)}")
        if not isinstance(props, dict):
            raise ValueError(f"Material properties must be a dictionary at materials[{name!r}], got {type(props)}")

        is_dispersive = bool(props.get("dielectric_dispersion", False))
        if is_dispersive:
            allowed_keys = {"permittivity", "dielectric_dispersion", "dielectric_file", "dispersion", "data_format", "data_unit"}
            unknown = set(props.keys()) - allowed_keys
            if unknown:
                raise ValueError(
                    f"Unknown material keys detected at materials[{name!r}]: {sorted(unknown)} "
                    f"(allowed: {sorted(allowed_keys)})"
                )
            if "permittivity" in props:
                raise ValueError(
                    f"Material at materials[{name!r}]: 'permittivity' is not allowed when 'dielectric_dispersion' is True"
                )

            has_file = "dielectric_file" in props
            has_disp = "dispersion" in props
            if has_file and has_disp:
                raise ValueError(
                    f"Material at materials[{name!r}] cannot specify both 'dielectric_file' and 'dispersion'"
                )
            if not has_file and not has_disp:
                raise ValueError(
                    f"Material at materials[{name!r}] requires 'dielectric_file' or 'dispersion' when 'dielectric_dispersion' is True"
                )

            if has_file:
                caller_dir = infer_caller_dir(
                    skip_files=("builder.py", "solver.py", "__init__.py", "path_utils.py")
                )
                resolved_file = resolve_data_path(
                    props["dielectric_file"],
                    config_dir=config_dir,
                    caller_dir=str(caller_dir) if caller_dir is not None else None,
                )
                materials[name] = create_material(
                    name=name,
                    dielectric_dispersion=True,
                    user_dielectric_file=resolved_file,
                    data_format=props.get("data_format", "freq-eps"),
                    data_unit=props.get("data_unit", "thz"),
                )
            else:
                if "data_format" in props or "data_unit" in props:
                    raise ValueError(
                        f"Material at materials[{name!r}]: 'data_format'/'data_unit' are not used with in-memory dispersion"
                    )
                disp = props["dispersion"]
                if not isinstance(disp, dict):
                    raise ValueError(f"Material at materials[{name!r}].dispersion must be a dict, got {type(disp)}")
                allowed_disp_keys = {"wavelengths_um", "eps", "n", "k"}
                unknown_disp = set(disp.keys()) - allowed_disp_keys
                if unknown_disp:
                    raise ValueError(
                        f"Unknown dispersion keys detected at materials[{name!r}].dispersion: {sorted(unknown_disp)} "
                        f"(allowed: {sorted(allowed_disp_keys)})"
                    )
                if "wavelengths_um" not in disp:
                    raise ValueError(f"Material at materials[{name!r}].dispersion requires 'wavelengths_um'")

                has_eps = "eps" in disp
                has_n = "n" in disp
                if has_eps and has_n:
                    raise ValueError(f"Material at materials[{name!r}].dispersion cannot specify both 'eps' and 'n/k'")
                if not has_eps and not has_n:
                    raise ValueError(f"Material at materials[{name!r}].dispersion requires 'eps' or 'n' (+ optional 'k')")
                if has_eps and ("k" in disp):
                    raise ValueError(f"Material at materials[{name!r}].dispersion: when 'eps' is provided, 'k' is not allowed")

                materials[name] = create_material(
                    name=name,
                    dielectric_dispersion=True,
                    user_dielectric_wavelengths_um=disp["wavelengths_um"],
                    user_dielectric_eps=disp.get("eps"),
                    user_dielectric_n=disp.get("n"),
                    user_dielectric_k=disp.get("k"),
                )
        else:
            allowed_keys = {"permittivity", "dielectric_dispersion"}
            unknown = set(props.keys()) - allowed_keys
            if unknown:
                raise ValueError(
                    f"Unknown material keys detected at materials[{name!r}]: {sorted(unknown)} "
                    f"(allowed: {sorted(allowed_keys)})"
                )
            if "permittivity" not in props:
                raise ValueError(
                    f"Material at materials[{name!r}] requires 'permittivity' when 'dielectric_dispersion' is False"
                )
            materials[name] = create_material(name=name, permittivity=props["permittivity"])

    return materials


# Function that was previously TorchrditConfig._add_layers
def _add_layers(
    solver: Union["RCWASolver", "RDITSolver"], layers_list: List[Dict[str, Any]], materials: Dict[str, Any]
) -> None:
    """Add layers to an existing solver instance.

    This internal utility function adds multiple layers to a solver from a list of
    layer configurations and a dictionary of materials.

    Args:
        solver (Union["RCWASolver", "RDITSolver"]): The solver to add layers to.
        layers_list (List[Dict[str, Any]]): List of layer configurations. Each dictionary
            should contain at minimum a "material" key and a "thickness" key.
            Optional keys include ``"slice_count"`` to replicate a layer across
            multiple identical slices for improved accuracy on thick layers.
        materials (Dict[str, Any]): Dictionary of material objects with material names as keys.

    Examples:
    ```python
    layers = [
        {"material": "SiO2", "thickness": 0.2, "is_homogeneous": True},
        {"material": "Si", "thickness": 0.5, "is_homogeneous": False},
        {"material": "Si", "thickness": 1.0, "slice_count": 4},
    ]
    _add_layers(solver, layers, materials_dict)
    ```

    Keywords:
        layers, materials, thickness, solver configuration, layer stack
    """
    allowed_keys = {"type", "material", "thickness", "is_homogeneous", "is_optimize", "slice_count"}
    for idx, layer in enumerate(layers_list):
        if not isinstance(layer, dict):
            raise ValueError(f"Layer configuration at layers[{idx}] must be a dict, got {type(layer)}")
        unknown = set(layer.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown layer keys detected at layers[{idx}]: {sorted(unknown)} (allowed: {sorted(allowed_keys)})")
        if "material" not in layer or "thickness" not in layer:
            raise ValueError(f"Layer configuration at layers[{idx}] must include 'material' and 'thickness'")

        material_value = layer["material"]
        if isinstance(material_value, MaterialClass):
            material_name = material_value.name
            material_obj = material_value
        elif isinstance(material_value, str):
            material_name = material_value
            if material_name not in materials:
                raise ValueError(f"Unknown material {material_name!r} at layers[{idx}]")
            material_obj = materials[material_name]
        else:
            raise ValueError(f"Invalid material type: {type(layer['material'])}")

        thickness_val = layer["thickness"]
        if isinstance(thickness_val, torch.Tensor):
            thickness = thickness_val.to(device=solver.device, dtype=solver.tfloat)
        else:
            thickness = torch.as_tensor(thickness_val, dtype=solver.tfloat, device=solver.device)
        is_homogeneous = layer.get("is_homogeneous", True)
        is_optimize = layer.get("is_optimize", False)

        slice_count = layer.get("slice_count", 1)
        if isinstance(slice_count, torch.Tensor):
            slice_count = int(slice_count.item())
        else:
            slice_count = int(slice_count)
        if slice_count < 1:
            slice_count = 1

        solver.add_layer(
            material_name=material_obj,
            thickness=thickness,
            is_homogeneous=is_homogeneous,
            is_optimize=is_optimize,
            slice_count=slice_count,
        )


# Function that was previously TorchrditConfig.flip_config
def flip_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a flipped version of the configuration with reversed layer stack.

    This function creates a new configuration with the layer stack reversed and
    the transmission and reflection materials swapped. This is useful for simulating
    the same structure from the opposite direction.

    Args:
        config (Dict[str, Any]): Original configuration dictionary.

    Returns:
        Dict[str, Any]: Flipped configuration with reversed layer stack and
            swapped transmission/reflection materials.

    Examples:
    ```python
    original_config = {
        "layers": [{"material": "Air"}, {"material": "Si"}, {"material": "SiO2"}],
        "trn_material": "Air",
        "ref_material": "SiO2"
    }
    flipped = flip_config(original_config)
    flipped["layers"]
    # [{"material": "SiO2"}, {"material": "Si"}, {"material": "Air"}]
    flipped["trn_material"]
    # "SiO2"
    flipped["ref_material"]
    # "Air"
    ```

    Keywords:
        configuration, flip, reverse, layer stack, transmission, reflection
    """
    flipped_config = config.copy()
    flipped_config["layers"] = list(reversed(config["layers"]))

    if "trn_material" in config and "ref_material" in config:
        flipped_config["trn_material"], flipped_config["ref_material"] = (
            config["ref_material"],
            config["trn_material"],
        )
    elif "trn_material" in config:
        flipped_config["ref_material"] = config["trn_material"]
        del flipped_config["trn_material"]
    elif "ref_material" in config:
        flipped_config["trn_material"] = config["ref_material"]
        del flipped_config["ref_material"]

    return flipped_config


class SolverBuilder:
    """Builder for creating and configuring electromagnetic solver instances.
    
    This class implements the Builder pattern for constructing and configuring
    RCWA and R-DIT solver instances. It provides a fluent interface with method
    chaining to set various solver parameters.
    
    The SolverBuilder is the recommended way to create and configure solvers
    in the TorchRDIT package, allowing for a clean and expressive API.
    
    Attributes:
        Various internal attributes storing the configuration state.
    
    Examples:
    ```python
    # Create a basic RCWA solver with default settings
    from torchrdit.builder import SolverBuilder
    from torchrdit.constants import Algorithm
    solver = SolverBuilder().build()

    # Create a more customized RCWA solver
    solver = SolverBuilder() \\
        .with_algorithm(Algorithm.RCWA) \\
        .with_wavelengths(1.55) \\
        .with_k_dimensions([5, 5]) \\
        .with_device("cuda") \\
        .build()

    # Create an R-DIT solver with custom order
    solver = SolverBuilder() \\
        .with_algorithm(Algorithm.RDIT) \\
        .with_rdit_order(8) \\
        .build()

    # Load configuration from a JSON file
    solver = SolverBuilder().from_config("config.json").build()
    ```
    
    Keywords:
        builder pattern, solver configuration, RCWA, R-DIT, electromagnetic solver
    """

    def __init__(self):
        """Initialize the builder with default values for all solver parameters."""
        # Default values
        self._algorithm_type = Algorithm.RCWA
        self._algorithm_instance = None
        self._precision = Precision.SINGLE
        self._lam0 = np.array([1.0])
        self._lengthunit = "um"
        self._grids = [512, 512]
        self._harmonics = [3, 3]
        self._materials = []
        self._materials_dict = {}
        self._t1 = torch.tensor([[1.0, 0.0]])
        self._t2 = torch.tensor([[0.0, 1.0]])
        self._is_use_FFF = False
        self._device = "cpu"
        self._layers = []
        self._rdit_order = None
        self._config_path = None
        self._trn_material = None
        self._ref_material = None
        self._fff_vector_scheme = "POL"
        self._fff_fourier_weight = 1e-2
        self._fff_smoothness_weight = 1e-3
        self._fff_vector_steps = 1

    def with_algorithm(self, algorithm_type: Algorithm) -> "SolverBuilder":
        """Set the algorithm type for the solver.

        Args:
            algorithm_type (Algorithm): The algorithm to use, either Algorithm.RCWA
                or Algorithm.RDIT.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.constants import Algorithm
        builder = SolverBuilder().with_algorithm(Algorithm.RCWA)
        builder = SolverBuilder().with_algorithm(Algorithm.RDIT)
        ```

        Keywords:
            algorithm, RCWA, R-DIT, solver configuration
        """
        self._algorithm_type = algorithm_type
        return self

    def with_algorithm_instance(self, algorithm: SolverAlgorithm) -> "SolverBuilder":
        """Set a specific algorithm instance directly.

        This is an advanced method that allows directly setting a custom algorithm
        instance rather than using the built-in implementations.

        Args:
            algorithm (SolverAlgorithm): A custom algorithm instance that implements
                the SolverAlgorithm interface.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        # Advanced usage with custom algorithm
        from torchrdit.constants import Algorithm
        from torchrdit.builder import SolverBuilder
        custom_algorithm = RCWAAlgorithm(None)  # Configure as needed
        builder = SolverBuilder().with_algorithm_instance(custom_algorithm)
        ```

        Keywords:
            custom algorithm, algorithm instance, advanced configuration
        """
        self._algorithm_instance = algorithm
        return self

    def with_precision(self, precision: Precision) -> "SolverBuilder":
        """Set the numerical precision for computations.

        Args:
            precision (Precision): The precision to use, either Precision.SINGLE
                for float32 or Precision.DOUBLE for float64.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        from torchrdit.constants import Precision
        # Use single precision (float32)
        builder = SolverBuilder().with_precision(Precision.SINGLE)
        # Use double precision (float64) for higher accuracy
        builder = SolverBuilder().with_precision(Precision.DOUBLE)
        ```

        Keywords:
            precision, float32, float64, numerical accuracy
        """
        self._precision = precision
        return self

    def with_wavelengths(self, lam0: Union[float, np.ndarray]) -> "SolverBuilder":
        """Set the wavelengths for the solver to compute solutions at.

        Args:
            lam0 (Union[float, np.ndarray]): The wavelength(s) to solve for. Can be
                a single value or an array of values for spectral calculations.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        import numpy as np
        # Set a single wavelength (1.55 μm)
        builder = SolverBuilder().with_wavelengths(1.55)
        # Set multiple wavelengths for spectral calculations
        wavelengths = np.linspace(1.2, 1.8, 100)  # 100 points from 1.2 to 1.8 μm
        builder = SolverBuilder().with_wavelengths(wavelengths)
        ```

        Keywords:
            wavelength, spectral, optical frequency, electromagnetic spectrum
        """
        if isinstance(lam0, float) or isinstance(lam0, int):
            self._lam0 = np.array([float(lam0)])
        else:
            self._lam0 = np.array(lam0)
        return self

    def with_length_unit(self, unit: str) -> "SolverBuilder":
        """Set the length unit for all dimensional parameters.

        Args:
            unit (str): The length unit to use (e.g., 'um', 'nm', 'mm').

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Set length unit to micrometers
        builder = SolverBuilder().with_length_unit('um')
        # Set length unit to nanometers
        builder = SolverBuilder().with_length_unit('nm')
        ```

        Keywords:
            length unit, dimensions, scale, micrometers, nanometers
        """
        self._lengthunit = unit
        return self

    def with_real_dimensions(self, grids: List[int]) -> "SolverBuilder":
        """Set the dimensions in real space for discretization.

        Args:
            grids (List[int]): The dimensions in real space as [height, width].

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Set real space dimensions to 512x512
        builder = SolverBuilder().with_real_dimensions([512, 512])
        # Set real space dimensions to 1024x1024 for higher resolution
        builder = SolverBuilder().with_real_dimensions([1024, 1024])
        ```

        Keywords:
            real dimensions, discretization, spatial resolution, grid size
        """
        self._grids = grids
        return self

    def with_k_dimensions(self, harmonics: List[int]) -> "SolverBuilder":
        """Set the dimensions in k-space (Fourier space) for harmonics.

        Args:
            harmonics (List[int]): The dimensions in k-space as [height, width].
                Higher values include more harmonics for better accuracy at
                the cost of computational complexity.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Set k-space dimensions to 3x3 (9 harmonics)
        builder = SolverBuilder().with_k_dimensions([3, 3])
        # Set k-space dimensions to 5x5 (25 harmonics) for higher accuracy
        builder = SolverBuilder().with_k_dimensions([5, 5])
        ```

        Keywords:
            k-space, Fourier harmonics, diffraction orders, computational accuracy
        """
        self._harmonics = harmonics
        return self

    def with_materials(self, materials: List[Any]) -> "SolverBuilder":
        """Set the list of materials for the simulation.

        Args:
            materials (List[Any]): List of material objects to use in the simulation.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        # Set materials for the simulation
        from torchrdit.utils import create_material
        from torchrdit.builder import SolverBuilder
        air = create_material(name="Air", permittivity=1.0)
        silicon = create_material(name="Si", permittivity=12.0)
        builder = SolverBuilder().with_materials([air, silicon])
        ```

        Keywords:
            materials, permittivity, dielectric properties, optical materials
        """
        self._materials = materials
        self._materials_dict = {mat.name: mat for mat in materials if hasattr(mat, "name")}
        return self

    def add_material(self, material: Any) -> "SolverBuilder":
        """Add a single material to the materials list.

        Args:
            material (Any): The material object to add to the simulation.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        # Add materials one by one
        from torchrdit.utils import create_material
        from torchrdit.builder import SolverBuilder
        builder = SolverBuilder()
        builder.add_material(create_material(name="Air", permittivity=1.0))
        builder.add_material(create_material(name="Si", permittivity=12.0))
        ```

        Keywords:
            material, add material, permittivity, optical properties
        """
        self._materials.append(material)
        if hasattr(material, "name"):
            self._materials_dict[material.name] = material
        return self

    def with_trn_material(self, material: Union[str, Any]) -> "SolverBuilder":
        """Set the transmission material for the simulation.

        The transmission material defines the material properties of the medium
        on the transmission side of the structure.

        Args:
            material (Union[str, Any]): Either a material name (string) that exists
                in the material list, or a material object that will be added to
                the material list.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Set transmission material by name (must already be in materials list)
        builder = SolverBuilder().with_trn_material("Air")
        # Set transmission material by object
        from torchrdit.utils import create_material
        air = create_material(name="Air", permittivity=1.0)
        builder = SolverBuilder().with_trn_material(air)
        ```

        Keywords:
            transmission material, output medium, optical interface
        """
        if hasattr(material, "name"):
            # Material instance provided, add it to materials if not already there
            if material.name not in self._materials_dict:
                self.add_material(material)
            self._trn_material = material.name
        else:
            # Assume it's a material name string
            self._trn_material = material
        return self

    def with_ref_material(self, material: Union[str, Any]) -> "SolverBuilder":
        """Set the reflection material (or incident material) for the simulation.

        The reflection material (or incident material) defines the material properties of the medium
        on the reflection side (or incident side) of the structure.

        Args:
            material (Union[str, Any]): Either a material name (string) that exists
                in the material list, or a material object that will be added to
                the material list.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Set reflection material by name (must already be in materials list)
        builder = SolverBuilder().with_ref_material("Si")
        # Set reflection material by object
        from torchrdit.utils import create_material
        silicon = create_material(name="Si", permittivity=12.0)
        builder = SolverBuilder().with_ref_material(silicon)
        ```

        Keywords:
            reflection material, incident material, optical interface
        """
        if hasattr(material, "name"):
            # Material instance provided, add it to materials if not already there
            if material.name not in self._materials_dict:
                self.add_material(material)
            self._ref_material = material.name
        else:
            # Assume it's a material name string
            self._ref_material = material
        return self

    def with_lattice_vectors(self, t1: torch.Tensor, t2: torch.Tensor) -> "SolverBuilder":
        """Set the lattice vectors defining the unit cell geometry.

        Args:
            t1 (torch.Tensor): First lattice vector.
            t2 (torch.Tensor): Second lattice vector.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        # Set square lattice with 1 μm period
        import torch
        from torchrdit.builder import SolverBuilder
        t1 = torch.tensor([[1.0, 0.0]])
        t2 = torch.tensor([[0.0, 1.0]])
        builder = SolverBuilder().with_lattice_vectors(t1, t2)
        # Set rectangular lattice
        t1 = torch.tensor([[1.5, 0.0]])
        t2 = torch.tensor([[0.0, 1.0]])
        builder = SolverBuilder().with_lattice_vectors(t1, t2)
        ```

        Keywords:
            lattice vectors, unit cell, periodicity, photonic crystal
        """
        if not isinstance(t1, torch.Tensor):
            try:
                t1 = torch.tensor(t1)
            except (TypeError, ValueError, RuntimeError) as e:
                raise ValueError(f"Invalid input t1 [{t1}]: {str(e)}")
        if not isinstance(t2, torch.Tensor):
            try:
                t2 = torch.tensor(t2)
            except (TypeError, ValueError, RuntimeError) as e:
                raise ValueError(f"Invalid input t2 [{t2}]: {str(e)}")
        self._t1 = t1
        self._t2 = t2
        return self

    def with_fff(self, use_fff: bool) -> "SolverBuilder":
        """Set whether to use Fast Fourier Factorization for improved convergence.

        Fast Fourier Factorization (FFF) improves the convergence of the RCWA
        algorithm, especially for metallic structures and TM polarization.

        Args:
            use_fff (bool): Whether to use Fast Fourier Factorization.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Enable Fast Fourier Factorization (default)
        builder = SolverBuilder().with_fff(True)
        # Disable Fast Fourier Factorization
        builder = SolverBuilder().with_fff(False)
        ```

        Keywords:
            Fast Fourier Factorization, FFF, convergence, numerical stability
        """
        self._is_use_FFF = use_fff
        return self

    def with_fff_vector_options(
        self,
        *,
        scheme: Optional[str] = None,
        fourier_weight: Optional[float] = None,
        smoothness_weight: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> "SolverBuilder":
        """Configure tangent vector field generation behaviour.

        Args:
            scheme: Tangent field scheme used by Fast Fourier Factorization. Supported
                values mirror :func:`torchrdit.vector.compute_tangent_field`:

                * ``"POL"`` – polarization-preserving tangents normalized by the global maximum.
                * ``"NORMAL"`` – per-pixel unit tangents for geometry-driven factorization.
                * ``"JONES"`` – convert refined tangents into Jones vectors after the solve.
                * ``"JONES_DIRECT"`` – execute the Newton solve directly in Jones space.

                Defaults to the builder's current choice (``"POL"`` for new instances).
            fourier_weight: Weight for the Fourier-domain loss term.
            smoothness_weight: Weight for the spatial smoothness penalty.
            steps: Number of Newton iterations for tangent field refinement.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Raises:
            ValueError: If ``steps`` is provided and less than 1.

        Keywords:
            tangent field, normal vector method, Fourier factorization, options
        """
        if scheme is not None:
            self._fff_vector_scheme = scheme
        if fourier_weight is not None:
            self._fff_fourier_weight = float(fourier_weight)
        if smoothness_weight is not None:
            self._fff_smoothness_weight = float(smoothness_weight)
        if steps is not None:
            steps_int = int(steps)
            if steps_int < 1:
                raise ValueError("Vector field steps must be a positive integer.")
            self._fff_vector_steps = steps_int
        return self

    def with_device(self, device: Union[str, torch.device]) -> "SolverBuilder":
        """Set the computation device (CPU or GPU).

        Args:
            device (Union[str, torch.device]): The device to use for computations,
                either as a string ('cpu', 'cuda', 'cuda:0', etc.) or as a torch.device.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Use CPU for computations
        builder = SolverBuilder().with_device('cpu')
        # Use GPU for computations
        builder = SolverBuilder().with_device('cuda')
        # Use specific GPU
        builder = SolverBuilder().with_device('cuda:1')
        ```

        Keywords:
            device, CPU, GPU, CUDA, computation acceleration
        """
        self._device = device
        return self

    def with_rdit_order(self, order: int) -> "SolverBuilder":
        """Set the R-DIT order for the R-DIT algorithm.
        
        The R-DIT order determines the approximation used in the diffraction
        interface theory. Higher orders provide better accuracy at the cost of
        computational efficiency.
        
        Note:
            This parameter is only used when the R-DIT algorithm is selected.
        
        Args:
            order (int): The order of the R-DIT algorithm (typically 1-10).
                Higher values provide more accurate results.
                
        Returns:
            SolverBuilder: The builder instance for method chaining.
        
        Examples:
        ```python
        # Use R-DIT algorithm with order 8
        from torchrdit.constants import Algorithm
        from torchrdit.builder import SolverBuilder
        builder = SolverBuilder() \\
            .with_algorithm(Algorithm.RDIT) \\
            .with_rdit_order(8)
        ```
        
        Keywords:
            R-DIT, order, approximation, accuracy, performance
        """
        self._rdit_order = order
        return self

    def add_layer(self, layer_config: Dict[str, Any]) -> "SolverBuilder":
        """Add a layer configuration to be added after solver creation.

        Args:
            layer_config (Dict[str, Any]): Configuration dictionary for a layer,
                containing at minimum "material" and "thickness" keys.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Add a homogeneous silicon layer
        builder = SolverBuilder()
        builder.add_layer({
            "material": "Si",
            "thickness": 0.5,
            "is_homogeneous": True
        })
        # Add a non-homogeneous (patterned) layer
        builder.add_layer({
            "material": "SiO2",
            "thickness": 0.2,
            "is_homogeneous": False
        })
        ```

        Keywords:
            layer, material, thickness, homogeneous, patterned
        """
        self._layers.append(layer_config)
        return self

    def from_config(self, config: Union[str, Dict[str, Any]], flip: bool = False) -> "SolverBuilder":
        """Configure the builder from a JSON configuration file or dictionary.

        This method allows loading all solver configuration from a JSON file or
        dictionary, which is useful for storing and sharing complex configurations.

        Args:
            config (Union[str, Dict[str, Any]]): Either a path to a JSON configuration
                file or a configuration dictionary.
            flip (bool, optional): Whether to flip the configuration (reverse layer
                stack and swap transmission/reflection materials). Defaults to False.

        Returns:
            SolverBuilder: The builder instance for method chaining.

        Notes:
            Fast Fourier Factorization (FFF) options in JSON configs use the
            solver-style keys: ``is_use_fff``, ``fff_vector_scheme``,
            ``fff_fourier_weight``, ``fff_smoothness_weight``, and
            ``fff_vector_steps``.

        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        # Load configuration from a JSON file
        builder = SolverBuilder().from_config("config.json")
        # Load and flip configuration (reverse layer stack)
        builder = SolverBuilder().from_config("config.json", flip=True)
        # Load from a dictionary
        config_dict = {
            "algorithm": "RDIT",
            "wavelengths": [1.55],
            "harmonics": [5, 5],
            "is_use_fff": True,
            "fff_vector_scheme": "POL",
            "materials": {...},
            "layers": [...]
        }
        builder = SolverBuilder().from_config(config_dict)
        ```

        Raises:
            ValueError: If unknown configuration keys are detected.

        Keywords:
            configuration, JSON, file loading, serialization, flip
        """

        from .config_io import detect_config_shape, load_config

        # Store original config path if it's a string
        if isinstance(config, (str, os.PathLike)):
            self._config_path = str(config)

        config, config_dir = load_config(config)

        # Apply flip if needed
        if flip:
            config = flip_config(config)

        detect_config_shape(config)  # validates config structure
        # Define valid configuration keys
        valid_keys = {
            "algorithm",
            "precision",
            "wavelengths",
            "length_unit",
            "grids",
            "harmonics",
            "maxG",
            "lattice_vectors",
            "use_fff",
            "is_use_fff",
            "fff_vector_scheme",
            "fff_fourier_weight",
            "fff_smoothness_weight",
            "fff_vector_steps",
            "device",
            "rdit_order",
            "materials",
            "layers",
            "trn_material",
            "ref_material",
            "sources",
            "vars",
            "output",
        }

        # Create case-insensitive mapping of keys
        case_insensitive_config = {}
        for key, value in config.items():
            case_insensitive_config[key.lower()] = (key, value)

        if "harmonics" in case_insensitive_config:
            harmonics_value = case_insensitive_config["harmonics"][1]
            if isinstance(harmonics_value, str) and harmonics_value.lower() == "auto":
                raise ValueError("harmonics='auto' is not supported")


        # Check for unknown keys (case insensitive)
        unknown_keys = set()
        for key in case_insensitive_config:
            if key not in {valid_key.lower() for valid_key in valid_keys}:
                unknown_keys.add(case_insensitive_config[key][0])  # Add original key to unknown keys

        if unknown_keys:
            raise ValueError(
                f"Unknown configuration keys detected: {unknown_keys}. "
                f"Valid keys are: {valid_keys}. Use grids/harmonics for dimensions."
            )

        # Extract parameters from config (case insensitive)
        if "algorithm" in case_insensitive_config:
            algo_value = case_insensitive_config["algorithm"][1]
            if isinstance(algo_value, Algorithm):
                self._algorithm_type = algo_value
            elif isinstance(algo_value, str):
                self._algorithm_type = Algorithm[algo_value.upper()]
            else:
                raise ValueError(f"algorithm must be a string or Algorithm, got {type(algo_value)}")
        if "precision" in case_insensitive_config:
            prec_value = case_insensitive_config["precision"][1]
            if isinstance(prec_value, Precision):
                self._precision = prec_value
            elif isinstance(prec_value, str):
                self._precision = Precision[prec_value.upper()]
            else:
                raise ValueError(f"precision must be a string or Precision, got {type(prec_value)}")
        if "wavelengths" in case_insensitive_config:
            self._lam0 = np.array(case_insensitive_config["wavelengths"][1])
        if "length_unit" in case_insensitive_config:
            self._lengthunit = case_insensitive_config["length_unit"][1]
        if "grids" in case_insensitive_config:
            self._grids = case_insensitive_config["grids"][1]
        if "harmonics" in case_insensitive_config:
            self._harmonics = case_insensitive_config["harmonics"][1]
        if "lattice_vectors" in case_insensitive_config:
            lattice_vectors = case_insensitive_config["lattice_vectors"][1]
            t1_data = lattice_vectors["t1"]
            t2_data = lattice_vectors["t2"]
            self._t1 = t1_data if isinstance(t1_data, torch.Tensor) else torch.as_tensor(t1_data)
            self._t2 = t2_data if isinstance(t2_data, torch.Tensor) else torch.as_tensor(t2_data)
        if "is_use_fff" in case_insensitive_config:
            self._is_use_FFF = case_insensitive_config["is_use_fff"][1]
        elif "use_fff" in case_insensitive_config:
            self._is_use_FFF = case_insensitive_config["use_fff"][1]
        if "device" in case_insensitive_config:
            self._device = case_insensitive_config["device"][1]
        if "rdit_order" in case_insensitive_config:
            self._rdit_order = case_insensitive_config["rdit_order"][1]

        fff_vector_opts = {}
        if "fff_vector_scheme" in case_insensitive_config:
            fff_vector_opts["scheme"] = case_insensitive_config["fff_vector_scheme"][1]
        if "fff_fourier_weight" in case_insensitive_config:
            fff_vector_opts["fourier_weight"] = case_insensitive_config["fff_fourier_weight"][1]
        if "fff_smoothness_weight" in case_insensitive_config:
            fff_vector_opts["smoothness_weight"] = case_insensitive_config["fff_smoothness_weight"][1]
        if "fff_vector_steps" in case_insensitive_config:
            fff_vector_opts["steps"] = case_insensitive_config["fff_vector_steps"][1]

        if fff_vector_opts:
            self.with_fff_vector_options(**fff_vector_opts)

        # Process materials and layers
        if config_dir is None and getattr(self, "_config_path", None):
            try:
                config_dir = str(Path(self._config_path).resolve().parent)
            except OSError:
                config_dir = None
        if "materials" in case_insensitive_config:
            materials_dict = _create_materials(case_insensitive_config["materials"][1], config_dir=config_dir)
            self._materials = list(materials_dict.values())
            self._materials_dict = materials_dict

        if "layers" in case_insensitive_config:
            self._layers = case_insensitive_config["layers"][1]

        # Handle transmission and reflection material references
        # These will be processed after the solver is built
        if "trn_material" in case_insensitive_config:
            self._trn_material = case_insensitive_config["trn_material"][1]
        if "ref_material" in case_insensitive_config:
            self._ref_material = case_insensitive_config["ref_material"][1]

        return self

    def build(self):
        """Build and return the configured solver instance.
        
        This method creates a solver instance based on all the configuration
        parameters that have been set on the builder.
        
        Returns:
            Union[RCWASolver, RDITSolver]: The configured solver instance.
        
        Examples:
        ```python
        from torchrdit.builder import SolverBuilder
        from torchrdit.constants import Algorithm
        # Build a basic RCWA solver
        solver = SolverBuilder().with_algorithm(Algorithm.RCWA).build()
        # Build a fully configured solver
        solver = SolverBuilder() \\
            .with_algorithm(Algorithm.RDIT) \\
            .with_rdit_order(8) \\
            .with_wavelengths([1.3, 1.4, 1.5, 1.6]) \\
            .with_k_dimensions([5, 5]) \\
            .with_device("cuda") \\
            .build()
        ```
        
        Keywords:
            build, create solver, instantiate, solver instance
        """
        # Lazy import to avoid circular dependencies
        from .solver import RCWASolver, RDITSolver

        # Create the appropriate solver type based on algorithm
        if self._algorithm_type == Algorithm.RCWA:
            solver = RCWASolver(
                lam0=self._lam0,
                lengthunit=self._lengthunit,
                grids=self._grids,
                harmonics=self._harmonics,
                materiallist=self._materials,
                t1=self._t1,
                t2=self._t2,
                is_use_FFF=self._is_use_FFF,
                fff_vector_scheme=self._fff_vector_scheme,
                fff_fourier_weight=self._fff_fourier_weight,
                fff_smoothness_weight=self._fff_smoothness_weight,
                fff_vector_steps=self._fff_vector_steps,
                precision=self._precision,
                device=self._device,
            )

            # If custom algorithm instance provided, override the default
            if self._algorithm_instance:
                solver.algorithm = self._algorithm_instance

        else:  # RDIT
            solver = RDITSolver(
                lam0=self._lam0,
                lengthunit=self._lengthunit,
                grids=self._grids,
                harmonics=self._harmonics,
                materiallist=self._materials,
                t1=self._t1,
                t2=self._t2,
                is_use_FFF=self._is_use_FFF,
                fff_vector_scheme=self._fff_vector_scheme,
                fff_fourier_weight=self._fff_fourier_weight,
                fff_smoothness_weight=self._fff_smoothness_weight,
                fff_vector_steps=self._fff_vector_steps,
                precision=self._precision,
                device=self._device,
            )

            # If custom algorithm instance provided, override the default
            if self._algorithm_instance:
                solver.algorithm = self._algorithm_instance

            # Set RDIT order if specified
            if self._rdit_order is not None:
                solver.set_rdit_order(self._rdit_order)

        # Add layers if specified
        if self._layers:
            if not self._materials_dict:
                has_str_material = any(isinstance(layer.get("material"), str) for layer in self._layers)
                if has_str_material:
                    raise ValueError("layers specified but no materials provided")
            _add_layers(solver, self._layers, self._materials_dict)

        # Set transmission and reflection materials if specified
        if hasattr(self, "_trn_material") and self._trn_material is not None and hasattr(solver, "update_trn_material"):
            solver.update_trn_material(self._materials_dict.get(self._trn_material, self._trn_material))

        if hasattr(self, "_ref_material") and self._ref_material is not None and hasattr(solver, "update_ref_material"):
            solver.update_ref_material(self._materials_dict.get(self._ref_material, self._ref_material))

        return solver
