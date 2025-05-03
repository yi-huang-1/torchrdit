"""Module for defining and managing material properties in TorchRDIT electromagnetic simulations.

This module provides classes and utilities for creating, managing, and manipulating
material properties used in electromagnetic simulations with TorchRDIT. It handles
both dispersive (wavelength/frequency dependent) and non-dispersive materials with
a unified interface.

The material system is designed with a proxy pattern, separating material property
definitions from their data loading and processing. This allows for efficient handling
of different material data formats and sources, including:
- Direct permittivity/permeability specification
- Refractive index and extinction coefficient (n, k) specification
- Loading from data files with different formats (freq-eps, wl-eps, freq-nk, wl-nk)

Classes:
    MaterialClass: Main class for representing materials with their electromagnetic properties.

Functions:
    (No module-level functions, but MaterialClass provides factory methods)

Examples:
```python
from torchrdit.utils import create_material
# Create a simple material with constant permittivity
air = create_material(name="air", permittivity=1.0)
silicon = create_material(name="silicon", permittivity=11.7)
# Create a material from refractive index
glass = create_material(name="glass", permittivity=2.25)  # n=1.5
# Create a material with complex permittivity (lossy)
gold = create_material(name="gold", permittivity=complex(-10.0, 1.5))

# Create dispersive material from data file with wavelength-permittivity data
silica = create_material(
    name="silica",
    dielectric_dispersion=True,
    user_dielectric_file="materials/SiO2.txt",
    data_format="wl-eps",
    data_unit="um"
)

# Create dispersive material from data file with frequency-nk data
silicon_dispersive = create_material(
    name="silicon_disp",
    dielectric_dispersion=True,
    user_dielectric_file="materials/Si.txt",
    data_format="freq-nk",
    data_unit="thz"
)

# Using materials in solvers
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import torch
# Create a solver and add materials
solver = create_solver(algorithm=Algorithm.RCWA)
solver.add_materials([air, silicon, glass])
# Add layers using these materials
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
solver.add_layer(material_name="glass", thickness=torch.tensor(0.1))
# Set input/output materials
solver.update_ref_material("air")
solver.update_trn_material("air")
```

Keywords:
    materials, permittivity, permeability, optical properties, dispersive materials,
    refractive index, extinction coefficient, material data, complex permittivity,
    electromagnetic properties, dielectric function, optical constants, wavelength-dependent
"""
import os
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch

from .constants import frequnit_dict, lengthunit_dict, C_0
from .material_proxy import MaterialDataProxy, UnitConverter


class MaterialClass:
    """Class for representing materials and their electromagnetic properties in TorchRDIT.
    
    This class implements a comprehensive representation of materials used in electromagnetic
    simulations, supporting both dispersive (wavelength/frequency dependent) and
    non-dispersive (constant) material properties. It provides a unified interface
    for managing permittivity and permeability data regardless of the source format.
    
    MaterialClass uses a proxy pattern for handling material data loading and processing,
    allowing it to support multiple data formats and unit systems. For dispersive
    materials, it can load data from files and perform polynomial fitting to interpolate
    property values at specific wavelengths needed for simulations.
    
    Attributes:
        name (str): Name identifier for the material.
        er (torch.Tensor): Complex permittivity tensor.
        ur (torch.Tensor): Permeability tensor.
        isdispersive_er (bool): Whether the material has wavelength/frequency-dependent permittivity.
        data_format (str): Format of the loaded data file for dispersive materials.
        data_unit (str): Unit used in the data file for dispersive materials.
        fitted_data (Dict[str, Any]): Fitted profile data for dispersive materials.
    
    Note:
        Users typically create MaterialClass instances through the `create_material` 
        function rather than directly instantiating this class.
    
    Examples:
    ```python
    from torchrdit.utils import create_material
    # Material with constant permittivity
    silicon = create_material(name="silicon", permittivity=11.7)
    print(f"Material: {silicon.name}, ε = {silicon.er.real:.1f}")
    # Material: silicon, ε = 11.7
    # Material with complex permittivity (lossy)
    gold = create_material(name="gold", permittivity=complex(-10.0, 1.5))
    print(f"Material: {gold.name}, ε = {gold.er.real:.1f}{gold.er.imag:+.1f}j")
    # Material: gold, ε = -10.0+1.5j
    
    # Load permittivity data from wavelength-permittivity data file
    silica = create_material(
        name="silica",
        dielectric_dispersion=True,
        user_dielectric_file="materials/SiO2.txt",
        data_format="wl-eps",
        data_unit="um"
    )
    # Use in a simulation with specific wavelengths
    import numpy as np
    wavelengths = np.array([1.31, 1.55])
    permittivity = silica.get_permittivity(wavelengths, 'um')
    print(f"Silica permittivity at λ=1.55 μm: {permittivity[1].real:.4f}")
    # Silica permittivity at λ=1.55 μm: 2.1521
    ```
    
    Keywords:
        material properties, permittivity, permeability, optical constants,
        dispersive materials, refractive index, wavelength-dependent properties,
        material data, complex permittivity, dielectric function
    """
    
    # Class-level shared proxy for efficiency
    _shared_proxy = MaterialDataProxy()
    
    # Supported data formats
    _SUPPORTED_FORMATS = ['freq-eps', 'wl-eps', 'freq-nk', 'wl-nk']
    
    def __init__(self,
                 name: str = 'material1',
                 permittivity: float = 1.0,
                 permeability: float = 1.0,
                 dielectric_dispersion: bool = False,
                 user_dielectric_file: str = '',
                 data_format: str = 'freq-eps',
                 data_unit: str = 'thz',
                 max_poly_fit_order: int = 10,
                 data_proxy: Optional[MaterialDataProxy] = None
                 ) -> None:
        """Initialize a MaterialClass instance with electromagnetic properties.

        Creates a new material with specified properties. For non-dispersive materials,
        only name, permittivity, and permeability need to be specified. For dispersive
        materials (with wavelength/frequency-dependent properties), additional parameters
        are required to specify the data source and format.

        Args:
            name: Unique identifier for the material. Used when referencing the material
                 in solvers and layer definitions.
            permittivity: Relative permittivity (εr) for non-dispersive materials.
                         Can be a real or complex value to represent lossless or lossy materials.
                         This value is ignored for dispersive materials.
            permeability: Relative permeability (μr) of the material. Default is 1.0
                         (non-magnetic material).
            dielectric_dispersion: Whether the material has wavelength/frequency-dependent
                                  permittivity. If True, a data file must be provided.
            user_dielectric_file: Path to the data file containing the dispersive properties.
                                 Required if dielectric_dispersion is True.
            data_format: Format of the data in the file. Must be one of:
                        'freq-eps': Frequency and permittivity (real, imaginary)
                        'wl-eps': Wavelength and permittivity (real, imaginary)
                        'freq-nk': Frequency and refractive index (n, k)
                        'wl-nk': Wavelength and refractive index (n, k)
            data_unit: Unit of the frequency or wavelength in the data file (e.g., 'thz', 'um').
            max_poly_fit_order: Maximum polynomial order for fitting dispersive data.
                               Higher values provide more accurate fits for complex
                               dispersion curves but may lead to overfitting.
            data_proxy: Custom data proxy instance for handling material data loading.
                       Uses the shared class-level proxy if None.
        
        Raises:
            ValueError: If dispersive material is missing a data file,
                       if the data file doesn't exist, or if the data format is invalid.
                       
        Examples:
        ```python
        from torchrdit.materials import MaterialClass
        # Create a simple air material
        air = MaterialClass(name="air", permittivity=1.0)
        
        # Create a lossy metal with complex permittivity
        gold = MaterialClass(
            name="gold", 
            permittivity=complex(-10.0, 1.5)
        )
        
        # Create a dispersive material from data file
        silica = MaterialClass(
            name="silica",
            dielectric_dispersion=True,
            user_dielectric_file="materials/SiO2.txt",
            data_format="wl-eps",
            data_unit="um"
        )
        ```
        
        Keywords:
            material creation, permittivity, permeability, dispersive material,
            optical constants, electromagnetic properties, material initialization
        """
        self._name = name
        self._data_format = data_format.lower()
        self._data_unit = data_unit.lower()
        self._max_poly_order = max_poly_fit_order
        self._perm_cache: Dict[Tuple[Tuple[float, ...], str], torch.Tensor] = {}

        # Validate data format if dispersive
        if dielectric_dispersion and self._data_format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported data format: {data_format}. " 
                f"Expected one of: {', '.join(self._SUPPORTED_FORMATS)}")

        # Use the provided proxy or the shared class proxy
        self._data_proxy = data_proxy or self._shared_proxy
        self._fitted_data: Dict[str, Any] = {}

        if not dielectric_dispersion:
            # Non-dispersive material
            self._isdiedispersive = False
            self._loadeder = None
            self._er = torch.tensor(permittivity, dtype=torch.complex64)
        else:
            # Dispersive material
            self._isdiedispersive = True
            
            # Validate data file
            if not user_dielectric_file:
                raise ValueError('File path of the dispersive data must be defined!')
                
            if not os.path.exists(user_dielectric_file):
                raise ValueError(f"Material data file not found: {user_dielectric_file}")
                
            # Load the data using the proxy
            try:
                self._loadeder = self._data_proxy.load_data(
                    user_dielectric_file, self._data_format, self._data_unit, 'um')
                self._er = None
            except ValueError as e:
                raise ValueError(f"Error loading dispersive data for {name}: {str(e)}")

        self._ur = torch.tensor(permeability, dtype=torch.float32)

    @property
    def isdispersive_er(self) -> bool:
        """Whether the material has dispersive permittivity."""
        return self._isdiedispersive

    @property
    def er(self) -> torch.Tensor:
        """Permittivity tensor of the material."""
        return self._er

    @property
    def ur(self) -> torch.Tensor:
        """Permeability tensor of the material."""
        return self._ur

    @property
    def name(self) -> str:
        """Name of the material."""
        return self._name

    @property
    def fitted_data(self) -> Dict[str, Any]:
        """Fitted profile data for dispersive materials."""
        return self._fitted_data
        
    @property
    def data_format(self) -> str:
        """Format of the material data."""
        return self._data_format
        
    @property
    def data_unit(self) -> str:
        """Unit of the material data."""
        return self._data_unit
        
    @property
    def supported_formats(self) -> List[str]:
        """List of supported data formats."""
        return self._SUPPORTED_FORMATS.copy()

    def get_permittivity(self, 
                         wavelengths: np.ndarray, 
                         wavelength_unit: str = 'um') -> torch.Tensor:
        """
        Get the material's permittivity at specified wavelengths.
        
        This is a standardized interface that handles both dispersive and 
        non-dispersive materials.
        
        Args:
            wavelengths: Array of wavelengths to calculate permittivity at
            wavelength_unit: Unit of the provided wavelengths
            
        Returns:
            Permittivity tensor at the specified wavelengths
            
        Raises:
            ValueError: If the wavelengths are out of the available data range
        """
        if not self._isdiedispersive:
            return self._er
            
        # Load dispersive data
        self.load_dispersive_er(wavelengths, wavelength_unit)
        return self._er

    def load_dispersive_er(self,
                           lam0: np.ndarray,
                           lengthunit: str = 'um') -> None:
        """
        Load the dispersive profile and fit with the wavelengths to be simulated.

        Args:
            lam0: Wavelengths to simulate
            lengthunit: Length unit used in the solver

        Raises:
            ValueError: If wavelengths are out of the available data range
        """
        if not self._isdiedispersive or self._loadeder is None:
            return
            
        # Check cache for previously calculated permittivity
        cache_key = (tuple(lam0), lengthunit)
        if cache_key in self._perm_cache:
            self._er = self._perm_cache[cache_key]
            return
            
        min_ref_pts = 6
        
        # Convert simulation wavelengths to um (internal proxy unit)
        converter = UnitConverter()
        sim_wavelengths = converter.convert_length(lam0, lengthunit, 'um')
            
        # Get processed data from the proxy
        wl_list = self._loadeder[:, 0]  # Wavelengths in um
        
        lam_min = np.min(sim_wavelengths)
        lam_max = np.max(sim_wavelengths)
        
        # Check if simulation wavelengths are in range
        if lam_min < np.min(wl_list) or lam_max > np.max(wl_list):
            raise ValueError(
                f"Required wavelengths for material [{self.name}] are out of range "
                f"[{np.min(wl_list):.2f}um, {np.max(wl_list):.2f}um]!")
        
        # Find indices for the data range we need
        def wl_inds(ind_x: float, array: np.ndarray) -> int:
            return np.abs(array - ind_x).argmin()

        wl_ind1, wl_ind2 = tuple(
            map(wl_inds, (lam_min, lam_max), (wl_list, wl_list)))
        wl_ind2 = wl_ind2 + 1
        
        # Ensure enough reference points
        if wl_ind2 - wl_ind1 < min_ref_pts:
            lft = min_ref_pts // 2
            rft = min_ref_pts - lft
            wl_ind1 = max(0, wl_ind1 - lft)
            wl_ind2 = min(len(wl_list), wl_ind2 + rft)

        # Extract data subset (using a view, not a copy)
        filtered_data = self._loadeder[wl_ind1:wl_ind2, :]
        
        # Use the proxy to extract permittivity at simulation wavelengths
        eps_real, eps_imag = self._data_proxy.extract_permittivity(
            filtered_data, sim_wavelengths, fit_order=self._max_poly_order)
        
        # Store the fitted data for reference
        self._fitted_data = {
            'wavelengths': filtered_data[:, 0],
            'data_eps1': filtered_data[:, 1],
            'data_eps2': filtered_data[:, 2],
            'fitted_eps1': eps_real,
            'fitted_eps2': eps_imag
        }
        
        # Set the permittivity tensor
        self._er = torch.tensor(eps_real(sim_wavelengths) - 1j*eps_imag(sim_wavelengths))
        
        # Cache the result
        self._perm_cache[cache_key] = self._er

    @classmethod
    def from_nk_data(cls, 
                    name: str, 
                    n: float, 
                    k: float = 0.0, 
                    permeability: float = 1.0) -> 'MaterialClass':
        """
        Create a material from refractive index and extinction coefficient.
        
        Args:
            name: Name of the material
            n: Refractive index
            k: Extinction coefficient
            permeability: Relative permeability
            
        Returns:
            Instantiated MaterialClass with permittivity derived from n and k
        """
        # Calculate complex permittivity from n and k
        permittivity = (n**2 - k**2) + 1j*(2*n*k)
        
        return cls(
            name=name, 
            permittivity=permittivity.real,  # We use the real part for initialization
            permeability=permeability,
            dielectric_dispersion=False
        )

    @classmethod
    def from_data_file(cls, 
                      name: str, 
                      file_path: str, 
                      data_format: str = 'wl-eps', 
                      data_unit: str = 'um',
                      permeability: float = 1.0,
                      max_poly_fit_order: int = 10) -> 'MaterialClass':
        """
        Create a dispersive material from a data file.
        
        Args:
            name: Name of the material
            file_path: Path to the data file
            data_format: Format of the data file ('freq-eps', 'wl-eps', etc.)
            data_unit: Unit of the data in the file ('thz', 'um', etc.)
            permeability: Relative permeability
            max_poly_fit_order: Maximum polynomial order for fitting
            
        Returns:
            Instantiated dispersive MaterialClass
            
        Raises:
            ValueError: If the file doesn't exist or has an invalid format
        """
        return cls(
            name=name, 
            dielectric_dispersion=True, 
            user_dielectric_file=file_path,
            data_format=data_format, 
            data_unit=data_unit,
            permeability=permeability,
            max_poly_fit_order=max_poly_fit_order
        )

    def clear_cache(self) -> None:
        """Clear the permittivity cache to free memory."""
        self._perm_cache.clear()

    def __str__(self) -> str:
        """String representation of the material."""
        if self._isdiedispersive:
            return f"Material {self._name} (dispersive, {self._data_format})"
        else:
            return f"Material {self._name} (ε={self._er.real:.3f})"

    def __repr__(self) -> str:
        """Detailed representation of the material."""
        if self._isdiedispersive:
            return f"MaterialClass(name='{self._name}', dispersive=True, format='{self._data_format}')"
        else:
            return f"MaterialClass(name='{self._name}', permittivity={self._er.real}, permeability={self._ur.item()})"

