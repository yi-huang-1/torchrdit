""" This file defines the material class to manage all materials. """
import os
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch

from .constants import frequnit_dict, lengthunit_dict, C_0
from .material_proxy import MaterialDataProxy, UnitConverter


class MaterialClass:
    """
    Class of materials used in the RCWA solver.
    
    This class implements a proxy pattern for handling material data with different
    units and formats. The material data can be dispersive (wavelength/frequency dependent) 
    or non-dispersive (fixed values).
    
    Attributes:
        name (str): Name of the material
        er (torch.Tensor): Permittivity tensor (complex for dispersive materials)
        ur (torch.Tensor): Permeability tensor
        isdispersive_er (bool): Whether the material has dispersive permittivity
        data_format (str): Format of the material data ('freq-eps', 'wl-eps', etc.)
        data_unit (str): Unit of the material data ('thz', 'um', etc.)
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
        """
        Initialize the MaterialClass instance.

        Args:
            name: Name of the material.
            permittivity: Relative permittivity (used for non-dispersive materials).
            permeability: Relative permeability.
            dielectric_dispersion: If the permittivity of material is dispersive.
            user_dielectric_file: Path of the user-defined dispersive dielectric data.
            data_format: Format of the user-defined data ('freq-eps', 'wl-eps', 'freq-nk', or 'wl-nk').
            data_unit: Unit of frequency or wavelength in the data file.
            max_poly_fit_order: Max polynomial fit order for dispersive data.
            data_proxy: Custom data proxy instance (uses shared proxy if None).
        
        Raises:
            ValueError: If dispersive material is missing a data file,
                       if the data file doesn't exist, or if the data format is invalid.
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
        self._er = torch.tensor(eps_real - 1j*eps_imag)
        
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

