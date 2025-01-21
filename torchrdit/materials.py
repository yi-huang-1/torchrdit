""" This file defines the material class to manage all materials. """
import warnings
import numpy as np
import torch

from .constants import frequnit_dict, lengthunit_dict, C_0


class MaterialClass():
    """Class of materials used in the RCWA solver
    """

    def __init__(self,
                 name: str = 'material1',  # Name of the material
                 permittivity: float = 1.0,  # relative permittivity
                 permeability: float = 1.0,  # relative permeability
                 # true or false if the permittivity of material is dispersive
                 dielectric_dispersion: bool = False,
                 # path of the user-defined dispersive dielectric data,
                 user_dielectric_file: str = '',
                 data_format: str = 'freq-eps',  # format of the user-difined data
                 data_unit: str = 'thz',  # unit of frequency or wavelength
                 max_poly_fit_order: int = 10,  # max polynomial fit order
                 ) -> None:
        """
        Initialize the MaterialClass instance.

        Args:
            name (str): Name of the material.
            permittivity (float): Relative permittivity.
            permeability (float): Relative permeability.
            dielectric_dispersion (bool): If the permittivity of material is dispersive.
            user_dielectric_file (str): Path of the user-defined dispersive dielectric data.
            data_format (str): Format of the user-defined data.
            data_unit (str): Unit of frequency or wavelength.
            max_poly_fit_order (int): Max polynomial fit order.
        """
        self._name = name
        self._data_format = data_format
        self._data_unit = data_unit
        self._max_poly_order = max_poly_fit_order

        self._fitted_data = {}

        if dielectric_dispersion is False:
            self._isdiedispersive = False
            self._loadeder = None
            self._er = torch.tensor(permittivity)
        else:
            self._isdiedispersive = True
            if user_dielectric_file is not None:
                skip_header = self._check_header(user_dielectric_file)
                self._loadeder = np.loadtxt(user_dielectric_file, skiprows=1 if skip_header else 0).astype(np.float32)
                self._er = None
            else:
                raise ValueError(
                    'File path of the dispersive data must be defined!')

        self._ur = torch.tensor(permeability)

    @staticmethod
    def _check_header(filename):
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            # Check if the first line contains non-numeric characters
            return not all(char.isdigit() or char in '.- ' for char in first_line)

    @property
    def isdispersive_er(self):
        """isdispersive_er.
        returns whether material is dispersive.
        """
        return self._isdiedispersive

    @property
    def er(self):
        """er.
        returns permittivity of the material.
        """
        return self._er

    @property
    def ur(self):
        """ur.
        returns permeability of the meterial.
        """
        return self._ur

    @property
    def name(self):
        """name.
        returns material name.
        """
        return self._name

    @property
    def fitted_data(self):
        """fitted_data.
        returns fitted profile.
        """
        return self._fitted_data

    def _extract_laoded_data(self,
                             lengthunit: str = 'um',  # length unit used in the solver
                             ):
        """_extract_laoded_data.

        Extracts the loaded permittivity profile.

        Args:
            lengthunit (str): lengthunit
        """
        if self._loadeder is None:
            return None

        disp_er = self._loadeder.copy()
        calculated_er = np.zeros((disp_er.shape[0], 3))

        if self._data_format == 'freq-eps':
            calculated_er[:, 0] = C_0 / (disp_er[:, 0] *
                                   frequnit_dict[self._data_unit]) / lengthunit_dict[lengthunit]
            calculated_er[:, 1] = disp_er[:, 1]
            if disp_er.shape[1] > 2:
                calculated_er[:, 2] = disp_er[:, 2]
            else:
                calculated_er[:, 2] = np.zeros_like(disp_er[:, 0])
        elif self._data_format == 'wl-eps':
            calculated_er[:, 0] = disp_er[:, 0] * \
                lengthunit_dict[self._data_unit] / lengthunit_dict[lengthunit]
            calculated_er[:, 1] = disp_er[:, 1]
            if disp_er.shape[1] > 2:
                calculated_er[:, 2] = disp_er[:, 2]
            else:
                calculated_er[:, 2] = np.zeros_like(disp_er[:, 0])
        elif self._data_format == 'freq-nk':
            calculated_er[:, 0] = C_0 / (disp_er[:, 0] *
                                   frequnit_dict[self._data_unit]) / lengthunit_dict[lengthunit]
            data_n = disp_er[:, 1]
            if disp_er.shape[1] > 2:
                data_k = disp_er[:, 2]
            else:
                data_k = np.zeros_like(disp_er[:, 0])

            complex_permittivity = (data_n + 1j*data_k) ** 2
            calculated_er[:, 1] = np.real(complex_permittivity)
            calculated_er[:, 2] = np.imag(complex_permittivity)
        elif self._data_format == 'wl-nk':
            calculated_er[:, 0] = disp_er[:, 0] * \
                lengthunit_dict[self._data_unit] / lengthunit_dict[lengthunit]
            data_n = disp_er[:, 1]
            if disp_er.shape[1] > 2:
                data_k = disp_er[:, 2]
            else:
                data_k = np.zeros_like(disp_er[:, 0])

            complex_permittivity = (data_n + 1j*data_k) ** 2
            calculated_er[:, 1] = np.real(complex_permittivity)
            calculated_er[:, 2] = np.imag(complex_permittivity)

        # Sort the wavelengths
        if (calculated_er[0, 0] > calculated_er[-1, 0]):
            disp_er_sorted = calculated_er[::-1, :]
        else:
            disp_er_sorted =calculated_er 

        return disp_er_sorted

    def load_dispersive_er(self,
                           lam0: np.ndarray,
                           lengthunit: str = 'um',  # length unit used in the solver
                           ) -> None:
        """load_dispersive_er.

        Laod the dispersive profile and fit with the frequencies to be simulated.

        Args:
            lam0 (np.ndarray): lam0
            lengthunit (str): lengthunit

        Returns:
            None:
        """
        min_ref_pts = 6

        disp_er_sorted = self._extract_laoded_data(lengthunit=lengthunit)

        wl_list = disp_er_sorted[:, 0]

        lam_min = np.min(lam0)
        lam_max = np.max(lam0)

        if (lam_min < np.min(wl_list) or (lam_max > np.max(wl_list))):
            raise ValueError(
                f"Required frequencies of the material [{self.name}] are out of range [{np.min(wl_list):.2f}, {np.max(wl_list):.2f}]!")

        def wl_inds(ind_x, array):
            return np.abs(array - ind_x).argmin()

        wl_ind1, wl_ind2 = tuple(
            map(wl_inds, (lam_min, lam_max), (wl_list, wl_list)))
        wl_ind2 = wl_ind2 + 1
        if wl_ind2 - wl_ind1 < min_ref_pts:
            lft = min_ref_pts // 2
            rft = min_ref_pts - lft
            wl_ind1 = wl_ind1 - lft
            wl_ind2 = wl_ind2 + rft

        wls = wl_list[wl_ind1: wl_ind2]
        data_eps1 = disp_er_sorted[wl_ind1:wl_ind2, 1]
        data_eps2 = disp_er_sorted[wl_ind1:wl_ind2, 2]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.exceptions.RankWarning)
            ze1 = np.polyfit(
                wls, data_eps1, self._max_poly_order)
            ze2 = np.polyfit(
                wls, data_eps2, self._max_poly_order)

        pe1 = np.poly1d(ze1)
        pe2 = np.poly1d(ze2)

        # Save data
        self._fitted_data['wavelengths'] = wls
        self._fitted_data['data_eps1'] = data_eps1
        self._fitted_data['data_eps2'] = data_eps2
        self._fitted_data['fitted_crv1'] = pe1
        self._fitted_data['fitted_crv2'] = pe2

        self._er = torch.tensor(pe1(lam0) - 1j*pe2(lam0))

