import numpy as np
import warnings
import matplotlib.pyplot as plt

from .constants import *


class materials():
    """Class of materials used in the RCWA solver
    """

    def __init__(self,
                 name: str = 'material1',  # Name of the material
                 permittivity: float = 1.0,  # relative permittivity
                 permeability: float = 1.0,  # relative permeability
                 # true or false if the permittivity of material is dispersive
                 dielectric_dispersion: bool = False,
                 # path of the user-defined dispersive dielectric data,
                 user_dielectric_file: str = None,
                 data_format: str = 'freq-eps',  # format of the user-difined data
                 data_unit: str = 'thz',  # unit of frequency or wavelength
                 max_poly_fit_order: int = 10,  # max polynomial fit order
                 ) -> None:

        self._name = name
        self._data_format = data_format
        self._data_unit = data_unit
        self._max_poly_order = max_poly_fit_order

        if dielectric_dispersion == False:
            self._isdiedispersive = False
            self._er = permittivity
        else:
            self._isdiedispersive = True
            if user_dielectric_file == None:
                raise ValueError(
                    'File path of the dispersive data must be defined!')
            else:
                self._loadeder = np.loadtxt(user_dielectric_file)
                self._er = None

        self._ur = permeability

    @property
    def isdispersive_er(self):
        return self._isdiedispersive

    @property
    def er(self):
        return self._er

    @property
    def ur(self):
        return self._ur

    @property
    def name(self):
        return self._name

    def _extract_laoded_data(self,
                             lengthunit: str = 'um',  # length unit used in the solver
                             ):
        """_extract_laoded_data.

        Extracts the loaded permittivity profile.

        Args:
            lengthunit (str): lengthunit
        """
        disp_er = self._loadeder.copy()

        if self._data_format == 'freq-eps':
            disp_er[:, 0] = C_0 / (disp_er[:, 0] *
                                   frequnit_dict[self._data_unit]) / lengthunit_dict[lengthunit]
        elif self._data_format == 'wl-eps':
            disp_er[:, 0] = disp_er[:, 0] * \
                lengthunit_dict[self._data_unit] / lengthunit_dict[lengthunit]
        elif self._data_format == 'freq-nk':
            disp_er[:, 0] = C_0 / (disp_er[:, 0] *
                                   frequnit_dict[self._data_unit]) / lengthunit_dict[lengthunit]
            data_n = disp_er[:, 1]
            data_k = disp_er[:, 2]
            complex_permittivity = (data_n + 1j*data_k) ** 2
            disp_er[:, 1] = np.real(complex_permittivity)
            disp_er[:, 2] = np.imag(complex_permittivity)
        elif self._data_format == 'wl-nk':
            disp_er[:, 0] = disp_er[:, 0] * \
                lengthunit_dict[self._data_unit] / lengthunit_dict[lengthunit]
            data_n = disp_er[:, 1]
            data_k = disp_er[:, 2]
            complex_permittivity = (data_n + 1j*data_k) ** 2
            disp_er[:, 1] = np.real(complex_permittivity)
            disp_er[:, 2] = np.imag(complex_permittivity)

        # Sort the wavelengths
        if (disp_er[0, 0] > disp_er[-1, 0]):
            disp_er_sorted = disp_er[::-1, :]
        else:
            disp_er_sorted = disp_er

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

        disp_er_sorted = self._extract_laoded_data(lengthunit=lengthunit)

        wl_list = disp_er_sorted[:, 0]

        lam_min = np.min(lam0)
        lam_max = np.max(lam0)

        if (lam_min < np.min(wl_list) or (lam_max > np.max(wl_list))):
            raise ValueError(
                f"Required frequencies of the material [{self.name}] are out of range!")

        def wl_inds(x, array): return np.abs(array - x).argmin()
        wl_ind1, wl_ind2 = tuple(
            map(wl_inds, (lam_min, lam_max), (wl_list, wl_list)))
        wl_ind2 = wl_ind2 + 1

        wls = wl_list[wl_ind1: wl_ind2]
        data_eps1 = disp_er_sorted[wl_ind1:wl_ind2, 1]
        data_eps2 = disp_er_sorted[wl_ind1:wl_ind2, 2]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            ze1 = np.polyfit(
                wls, data_eps1, self._max_poly_order)
            ze2 = np.polyfit(
                wls, data_eps2, self._max_poly_order)

        pe1 = np.poly1d(ze1)
        pe2 = np.poly1d(ze2)

        # Save data
        self._fitted_data = dict()
        self._fitted_data['wavelengths'] = wls
        self._fitted_data['data_eps1'] = data_eps1
        self._fitted_data['data_eps2'] = data_eps2
        self._fitted_data['fitted_crv1'] = pe1
        self._fitted_data['fitted_crv2'] = pe2

        self._er = pe1(lam0) - 1j*pe2(lam0)

    def display_dispersive_profile(self,
                                   lengthunit: str = 'um',  # length unit used in the solver
                                   ):
        """display_dispersive_profile.

        Plot the original and fitted dispersive profile.

        Args:
            lengthunit (str): lengthunit
        """

        if self._isdiedispersive == True:
            disp_er_sorted = self._extract_laoded_data(lengthunit=lengthunit)
            fig, ax = plt.subplots()
            ln1 = ax.plot(disp_er_sorted[:, 0],
                          disp_er_sorted[:, 1], 'r-', label='e\'')
            # ax.legend(loc='best')
            ax2 = ax.twinx()
            ln2 = ax2.plot(disp_er_sorted[:, 0],
                           disp_er_sorted[:, 2], 'g--', label='e\"')
            # ax2.legend(loc='best')
            ax.set_title(f"Permittivity [{self._name}]")
            ax.set_xlabel(f"Wavelength [{lengthunit}]")
            ax.set_ylabel('Eps\' [Real Part]')
            ax2.set_ylabel('Eps\" [Imag Part]')
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='best')

            return fig, ax
        else:
            print(f"The material [{self._name}] has no dispersive attribute.")
            return 0
