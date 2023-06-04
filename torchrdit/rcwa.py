""" This file defines all operations related to solver computations."""
from typing import Union, Optional

import numpy as np
import torch
# import datetime

from torch.linalg import inv as tinv
from torch.linalg import solve as tsolve
from .cell import Cell3D
from .utils import blockmat2x2, redhstar, EigComplex, init_smatrix



class FourierBaseSover(Cell3D):
    """Base Class of Fourier Domain Solver"""

    def __init__(self,
                 batch_size: int = 1,  # batch size of data set
                 # wavelengths (frequencies) to be solved
                 lam0: np.ndarray = np.array([1.0]),
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: Union[list, None] = None,  # dimensions in real space: (H, W)
                 kdim: Union[list, None] = None,  # dimensions in k space: (kH, kW)
                 materiallist: Union[list, None] = None,  # list of materials
                 t1: Union[list, None] = None,  # lattice vector in real space
                 t2: Union[list, None] = None,  # lattice vector in real space
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(batch_size, lengthunit, rdim,
                         kdim, materiallist, t1, t2, device)

        # Create a task time stamp
        # task_time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Set free space wavelength
        if isinstance(lam0, float):
            self._lam0 = np.array([lam0])
        elif isinstance(lam0, np.ndarray):
            self._lam0 = lam0

        self.n_freqs = len(self._lam0)
        self.kinc = 0
        self.k_0 = 0

        self.src = None
        self.reci_t1 = None
        self.reci_t2 = None
        self.tlam0 = None
        self.mesh_fp = None
        self.mesh_fq = None

        # Initialize materials, will fit the dispersive profile if applicable
        self._init_dispersive_materials()

        self.n_solve = 0
        self.smat_layers = {}

        # solver mode: [eval] [opt]
        self.solver_mode = 'eval'

        # self.solver_logger.info(f"Solver has been initialized.")

    def add_source(self,
                   theta: Union[np.float32, np.float64],
                   phi: Union[np.float32, np.float64],
                   pte: Union[np.float32, np.float64],
                   ptm: Union[np.float32, np.float64],
                   norm_te_dir: str = 'y') -> dict:
        """add_source.

        Creates and returns a source object.

        Args:
            theta (Union[np.float32, np.float64]): theta
            phi (Union[np.float32, np.float64]): phi
            pte (Union[np.float32, np.float64]): pte
            ptm (Union[np.float32, np.float64]): ptm
            norm_te_dir (str): norm_te_dir, choose a direction that of normal incidence.

        Returns:
            dict:
        """
        source_ditc = {}
        source_ditc['theta'] = theta
        source_ditc['phi'] = phi
        source_ditc['pte'] = pte
        source_ditc['ptm'] = ptm
        source_ditc['norm_te_dir'] = norm_te_dir

        return source_ditc

    def _solve_nonhomo_layer(self,
                             layer_index: int,
                             p_mat_i: torch.Tensor,
                             q_mat_i: torch.Tensor,
                             mat_w0: torch.Tensor,
                             mat_v0: torch.Tensor) -> dict:
        """_solve_nonhomo_layer.

        Fucntion of solving non-homogenous layers in the parent class. It needs to be implemented
        in the child classes with concrete algorithm.

        Args:
            layer_index (int): layer_index
            p_mat_i (torch.Tensor): p_mat_i
            q_mat_i (torch.Tensor): q_mat_i
            mat_w0 (torch.Tensor): mat_w0
            mat_v0 (torch.Tensor): mat_v0

        Returns:
            dict:
        """
        raise NotImplementedError("self._solve not implemented.")

    def _solve(self) -> dict:


        # kx_0, ky_0: (n_batches, n_freqs, n_harmonics, kHw)
        kx_0 = self.kinc[:, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - \
            (self.mesh_fp.unsqueeze(0).unsqueeze(0) * \
                self.reci_t1[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                self.mesh_fq.unsqueeze(0).unsqueeze(0) * \
                    self.reci_t2[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / \
                self.k_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-2) + 0*1j
        ky_0 = self.kinc[:, 1].unsqueeze(-1).unsqueeze(-2) - \
            (self.mesh_fp.unsqueeze(0).unsqueeze(0) * \
                self.reci_t1[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                self.mesh_fq.unsqueeze(0).unsqueeze(0) * \
                    self.reci_t2[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / \
                self.k_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-2) + 0*1j


        # Add small perturbations to make sure matrices are inversable
        if torch.any(kx_0 == 0.0):
            kx_0 = kx_0 + 1.0e-13
        if torch.any(ky_0 == 0.0):
            ky_0 = ky_0 + 1.0e-13


        if self.layer_manager.is_ref_dispersive is False:
            kz_ref_0 = torch.conj(torch.sqrt(
                torch.conj(self.ur1)*torch.conj(self.er1) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_ref_0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur1)*torch.conj(self.er1)).unsqueeze(-1).unsqueeze(-2) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))

        if self.layer_manager.is_trn_dispersive is False:
            kz_trn_0 = torch.conj(torch.sqrt(
                torch.conj(self.ur2)*torch.conj(self.er2) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_trn_0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur2)*torch.conj(self.er2)).unsqueeze(-1).unsqueeze(-2) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))


        # Transform to diagnal matrices
        n_harmonics = self.kdim[0] * self.kdim[1]

        ident_mat_k = torch.eye(n_harmonics, dtype=self.tcomplex, device=self.device)

        mat_kx = kx_0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * ident_mat_k
        mat_ky = ky_0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * ident_mat_k
        mat_kz_ref = kz_ref_0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * ident_mat_k
        mat_kz_trn = kz_trn_0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * ident_mat_k


        ident_mat = ident_mat_k.unsqueeze(0).unsqueeze(0).expand(self.n_batches, self.n_freqs, -1, -1)
        zero_mat = torch.zeros(
            size=(self.n_batches, self.n_freqs ,n_harmonics, n_harmonics), dtype=self.tcomplex, device=self.device)

        # Calculate eigen-modes of the gap medium

        mat_kx_ky = mat_kx @ mat_ky
        mat_ky_kx = mat_ky @ mat_kx
        mat_kx_kx = mat_kx @ mat_kx
        mat_ky_ky = mat_ky @ mat_ky

        mat_kz = torch.conj(torch.sqrt(ident_mat - mat_kx_kx - mat_ky_ky))

        ident_mat_kx_ky = ident_mat - mat_kx_ky

        # q_mat = blockmat2x2([[mat_kx @ mat_ky, ident_mat - mat_kx @ mat_kx],
        #                  [mat_ky @ mat_ky - ident_mat, -mat_kx @ mat_ky]])
        q_mat = blockmat2x2([[mat_kx_ky, ident_mat_kx_ky],
                         [- ident_mat_kx_ky, - mat_kx_ky]])

        mat_w0 = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])

        mat_lam = blockmat2x2([[1j*mat_kz, zero_mat], [zero_mat, 1j*mat_kz]])

        mat_v0 = q_mat @ tinv(mat_lam)


        # Initialize global scattering matrix
        smat_global = init_smatrix(shape=(self.n_batches, self.n_freqs,
                          2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex, device=self.device)
        smat_layer = {}

        # ============================================================================
        # Main Loop
        # ============================================================================
        # Build Eigen Value Problem

        for n_layer in range(self.layer_manager.nlayer):
            # Transform dimensions to (n_batches, n_freqs, n_harmonics, n_harmonics)
            if (self.layer_manager.layers[n_layer].is_solved is False) \
                or (self.layer_manager.layers[n_layer].is_optimize is True):

                # no need to solve the non-optimized layers in 'opt' mode.
                if self.solver_mode == 'opt':
                    self.layer_manager.layers[n_layer].is_solved = True

                if self.layer_manager.layers[n_layer].is_homogeneous is False:
                    if self.layer_manager.layers[n_layer].is_dispersive is False:
                        toeplitz_er = self.expand_dims(self.layer_manager.layers[n_layer].kermat)
                    else:
                        toeplitz_er = self.layer_manager.layers[n_layer].kermat

                    # permeability always non-dispersive
                    # Transform dimensions to (n_batches, n_freqs, n_harmonics, n_harmonics)
                    toeplitz_ur = self.expand_dims(self.layer_manager.layers[n_layer].kurmat)

                    solve_ter_mky = tsolve(toeplitz_er, mat_ky)
                    solve_ter_mkx = tsolve(toeplitz_er, mat_kx)
                    solve_tur_mky = tsolve(toeplitz_ur, mat_ky)
                    solve_tur_mkx = tsolve(toeplitz_ur, mat_kx)

                    # p_mat_i = blockmat2x2([[mat_kx @ tinv(toeplitz_er) @ mat_ky,
                    #                    toeplitz_ur - mat_kx @ tinv(toeplitz_er) @ mat_kx],
                    #                   [mat_ky @ tinv(toeplitz_er) @ mat_ky - toeplitz_ur,
                    #                  - mat_ky @ tinv(toeplitz_er) @ mat_kx]])
                    p_mat_i = blockmat2x2([[mat_kx @ solve_ter_mky,
                                       toeplitz_ur - mat_kx @ solve_ter_mkx],
                                      [mat_ky @ solve_ter_mky - toeplitz_ur,
                                     - mat_ky @ solve_ter_mkx]])

                    # q_mat_i = blockmat2x2([[mat_kx @ tinv(toeplitz_ur) @ mat_ky,
                    #                    toeplitz_er - mat_kx @ tinv(toeplitz_ur) @ mat_kx],
                    #                   [mat_ky @ tinv(toeplitz_ur) @ mat_ky - toeplitz_er,
                    #                  - mat_ky @ tinv(toeplitz_ur) @ mat_kx]])
                    q_mat_i = blockmat2x2([[mat_kx @ solve_tur_mky,
                                       toeplitz_er - mat_kx @ solve_tur_mkx],
                                      [mat_ky @ solve_tur_mky - toeplitz_er,
                                     - mat_ky @ solve_tur_mkx]])

                    smat_layer = self._solve_nonhomo_layer(layer_index=n_layer,
                                              p_mat_i=p_mat_i,
                                              q_mat_i=q_mat_i,
                                              mat_w0=mat_w0,
                                              mat_v0=mat_v0)

                elif self.layer_manager.layers[n_layer].is_homogeneous is True:
                    if self.layer_manager.layers[n_layer].is_dispersive is True:
                        toeplitz_er = self.expand_dims(torch.tensor(
                            self._matlib[self.layer_manager.layers[n_layer].material_name].er,
                            device=self.device)).to(self.tcomplex)
                        toeplitz_ur = torch.tensor(
                            self._matlib[self.layer_manager.layers[n_layer].material_name].ur,
                            device=self.device).to(self.tcomplex)

                    else:
                        toeplitz_er = torch.tensor(
                            self._matlib[self.layer_manager.layers[n_layer].material_name].er, device=self.device)
                        toeplitz_ur = torch.tensor(
                            self._matlib[self.layer_manager.layers[n_layer].material_name].ur, device=self.device)

                    # q_mat_i = 1 / toeplitz_ur * blockmat2x2([[mat_kx @ mat_ky,
                    #                             toeplitz_ur*toeplitz_er*ident_mat - mat_kx @  mat_kx],
                    #                            [mat_ky @ mat_ky - toeplitz_ur*toeplitz_er*ident_mat,
                    #                             - mat_ky @ mat_kx]])
                    toep_ur_er = toeplitz_ur * toeplitz_er * ident_mat
                    conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er) * ident_mat

                    q_mat_i = 1 / toeplitz_ur * blockmat2x2([[mat_kx_ky,
                                                toep_ur_er - mat_kx_kx],
                                               [mat_ky_ky - toep_ur_er,
                                                - mat_ky_kx]])

                    mat_kz_i = torch.conj(torch.sqrt(
                        conj_toep_ur_er - \
                            mat_kx ** 2 - mat_ky ** 2 + 0*1j)).to(self.tcomplex)

                    dmat_lam_i = blockmat2x2(
                        [[1j*mat_kz_i, zero_mat], [zero_mat, 1j*mat_kz_i]])

                    mat_w_i = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])
                    mat_v_i = q_mat_i @ tinv(dmat_lam_i)

                    mat_x_i = torch.linalg.matrix_exp(- dmat_lam_i * \
                        self.expand_dims(self.k_0) \
                        * self.expand_dims(self.layer_manager.layers[n_layer]\
                                           .thickness.to(self.device))).to(self.tcomplex)

                    # Calculate Layer Scattering Matrix
                    atwi = tsolve(mat_w_i, mat_w0)
                    atvi = tsolve(mat_v_i, mat_v0)
                    mat_a_i = atwi + atvi
                    mat_b_i = atwi - atvi

                    solve_ai_xi = tsolve(mat_a_i, mat_x_i)

                    mat_xi_bi = mat_x_i @ mat_b_i

                    # mat_d_i = mat_a_i - mat_x_i @ mat_b_i @ tinv(mat_a_i) @ mat_x_i @ mat_b_i
                    mat_d_i = mat_a_i - mat_xi_bi @ solve_ai_xi @ mat_b_i

                    # smat_layer['S11'] = tsolve(
                    #     mat_d_i, mat_x_i @ mat_b_i @ tinv(mat_a_i) @ mat_x_i @ mat_a_i - mat_b_i)
                    smat_layer['S11'] = tsolve(
                        mat_d_i, mat_xi_bi @ solve_ai_xi @ mat_a_i - mat_b_i)
                    # smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tinv(mat_a_i) @ mat_b_i)
                    smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
                    smat_layer['S21'] = smat_layer['S12']
                    smat_layer['S22'] = smat_layer['S11']

                self.smat_layers[n_layer] = smat_layer

            # Update global scattering matrix
            smat_global = redhstar(smat_global, self.smat_layers[n_layer])

        # ============================================================================
        # End Main Loop
        # ============================================================================

        # Connect to reflection region
        if self.layer_manager.is_ref_dispersive is False:
            # q_mat_1 = (1/self.ur1) * blockmat2x2([[mat_kx @ mat_ky,
            #                                 self.ur1 * self.er1 * ident_mat - mat_kx @ mat_kx],
            #                                 [mat_ky @ mat_ky - self.ur1 * self.er1 * ident_mat,
            #                                 - mat_ky @ mat_kx]])
            q_mat_1 = (1/self.ur1) * blockmat2x2([[mat_kx_ky,
                                            self.ur1 * self.er1 * ident_mat - mat_kx_kx],
                                            [mat_ky_ky - self.ur1 * self.er1 * ident_mat,
                                            - mat_ky_kx]])
        else:
            # q_mat_1 = (1/self.ur1) * blockmat2x2([[mat_kx @ mat_ky,
            #                                 self.expand_dims(self.ur1 * self.er1) * ident_mat - mat_kx @ mat_kx],
            #                                 [mat_ky @ mat_ky - self.expand_dims(self.ur1 * self.er1) * ident_mat,
            #                                 - mat_ky @ mat_kx]])
            q_mat_1 = (1/self.ur1) * blockmat2x2([[mat_kx_ky,
                                            self.expand_dims(self.ur1 * self.er1) * ident_mat - mat_kx_kx],
                                            [mat_ky_ky - self.expand_dims(self.ur1 * self.er1) * ident_mat,
                                            - mat_ky_kx]])

        mat_w_ref = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])
        mat_lam_ref = blockmat2x2([[1j * mat_kz_ref, zero_mat], [zero_mat, 1j*mat_kz_ref]])


        mat_v_ref = q_mat_1 @ tinv(mat_lam_ref)

        smat_ref = {}
        atw1 = tsolve(mat_w0, mat_w_ref)
        atv1 = tsolve(mat_v0, mat_v_ref)
        mat_a_1 = atw1 + atv1
        mat_b_1 = atw1 - atv1
        smat_ref['S11'] = - tsolve(mat_a_1, mat_b_1)
        smat_ref['S12'] = 2 * tinv(mat_a_1)
        # smat_ref['S21'] = 1/2 * (mat_a_1 - mat_b_1 @ tinv(mat_a_1) @ mat_b_1)
        smat_ref['S21'] = 1/2 * (mat_a_1 - mat_b_1 @ tsolve(mat_a_1, mat_b_1))
        smat_ref['S22'] = mat_b_1 @ tinv(mat_a_1)

        smat_global = redhstar(smat_ref, smat_global)

        # Connect to transmission region
        if self.layer_manager.is_trn_dispersive is False:
            # q_mat_2 = (1/self.ur2) * blockmat2x2([[mat_kx @ mat_ky,
            #                                 self.ur2 * self.er2 * ident_mat - mat_kx @ mat_kx],
            #                                 [mat_ky @ mat_ky - self.ur2 * self.er2 * ident_mat,
            #                                 - mat_ky @ mat_kx]])
            q_mat_2 = (1/self.ur2) * blockmat2x2([[mat_kx_ky,
                                            self.ur2 * self.er2 * ident_mat - mat_kx_kx],
                                            [mat_ky_ky - self.ur2 * self.er2 * ident_mat,
                                            - mat_ky_kx]])
        else:
            # q_mat_2 = (1/self.ur2) * blockmat2x2([[mat_kx @ mat_ky,
            #                                   self.expand_dims(self.ur2 * self.er2) * ident_mat - mat_kx @ mat_kx],
            #                                  [mat_ky @ mat_ky - self.expand_dims(self.ur2 * self.er2) * ident_mat,
            #                                   - mat_ky @ mat_kx]])
            q_mat_2 = (1/self.ur2) * blockmat2x2([[mat_kx_ky,
                                              self.expand_dims(self.ur2 * self.er2) * ident_mat - mat_kx_kx],
                                             [mat_ky_ky - self.expand_dims(self.ur2 * self.er2) * ident_mat,
                                              - mat_ky_kx]])

        # mat_w_trn = blockmat2x2([[I, zero_mat], [zero_mat, I]])
        mat_w_trn = mat_w_ref
        mat_lam_trn = blockmat2x2([[1j * mat_kz_trn, zero_mat], [zero_mat, 1j*mat_kz_trn]])
        mat_v_trn = q_mat_2 @ tinv(mat_lam_trn)

        smat_trn = {}
        atw2 = tsolve(mat_w0, mat_w_trn)
        atv2 = tsolve(mat_v0, mat_v_trn)
        mat_a_2 = atw2 + atv2
        mat_b_2 = atw2 - atv2
        smat_trn['S11'] = mat_b_2 @ tinv(mat_a_2)
        # smat_trn['S12'] = 1/2 * (mat_a_2 - mat_b_2 @ tinv(mat_a_2) @ mat_b_2)
        smat_trn['S12'] = 1/2 * (mat_a_2 - mat_b_2 @ tsolve(mat_a_2, mat_b_2))
        smat_trn['S21'] = 2 * tinv(mat_a_2)
        smat_trn['S22'] = - tsolve(mat_a_2, mat_b_2)

        smat_global = redhstar(smat_global, smat_trn)

        # Compute polarization vector
        norm_vec = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self.tcomplex, device=self.device)

        if np.abs(self.src['theta']) < 1e-3:
            if self.src['norm_te_dir'] == 'y':
                ate = torch.tensor(
                    [0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device).unsqueeze(0).repeat(self.n_freqs, 1)
            elif self.src['norm_te_dir'] == 'x':
                ate = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=self.tcomplex, device=self.device).unsqueeze(0).repeat(self.n_freqs, 1)
        else:
            ate = torch.cross(self.kinc, norm_vec.unsqueeze(0).repeat(self.n_freqs, 1))
            ate = ate / torch.norm(ate, dim=1).unsqueeze(-1)

        atm = torch.cross(ate, self.kinc, dim=1)
        atm = atm / torch.norm(atm)

        pol_vec = self.src['pte'] * ate + self.src['ptm'] * atm
        pol_vec = pol_vec / torch.norm(pol_vec, dim=1).unsqueeze(-1)

        # Calculate electric field source vector
        delta = torch.zeros(
            size=(self.n_freqs, n_harmonics), dtype=self.tcomplex, device=self.device)

        delta[:, (self.kdim[1] // 2) * self.kdim[0] +(self.kdim[0] // 2)] = 1
        esrc = torch.cat((pol_vec[:, 0].unsqueeze(-1) * delta,
                         pol_vec[:, 1].unsqueeze(-1) * delta), dim=1)

        # Calculate source vectors
        csrc = tsolve(mat_w_ref, esrc.unsqueeze(0).unsqueeze(-1))

        # Calculate reflected fields
        cref = smat_global['S11'] @ csrc
        eref = mat_w_ref @ cref

        ref_field_x = eref[:, :, 0: n_harmonics, :]
        ref_field_y = eref[:, :, n_harmonics: 2*n_harmonics, :]

        # ref_field_z = - (mat_kx @ tinv(mat_kz_ref)) @ ref_field_x \
        #      - (mat_ky @ tinv(mat_kz_ref)) @ ref_field_y
        ref_field_z = - mat_kx @ tsolve(mat_kz_ref, ref_field_x) \
             - mat_ky @ tsolve(mat_kz_ref, ref_field_y)

        # Calculate transmitted fields
        ctrn = smat_global['S21'] @ csrc
        etrn = mat_w_trn @ ctrn
        trn_field_x = etrn[:, :, 0: n_harmonics, :]
        trn_field_y = etrn[:, :, n_harmonics: 2*n_harmonics, :]
        # trn_field_z = - (mat_kx @ tinv(mat_kz_trn)) @ trn_field_x \
        #      - (mat_ky @ tinv(mat_kz_trn)) @ trn_field_y
        trn_field_z = - mat_kx @ tsolve(mat_kz_trn, trn_field_x) \
             - mat_ky @ tsolve(mat_kz_trn, trn_field_y)

        # Calculate diffraction efficiences
        ref_diff_efficiency = torch.reshape(torch.real(self.ur2/self.ur1*mat_kz_ref / \
            self.kinc[:, 2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) @
                            (torch.real(ref_field_x) ** 2 + torch.imag(ref_field_x) ** 2 +
                             torch.real(ref_field_y) ** 2 + torch.imag(ref_field_y) ** 2 +
                             + torch.real(ref_field_z) ** 2 + torch.imag(ref_field_z) ** 2),
                            shape=(self.n_batches,
                                   self.n_freqs,
                                   self.kdim[0],
                                   self.kdim[1])).transpose(dim0=-2, dim1=-1)

        trn_diff_efficiency = torch.reshape(torch.real(self.ur1/self.ur2*mat_kz_trn / \
            self.kinc[:, 2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) @
                            (torch.real(trn_field_x) ** 2 + torch.imag(trn_field_x) ** 2 +
                             torch.real(trn_field_y) ** 2 + torch.imag(trn_field_y) ** 2 +
                             + torch.real(trn_field_z) ** 2 + torch.imag(trn_field_z) ** 2),
                            shape=(self.n_batches,
                                   self.n_freqs,
                                   self.kdim[0],
                                   self.kdim[1])).transpose(dim0=-2, dim1=-1)

        # Calculate overall reflectance & Transmittance
        total_ref_efficiency = torch.sum(ref_diff_efficiency, dim=(-1, -2))
        total_trn_efficiency = torch.sum(trn_diff_efficiency, dim=(-1, -2))

        # Store output data
        data = {}
        data['rx'] = torch.reshape(ref_field_x, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['ry'] = torch.reshape(ref_field_y, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['rz'] = torch.reshape(ref_field_z, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['tx'] = torch.reshape(trn_field_x, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['ty'] = torch.reshape(trn_field_y, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['tz'] = torch.reshape(trn_field_z, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['RDE'] = ref_diff_efficiency
        data['TDE'] = trn_diff_efficiency
        data['REF'] = total_ref_efficiency
        data['TRN'] = total_trn_efficiency
        data['kzref'] = mat_kz_ref
        data['kztrn'] = mat_kz_trn
        data['kinc'] = self.kinc
        data['kx'] = torch.squeeze(kx_0)
        data['ky'] = torch.squeeze(ky_0)

        return data

    def solve(self,
              source: dict) -> dict:
        """solve.

        This function call the specified algorithm to solve the models.

        Args:
            source (dict): source

        Returns:
            dict:
        """

        # self.n_freqs = len(source.lam0)
        self.src = source

        if self.n_solve == 0:
            self._pre_solve()

        data = self._solve()

        self.n_solve = self.n_solve + 1

        return data

    def _pre_solve(self) -> None:
        """_pre_solve.

        Check parameters before solving.

        Args:

        Returns:
            None:
        """
        # Calculate refrective index of external medium
        refractive_1 = torch.sqrt(self.ur1 * self.er1)
        # refractive_2 = torch.sqrt(self.ur2 * self.er2)

        # kinc: dimensions (n_freqs, 3)
        self.kinc = refractive_1.unsqueeze(-1) * torch.tensor([np.sin(self.src['theta']) * np.cos(self.src['phi']),
                                                     np.sin(self.src['theta']) *
                                                     np.sin(self.src['phi']),
                                                     np.cos(self.src['theta'])],
                                                    dtype=self.tfloat,
                                                    device=self.device).unsqueeze(0).repeat(self.n_freqs, 1)

        # Calculate reciprocal lattice vectors
        d_v = self.lattice_t1[:, 0] * self.lattice_t2[:, 1] - self.lattice_t2[:, 0] * self.lattice_t1[:, 1]

        self.reci_t1 = 2 * torch.pi * \
            torch.cat(((+self.lattice_t2[:, 1] / d_v).unsqueeze(-1),
                       (-self.lattice_t2[:, 0] / d_v).unsqueeze(-1)), dim=1)
        self.reci_t2 = 2 * torch.pi * \
            torch.cat(((-self.lattice_t1[:, 1] / d_v).unsqueeze(-1),
                       (+self.lattice_t1[:, 0] / d_v).unsqueeze(-1)), dim=1)

        # Calculate wave vector expansion
        self.tlam0 = torch.tensor(
            self.lam0, dtype=self.tfloat, device=self.device)
        self.k_0 = 2 * torch.pi / self.tlam0  # k0 with dimensions: n_freqs

        f_p = torch.arange(start=-np.floor(self.kdim[0] / 2), end=np.floor(
            self.kdim[0] / 2) + 1, dtype=torch.int32, device=self.device)
        f_q = torch.arange(start=-np.floor(self.kdim[1] / 2), end=np.floor(
            self.kdim[1] / 2) + 1, dtype=torch.int32, device=self.device)
        [self.mesh_fq, self.mesh_fp] = torch.meshgrid(f_q, f_p, indexing='xy')

        # check if options are set correctly
        for n_layer in range(self.layer_manager.nlayer):
            if self.layer_manager.layers[n_layer].is_homogeneous is False:
                if self.layer_manager.layers[n_layer].ermat is None:
                    # if not homogenous material, must be with a pattern.
                    # if no pattern assigned before solving, the material will be set as homogeneous
                    self.layer_manager.replace_layer_to_homogeneous(layer_index=n_layer)
                    self.n_solve = 0

                    print(
                        f"Warning: Layer {n_layer} has no pattern assigned, and was changed to homogeneous")


    def eval(self):
        """eval.

        Set the solver to 'eval' mode.
        """
        self.solver_mode = 'eval'

    def optimization(self):
        """optimization.

        Set the solver to 'opt' mode.
        """
        self.solver_mode = 'opt'


    def update_er_with_mask(self,
                            mask: torch.Tensor,
                            layer_index: int,
                            bg_material: str = 'air',
                            set_grad: bool = False) -> None:
        """update_er_with_mask.

        To update the layer permittivity (or permeability) distribution in a specified layer.

        Args:
            mask (torch.Tensor): mask, the new binary pattern mask to be updated.
            layer_index (int): layer_index
            bg_material (str): bg_material, the background material of the pattern (mask == 0).
            set_grad (bool): set_grad, manually set the requires_grad True of the pattern.

        Returns:
            None:
        """

        n_batches, ndim1, ndim2 = mask.size()
        if n_batches != self.n_batches:
            raise ValueError(
                f"Input batch size {n_batches} doesn't match the existing size {self.n_batches}!")
        if (ndim1 != self.rdim[0]) or (ndim2 != self.rdim[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)
            self.n_solve = 0

        er_bg = self._get_bg(layer_index=layer_index, param='er')

        if self.layer_manager.layers[layer_index].is_dispersive is False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        if bg_material == 'air':
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                (er_bg - 1) * mask_format

        if set_grad is True:
            self.layer_manager.layers[layer_index].ermat.requires_grad = True
        # self.layer_manager.layers[layer_index].kermat = self.gen_toeplitz_matrix(
        #     layer_index=layer_index, param='er', is_dispersive=self.layer_manager.layers[layer_index].is_dispersive)
        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='er')

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        if self.n_solve == 0:
            self.layer_manager.layers[layer_index].urmat = self._get_bg(
                layer_index=layer_index, param='ur')
            # self.layer_manager.layers[layer_index].kurmat = self.gen_toeplitz_matrix(
            #     layer_index=layer_index, param='ur', is_dispersive=False)
            self.layer_manager.gen_toeplitz_matrix(
                layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='ur')


    def update_layer_thickness(self,
                               layer_index: int,
                               thickness: torch.Tensor):
        """update_layer_thickness.

        Update the thickness of the specified layer.

        Args:
            layer_index (int): layer_index
            thickness (torch.Tensor): thickness
        """

        self.layer_manager.update_layer_thickness(layer_index=layer_index, thickness=thickness)
        self.layer_manager.layers[layer_index].is_solved = False

    def _get_bg(self, layer_index: int, param='er') -> Union[torch.Tensor, None]:
        """_get_bg.

        Returns the background matrix of the specified layer.

        Args:
            layer_index (int): layer_index
            param:

        Returns:
            torch.Tensor:
        """

        ret_mat = None

        if layer_index < self.layer_manager.nlayer:
            if self.layer_manager.layers[layer_index].is_dispersive is True:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == 'er':
                    ret_mat = torch.tensor(self._matlib[material_name].er)\
                    .unsqueeze(0).repeat(self.n_batches, 1).unsqueeze(2)\
                    .unsqueeze(2).repeat(1, 1, self.rdim[0], self.rdim[1])\
                    .to(self.device).to(self.tcomplex)
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur
                    ret_mat = param_val * torch.ones(size=(self.n_batches, self.rdim[0], self.rdim[1]),
                                                  dtype=self.tcomplex, device=self.device)
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

            else:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == 'er':
                    param_val = self._matlib[material_name].er
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

                ret_mat = param_val * torch.ones(size=(self.n_batches, self.rdim[0], self.rdim[1]),
                                              dtype=self.tcomplex, device=self.device)
        else:
            raise ValueError("The index exceeds the max layer number.")

        return ret_mat

    def _init_dispersive_materials(self) -> None:
        """_init_dispersive_materials.
        
        Initialize the dispersive profile of materials.

        Args:

        Returns:
            None:
        """
        for imat in self._matlib.values():
            if imat.isdispersive_er is True:
                # if type(imat.er) == type(None):
                # if isinstance(imat.er, NoneType):
                if imat.er is None:
                    imat.load_dispersive_er(
                        self._lam0, self._lenunit)

    def expand_dims(self, mat: torch.Tensor) -> Optional[torch.Tensor]:
        """Function that expands the input matrix to a standard output dimension without layer information:
            (n_batches, n_freqs, n_harmonics, n_harmonics)

        Args:
            mat (torch.Tensor): input tensor matrxi
        """

        ret = None

        if mat.ndim == 1 and mat.shape[0] == self.n_freqs:
            # The input tensor with dimension (n_freqs)
            ret = mat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif mat.ndim == 1 and mat.shape[0] == self.n_batches:
            ret = mat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif mat.ndim == 2:
            # The input matrix with dimension (n_harmonics, n_harmonics)
            ret = mat.unsqueeze(0).unsqueeze(0).repeat(1, self.n_freqs, 1, 1)
        elif mat.ndim == 3 and mat.shape[0] == self.n_freqs:
            # The input matrix with dimension (n_freq, n_harmonics, n_harmonics)
            ret = mat.unsqueeze(0)
        elif mat.ndim == 3:
            # The input matrix with dimension (n_batches, n_harmonics, n_harmonics)
            if mat.shape[0] == 1:
                ret = mat.unsqueeze(1).repeat(self.n_batches, self.n_freqs, 1, 1)
            elif mat.shape[0] == self.n_batches:
                ret = mat.unsqueeze(1).repeat(1, self.n_freqs, 1, 1)
        elif mat.ndim == 4 and mat.shape[1] == self.n_freqs and mat.shape[0] == 1:
            # The input matrix with dimension (n_batches, n_freqs, n_harmonics, n_harmonics)
            ret = mat.repeat(self.n_batches, 1, 1, 1)
        else:
            raise RuntimeError("Not Listed in the Case")

        return ret

class RCWASolver(FourierBaseSover):
    """RCWASolver.

    This is the implemented solver class for RCWA.
    """
    def __init__(self,
                 batch_size: int = 1,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: Union[list, None] = None,
                 kdim: Union[list, None] = None,
                 materiallist: Union[list, None] = None,
                 t1: Union[list, None] = None,
                 t2: Union[list, None] = None,
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(batch_size, lam0, lengthunit, rdim, kdim, materiallist, t1, t2, device)

    def _solve_nonhomo_layer(self,
                             layer_index: int,
                             p_mat_i: torch.Tensor,
                             q_mat_i: torch.Tensor,
                             mat_w0: torch.Tensor,
                             mat_v0: torch.Tensor) -> dict:
        """_solve_nonhomo_layer.

        Solves the non-homogenous layer using RCWA methods.

        Args:
            layer_index (int): layer_index
            p_mat_i (torch.Tensor): p_mat_i
            q_mat_i (torch.Tensor): q_mat_i
            mat_w0 (torch.Tensor): mat_w0
            mat_v0 (torch.Tensor): mat_v0

        Returns:
            dict:
        """
        smat_layer = {}

        # Compute Eigen Modes
        mat_lam_i, mat_w_i = EigComplex.apply(p_mat_i @ q_mat_i)

        dmat_lam_i = torch.sqrt(mat_lam_i).unsqueeze(-2) * \
            torch.eye(mat_lam_i.shape[-1],
                      dtype=self.tcomplex, device=self.device)

        mat_v_i = q_mat_i @ mat_w_i @ tinv(dmat_lam_i)
        mat_x_i = torch.linalg.matrix_exp(- dmat_lam_i * self.expand_dims(self.k_0)\
                                     * self.expand_dims(self.layer_manager.layers[layer_index]\
                                                        .thickness.to(self.device))).to(self.tcomplex)

        # Calculate Layer Scattering Matrix
        mat_a_i = tsolve(mat_w_i, mat_w0) + tsolve(mat_v_i, mat_v0)
        mat_b_i = tsolve(mat_w_i, mat_w0) - tsolve(mat_v_i, mat_v0)
        # mat_d_i = mat_a_i - mat_x_i @ mat_b_i @ tinv(mat_a_i) @ mat_x_i @ mat_b_i

        mat_xb_i = mat_x_i @ mat_b_i
        mat_d_i = mat_a_i - mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_b_i

        # smat_layer['S11'] = tsolve(
        #     mat_d_i, mat_x_i @ mat_b_i @ tinv(mat_a_i) @ mat_x_i @ mat_a_i - mat_b_i)
        smat_layer['S11'] = tsolve(
            mat_d_i, mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_a_i - mat_b_i)
        # smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tinv(mat_a_i) @ mat_b_i)
        smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
        smat_layer['S21'] = smat_layer['S12']
        smat_layer['S22'] = smat_layer['S11']

        return smat_layer


class RDITSolver(FourierBaseSover):
    """RDITSolver.

    This is the implemented solver class for R-DIT.
    """


    def __init__(self,
                 batch_size: int = 1,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: Union[list, None] = None,
                 kdim: Union[list, None] = None,
                 materiallist: Union[list, None] = None,
                 t1: Union[list, None] = None,
                 t2: Union[list, None] = None,
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(batch_size, lam0, lengthunit, rdim, kdim, materiallist, t1, t2, device)
        self._rdit_orders = 10

    def _solve_nonhomo_layer(self,
                             layer_index: int,
                             p_mat_i: torch.Tensor,
                             q_mat_i: torch.Tensor,
                             mat_w0: torch.Tensor,
                             mat_v0: torch.Tensor) -> dict:
        """_solve_nonhomo_layer.

        Solves the non-homogenous layer using R-DIT methods.

        Args:
            layer_index (int): layer_index
            p_mat_i (torch.Tensor): p_mat_i
            q_mat_i (torch.Tensor): q_mat_i
            mat_w0 (torch.Tensor): mat_w0
            mat_v0 (torch.Tensor): mat_v0

        Returns:
            dict:
        """
        n_harmonics = self.kdim[0] * self.kdim[1]

        smat_layer = {}

        # Construct T matrix
        delta_h = self.expand_dims(self.k_0) * \
            self.expand_dims(self.layer_manager.layers[layer_index]\
                             .thickness.to(self.device)) / 2.0
        tmat_a_i = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                        device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)
        tmat_b_i = torch.zeros(size=(self.n_batches, self.n_freqs, 2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex,
                          device=self.device)
        tmat_c_i = torch.zeros(size=(self.n_batches, self.n_freqs, 2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex,
                          device=self.device)
        tmat_d_i = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                        device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)

        p_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                            device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)
        q_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                            device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)

        for irdit_order in range(1, self._rdit_orders + 1):
            if (irdit_order % 2) == 0:  # even orders
                p_fcoef = p_fcoef @ q_mat_i
                q_fcoef = q_fcoef @ p_mat_i
                fac = (delta_h.to(self.tfloat)**irdit_order / np.math.factorial(irdit_order))
                tmat_a_i = tmat_a_i + fac * p_fcoef
                tmat_d_i = tmat_d_i + fac * q_fcoef
            else:  # odd orders
                p_fcoef = p_fcoef @ p_mat_i
                q_fcoef = q_fcoef @ q_mat_i
                fac = (delta_h.to(self.tfloat)**irdit_order / np.math.factorial(irdit_order))
                tmat_b_i = tmat_b_i + fac * p_fcoef
                tmat_c_i = tmat_c_i + fac * q_fcoef

        # Construct some helper functions
        mat_g1 = tmat_a_i @ mat_w0 + tmat_b_i @ mat_v0
        mat_g2 = - tmat_c_i @ mat_w0 - tmat_d_i @ mat_v0

        mat_xx1 = tsolve(mat_g2, tmat_c_i @ mat_w0 - tmat_d_i @ mat_v0)
        mat_xx2 = tsolve(mat_g1, tmat_a_i @ mat_w0 - tmat_b_i @ mat_v0)

        mat_yyi = mat_xx1 - mat_xx2
        mat_zzi = mat_xx1 + mat_xx2

        smat_layer['S11'] = mat_yyi / 2.0
        smat_layer['S12'] = mat_zzi / 2.0
        smat_layer['S21'] = smat_layer['S12']
        smat_layer['S22'] = smat_layer['S11']

        return smat_layer

    def set_rdit_order(self, rdit_order: int) -> None:
        """set_rdit_order.

        Set the order of R-DIT.

        Args:
            rdit_order (int): rdit_order

        Returns:
            None:
        """
        self._rdit_orders = rdit_order
