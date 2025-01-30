""" This file defines all operations related to solver computations."""
from typing import Union, Optional

import numpy as np
import torch
import math
import json
import os

from torch.linalg import inv as tinv
from torch.linalg import solve as tsolve
from torch.nn.functional import conv2d as tconv2d
from .cell import Cell3D, CellType
from .utils import blockmat2x2, redhstar, EigComplex, init_smatrix, blur_filter, create_material
from .constants import Algorithm, Precision


class TorchrditConfig:
    def __init__(self):
        """
        Initialize an empty TorchrditConfig instance. This class no longer stores
        configuration or solver state, only provides functionality to create solvers.
        """

    @staticmethod
    def create_solver(config):
        """
        Create and return a solver based on the provided configuration.
        """
        # Check if the input is a string assuming it could be a JSON file path
        if isinstance(config, str):
            base_path = os.path.dirname(config)
            with open(config, 'r') as file:
                config = json.load(file)
        else:
            base_path = os.getcwd()

        # Extract solver settings
        algorithm = Algorithm[config["solver"]["algorithm"].upper()]
        precision = Precision[config["solver"]["precision"].upper()]
        lam0 = np.array(config["solver"]["wavelengths"])
        lengthunit = config["solver"].get("lengthunit", "um")
        rdim = config["solver"].get("rdim", [512, 512])
        kdim = config["solver"].get("kdim", [9, 9])
        is_use_FFF = config["solver"].get("is_use_FFF", True)

        # Extract lattice vectors
        t1 = torch.tensor(config["lattice"]["t1"])
        t2 = torch.tensor(config["lattice"]["t2"])

        # Initialize the solver with the new configuration
        solver = SolverConstructer.creat_sovler(
            algorithm=algorithm,
            precision=precision,
            lam0=lam0,
            lengthunit=lengthunit,
            rdim=rdim,
            kdim=kdim,
            t1=t1,
            t2=t2,
            is_use_FFF=is_use_FFF,
            device=config.get("device", "cpu")
        )

        # Create materials
        materials = TorchrditConfig._create_materials(config["materials"], base_path=base_path)

        # Add layers dynamically
        TorchrditConfig._add_layers(solver, config["layers"], materials)

        # Update transmission and reflection materials if specified
        if "trn_material" in config:
            trn_material = materials[config["trn_material"]]
            solver.update_trn_material(trn_material=trn_material)
        
        if "ref_material" in config:
            ref_material = materials[config["ref_material"]]
            solver.update_ref_material(ref_material=ref_material)

        return solver

    @staticmethod
    def _create_materials(materials_dict, base_path):
        """
        Create material objects from dictionary.
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

    @staticmethod
    def _add_layers(solver, layers_list, materials):
        """
        Add layers to the solver.
        """
        for layer in layers_list:
            material_name = layer["material"]
            thickness = torch.tensor(layer["thickness"], dtype=torch.float32)
            is_homogeneous = layer.get("is_homogeneous", True)
            is_optimize = layer.get("is_optimize", False)

            solver.add_layer(
                material_name=materials[material_name],
                thickness=thickness,
                is_homogeneous=is_homogeneous,
                is_optimize=is_optimize
            )

class SolverConstructer():
            
    def __init__(self) -> None:
        pass

    @staticmethod
    def creat_sovler(algorithm: Algorithm = Algorithm.RDIT,
                     precision: Precision = Precision.SINGLE,
                     lam0: np.ndarray = np.array([1.0]),
                     lengthunit: str = 'um',  # length unit used in the solver
                     rdim: list = [512, 512],  # dimensions in real space: (H, W)
                     kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                     materiallist: list = [],  # list of materials
                     t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                     t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                     is_use_FFF: bool = True, # if use Fast Fourier Factorization
                     device: Union[str, torch.device] = 'cpu'):
            
        if precision == Precision.SINGLE:
            if algorithm == Algorithm.RCWA:
                inst = RCWARolverFloat(lam0=lam0,
                                lengthunit=lengthunit,
                                rdim=rdim,
                                kdim=kdim,
                                materiallist=materiallist,
                                t1=t1,
                                t2=t2,
                                is_use_FFF=is_use_FFF,
                                device=device)
            else:
                if algorithm != Algorithm.RDIT:
                    print(f"Unknown selected algorithm, set to [R-DIT]")
                    
                inst = RDITSolverFloat(lam0=lam0,
                                lengthunit=lengthunit,
                                rdim=rdim,
                                kdim=kdim,
                                materiallist=materiallist,
                                t1=t1,
                                t2=t2,
                                is_use_FFF=is_use_FFF,
                                device=device)
        else: 
            if precision != Precision.DOUBLE:
                print(f"Unknown precision, set to [double]")
            
            if algorithm == Algorithm.RCWA:
                inst = RCWASolverDouble(lam0=lam0,
                                lengthunit=lengthunit,
                                rdim=rdim,
                                kdim=kdim,
                                materiallist=materiallist,
                                t1=t1,
                                t2=t2,
                                is_use_FFF=is_use_FFF,
                                device=device)
            else:
                if algorithm != Algorithm.RDIT:
                    print(f"Unknown selected algorithm, set to [R-DIT]")
                    
                inst = RDITSolverDouble(lam0=lam0,
                                lengthunit=lengthunit,
                                rdim=rdim,
                                kdim=kdim,
                                materiallist=materiallist,
                                t1=t1,
                                t2=t2,
                                is_use_FFF=is_use_FFF,
                                device=device)
            

        return inst
        
        
        
class FourierBaseSover(Cell3D):
    """Base Class of Fourier Domain Solver"""

    mesh_fp = torch.empty((3,3))
    mesh_fq = torch.empty((3,3)) 

    reci_t1 = torch.empty((2,)) 
    reci_t2 = torch.empty((2,)) 
    tlam0 = torch.empty((1,)) 
    k_0 = torch.empty((1,))

    is_use_FFF = True
    is_solve_batch = True

    def __init__(self,
                 # wavelengths (frequencies) to be solved
                 lam0: np.ndarray = np.array([1.0]),
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True, # if use Fast Fourier Factorization
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(lengthunit, rdim,
                         kdim, materiallist, t1, t2, device)

        # Set free space wavelength
        if isinstance(lam0, float):
            self._lam0 = np.array([lam0])
        elif isinstance(lam0, np.ndarray):
            self._lam0 = lam0

        self.n_freqs = len(self._lam0)
        self.kinc = torch.zeros((1, 3))

        self.src = {}

        self.smat_layers = {}

        # solver mode: [eval] [opt]
        self.solver_mode = 'eval'
        self.is_use_FFF = is_use_FFF

    @property
    def lam0(self):
        """lam0.
        attribute.
        """
        return self._lam0

    @lam0.setter
    def lam0(self, value):
        if isinstance(value, float):
            self._lam0 = np.array([value])
        elif isinstance(value, np.ndarray):
            self._lam0 = value
        else:
            raise ValueError(f"Wrong input type {type(value)}")

    def add_source(self,
                   theta: float,
                   phi: float,
                   pte: float,
                   ptm: float,
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
        
    def _solve_nonhomo_layer_single_frequency(self,
                             layer_index: int,
                             p_mat_i: torch.Tensor,
                             q_mat_i: torch.Tensor,
                             mat_w0: torch.Tensor,
                             mat_v0: torch.Tensor,
                            freq_index: int) -> dict:
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


    def _solve_batch_frequencies(self, **kwargs) -> dict:

        # kx_0, ky_0: (n_freqs, kdim[0], kdim[1])
        kx_0 = self.kinc[:, 0, None, None] - \
            (self.mesh_fp[None, :, :] * self.reci_t1[0, None, None] + \
                self.mesh_fq[None, :, :] * self.reci_t2[0, None, None]) / self.k_0[:, None, None] + 0*1j

        ky_0 = self.kinc[:, 1, None, None] - \
            (self.mesh_fp[None, :, :] * self.reci_t1[1, None, None] + \
                self.mesh_fq[None, :, :] * self.reci_t2[1, None, None]) / self.k_0[:, None, None] + 0*1j

        # Add relaxition
        if torch.any(kx_0 == 0.0):
            ind = torch.nonzero(kx_0 == 0.0)
            kx_0[ind[:, 0], ind[:, 1]] = kx_0[ind[:, 0], ind[:, 1]] + 1e-6
        if torch.any(ky_0 == 0.0):
            ind = torch.nonzero(ky_0 == 0.0)
            ky_0[ind[:, 0], ind[:, 1]] = ky_0[ind[:, 0], ind[:, 1]] + 1e-6

        if self.layer_manager.is_ref_dispersive is False:
            kz_ref_0 = torch.conj(torch.sqrt(
                torch.conj(self.ur1)*torch.conj(self.er1) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_ref_0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur1)*torch.conj(self.er1))[:, None, None] - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))

        if self.layer_manager.is_trn_dispersive is False:
            kz_trn_0 = torch.conj(torch.sqrt(
                torch.conj(self.ur2)*torch.conj(self.er2) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_trn_0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur2)*torch.conj(self.er2))[:, None, None] - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))

        # Transform to diagnal matrices
        n_harmonics = self.kdim[0] * self.kdim[1]

        self.ident_mat_k = torch.eye(n_harmonics, dtype=self.tcomplex, device=self.device)
        self.ident_mat_k2 = torch.eye(2 * n_harmonics, dtype=self.tcomplex, device=self.device)

        mat_kx = kx_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_ky = ky_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_ref = kz_ref_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_trn = kz_trn_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)


        ident_mat = self.ident_mat_k[None, :, :].expand(self.n_freqs, -1, -1)
        zero_mat = torch.zeros(
            size=(self.n_freqs ,n_harmonics, n_harmonics), dtype=self.tcomplex, device=self.device)
        
        # Calculate eigen-modes of the gap medium

        mat_kx_ky = mat_kx * mat_ky
        mat_ky_kx = mat_ky * mat_kx
        mat_kx_kx = mat_kx * mat_kx
        mat_ky_ky = mat_ky * mat_ky

        mat_kx_diag = self.to_diag(mat_kx)
        mat_ky_diag = self.to_diag(mat_ky)

        mat_kz = torch.conj(torch.sqrt(1.0 - mat_kx_kx - mat_ky_ky))

        ident_mat_kx_kx = 1.0 - mat_kx_kx
        ident_mat_ky_ky = 1.0 - mat_ky_ky

        mat_w0 = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])

        inv_mat_lam = 1 / (1j*mat_kz)
        
        mat_v0 = blockmat2x2([[self.to_diag(mat_kx_ky * inv_mat_lam), self.to_diag(ident_mat_kx_kx * inv_mat_lam)],
                  [self.to_diag(- ident_mat_ky_ky * inv_mat_lam), self.to_diag(- mat_kx_ky * inv_mat_lam)]])

        # Initialize global scattering matrix
        smat_global = init_smatrix(shape=(self.n_freqs,
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

                    # assuming permeability always non-dispersive
                    # Transform dimensions to (n_freqs, n_harmonics, n_harmonics)
                    toeplitz_ur = self.expand_dims(self.layer_manager.layers[n_layer].kurmat)

                    solve_ter_mky = tsolve(toeplitz_er, mat_ky_diag)
                    solve_ter_mkx = tsolve(toeplitz_er, mat_kx_diag)
                    solve_tur_mky = tsolve(toeplitz_ur, mat_ky_diag)
                    solve_tur_mkx = tsolve(toeplitz_ur, mat_kx_diag)

                    p_mat_i = blockmat2x2([[mat_kx_diag @ solve_ter_mky,
                                       toeplitz_ur - mat_kx_diag @ solve_ter_mkx],
                                      [mat_ky_diag @ solve_ter_mky - toeplitz_ur,
                                     - mat_ky_diag @ solve_ter_mkx]])

                    if self.is_use_FFF is True: 
                        delta_toeplitz_er = toeplitz_er - tinv(self.reciprocal_toeplitz_er)
                        q_mat_i = blockmat2x2([[mat_kx_diag @ solve_tur_mky - delta_toeplitz_er @ self.n_xy,
                                           toeplitz_er - mat_kx_diag @ solve_tur_mkx - delta_toeplitz_er @ self.n_yy],
                                          [mat_ky_diag @ solve_tur_mky - toeplitz_er + delta_toeplitz_er @ self.n_xx,
                                         delta_toeplitz_er @ self.n_xy - mat_ky_diag @ solve_tur_mkx]])
                    else:
                        q_mat_i = blockmat2x2([[mat_kx_diag @ solve_tur_mky,
                                           toeplitz_er - mat_kx_diag @ solve_tur_mkx],
                                          [mat_ky_diag @ solve_tur_mky - toeplitz_er,
                                         - mat_ky_diag @ solve_tur_mkx]])
                        
                    smat_layer = self._solve_nonhomo_layer(layer_index=n_layer,
                                              p_mat_i=p_mat_i,
                                              q_mat_i=q_mat_i,
                                              mat_w0=mat_w0,
                                              mat_v0=mat_v0)

                elif self.layer_manager.layers[n_layer].is_homogeneous is True:
                    if self.layer_manager.layers[n_layer].is_dispersive is True:
                        toeplitz_er = self._matlib[self.layer_manager.layers[n_layer].material_name].er.detach().to(self.tcomplex).to(self.device)
                        toeplitz_ur = self._matlib[self.layer_manager.layers[n_layer].material_name].ur.detach().to(self.tcomplex).to(self.device)

                        # ===========================================================
                        toep_ur_er = toeplitz_ur * toeplitz_er
                        conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                        mat_kz_i = torch.conj(torch.sqrt(
                            conj_toep_ur_er[:, None] - mat_kx ** 2 - mat_ky ** 2) + 0*1j).to(self.tcomplex)

                        inv_dmat_lam_i = 1 / (1j*mat_kz_i)
                        mat_v_i = 1 / toeplitz_ur * blockmat2x2([[self.to_diag(mat_kx_ky * inv_dmat_lam_i),
                                                                  self.to_diag((toep_ur_er[:, None] - mat_kx_kx) * inv_dmat_lam_i)],
                                                                 [self.to_diag((mat_ky_ky - toep_ur_er[:, None]) * inv_dmat_lam_i),
                                                                  - self.to_diag(mat_ky_kx * inv_dmat_lam_i)]])


                        # ===========================================================
                    else:
                        toeplitz_er = self._matlib[self.layer_manager.layers[n_layer].material_name].er.detach().clone().to(self.device)
                        toeplitz_ur = self._matlib[self.layer_manager.layers[n_layer].material_name].ur.detach().clone().to(self.device)
                        # ===========================================================
                        toep_ur_er = toeplitz_ur * toeplitz_er
                        conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                        mat_kz_i = torch.conj(torch.sqrt(
                            conj_toep_ur_er - mat_kx ** 2 - mat_ky ** 2) + 0*1j).to(self.tcomplex)

                        inv_dmat_lam_i = 1 / (1j*mat_kz_i)
                        mat_v_i = 1 / toeplitz_ur * blockmat2x2([[self.to_diag(mat_kx_ky * inv_dmat_lam_i),
                                                                  self.to_diag((toep_ur_er - mat_kx_kx) * inv_dmat_lam_i)],
                                                                 [self.to_diag((mat_ky_ky - toep_ur_er) * inv_dmat_lam_i),
                                                                  - self.to_diag(mat_ky_kx * inv_dmat_lam_i)]])


                        # ===========================================================

                    mat_x_i = - 1j*mat_kz_i * self.k_0[:, None] \
                        * self.layer_manager.layers[n_layer].thickness.to(self.device).to(self.tcomplex)
                    mat_x_i_diag = torch.concat([mat_x_i, mat_x_i], dim=1)
                    mat_x_i = self.to_diag(torch.exp(mat_x_i_diag))

                    # Calculate Layer Scattering Matrix
                    atwi = mat_w0
                    atvi = tsolve(mat_v_i, mat_v0)
                    mat_a_i = atwi + atvi
                    mat_b_i = atwi - atvi

                    solve_ai_xi = tsolve(mat_a_i, mat_x_i)           

                    mat_xi_bi = mat_x_i @ mat_b_i

                    mat_d_i = mat_a_i - mat_xi_bi @ solve_ai_xi @ mat_b_i

                    smat_layer['S11'] = tsolve(
                        mat_d_i, mat_xi_bi @ solve_ai_xi @ mat_a_i - mat_b_i)
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
        inv_mat_lam_ref = 1 / (1j * mat_kz_ref)

        if self.layer_manager.is_ref_dispersive is False:
            mat_v_ref = (1/self.ur1) * blockmat2x2([[self.to_diag(mat_kx_ky * inv_mat_lam_ref),
                                             self.to_diag((self.ur1 * self.er1 - mat_kx_kx) * inv_mat_lam_ref)],
                                            [self.to_diag((mat_ky_ky - self.ur1 * self.er1) * inv_mat_lam_ref),
                                            - self.to_diag(mat_ky_kx * inv_mat_lam_ref)]])
        else:
            mat_v_ref = (1/self.ur1) * blockmat2x2([[self.to_diag(mat_kx_ky * inv_mat_lam_ref),
                                             self.to_diag(((self.ur1 * self.er1)[:, None] - mat_kx_kx) * inv_mat_lam_ref)],
                                            [self.to_diag((mat_ky_ky - (self.ur1 * self.er1)[:, None]) * inv_mat_lam_ref),
                                            - self.to_diag(mat_ky_kx * inv_mat_lam_ref)]])
        mat_w_ref = self.ident_mat_k2[None, :, :].expand(self.n_freqs, -1, -1)

        smat_ref = {}
        atw1 = mat_w_ref
        atv1 = tsolve(mat_v0, mat_v_ref)

        mat_a_1 = atw1 + atv1
        mat_b_1 = atw1 - atv1
        inv_mat_a_1 = tinv(mat_a_1)
        inv_mat_a_1_mat_b_1 = inv_mat_a_1 @ mat_b_1
        smat_ref['S11'] = - inv_mat_a_1_mat_b_1
        smat_ref['S12'] = 2 * inv_mat_a_1
        smat_ref['S21'] = 0.5 * (mat_a_1 - mat_b_1 @ inv_mat_a_1_mat_b_1)
        smat_ref['S22'] = mat_b_1 @ inv_mat_a_1

        smat_global = redhstar(smat_ref, smat_global)

        # Connect to transmission region
        inv_mat_lam_trn = 1 / (1j * mat_kz_trn) 

        if self.layer_manager.is_trn_dispersive is False:
            mat_v_trn = (1/self.ur2) * blockmat2x2([[self.to_diag(mat_kx_ky * inv_mat_lam_trn),
                                            self.to_diag((self.ur2 * self.er2 - mat_kx_kx) * inv_mat_lam_trn)],
                                            [self.to_diag((mat_ky_ky - self.ur2 * self.er2) * inv_mat_lam_trn),
                                            - self.to_diag(mat_ky_kx * inv_mat_lam_trn)]])
        else:
            mat_v_trn = (1/self.ur2) * blockmat2x2([[self.to_diag(mat_kx_ky * inv_mat_lam_trn),
                                                     self.to_diag(((self.ur2 * self.er2)[:, None] - mat_kx_kx) * inv_mat_lam_trn)],
                                                    [self.to_diag((mat_ky_ky - (self.ur2 * self.er2)[:, None]) * inv_mat_lam_trn),
                                            - self.to_diag(mat_ky_kx * inv_mat_lam_trn)]])

        mat_w_trn = mat_w_ref
        smat_trn = {}
        # atw2 = tsolve(mat_w0, mat_w_trn)
        atw2 = mat_w_trn
        atv2 = tsolve(mat_v0, mat_v_trn)

        mat_a_2 = atw2 + atv2
        mat_b_2 = atw2 - atv2
        inv_mat_a_2 = tinv(mat_a_2)
        inv_mat_a_2_mat_b_2 = inv_mat_a_2 @ mat_b_2
        smat_trn['S11'] = mat_b_2 @ inv_mat_a_2
        smat_trn['S12'] = 0.5 * (mat_a_2 - mat_b_2 @ inv_mat_a_2_mat_b_2)
        smat_trn['S21'] = 2 * inv_mat_a_2
        smat_trn['S22'] = - inv_mat_a_2_mat_b_2

        smat_global = redhstar(smat_global, smat_trn)

        smat_structure = {key: value.detach().clone() for key, value in smat_global.items()}


        # Compute polarization vector
        norm_vec = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self.tcomplex, device=self.device)

        ate = torch.empty_like(norm_vec)
        if np.abs(self.src['theta']) < 1e-3:
            if 'norm_te_dir' not in self.src:
                ate = torch.tensor([0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)[None, :].repeat(self.n_freqs, 1)
            else:
                if self.src['norm_te_dir'] == 'y':
                    ate = torch.tensor(
                        [0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)[None, :].repeat(self.n_freqs, 1)
                elif self.src['norm_te_dir'] == 'x':
                    ate = torch.tensor(
                        [1.0, 0.0, 0.0], dtype=self.tcomplex, device=self.device)[None, :].repeat(self.n_freqs, 1)
        else:
            ate = torch.cross(self.kinc, norm_vec[None, :].repeat(self.n_freqs, 1), dim=1)
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
        csrc = tsolve(mat_w_ref, esrc.unsqueeze(-1))

        # Calculate reflected fields
        cref = smat_global['S11'] @ csrc
        eref = mat_w_ref @ cref

        ref_field_x = eref[:, 0: n_harmonics, :]
        ref_field_y = eref[:, n_harmonics: 2*n_harmonics, :]

        ref_field_z = - mat_kx_diag @ tsolve(self.to_diag(mat_kz_ref), ref_field_x) \
             - mat_ky_diag @ tsolve(self.to_diag(mat_kz_ref), ref_field_y)

        # Calculate transmitted fields
        ctrn = smat_global['S21'] @ csrc
        etrn = mat_w_trn @ ctrn
        trn_field_x = etrn[:, 0: n_harmonics, :]
        trn_field_y = etrn[:, n_harmonics: 2*n_harmonics, :]
        trn_field_z = - mat_kx_diag @ tsolve(self.to_diag(mat_kz_trn), trn_field_x) \
             - mat_ky_diag @ tsolve(self.to_diag(mat_kz_trn), trn_field_y)

        # Calculate diffraction efficiences
        ref_diff_efficiency = torch.reshape(torch.real(self.ur2/self.ur1*self.to_diag(mat_kz_ref) / \
            self.kinc[:, 2, None, None]) @
                            (torch.real(ref_field_x) ** 2 + torch.imag(ref_field_x) ** 2 +
                             torch.real(ref_field_y) ** 2 + torch.imag(ref_field_y) ** 2 +
                             + torch.real(ref_field_z) ** 2 + torch.imag(ref_field_z) ** 2),
                            shape=(self.n_freqs,
                                   self.kdim[0],
                                   self.kdim[1])).transpose(dim0=-2, dim1=-1)

        trn_diff_efficiency = torch.reshape(torch.real(self.ur1/self.ur2*self.to_diag(mat_kz_trn) / \
            self.kinc[:, 2, None, None]) @
                            (torch.real(trn_field_x) ** 2 + torch.imag(trn_field_x) ** 2 +
                             torch.real(trn_field_y) ** 2 + torch.imag(trn_field_y) ** 2 +
                             + torch.real(trn_field_z) ** 2 + torch.imag(trn_field_z) ** 2),
                            shape=(self.n_freqs,
                                   self.kdim[0],
                                   self.kdim[1])).transpose(dim0=-2, dim1=-1)


        # Calculate overall reflectance & Transmittance
        total_ref_efficiency = torch.sum(ref_diff_efficiency, dim=(-1, -2))
        total_trn_efficiency = torch.sum(trn_diff_efficiency, dim=(-1, -2))

        # Store output data
        data = {}
        data['smat_structure'] = smat_structure 
        data['smat_layers'] = self.smat_layers
        data['rx'] = torch.reshape(ref_field_x, shape=(
            self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['ry'] = torch.reshape(ref_field_y, shape=(
            self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['rz'] = torch.reshape(ref_field_z, shape=(
            self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['tx'] = torch.reshape(trn_field_x, shape=(
            self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['ty'] = torch.reshape(trn_field_y, shape=(
            self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['tz'] = torch.reshape(trn_field_z, shape=(
            self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
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

    
    def _solve_single_frequency(self, **kwargs) -> dict:

        # kx_0, ky_0: (n_freqs, kdim[0], kdim[1])
        kx_0 = self.kinc[:, 0, None, None] - \
            (self.mesh_fp[None, :, :] * self.reci_t1[0, None, None] + \
                self.mesh_fq[None, :, :] * self.reci_t2[0, None, None]) / self.k_0[:, None, None] + 0*1j

        ky_0 = self.kinc[:, 1, None, None] - \
            (self.mesh_fp[None, :, :] * self.reci_t1[1, None, None] + \
                self.mesh_fq[None, :, :] * self.reci_t2[1, None, None]) / self.k_0[:, None, None] + 0*1j


        # Add relaxition
        if torch.any(kx_0 == 0.0):
            ind = torch.nonzero(kx_0 == 0.0)
            kx_0[ind[:, 0], ind[:, 1]] = kx_0[ind[:, 0], ind[:, 1]] + 1e-6
        if torch.any(ky_0 == 0.0):
            ind = torch.nonzero(ky_0 == 0.0)
            ky_0[ind[:, 0], ind[:, 1]] = ky_0[ind[:, 0], ind[:, 1]] + 1e-6


        if self.layer_manager.is_ref_dispersive is False:
            kz_ref_0 = torch.conj(torch.sqrt(
                torch.conj(self.ur1)*torch.conj(self.er1) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_ref_0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur1)*torch.conj(self.er1))[:, None, None] - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))

        if self.layer_manager.is_trn_dispersive is False:
            kz_trn_0 = torch.conj(torch.sqrt(
                torch.conj(self.ur2)*torch.conj(self.er2) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_trn_0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur2)*torch.conj(self.er2))[:, None, None] - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))

        # Transform to diagnal matrices
        n_harmonics = self.kdim[0] * self.kdim[1]

        self.ident_mat_k = torch.eye(n_harmonics, dtype=self.tcomplex, device=self.device)
        self.ident_mat_k2 = torch.eye(2 * n_harmonics, dtype=self.tcomplex, device=self.device)

        mat_kx = kx_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_ky = ky_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_ref = kz_ref_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_trn = kz_trn_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)


        ident_mat = self.ident_mat_k[None, :, :].expand(self.n_freqs, -1, -1)
        zero_mat = torch.zeros(
            size=(self.n_freqs ,n_harmonics, n_harmonics), dtype=self.tcomplex, device=self.device)
        
        # Calculate eigen-modes of the gap medium

        mat_kx_ky = mat_kx * mat_ky
        mat_ky_kx = mat_ky * mat_kx
        mat_kx_kx = mat_kx * mat_kx
        mat_ky_ky = mat_ky * mat_ky

        mat_kx_diag = self.to_diag(mat_kx)
        mat_ky_diag = self.to_diag(mat_ky)

        mat_kz = torch.conj(torch.sqrt(1.0 - mat_kx_kx - mat_ky_ky))


        ident_mat_kx_kx = 1.0 - mat_kx_kx
        ident_mat_ky_ky = 1.0 - mat_ky_ky

        mat_w0 = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])

        inv_mat_lam = 1 / (1j*mat_kz)
        
        mat_v0 = blockmat2x2([[self.to_diag(mat_kx_ky * inv_mat_lam), self.to_diag(ident_mat_kx_kx * inv_mat_lam)],
                  [self.to_diag(- ident_mat_ky_ky * inv_mat_lam), self.to_diag(- mat_kx_ky * inv_mat_lam)]])
        
        data = {}

        for ind_freq  in range(self.n_freqs):
            
            # Initialize global scattering matrix
            smat_global = init_smatrix(shape=(
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

                    # # no need to solve the non-optimized layers in 'opt' mode.
                    # if self.solver_mode == 'opt':
                    #     self.layer_manager.layers[n_layer].is_solved = True

                    if self.layer_manager.layers[n_layer].is_homogeneous is False:
                        if self.layer_manager.layers[n_layer].is_dispersive is False:
                            toeplitz_er = self.expand_dims(self.layer_manager.layers[n_layer].kermat)[ind_freq,:,:]
                        else:
                            toeplitz_er = self.layer_manager.layers[n_layer].kermat[ind_freq,:,:]


                        # assuming permeability always non-dispersive
                        # Transform dimensions to (n_freqs, n_harmonics, n_harmonics)
                        toeplitz_ur = self.expand_dims(self.layer_manager.layers[n_layer].kurmat)[ind_freq,:,:]


                        solve_ter_mky = tsolve(toeplitz_er, mat_ky_diag[ind_freq,:,:])
                        solve_ter_mkx = tsolve(toeplitz_er, mat_kx_diag[ind_freq,:,:])
                        solve_tur_mky = tsolve(toeplitz_ur, mat_ky_diag[ind_freq,:,:])
                        solve_tur_mkx = tsolve(toeplitz_ur, mat_kx_diag[ind_freq,:,:])


                        p_mat_i = blockmat2x2([[mat_kx_diag[ind_freq,:,:] @ solve_ter_mky,
                                           toeplitz_ur - mat_kx_diag[ind_freq,:,:] @ solve_ter_mkx],
                                          [mat_ky_diag[ind_freq,:,:] @ solve_ter_mky - toeplitz_ur,
                                         - mat_ky_diag[ind_freq,:,:] @ solve_ter_mkx]])
                        

                        if self.is_use_FFF is True: 
                            if self.layer_manager.layers[n_layer].is_dispersive is False:
                                delta_toeplitz_er = toeplitz_er - tinv(self.reciprocal_toeplitz_er)
                            else:
                                delta_toeplitz_er = toeplitz_er - tinv(self.reciprocal_toeplitz_er[ind_freq,:,:])
                            q_mat_i = blockmat2x2([[mat_kx_diag[ind_freq,:,:] @ solve_tur_mky - delta_toeplitz_er @ self.n_xy,
                                               toeplitz_er - mat_kx_diag[ind_freq,:,:] @ solve_tur_mkx - delta_toeplitz_er @ self.n_yy],
                                              [mat_ky_diag[ind_freq,:,:] @ solve_tur_mky - toeplitz_er + delta_toeplitz_er @ self.n_xx,
                                             delta_toeplitz_er @ self.n_xy - mat_ky_diag[ind_freq,:,:] @ solve_tur_mkx]])
                        else:
                            q_mat_i = blockmat2x2([[mat_kx_diag[ind_freq,:,:] @ solve_tur_mky,
                                               toeplitz_er - mat_kx_diag[ind_freq,:,:] @ solve_tur_mkx],
                                              [mat_ky_diag[ind_freq,:,:] @ solve_tur_mky - toeplitz_er,
                                             - mat_ky_diag[ind_freq,:,:] @ solve_tur_mkx]])
                            
                        smat_layer = self._solve_nonhomo_layer_single_frequency(layer_index=n_layer,
                                                  p_mat_i=p_mat_i,
                                                  q_mat_i=q_mat_i,
                                                  mat_w0=mat_w0[ind_freq,:,:],
                                                  mat_v0=mat_v0[ind_freq,:,:],
                                                    freq_index=ind_freq)

                    elif self.layer_manager.layers[n_layer].is_homogeneous is True:
                        if self.layer_manager.layers[n_layer].is_dispersive is True:
                            toeplitz_er = self._matlib[self.layer_manager.layers[n_layer].material_name].er.detach().to(self.tcomplex).to(self.device)[ind_freq]
                            toeplitz_ur = self._matlib[self.layer_manager.layers[n_layer].material_name].ur.detach().to(self.tcomplex).to(self.device)[ind_freq]

                            
                            # ===========================================================
                            toep_ur_er = toeplitz_ur * toeplitz_er
                            conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                            mat_kz_i = torch.conj(torch.sqrt(
                                conj_toep_ur_er - mat_kx[ind_freq,:] ** 2 - mat_ky[ind_freq,:] ** 2) + 0*1j).to(self.tcomplex)
                            
                            inv_dmat_lam_i = 1 / (1j*mat_kz_i)
                            mat_v_i = 1 / toeplitz_ur * blockmat2x2([[self.to_diag(mat_kx_ky[ind_freq,:] * inv_dmat_lam_i),
                                                                      self.to_diag((toep_ur_er - mat_kx_kx[ind_freq,:]) * inv_dmat_lam_i)],
                                                                     [self.to_diag((mat_ky_ky[ind_freq,:] - toep_ur_er) * inv_dmat_lam_i),
                                                                      - self.to_diag(mat_ky_kx[ind_freq,:] * inv_dmat_lam_i)]])

                            # ===========================================================
                        else:
                            toeplitz_er = self._matlib[self.layer_manager.layers[n_layer].material_name].er.detach().clone().to(self.device)
                            toeplitz_ur = self._matlib[self.layer_manager.layers[n_layer].material_name].ur.detach().clone().to(self.device)
                            # ===========================================================
                            toep_ur_er = toeplitz_ur * toeplitz_er
                            conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                            mat_kz_i = torch.conj(torch.sqrt(
                                conj_toep_ur_er - mat_kx[ind_freq,:] ** 2 - mat_ky[ind_freq,:] ** 2) + 0*1j).to(self.tcomplex)

                            inv_dmat_lam_i = 1 / (1j*mat_kz_i)
                            mat_v_i = 1 / toeplitz_ur * blockmat2x2([[self.to_diag(mat_kx_ky[ind_freq,:] * inv_dmat_lam_i),
                                                                      self.to_diag((toep_ur_er - mat_kx_kx[ind_freq,:]) * inv_dmat_lam_i)],
                                                                     [self.to_diag((mat_ky_ky[ind_freq,:] - toep_ur_er) * inv_dmat_lam_i),
                                                                      - self.to_diag(mat_ky_kx[ind_freq,:] * inv_dmat_lam_i)]])


                            # ===========================================================


                        mat_x_i = - 1j*mat_kz_i * self.k_0[ind_freq] \
                            * self.layer_manager.layers[n_layer].thickness.to(self.device).to(self.tcomplex)

                        mat_x_i_diag = torch.concat([mat_x_i, mat_x_i], dim=0)
                        mat_x_i = self.to_diag(torch.exp(mat_x_i_diag))

                        # Calculate Layer Scattering Matrix
                        atwi = mat_w0[ind_freq,:,:]
                        atvi = tsolve(mat_v_i, mat_v0[ind_freq,:,:])
                        mat_a_i = atwi + atvi
                        mat_b_i = atwi - atvi
                        
                        solve_ai_xi = tsolve(mat_a_i, mat_x_i)           

                        mat_xi_bi = mat_x_i @ mat_b_i

                        mat_d_i = mat_a_i - mat_xi_bi @ solve_ai_xi @ mat_b_i

                        smat_layer['S11'] = tsolve(
                            mat_d_i, mat_xi_bi @ solve_ai_xi @ mat_a_i - mat_b_i)
                        smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
                        smat_layer['S21'] = smat_layer['S12']
                        smat_layer['S22'] = smat_layer['S11']


                    self.smat_layers[n_layer] = smat_layer
                    
                    # raise RuntimeError("Debug homo layer homo")

                # Update global scattering matrix
                smat_global = redhstar(smat_global, self.smat_layers[n_layer])
            # ============================================================================
            # End Main Loop
            # ============================================================================

            # Connect to reflection region
            inv_mat_lam_ref = 1 / (1j * mat_kz_ref[ind_freq,:])

            if self.layer_manager.is_ref_dispersive is False:
                mat_v_ref = (1/self.ur1) * blockmat2x2([[self.to_diag(mat_kx_ky[ind_freq,:] * inv_mat_lam_ref),
                                                 self.to_diag((self.ur1 * self.er1 - mat_kx_kx[ind_freq,:]) * inv_mat_lam_ref)],
                                                [self.to_diag((mat_ky_ky[ind_freq,:] - self.ur1 * self.er1) * inv_mat_lam_ref),
                                                - self.to_diag(mat_ky_kx[ind_freq,:] * inv_mat_lam_ref)]])
            else:
                mat_v_ref = (1/self.ur1) * blockmat2x2([[self.to_diag(mat_kx_ky[ind_freq,:] * inv_mat_lam_ref),
                                                 self.to_diag(((self.ur1 * self.er1)[:, None] - mat_kx_kx[ind_freq,:]) * inv_mat_lam_ref)],
                                                [self.to_diag((mat_ky_ky[ind_freq,:] - (self.ur1 * self.er1)[:, None]) * inv_mat_lam_ref),
                                                - self.to_diag(mat_ky_kx[ind_freq,:] * inv_mat_lam_ref)]])
            # mat_w_ref = self.ident_mat_k2[None, :, :].expand(self.n_freqs, -1, -1)
            mat_w_ref = self.ident_mat_k2


            smat_ref = {}
            atw1 = mat_w_ref
            atv1 = tsolve(mat_v0[ind_freq,:,:], mat_v_ref)
            

            mat_a_1 = atw1 + atv1
            mat_b_1 = atw1 - atv1
            inv_mat_a_1 = tinv(mat_a_1)
            inv_mat_a_1_mat_b_1 = inv_mat_a_1 @ mat_b_1
            smat_ref['S11'] = - inv_mat_a_1_mat_b_1
            smat_ref['S12'] = 2 * inv_mat_a_1
            smat_ref['S21'] = 0.5 * (mat_a_1 - mat_b_1 @ inv_mat_a_1_mat_b_1)
            smat_ref['S22'] = mat_b_1 @ inv_mat_a_1

            smat_global = redhstar(smat_ref, smat_global)
            
            

            # Connect to transmission region
            inv_mat_lam_trn = 1 / (1j * mat_kz_trn[ind_freq,:]) 

            if self.layer_manager.is_trn_dispersive is False:
                mat_v_trn = (1/self.ur2) * blockmat2x2([[self.to_diag(mat_kx_ky[ind_freq,:] * inv_mat_lam_trn),
                                                self.to_diag((self.ur2 * self.er2 - mat_kx_kx[ind_freq,:]) * inv_mat_lam_trn)],
                                                [self.to_diag((mat_ky_ky[ind_freq,:] - self.ur2 * self.er2) * inv_mat_lam_trn),
                                                - self.to_diag(mat_ky_kx[ind_freq,:] * inv_mat_lam_trn)]])
            else:
                mat_v_trn = (1/self.ur2) * blockmat2x2([[self.to_diag(mat_kx_ky[ind_freq,:] * inv_mat_lam_trn),
                                                         self.to_diag(((self.ur2 * self.er2)[ind_freq, None] - mat_kx_kx[ind_freq,:]) * inv_mat_lam_trn)],
                                                        [self.to_diag((mat_ky_ky[ind_freq,:] - (self.ur2 * self.er2)[ind_freq, None]) * inv_mat_lam_trn),
                                                - self.to_diag(mat_ky_kx[ind_freq,:] * inv_mat_lam_trn)]])

            mat_w_trn = mat_w_ref
            smat_trn = {}
            # atw2 = tsolve(mat_w0, mat_w_trn)
            atw2 = mat_w_trn
            atv2 = tsolve(mat_v0[ind_freq,:,:], mat_v_trn)


            mat_a_2 = atw2 + atv2
            mat_b_2 = atw2 - atv2
            inv_mat_a_2 = tinv(mat_a_2)
            inv_mat_a_2_mat_b_2 = inv_mat_a_2 @ mat_b_2
            smat_trn['S11'] = mat_b_2 @ inv_mat_a_2
            smat_trn['S12'] = 0.5 * (mat_a_2 - mat_b_2 @ inv_mat_a_2_mat_b_2)
            smat_trn['S21'] = 2 * inv_mat_a_2
            smat_trn['S22'] = - inv_mat_a_2_mat_b_2

            smat_global = redhstar(smat_global, smat_trn)

            smat_structure = {key: value.detach().clone() for key, value in smat_global.items()}


            # Compute polarization vector
            norm_vec = torch.tensor(
                [0.0, 0.0, 1.0], dtype=self.tcomplex, device=self.device)

            ate = torch.empty_like(norm_vec)
            if np.abs(self.src['theta']) < 1e-3:
                if self.src['norm_te_dir'] == 'y':
                    ate = torch.tensor(
                        [0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)
                elif self.src['norm_te_dir'] == 'x':
                    ate = torch.tensor(
                        [1.0, 0.0, 0.0], dtype=self.tcomplex, device=self.device)
            else:
                ate = torch.cross(self.kinc[ind_freq,:], norm_vec, dim=0)
                ate = ate / torch.norm(ate, dim=0)
            

            atm = torch.cross(ate, self.kinc[ind_freq,:], dim=0)
            atm = atm / torch.norm(atm)
            

            pol_vec = self.src['pte'] * ate + self.src['ptm'] * atm
            pol_vec = pol_vec / torch.norm(pol_vec, dim=0)
            

            # Calculate electric field source vector
            delta = torch.zeros(
                size=(n_harmonics,), dtype=self.tcomplex, device=self.device)

            delta[(self.kdim[1] // 2) * self.kdim[0] +(self.kdim[0] // 2)] = 1
            esrc = torch.cat((pol_vec[0] * delta,
                             pol_vec[1] * delta), dim=0)

            # Calculate source vectors
            csrc = tsolve(mat_w_ref, esrc.unsqueeze(-1))
            
            

            # Calculate reflected fields
            cref = smat_global['S11'] @ csrc
            eref = mat_w_ref @ cref
            
            ref_field_x = eref[0: n_harmonics, :]
            ref_field_y = eref[n_harmonics: 2*n_harmonics, :]


            ref_field_z = - mat_kx_diag[ind_freq,:,:] @ tsolve(self.to_diag(mat_kz_ref)[ind_freq,:,:], ref_field_x) \
                - mat_ky_diag[ind_freq,:,:] @ tsolve(self.to_diag(mat_kz_ref)[ind_freq,:,:], ref_field_y)

            
            # Calculate transmitted fields
            ctrn = smat_global['S21'] @ csrc
            etrn = mat_w_trn @ ctrn
            trn_field_x = etrn[0: n_harmonics, :]
            trn_field_y = etrn[n_harmonics: 2*n_harmonics, :]
            trn_field_z = - mat_kx_diag[ind_freq,:,:] @ tsolve(self.to_diag(mat_kz_trn)[ind_freq,:,:], trn_field_x) \
                 - mat_ky_diag[ind_freq,:,:] @ tsolve(self.to_diag(mat_kz_trn)[ind_freq,:,:], trn_field_y)
                

            # Calculate diffraction efficiences
            ref_diff_efficiency = torch.reshape(torch.real(self.ur2/self.ur1*self.to_diag(mat_kz_ref)[ind_freq,:,:] / \
                self.kinc[ind_freq, 2, None, None]) @
                                (torch.real(ref_field_x) ** 2 + torch.imag(ref_field_x) ** 2 +
                                 torch.real(ref_field_y) ** 2 + torch.imag(ref_field_y) ** 2 +
                                 + torch.real(ref_field_z) ** 2 + torch.imag(ref_field_z) ** 2),
                                shape=(self.kdim[0],
                                       self.kdim[1])).transpose(dim0=-2, dim1=-1)

            trn_diff_efficiency = torch.reshape(torch.real(self.ur1/self.ur2*self.to_diag(mat_kz_trn)[ind_freq,:,:] / \
                self.kinc[ind_freq, 2, None, None]) @
                                (torch.real(trn_field_x) ** 2 + torch.imag(trn_field_x) ** 2 +
                                 torch.real(trn_field_y) ** 2 + torch.imag(trn_field_y) ** 2 +
                                 + torch.real(trn_field_z) ** 2 + torch.imag(trn_field_z) ** 2),
                                shape=(self.kdim[0],
                                       self.kdim[1])).transpose(dim0=-2, dim1=-1)


            # Calculate overall reflectance & Transmittance
            total_ref_efficiency = torch.sum(ref_diff_efficiency, dim=(-1, -2))
            total_trn_efficiency = torch.sum(trn_diff_efficiency, dim=(-1, -2))
            
            
            # raise RuntimeError("Debug")

            # Store output data
            if ind_freq == 0:
                data['rx'] = torch.reshape(ref_field_x, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]
                data['ry'] = torch.reshape(ref_field_y, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]
                data['rz'] = torch.reshape(ref_field_z, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]
                data['tx'] = torch.reshape(trn_field_x, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]
                data['ty'] = torch.reshape(trn_field_y, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]
                data['tz'] = torch.reshape(trn_field_z, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]
                data['RDE'] = ref_diff_efficiency[None,:,:]
                data['TDE'] = trn_diff_efficiency[None,:,:]
                data['REF'] = total_ref_efficiency[None]
                data['TRN'] = total_trn_efficiency[None]
            else:
                data['rx'] = torch.cat((data['rx'],
                                       torch.reshape(ref_field_x, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]),dim=0)
                data['ry'] = torch.cat((data['ry'],
                                       torch.reshape(ref_field_y, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]),dim=0)
                data['rz'] = torch.cat((data['rz'],
                                       torch.reshape(ref_field_z, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]),dim=0)
                data['tx'] = torch.cat((data['tx'],
                                       torch.reshape(trn_field_x, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]),dim=0)
                data['ty'] = torch.cat((data['ty'],
                                       torch.reshape(trn_field_y, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]),dim=0)
                data['tz'] = torch.cat((data['tz'],
                                       torch.reshape(trn_field_z, shape=(
                    self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)[None,:,:]),dim=0)
                data['RDE'] = torch.cat((data['RDE'],
                                        ref_diff_efficiency[None,:,:]), dim=0)
                
                data['TDE'] = torch.cat((data['TDE'],
                                        trn_diff_efficiency[None,:,:]), dim=0)
                data['REF'] = torch.cat((data['REF'],
                                        total_ref_efficiency[None]), dim=0)
                data['TRN'] = torch.cat((data['TRN'],
                                        total_trn_efficiency[None]), dim=0)
                
            
                # raise RuntimeError("Debug")
        data['smat_structure'] = smat_structure 
        data['smat_layers'] = self.smat_layers
        data['kzref'] = mat_kz_ref
        data['kztrn'] = mat_kz_trn
        data['kinc'] = self.kinc
        data['kx'] = torch.squeeze(kx_0)
        data['ky'] = torch.squeeze(ky_0)
            
        return data

    
    def solve(self,
              source: dict,
              is_sovle_batch: bool = True,
              **kwargs) -> dict:
        """solve.

        This function call the specified algorithm to solve the models.

        Args:
            source (dict): source

        Returns:
            dict:
        """

        # self.n_freqs = len(source.lam0)
        self.src = source
        self.is_solve_batch = is_sovle_batch

        self._pre_solve()

        if self.is_solve_batch is True:
            data = self._solve_batch_frequencies()
        else:
            data = self._solve_single_frequency()

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
        d_v = self.lattice_t1[0] * self.lattice_t2[1] - self.lattice_t2[0] * self.lattice_t1[1]

        self.reci_t1 = 2 * torch.pi * \
            torch.cat(((+self.lattice_t2[1] / d_v).unsqueeze(0),
                       (-self.lattice_t2[0] / d_v).unsqueeze(0)), dim=0)
        self.reci_t2 = 2 * torch.pi * \
            torch.cat(((-self.lattice_t1[1] / d_v).unsqueeze(0),
                       (+self.lattice_t1[0] / d_v).unsqueeze(0)), dim=0)

        # Calculate wave vector expansion
        self.tlam0 = torch.tensor(
            self.lam0, dtype=self.tfloat, device=self.device)
        self.k_0 = 2 * torch.pi / self.tlam0  # k0 with dimensions: n_freqs

        f_p = torch.arange(start=-np.floor(self.kdim[0] / 2), end=np.floor(
            self.kdim[0] / 2) + 1, dtype=self.tint, device=self.device)
        f_q = torch.arange(start=-np.floor(self.kdim[1] / 2), end=np.floor(
            self.kdim[1] / 2) + 1, dtype=self.tint, device=self.device)
        [self.mesh_fq, self.mesh_fp] = torch.meshgrid(f_q, f_p, indexing='xy')

        # check if options are set correctly
        for n_layer in range(self.layer_manager.nlayer):
            if self.layer_manager.layers[n_layer].is_homogeneous is False:
                if self.layer_manager.layers[n_layer].ermat is None:
                    # if not homogenous material, must be with a pattern.
                    # if no pattern assigned before solving, the material will be set as homogeneous
                    self.layer_manager.replace_layer_to_homogeneous(layer_index=n_layer)

                    print(
                        f"Warning: Layer {n_layer} has no pattern assigned, and was changed to homogeneous")


    def to_diag(self, input_mat: torch.Tensor) -> torch.Tensor:
        n_harmonics = self.kdim[0] * self.kdim[1]
        if input_mat.shape[-1] == n_harmonics:
            return input_mat.unsqueeze(-2) * self.ident_mat_k
        else:
            return input_mat.unsqueeze(-2) * self.ident_mat_k2


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
                            method: str = 'FFT',
                            set_grad: bool = False) -> None:
        """update_er_with_mask.

        To update the layer permittivity (or permeability) distribution in a specified layer.

        Args:
            mask (torch.Tensor): mask, the new binary pattern mask to be updated.
            layer_index (int): layer_index
            bg_material (str): bg_material, the background material of the pattern (mask == 0).
            method (str): method of computing Toeplitz matrix. ['FFT': for all cell type; 'Analytical': only for Cartesian]
            set_grad (bool): set_grad, manually set the requires_grad True of the pattern.

        Returns:
            None:
        """

        ndim1, ndim2 = mask.size()
        if (ndim1 != self.rdim[0]) or (ndim2 != self.rdim[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)

        er_bg = self._get_bg(layer_index=layer_index, param='er')

        if self.layer_manager.layers[layer_index].is_dispersive is False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        self.layer_manager.layers[layer_index].mask_format = mask_format.to(self.tfloat)

        if bg_material == 'air':
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            if self._matlib[bg_material].er.ndim == 0:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                    (er_bg - 1) * mask_format
            else:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er[:, None, None] * (1 - mask_format) + \
                (er_bg - 1) * mask_format
        if set_grad is True:
            self.layer_manager.layers[layer_index].ermat.requires_grad = True
        
        if method == 'Analytical':
            if self.cell_type != CellType.Cartesian:
                print(f"method [{method}] does not support the cell type [{self.cell_type}], will use FFT instead.")
                method = 'FFT'
        elif method != 'Analytical' and method != 'FFT':
            method = 'FFT'

        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='er', method=method)

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        self.layer_manager.layers[layer_index].urmat = self._get_bg(
            layer_index=layer_index, param='ur')
        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='ur', method=method)

        
        if self.is_use_FFF is True: 
            _, _, self.n_xx, self.n_yy, self.n_xy = self._calculate_nv_field(mask=self.layer_manager.layers[layer_index].mask_format.squeeze())
            self.reciprocal_toeplitz_er = self.layer_manager._gen_toeplitz2d(1 / self.layer_manager.layers[layer_index].ermat,
                                                          nharmonic_1=self.kdim[0],
                                                            nharmonic_2=self.kdim[1],
                                                                        method='FFT')

    def update_er_with_mask_extern_NV(self,
                            mask: torch.Tensor,
                            nv_vectors: tuple,
                            layer_index: int,
                            bg_material: str = 'air',
                            method: str = 'FFT',
                            set_grad: bool = False) -> None:
        """update_er_with_mask.

        To update the layer permittivity (or permeability) distribution in a specified layer.

        Args:
            mask (torch.Tensor): mask, the new binary pattern mask to be updated.
            layer_index (int): layer_index
            bg_material (str): bg_material, the background material of the pattern (mask == 0).
            method (str): method of computing Toeplitz matrix. ['FFT': for all cell type; 'Analytical': only for Cartesian]
            set_grad (bool): set_grad, manually set the requires_grad True of the pattern.

        Returns:
            None:
        """

        ndim1, ndim2 = mask.size()
        if (ndim1 != self.rdim[0]) or (ndim2 != self.rdim[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)

        er_bg = self._get_bg(layer_index=layer_index, param='er')

        if self.layer_manager.layers[layer_index].is_dispersive is False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        self.layer_manager.layers[layer_index].mask_format = mask_format.to(self.tfloat)

        if bg_material == 'air':
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            if self._matlib[bg_material].er.ndim == 0:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                    (er_bg - 1) * mask_format
            else:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                (er_bg - 1) * mask_format

        if set_grad is True:
            self.layer_manager.layers[layer_index].ermat.requires_grad = True
        
        if method == 'Analytical':
            if self.cell_type != CellType.Cartesian:
                print(f"method [{method}] does not support the cell type [{self.cell_type}], will use FFT instead.")
                method = 'FFT'
        elif method != 'Analytical' and method != 'FFT':
            method = 'FFT'

        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='er', method=method)

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        self.layer_manager.layers[layer_index].urmat = self._get_bg(
            layer_index=layer_index, param='ur')
        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='ur', method=method)

        
        if self.is_use_FFF is True: 
            norm_vec_x = nv_vectors[0]
            norm_vec_y = nv_vectors[1]
            self.n_xx = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_x,
                                                   nharmonic_1=self.kdim[0],
                                                   nharmonic_2=self.kdim[1],
                                                     method='FFT')
            self.n_yy = self.layer_manager._gen_toeplitz2d(norm_vec_y * norm_vec_y,
                                                   nharmonic_1=self.kdim[0],
                                                   nharmonic_2=self.kdim[1],
                                                     method='FFT')
            self.n_xy = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_y,
                                                   nharmonic_1=self.kdim[0],
                                                   nharmonic_2=self.kdim[1],
                                                     method='FFT')
            self.reciprocal_toeplitz_er = self.layer_manager._gen_toeplitz2d(1 / self.layer_manager.layers[layer_index].ermat,
                                                          nharmonic_1=self.kdim[0],
                                                            nharmonic_2=self.kdim[1],
                                                                        method='FFT')

    def _calculate_nv_field(self, mask: torch.Tensor):
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=self.tfloat, device=self.device)[None, None, :, :]
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=self.tfloat, device=self.device)[None, None, :, :]

        mask = (mask > 0.5).to(self.tfloat).to(self.device)

        # use conv2d to compute the gradients
        gradient_x = tconv2d(mask[None, None, :, :], sobel_x, padding=1)
        gradient_y = tconv2d(mask[None, None, :, :], sobel_y, padding=1)
        gradient_mag = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        index_bond_vec = torch.nonzero(gradient_mag.squeeze())
        index_field_vec = torch.nonzero(gradient_mag.squeeze() == 0)


        blurred_mask = blur_filter(mask[None, None, :, :], radius=4, beta=2, num_blur=1, tfloat=self.tfloat, device=self.device)
        gradient_x_br = tconv2d(blurred_mask, sobel_x, padding=1)
        gradient_y_br = tconv2d(blurred_mask, sobel_y, padding=1)

        norm_vec_x = (gradient_mag * gradient_x_br).squeeze()
        norm_vec_y = (gradient_mag * gradient_y_br).squeeze()

        bondary_vec_x = norm_vec_x[index_bond_vec[:, 0], index_bond_vec[:, 1]]
        bondary_vec_y = norm_vec_y[index_bond_vec[:, 0], index_bond_vec[:, 1]]

        field_ind_i = index_field_vec[:, 0]
        field_ind_j = index_field_vec[:, 1]
        bond_ind_i = index_bond_vec[:, 0]
        bond_ind_j = index_bond_vec[:, 1]
        denom = torch.sqrt((field_ind_i[:, None] - bond_ind_i[None, :])**2 + (field_ind_j[:, None] - bond_ind_j[None, :])**2) + 1e-6
        # denom = torch.sqrt((index_field_vec[:, 0][:, None] - index_bond_vec[:, 0][None, :])**2 + (index_field_vec[:,1][:, None] - index_bond_vec[:,1][None, :])**2)
        
        norm_vec_x[index_field_vec[:,0], index_field_vec[:, 1]] = torch.sum(bondary_vec_x[None, :] / denom, dim=1)
        norm_vec_y[index_field_vec[:,0], index_field_vec[:, 1]] = torch.sum(bondary_vec_y[None, :] / denom, dim=1)

        denom_normal = torch.sqrt(norm_vec_x ** 2 + norm_vec_y ** 2) + 1e-6
        norm_vec_x = norm_vec_x / denom_normal
        norm_vec_y = norm_vec_y / denom_normal

        n_xx = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_x,
                                               nharmonic_1=self.kdim[0],
                                               nharmonic_2=self.kdim[1],
                                                 method='FFT')
        n_yy = self.layer_manager._gen_toeplitz2d(norm_vec_y * norm_vec_y,
                                               nharmonic_1=self.kdim[0],
                                               nharmonic_2=self.kdim[1],
                                                 method='FFT')
        n_xy = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_y,
                                               nharmonic_1=self.kdim[0],
                                               nharmonic_2=self.kdim[1],
                                                 method='FFT')
        
        return norm_vec_x, norm_vec_y, n_xx, n_yy, n_xy
        
       
    def get_nv_components(self, layer_index: int):
        if self.layer_manager.layers[layer_index].mask_format is not None:
            nx, ny, _, _, _ = self._calculate_nv_field(mask=self.layer_manager.layers[layer_index].mask_format.squeeze())
            return nx, ny
        else:
            return None, None

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

    def _get_bg(self, layer_index: int, param='er') -> torch.Tensor:
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
                    # ret_mat = torch.tensor(self._matlib[material_name].er).unsqueeze(1)\
                    # .unsqueeze(1).repeat(1, self.rdim[0], self.rdim[1]).to(self.device).to(self.tcomplex)
                    ret_mat = self._matlib[material_name].er.detach().clone().unsqueeze(1).unsqueeze(1).repeat(1, self.rdim[0], self.rdim[1]).to(self.device).to(self.tcomplex)
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur.detach().clone()
                    ret_mat = param_val * torch.ones(size=(self.rdim[0], self.rdim[1]),
                                                  dtype=self.tcomplex, device=self.device)
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

            else:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == 'er':
                    param_val = self._matlib[material_name].er.detach().clone()
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur.detach().clone()
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

                ret_mat = param_val * torch.ones(size=(self.rdim[0], self.rdim[1]),
                                              dtype=self.tcomplex, device=self.device)
        else:
            raise ValueError("The index exceeds the max layer number.")

        return ret_mat


    def expand_dims(self, mat: torch.Tensor) -> Optional[torch.Tensor]:
        """Function that expands the input matrix to a standard output dimension without layer information:
            (n_freqs, n_harmonics, n_harmonics)

        Args:
            mat (torch.Tensor): input tensor matrxi
        """

        ret = None

        if mat.ndim == 1 and mat.shape[0] == self.n_freqs:
            # The input tensor with dimension (n_freqs)
            ret = mat[:, None, None]
        elif mat.ndim == 2:
            # The input matrix with dimension (n_harmonics, n_harmonics)
            ret = mat.unsqueeze(0).repeat(self.n_freqs, 1, 1)
        else:
            raise RuntimeError("Not Listed in the Case")

        return ret


class RCWASolverDouble(FourierBaseSover):
    """RCWASolver.

    This is the implemented solver class for RCWA.
    """
    def __init__(self,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True,
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(lam0, lengthunit, rdim, kdim, materiallist, t1, t2, is_use_FFF, device)

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

        inv_dmat_lam_i = self.to_diag(1 / torch.sqrt(mat_lam_i))


        mat_v_i = q_mat_i @ mat_w_i @ inv_dmat_lam_i
        mat_x_i = torch.linalg.matrix_exp(self.to_diag(- torch.sqrt(mat_lam_i) * self.k_0[:, None] \
            * self.layer_manager.layers[layer_index].thickness.to(self.device))).to(self.tcomplex)

        # Calculate Layer Scattering Matrix
        mat_a_i = tsolve(mat_w_i, mat_w0) + tsolve(mat_v_i, mat_v0)
        mat_b_i = tsolve(mat_w_i, mat_w0) - tsolve(mat_v_i, mat_v0)

        mat_xb_i = mat_x_i @ mat_b_i
        mat_d_i = mat_a_i - mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_b_i

        smat_layer['S11'] = tsolve(
            mat_d_i, mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_a_i - mat_b_i)
        smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
        smat_layer['S21'] = smat_layer['S12']
        smat_layer['S22'] = smat_layer['S11']

        return smat_layer


    def _solve_nonhomo_layer_single_frequency(self,
                             layer_index: int,
                             p_mat_i: torch.Tensor,
                             q_mat_i: torch.Tensor,
                             mat_w0: torch.Tensor,
                             mat_v0: torch.Tensor,
                            freq_index: int) -> dict:
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

        inv_dmat_lam_i = self.to_diag(1 / torch.sqrt(mat_lam_i))

        mat_v_i = q_mat_i @ mat_w_i @ inv_dmat_lam_i
        mat_x_i = torch.linalg.matrix_exp(self.to_diag(- torch.sqrt(mat_lam_i) * self.k_0[freq_index] \
            * self.layer_manager.layers[layer_index].thickness.to(self.device))).to(self.tcomplex)


        # Calculate Layer Scattering Matrix
        mat_a_i = tsolve(mat_w_i, mat_w0) + tsolve(mat_v_i, mat_v0)
        mat_b_i = tsolve(mat_w_i, mat_w0) - tsolve(mat_v_i, mat_v0)

        mat_xb_i = mat_x_i @ mat_b_i
        mat_d_i = mat_a_i - mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_b_i

        smat_layer['S11'] = tsolve(
            mat_d_i, mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_a_i - mat_b_i)
        smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
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
        raise NotImplementedError("Invalid parameter for RCWA")

class RCWARolverFloat(RCWASolverDouble):
    """RCWASolver.

    This is the implemented solver class for RCWA.
    """
    def __init__(self,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True,
                 device: Union[str, torch.device] = 'cpu') -> None:

        self.tcomplex = torch.complex64
        self.tfloat = torch.float32
        self.tint = torch.int32
        self.nfloat = np.float32

        super().__init__(lam0, lengthunit, rdim, kdim, materiallist, t1, t2, is_use_FFF, device)

class RDITSolverDouble(FourierBaseSover):
    """RDITSolver.

    This is the implemented solver class for R-DIT.
    """


    def __init__(self,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True,
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(lam0, lengthunit, rdim, kdim, materiallist, t1, t2, is_use_FFF, device)
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
        delta_h = self.expand_dims(self.k_0) * self.layer_manager.layers[layer_index].thickness.to(self.device) / 2.0
        tmat_a_i = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                        device=self.device).unsqueeze(0).repeat(self.n_freqs, 1, 1)
        tmat_b_i = torch.zeros(size=(self.n_freqs, 2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex,
                          device=self.device)
        tmat_c_i = torch.zeros(size=(self.n_freqs, 2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex,
                          device=self.device)
        tmat_d_i = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                        device=self.device).unsqueeze(0).repeat(self.n_freqs, 1, 1)

        p_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                            device=self.device).unsqueeze(0).repeat(self.n_freqs, 1, 1)
        q_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                            device=self.device).unsqueeze(0).repeat(self.n_freqs, 1, 1)
        
        for irdit_order in range(1, self._rdit_orders + 1):
            if (irdit_order % 2) == 0:  # even orders
                p_fcoef = p_fcoef @ q_mat_i
                q_fcoef = q_fcoef @ p_mat_i
                fac = (delta_h.to(self.tfloat)**irdit_order / math.factorial(irdit_order))
                tmat_a_i = tmat_a_i + fac * p_fcoef
                tmat_d_i = tmat_d_i + fac * q_fcoef
            else:  # odd orders
                p_fcoef = p_fcoef @ p_mat_i
                q_fcoef = q_fcoef @ q_mat_i
                fac = (delta_h.to(self.tfloat)**irdit_order / math.factorial(irdit_order))
                tmat_b_i = tmat_b_i + fac * p_fcoef
                tmat_c_i = tmat_c_i + fac * q_fcoef

        # Construct some helper functions
        a_i_w0 = tmat_a_i @ mat_w0
        b_i_v0 = tmat_b_i @ mat_v0
        c_i_w0 = tmat_c_i @ mat_w0
        d_i_v0 = tmat_d_i @ mat_v0

        mat_g1 = a_i_w0 + b_i_v0
        mat_g2 = -c_i_w0 - d_i_v0

        mat_xx1 = tsolve(mat_g2, c_i_w0 - d_i_v0)
        mat_xx2 = tsolve(mat_g1, a_i_w0 - b_i_v0)

        mat_yyi = mat_xx1 - mat_xx2
        mat_zzi = mat_xx1 + mat_xx2

        smat_layer['S11'] = mat_yyi / 2.0
        smat_layer['S12'] = mat_zzi / 2.0
        smat_layer['S21'] = smat_layer['S12']
        smat_layer['S22'] = smat_layer['S11']
        return smat_layer
    
    def _solve_nonhomo_layer_single_frequency(self,
                                             layer_index: int,
                                             p_mat_i: torch.Tensor,
                                             q_mat_i: torch.Tensor,
                                             mat_w0: torch.Tensor,
                                             mat_v0: torch.Tensor,
                                              freq_index: int) -> dict:
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
        delta_h = self.k_0[freq_index] * self.layer_manager.layers[layer_index].thickness.to(self.device) / 2.0
        tmat_a_i = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                        device=self.device)
        tmat_b_i = torch.zeros(size=(2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex,
                          device=self.device)
        tmat_c_i = torch.zeros(size=(2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex,
                          device=self.device)
        tmat_d_i = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                        device=self.device)
        

        p_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                            device=self.device)
        q_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics, dtype=self.tcomplex,
                            device=self.device)
        
        
        for irdit_order in range(1, self._rdit_orders + 1):
            if (irdit_order % 2) == 0:  # even orders
                p_fcoef = p_fcoef @ q_mat_i
                q_fcoef = q_fcoef @ p_mat_i
                fac = (delta_h.to(self.tfloat)**irdit_order / math.factorial(irdit_order))
                tmat_a_i = tmat_a_i + fac * p_fcoef
                tmat_d_i = tmat_d_i + fac * q_fcoef
            else:  # odd orders
                p_fcoef = p_fcoef @ p_mat_i
                q_fcoef = q_fcoef @ q_mat_i
                fac = (delta_h.to(self.tfloat)**irdit_order / math.factorial(irdit_order))
                tmat_b_i = tmat_b_i + fac * p_fcoef
                tmat_c_i = tmat_c_i + fac * q_fcoef

                
        # Construct some helper functions
        a_i_w0 = tmat_a_i @ mat_w0
        b_i_v0 = tmat_b_i @ mat_v0
        c_i_w0 = tmat_c_i @ mat_w0
        d_i_v0 = tmat_d_i @ mat_v0

        mat_g1 = a_i_w0 + b_i_v0
        mat_g2 = -c_i_w0 - d_i_v0

        mat_xx1 = tsolve(mat_g2, c_i_w0 - d_i_v0)
        mat_xx2 = tsolve(mat_g1, a_i_w0 - b_i_v0)

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

class RDITSolverFloat(RDITSolverDouble):
    """RDITSolver.

    This is the implemented solver class for R-DIT.
    """
    def __init__(self,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True,
                 device: Union[str, torch.device] = 'cpu') -> None:

        self.tcomplex = torch.complex64
        self.tfloat = torch.float32
        self.tint = torch.int32
        self.nfloat = np.float32

        super().__init__(lam0, lengthunit, rdim, kdim, materiallist, t1, t2, is_use_FFF, device)
