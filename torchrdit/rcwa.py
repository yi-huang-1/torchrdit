import numpy as np
import torch
# import datetime

from torch.linalg import inv as tinv
from torch.linalg import solve as tsolve
from .cell import cell3d
from .utils import blockmat2x2, redhstar, eig_complex, init_smatrix
from typing import Union, Optional



class FourierBaseSover(cell3d):
    """Base Class of Fourier Domain Solver"""

    def __init__(self,
                 batch_size: int = 1,  # batch size of data set
                 # wavelengths (frequencies) to be solved
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3, 3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: list = [1.0, 0.0],  # lattice vector in real space
                 t2: list = [0.0, 1.0],  # lattice vector in real space
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(batch_size, lengthunit, rdim,
                         kdim, materiallist, t1, t2, device)

        # Create a task time stamp
        # task_time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Set free space wavelength
        if type(lam0) == self.nfloat or type(lam0) == float:
            self._lam0 = np.array([lam0])
        elif type(lam0) == np.ndarray:
            self._lam0 = lam0

        self.n_freqs = len(self._lam0)

        # Initialize materials, will fit the dispersive profile if applicable
        self._init_dispersive_materials()

        self.n_solve = 0
        self.S_layers = dict()

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
        
        
        source_ditc = dict()
        source_ditc['theta'] = theta
        source_ditc['phi'] = phi
        source_ditc['pte'] = pte
        source_ditc['ptm'] = ptm
        source_ditc['norm_te_dir'] = norm_te_dir

        return source_ditc

    def _solve_nonhomo_layer(self,
                             layer_index: int,
                             Pi: torch.Tensor,
                             Qi: torch.Tensor,
                             W0: torch.Tensor,
                             V0: torch.Tensor) -> dict:
        """_solve_nonhomo_layer.

        Fucntion of solving non-homogenous layers in the parent class. It needs to be implemented
        in the child classes with concrete algorithm.

        Args:
            layer_index (int): layer_index
            Pi (torch.Tensor): Pi
            Qi (torch.Tensor): Qi
            W0 (torch.Tensor): W0
            V0 (torch.Tensor): V0

        Returns:
            dict:
        """
        raise NotImplementedError("self._solve not implemented.")

    def _solve(self) -> dict:

        # kinc: dimensions (n_freqs, 3)
        kinc = self.n1.unsqueeze(-1) * torch.tensor([np.sin(self.src['theta']) * np.cos(self.src['phi']),
                                                     np.sin(self.src['theta']) *
                                                     np.sin(self.src['phi']),
                                                     np.cos(self.src['theta'])],
                                                    dtype=self.tfloat,
                                                    device=self.device).unsqueeze(0).repeat(self.n_freqs, 1)

        # Kx0, Ky0: (n_batches, n_freqs, kHW, kHw)
        Kx0 = kinc[:, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - (self.FP.unsqueeze(0).unsqueeze(0) * self.T1[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.FQ.unsqueeze(0).unsqueeze(0)
                                                                     * self.T2[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / self.k0.unsqueeze(0).unsqueeze(-1).unsqueeze(-2) + 0*1j
        Ky0 = kinc[:, 1].unsqueeze(-1).unsqueeze(-2) - (self.FP.unsqueeze(0).unsqueeze(0) * self.T1[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.FQ.unsqueeze(0).unsqueeze(0)
                                                        * self.T2[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / self.k0.unsqueeze(0).unsqueeze(-1).unsqueeze(-2) + 0*1j


        if self.layer_manager.is_ref_dispersive == False:
            Kzref0 = torch.conj(torch.sqrt(
                torch.conj(self.ur1)*torch.conj(self.er1) - Kx0 ** 2 - Ky0 ** 2 + 0*1j))
        else:
            Kzref0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur1)*torch.conj(self.er1)).unsqueeze(-1).unsqueeze(-2) - Kx0 ** 2 - Ky0 ** 2 + 0*1j))

        if self.layer_manager.is_trn_dispersive == False:
            Kztrn0 = torch.conj(torch.sqrt(
                torch.conj(self.ur2)*torch.conj(self.er2) - Kx0 ** 2 - Ky0 ** 2 + 0*1j))
        else:
            Kztrn0 = torch.conj(torch.sqrt(
                (torch.conj(self.ur2)*torch.conj(self.er2)).unsqueeze(-1).unsqueeze(-2) - Kx0 ** 2 - Ky0 ** 2 + 0*1j))

        # Transform to diagnal matrices
        kHW = self.kdim[0] * self.kdim[1]

        Ik = torch.eye(kHW, dtype=self.tcomplex, device=self.device)

        Kx = Kx0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * Ik
        Ky = Ky0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * Ik
        Kzref = Kzref0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * Ik
        Kztrn = Kztrn0.transpose(
            dim0=-2, dim1=-1).flatten(start_dim=-2).unsqueeze(-2) * Ik


        I = Ik.unsqueeze(0).unsqueeze(0).expand(self.n_batches, self.n_freqs, -1, -1)
        Z = torch.zeros(
            size=(self.n_batches, self.n_freqs ,kHW, kHW), dtype=self.tcomplex, device=self.device)

        # Calculate eigen-modes of the gap medium
        Kz = torch.conj(torch.sqrt(I - Kx @ Kx - Ky @ Ky))

        Q = blockmat2x2([[Kx @ Ky, I - Kx @ Kx], [Ky @ Ky - I, -Kx @ Ky]])
        W0 = blockmat2x2([[I, Z], [Z, I]])
        LAM = blockmat2x2([[1j*Kz, Z], [Z, 1j*Kz]])
        V0 = Q @ tinv(LAM)

        # Initialize global scattering matrix
        Sg = init_smatrix(shape=(self.n_batches, self.n_freqs,
                          2*kHW, 2*kHW), dtype=self.tcomplex, device=self.device)
        Si = dict()

        # ============================================================================
        # Main Loop
        # ============================================================================
        # Build Eigen Value Problem

        for nl in range(self.layer_manager.nlayer):
            # Transform dimensions to (n_batches, n_freqs, kHW, kHW)
            if (self.layer_manager.layers[nl].is_solved == False) or (self.layer_manager.layers[nl].is_optimize == True):

                # no need to solve the non-optimized layers in 'opt' mode.
                if self.solver_mode == 'opt':
                    self.layer_manager.layers[nl].is_solved = True

                if self.layer_manager.layers[nl].is_homogeneous == False:
                    if self.layer_manager.layers[nl].is_dispersive == False:
                        er = self.expand_dims(self.layer_manager.layers[nl].kermat)
                    else:
                        er = self.layer_manager.layers[nl].kermat

                    # permeability always non-dispersive
                    # Transform dimensions to (n_batches, n_freqs, kHW, kHW)
                    ur = self.expand_dims(self.layer_manager.layers[nl].kurmat)

                    Pi = blockmat2x2([[Kx @ tinv(er) @ Ky,
                                       ur - Kx @ tinv(er) @ Kx],
                                      [Ky @ tinv(er) @ Ky - ur,
                                     - Ky @ tinv(er) @ Kx]])

                    Qi = blockmat2x2([[Kx @ tinv(ur) @ Ky,
                                       er - Kx @ tinv(ur) @ Kx],
                                      [Ky @ tinv(ur) @ Ky - er,
                                     - Ky @ tinv(ur) @ Kx]])

                    Si = self._solve_nonhomo_layer(layer_index=nl,
                                              Pi=Pi,
                                              Qi=Qi,
                                              W0=W0,
                                              V0=V0)

                elif self.layer_manager.layers[nl].is_homogeneous == True:
                    if self.layer_manager.layers[nl].is_dispersive == True:
                        er = self.expand_dims(torch.tensor(
                            self._matlib[self.layer_manager.layers[nl].material_name].er, device=self.device)).to(self.tcomplex)
                        ur = torch.tensor(
                            self._matlib[self.layer_manager.layers[nl].material_name].ur, device=self.device).to(self.tcomplex)

                    else:
                        er = torch.tensor(
                            self._matlib[self.layer_manager.layers[nl].material_name].er, device=self.device)
                        ur = torch.tensor(
                            self._matlib[self.layer_manager.layers[nl].material_name].ur, device=self.device)

                    Qi = 1 / ur * blockmat2x2([[Kx @ Ky,
                                                ur*er*I - Kx @  Kx],
                                               [Ky @ Ky - ur*er*I,
                                                - Ky @ Kx]])

                    Kzi = torch.conj(torch.sqrt(
                        torch.conj(ur)*torch.conj(er)*I - Kx ** 2 - Ky ** 2 + 0*1j)).to(self.tcomplex)

                    dLAMi = blockmat2x2(
                        [[1j*Kzi, Z], [Z, 1j*Kzi]])

                    Wi = blockmat2x2([[I, Z], [Z, I]])
                    Vi = Qi @ tinv(dLAMi)

                    Xi = torch.linalg.matrix_exp(- dLAMi * self.expand_dims(self.k0)
                                                 * self.expand_dims(self.layer_manager.layers[nl].thickness.to(self.device))).to(self.tcomplex)

                    # Calculate Layer Scattering Matrix
                    atwi = tsolve(Wi, W0)
                    atvi = tsolve(Vi, V0)
                    Ai = atwi + atvi
                    Bi = atwi - atvi
                    Di = Ai - Xi @ Bi @ tinv(Ai) @ Xi @ Bi

                    Si['S11'] = tsolve(
                        Di, Xi @ Bi @ tinv(Ai) @ Xi @ Ai - Bi)
                    Si['S12'] = tsolve(Di, Xi) @ (Ai - Bi @ tinv(Ai) @ Bi)
                    Si['S21'] = Si['S12']
                    Si['S22'] = Si['S11']

                self.S_layers[nl] = Si

            # Update global scattering matrix
            Sg = redhstar(Sg, self.S_layers[nl])

        # ============================================================================
        # End Main Loop
        # ============================================================================

        # Connect to reflection region
        if self.layer_manager.is_ref_dispersive == False:
            Q1 = (1/self.ur1) * blockmat2x2([[Kx @ Ky,
                                            self.ur1 * self.er1 * I - Kx @ Kx],
                                            [Ky @ Ky - self.ur1 * self.er1 * I,
                                            - Ky @ Kx]])
        else:
            Q1 = (1/self.ur1) * blockmat2x2([[Kx @ Ky,
                                            self.expand_dims(self.ur1 * self.er1) * I - Kx @ Kx],
                                            [Ky @ Ky - self.expand_dims(self.ur1 * self.er1) * I,
                                            - Ky @ Kx]])
        Wref = blockmat2x2([[I, Z], [Z, I]])
        LAMref = blockmat2x2([[1j * Kzref, Z], [Z, 1j*Kzref]])
        Vref = Q1 @ tinv(LAMref)

        S1 = dict()
        atw1 = tsolve(W0, Wref)
        atv1 = tsolve(V0, Vref)
        A1 = atw1 + atv1
        B1 = atw1 - atv1
        S1['S11'] = - tsolve(A1, B1)
        S1['S12'] = 2 * tinv(A1)
        S1['S21'] = 1/2 * (A1 - B1 @ tinv(A1) * B1)
        S1['S22'] = B1 @ tinv(A1)

        Sg = redhstar(S1, Sg)

        # Connect to transmission region
        if self.layer_manager.is_trn_dispersive == False:
            Q2 = (1/self.ur2) * blockmat2x2([[Kx @ Ky,
                                            self.ur2 * self.er2 * I - Kx @ Kx],
                                            [Ky @ Ky - self.ur2 * self.er2 * I,
                                            - Ky @ Kx]])
        else:
            Q2 = (1/self.ur2) * blockmat2x2([[Kx @ Ky,
                                              self.expand_dims(self.ur2 * self.er2) * I - Kx @ Kx],
                                             [Ky @ Ky - self.expand_dims(self.ur2 * self.er2) * I,
                                              - Ky @ Kx]])

        Wtrn = blockmat2x2([[I, Z], [Z, I]])
        LAMtrn = blockmat2x2([[1j * Kztrn, Z], [Z, 1j*Kztrn]])
        Vtrn = Q2 @ tinv(LAMtrn)

        S2 = dict()
        atw2 = tsolve(W0, Wtrn)
        atv2 = tsolve(V0, Vtrn)
        A2 = atw2 + atv2
        B2 = atw2 - atv2
        S2['S11'] = B2 @ tinv(A2)
        S2['S12'] = 1/2 * (A2 - B2 @ tinv(A2) * B2)
        S2['S21'] = 2 * tinv(A2)
        S2['S22'] = - tsolve(A2, B2)

        Sg = redhstar(Sg, S2)

        # Compute polarization vector
        nn = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self.tcomplex, device=self.device)

        if np.abs(self.src['theta']) < 1e-3:
            if self.src['norm_te_dir'] == 'y':
                ate = torch.tensor(
                    [0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device).unsqueeze(0).repeat(self.n_freqs, 1)
            elif self.src['norm_te_dir'] == 'x':
                ate = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=self.tcomplex, device=self.device).unsqueeze(0).repeat(self.n_freqs, 1)
        else:
            ate = torch.cross(kinc, nn.unsqueeze(0).repeat(self.n_freqs, 1))
            ate = ate / torch.norm(ate, dim=1).unsqueeze(-1)

        atm = torch.cross(ate, kinc, dim=1)
        atm = atm / torch.norm(atm)

        EP = self.src['pte'] * ate + self.src['ptm'] * atm
        EP = EP / torch.norm(EP, dim=1).unsqueeze(-1)

        # Calculate electric field source vector
        delta = torch.zeros(
            size=(self.n_freqs, kHW), dtype=self.tcomplex, device=self.device)
        p0 = np.floor(self.kdim[0]/2)
        q0 = np.floor(self.kdim[1]/2)
        m0 = np.int32(q0 * self.kdim[0] + p0)
        delta[:, m0] = 1
        esrc = torch.cat((EP[:, 0].unsqueeze(-1) * delta,
                         EP[:, 1].unsqueeze(-1) * delta), dim=1)

        # Calculate source vectors
        csrc = tsolve(Wref, esrc.unsqueeze(0).unsqueeze(-1))

        # Calculate reflected fields
        cref = Sg['S11'] @ csrc
        eref = Wref @ cref

        rx = eref[:, :, 0: kHW, :]
        ry = eref[:, :, kHW: 2*kHW, :]

        rz = - (Kx @ tinv(Kzref)) @ rx \
             - (Ky @ tinv(Kzref)) @ ry

        # Calculate transmitted fields
        ctrn = Sg['S21'] @ csrc
        etrn = Wtrn @ ctrn
        tx = etrn[:, :, 0: kHW, :]
        ty = etrn[:, :, kHW: 2*kHW, :]
        tz = - (Kx @ tinv(Kztrn)) @ tx \
             - (Ky @ tinv(Kztrn)) @ ty

        # Calculate diffraction efficiences
        RDE = torch.reshape(torch.real(self.ur2/self.ur1*Kzref / kinc[:, 2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) @
                            (torch.real(rx) ** 2 + torch.imag(rx) ** 2 +
                             torch.real(ry) ** 2 + torch.imag(ry) ** 2 +
                             + torch.real(rz) ** 2 + torch.imag(rz) ** 2),
                            shape=(self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)

        TDE = torch.reshape(torch.real(self.ur1/self.ur2*Kztrn / kinc[:, 2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) @
                            (torch.real(tx) ** 2 + torch.imag(tx) ** 2 +
                             torch.real(ty) ** 2 + torch.imag(ty) ** 2 +
                             + torch.real(tz) ** 2 + torch.imag(tz) ** 2),
                            shape=(self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)

        # Calculate overall reflectance & Transmittance
        REF = torch.sum(RDE, dim=(-1, -2))
        TRN = torch.sum(TDE, dim=(-1, -2))

        # Store output data
        data = dict()
        data['rx'] = torch.reshape(rx, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['ry'] = torch.reshape(ry, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['rz'] = torch.reshape(rz, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['tx'] = torch.reshape(tx, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['ty'] = torch.reshape(ty, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['tz'] = torch.reshape(tz, shape=(
            self.n_batches, self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        data['RDE'] = RDE
        data['TDE'] = TDE
        data['REF'] = REF
        data['TRN'] = TRN
        data['kzref'] = Kzref
        data['kztrn'] = Kztrn
        data['kinc'] = kinc
        data['kx'] = torch.squeeze(Kx0)
        data['ky'] = torch.squeeze(Ky0)

        return data

    def solve(self,
              source: dict) -> dict:

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
        self.n1 = torch.sqrt(self.ur1 * self.er1)
        self.n2 = torch.sqrt(self.ur2 * self.er2)

        # Calculate reciprocal lattice vectors
        dv = self.t1[:, 0] * self.t2[:, 1] - self.t2[:, 0] * self.t1[:, 1]

        self.T1 = 2 * torch.pi * torch.cat(((+self.t2[:, 1] / dv).unsqueeze(-1), (-self.t2[:, 0] / dv).unsqueeze(-1)), dim=1)
        self.T2 = 2 * torch.pi * torch.cat(((-self.t1[:, 1] / dv).unsqueeze(-1), (+self.t1[:, 0] / dv).unsqueeze(-1)), dim=1)


        # Calculate wave vector expansion
        self.tlam0 = torch.tensor(
            self.lam0, dtype=torch.float64, device=self.device)
        self.k0 = 2 * torch.pi / self.tlam0  # k0 with dimensions: n_freqs

        fp = torch.arange(start=-np.floor(self.kdim[0] / 2), end=np.floor(
            self.kdim[0] / 2) + 1, dtype=torch.int32, device=self.device)
        fq = torch.arange(start=-np.floor(self.kdim[1] / 2), end=np.floor(
            self.kdim[1] / 2) + 1, dtype=torch.int32, device=self.device)
        [self.FQ, self.FP] = torch.meshgrid(fq, fp, indexing='xy')

        # check if options are set correctly
        for nl in range(self.layer_manager.nlayer):
            if self.layer_manager.layers[nl].is_homogeneous == False:
                if self.layer_manager.layers[nl].ermat == None:
                    # if not homogenous material, must be with a pattern.
                    # if no pattern assigned before solving, the material will be set as homogeneous
                    self.layer_manager.replace_layer_to_homogeneous(layer_index=nl)
                    self.n_solve = 0

                    print(
                        f"Warning: Layer {nl} has no pattern assigned, and was changed to homogeneous")


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
        elif (ndim1 != self.rdim[0]) or (ndim2 != self.rdim[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)
            self.n_solve = 0

        er_bg = self._get_bg(layer_index=layer_index, param='er')

        if self.layer_manager.layers[layer_index].is_dispersive == False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        if bg_material == 'air':
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                (er_bg - 1) * mask_format

        if set_grad == True:
            self.layer_manager.layers[layer_index].ermat.requires_grad = True
        self.layer_manager.layers[layer_index].kermat = self.gen_Toeplitz_matrix(
            layer_index=layer_index, param='er', is_dispersive=self.layer_manager.layers[layer_index].is_dispersive)

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        if self.n_solve == 0:
            self.layer_manager.layers[layer_index].urmat = self._get_bg(
                layer_index=layer_index, param='ur')
            self.layer_manager.layers[layer_index].kurmat = self.gen_Toeplitz_matrix(
                layer_index=layer_index, param='ur', is_dispersive=False)

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
        if layer_index < self.layer_manager.nlayer:
            if self.layer_manager.layers[layer_index].is_dispersive == True:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == 'er':
                    return torch.tensor(self._matlib[material_name].er).unsqueeze(0).repeat(self.n_batches, 1).unsqueeze(2).unsqueeze(2).repeat(1, 1, self.rdim[0], self.rdim[1]).to(self.device)
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur
                    return param_val * torch.ones(size=(self.n_batches, self.rdim[0], self.rdim[1]), dtype=self.tcomplex, device=self.device)
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

                return param_val * torch.ones(size=(self.n_batches, self.rdim[0], self.rdim[1]), dtype=self.tcomplex, device=self.device)
        else:
            raise ValueError("The index exceeds the max layer number.")

    def _init_dispersive_materials(self) -> None:
        """_init_dispersive_materials.
        
        Initialize the dispersive profile of materials.

        Args:

        Returns:
            None:
        """
        for imat in self._matlib.values():
            if imat.isdispersive_er == True:
                if type(imat.er) == type(None):
                    imat.load_dispersive_er(
                        self._lam0, self._lenunit)

    def expand_dims(self, mat: torch.Tensor) -> Optional[torch.Tensor]:
        """Function that expands the input matrix to a standard output dimension without layer information:
            (n_batches, n_freqs, kHW, kHW)

        Args:
            mat (torch.Tensor): input tensor matrxi
        """
        if mat.ndim == 1 and mat.shape[0] == self.n_freqs:
            # The input tensor with dimension (n_freqs)
            return mat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        elif mat.ndim == 1 and mat.shape[0] == self.n_batches:
            return mat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif mat.ndim == 2:
            # The input matrix with dimension (kHW, kHW)
            return mat.unsqueeze(0).unsqueeze(0).repeat(1, self.n_freqs, 1, 1)
        elif mat.ndim == 3 and mat.shape[0] == self.n_freqs:
            # The input matrix with dimension (n_freq, kHW, kHW)
            return mat.unsqueeze(0)
        elif mat.ndim == 3:
            # The input matrix with dimension (n_batches, kHW, kHW)
            if mat.shape[0] == 1:
                return mat.unsqueeze(1).repeat(self.n_batches, self.n_freqs, 1, 1)
            elif mat.shape[0] == self.n_batches:
                return mat.unsqueeze(1).repeat(1, self.n_freqs, 1, 1)
        elif mat.ndim == 4 and mat.shape[1] == self.n_freqs and mat.shape[0] == 1:
            # The input matrix with dimension (n_batches, n_freqs, kHW, kHW)
            return mat.repeat(self.n_batches, 1, 1, 1)
        else:
            raise RuntimeError("Not Listed in the Case")

class RCWASolver(FourierBaseSover):

    def __init__(self,
                 batch_size: int = 1,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],
                 kdim: list = [3, 3],
                 materiallist: list = [],
                 t1: list = [1, 0],
                 t2: list = [0, 1],
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(batch_size, lam0, lengthunit, rdim, kdim, materiallist, t1, t2, device)

    def _solve_nonhomo_layer(self,
                             layer_index: int,
                             Pi: torch.Tensor,
                             Qi: torch.Tensor,
                             W0: torch.Tensor,
                             V0: torch.Tensor) -> dict:
        """_solve_nonhomo_layer.

        Solves the non-homogenous layer using RCWA methods.

        Args:
            layer_index (int): layer_index
            Pi (torch.Tensor): Pi
            Qi (torch.Tensor): Qi
            W0 (torch.Tensor): W0
            V0 (torch.Tensor): V0

        Returns:
            dict:
        """
        nl = layer_index
        kHW = self.kdim[0] * self.kdim[1]

        Si = dict()

        # Compute Eigen Modes
        LAMi, Wi = eig_complex.apply(Pi @ Qi)

        dLAMi = torch.sqrt(LAMi).unsqueeze(-2) * \
            torch.eye(LAMi.shape[-1],
                      dtype=self.tcomplex, device=self.device)

        Vi = Qi @ Wi @ tinv(dLAMi)
        Xi = torch.linalg.matrix_exp(- dLAMi * self.expand_dims(self.k0)
                                     * self.expand_dims(self.layer_manager.layers[nl].thickness.to(self.device))).to(self.tcomplex)
        
        # Calculate Layer Scattering Matrix
        Ai = tsolve(Wi, W0) + tsolve(Vi, V0)
        Bi = tsolve(Wi, W0) - tsolve(Vi, V0)
        Di = Ai - Xi @ Bi @ tinv(Ai) @ Xi @ Bi

        Si['S11'] = tsolve(
            Di, Xi @ Bi @ tinv(Ai) @ Xi @ Ai - Bi)
        Si['S12'] = tsolve(Di, Xi) @ (Ai - Bi @ tinv(Ai) @ Bi)
        Si['S21'] = Si['S12']
        Si['S22'] = Si['S11']

        return Si


class RDITSolver(FourierBaseSover):

    def __init__(self,
                 batch_size: int = 1,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],
                 kdim: list = [3, 3],
                 materiallist: list = [],
                 t1: list = [1, 0],
                 t2: list = [0, 1],
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(batch_size, lam0, lengthunit, rdim, kdim, materiallist, t1, t2, device)
        self._rdit_orders = 10

    def _solve_nonhomo_layer(self,
                             layer_index: int,
                             Pi: torch.Tensor,
                             Qi: torch.Tensor,
                             W0: torch.Tensor,
                             V0: torch.Tensor) -> dict:
        """_solve_nonhomo_layer.

        Solves the non-homogenous layer using R-DIT methods.

        Args:
            layer_index (int): layer_index
            Pi (torch.Tensor): Pi
            Qi (torch.Tensor): Qi
            W0 (torch.Tensor): W0
            V0 (torch.Tensor): V0

        Returns:
            dict:
        """
        nl = layer_index
        kHW = self.kdim[0] * self.kdim[1]

        Si = dict()

        # Construct T matrix
        delta_h = self.expand_dims(self.k0) * self.expand_dims(self.layer_manager.layers[nl].thickness.to(self.device)) / 2.0
        TAi = torch.eye(2*kHW, 2*kHW, dtype=self.tcomplex,
                        device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)
        TBi = torch.zeros(size=(self.n_batches, self.n_freqs, 2*kHW, 2*kHW), dtype=self.tcomplex,
                          device=self.device)
        TCi = torch.zeros(size=(self.n_batches, self.n_freqs, 2*kHW, 2*kHW), dtype=self.tcomplex,
                          device=self.device)
        TDi = torch.eye(2*kHW, 2*kHW, dtype=self.tcomplex,
                        device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)

        p_fcoef = torch.eye(2*kHW, 2*kHW, dtype=self.tcomplex,
                            device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)
        q_fcoef = torch.eye(2*kHW, 2*kHW, dtype=self.tcomplex,
                            device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_batches, self.n_freqs, 1, 1)

        for irdit_order in range(1, self._rdit_orders + 1):
            if (irdit_order % 2) == 0:  # even orders
                p_fcoef = p_fcoef @ Qi
                q_fcoef = q_fcoef @ Pi
                fac = (delta_h.to(self.tfloat)**irdit_order / np.math.factorial(irdit_order))
                TAi = TAi + fac * p_fcoef
                TDi = TDi + fac * q_fcoef
            else:  # odd orders
                p_fcoef = p_fcoef @ Pi
                q_fcoef = q_fcoef @ Qi
                fac = (delta_h.to(self.tfloat)**irdit_order / np.math.factorial(irdit_order))
                TBi = TBi + fac * p_fcoef
                TCi = TCi + fac * q_fcoef

        # Construct some helper functions
        G1 = TAi @ W0 + TBi @ V0
        G2 = - TCi @ W0 - TDi @ V0

        XX1 = tsolve(G2, TCi @ W0 - TDi @ V0)
        XX2 = tsolve(G1, TAi @ W0 - TBi @ V0)

        YYi = XX1 - XX2
        ZZi = XX1 + XX2

        Si['S11'] = YYi / 2.0
        Si['S12'] = ZZi / 2.0
        Si['S21'] = Si['S12']
        Si['S22'] = Si['S11']

        return Si

    def set_rdit_order(self, rdit_order):
        self._rdit_orders = rdit_order

