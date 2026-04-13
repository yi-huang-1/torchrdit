"""External region connection for electromagnetic solvers.

Connects the global scattering matrix to reflection and transmission
regions. Extracted from ``solver.py`` to reduce module size.
"""

from __future__ import annotations

from torch.linalg import inv as tinv
from torch.linalg import solve as tsolve

from .numerics import safe_kz_reciprocal
from .utils import SMatrix, blockmat2x2, redhstar, to_diag_util


class ExternalRegionsMixin:
    """Mixin providing ``_connect_external_regions`` for FourierBaseSolver."""

    def _connect_external_regions(self, smat_global, matrices):
        """Connect the global scattering matrix to reflection and transmission regions.

        Args:
            smat_global: The global scattering matrix to connect to external regions
            matrices: Dictionary of matrices from setup_common_matrices

        Returns:
            tuple: (Updated global scattering matrix, V matrix for reflection, V matrix for transmission)
        """

        # Connect to reflection region — differentiable kz protection (see numerics.py)
        inv_mat_lam_ref = safe_kz_reciprocal(matrices["mat_kz_ref"], dtype=self.tcomplex)

        if self.layer_manager.is_ref_dispersive is False:
            mat_v_ref = (1 / self.ur1) * blockmat2x2(
                [
                    [
                        to_diag_util(matrices["mat_kx_ky"] * inv_mat_lam_ref, self.harmonics),
                        to_diag_util((self.ur1 * self.er1 - matrices["mat_kx_kx"]) * inv_mat_lam_ref, self.harmonics),
                    ],
                    [
                        to_diag_util((matrices["mat_ky_ky"] - self.ur1 * self.er1) * inv_mat_lam_ref, self.harmonics),
                        -to_diag_util(matrices["mat_ky_kx"] * inv_mat_lam_ref, self.harmonics),
                    ],
                ]
            )
        else:
            mat_v_ref = (1 / self.ur1) * blockmat2x2(
                [
                    [
                        to_diag_util(matrices["mat_kx_ky"] * inv_mat_lam_ref, self.harmonics),
                        to_diag_util(
                            ((self.ur1 * self.er1)[:, None] - matrices["mat_kx_kx"]) * inv_mat_lam_ref, self.harmonics
                        ),
                    ],
                    [
                        to_diag_util(
                            (matrices["mat_ky_ky"] - (self.ur1 * self.er1)[:, None]) * inv_mat_lam_ref, self.harmonics
                        ),
                        -to_diag_util(matrices["mat_ky_kx"] * inv_mat_lam_ref, self.harmonics),
                    ],
                ]
            )

        mat_w_ref = self.ident_mat_k2.unsqueeze(0).expand(self.n_freqs, -1, -1)

        # Calculate reflection region scattering matrix
        atw1 = mat_w_ref
        atv1 = tsolve(matrices["mat_v0"], mat_v_ref)

        mat_a_1 = atw1 + atv1
        mat_b_1 = atw1 - atv1
        inv_mat_a_1 = tinv(mat_a_1)
        inv_mat_a_1_mat_b_1 = inv_mat_a_1 @ mat_b_1
        smat_ref = SMatrix(
            S11=-inv_mat_a_1_mat_b_1,
            S12=2 * inv_mat_a_1,
            S21=0.5 * (mat_a_1 - mat_b_1 @ inv_mat_a_1_mat_b_1),
            S22=mat_b_1 @ inv_mat_a_1,
        )

        smat_global = redhstar(smat_ref, smat_global)

        # Connect to transmission region — differentiable kz protection (see numerics.py)
        inv_mat_lam_trn = safe_kz_reciprocal(matrices["mat_kz_trn"], dtype=self.tcomplex)

        if self.layer_manager.is_trn_dispersive is False:
            mat_v_trn = (1 / self.ur2) * blockmat2x2(
                [
                    [
                        to_diag_util(matrices["mat_kx_ky"] * inv_mat_lam_trn, self.harmonics),
                        to_diag_util((self.ur2 * self.er2 - matrices["mat_kx_kx"]) * inv_mat_lam_trn, self.harmonics),
                    ],
                    [
                        to_diag_util((matrices["mat_ky_ky"] - self.ur2 * self.er2) * inv_mat_lam_trn, self.harmonics),
                        -to_diag_util(matrices["mat_ky_kx"] * inv_mat_lam_trn, self.harmonics),
                    ],
                ]
            )
        else:
            mat_v_trn = (1 / self.ur2) * blockmat2x2(
                [
                    [
                        to_diag_util(matrices["mat_kx_ky"] * inv_mat_lam_trn, self.harmonics),
                        to_diag_util(
                            ((self.ur2 * self.er2)[:, None] - matrices["mat_kx_kx"]) * inv_mat_lam_trn, self.harmonics
                        ),
                    ],
                    [
                        to_diag_util(
                            (matrices["mat_ky_ky"] - (self.ur2 * self.er2)[:, None]) * inv_mat_lam_trn, self.harmonics
                        ),
                        -to_diag_util(matrices["mat_ky_kx"] * inv_mat_lam_trn, self.harmonics),
                    ],
                ]
            )

        mat_w_trn = mat_w_ref

        # Calculate transmission region scattering matrix
        atw2 = mat_w_trn
        atv2 = tsolve(matrices["mat_v0"], mat_v_trn)

        mat_a_2 = atw2 + atv2
        mat_b_2 = atw2 - atv2
        inv_mat_a_2 = tinv(mat_a_2)
        inv_mat_a_2_mat_b_2 = inv_mat_a_2 @ mat_b_2
        smat_trn = SMatrix(
            S11=mat_b_2 @ inv_mat_a_2,
            S12=0.5 * (mat_a_2 - mat_b_2 @ inv_mat_a_2_mat_b_2),
            S21=2 * inv_mat_a_2,
            S22=-inv_mat_a_2_mat_b_2,
        )

        smat_global = redhstar(smat_global, smat_trn)

        return smat_global, mat_v_ref, mat_v_trn
