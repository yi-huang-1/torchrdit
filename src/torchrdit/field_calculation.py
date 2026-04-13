"""Field and efficiency calculations for electromagnetic solvers.

Computes polarization vectors, electromagnetic fields, and diffraction
efficiencies. Extracted from ``solver.py`` to reduce module size.
"""

import torch
from torch.linalg import solve as tsolve

from .numerics import softplus_floor, softplus_magnitude_floor
from .utils import to_diag_util


class FieldCalculationMixin:
    """Mixin providing field/efficiency methods for FourierBaseSolver."""

    def _calculate_polarization(self, sources=None, kinc=None):
        """Unified polarization calculation for both single and batched sources.

        This method automatically detects whether it's dealing with a single source
        or multiple sources and handles both cases using tensorized operations.

        Args:
            sources: Optional list of source dictionaries for batched processing.
                    If None, uses self.src for single source processing.
            kinc: Optional kinc tensor to use instead of self.kinc. If None, uses self.kinc.

        Returns:
            Dictionary containing:
            - ate: TE polarization vectors
            - atm: TM polarization vectors
            - pol_vec: Combined polarization vectors
            - esrc: Electric field source vectors

            Shapes:
            - Single source: (n_freqs, 3) for vectors, (n_freqs, 2*n_harmonics_squared) for esrc
            - Batched sources: (n_sources, n_freqs, 3) for vectors, (n_sources, n_freqs, 2*n_harmonics_squared) for esrc
        """
        # Use passed kinc parameter or fall back to self.kinc
        if kinc is None:
            kinc = self.kinc

        # Detect input type and normalize
        if sources is None:
            # Single source mode - use self.src
            is_single = True
            sources_list = [self.src]
        elif isinstance(sources, dict):
            # Single source passed as dict
            is_single = True
            sources_list = [sources]
        else:
            # Batched sources
            is_single = False
            sources_list = sources

        n_sources = len(sources_list)
        n_harmonics_squared = self.harmonics[0] * self.harmonics[1]

        # Use tensorized parameter collection while preserving autograd
        def _to_float_tensor_preserve(x):
            if isinstance(x, torch.Tensor):
                # Ensure device/dtype match; preserves graph
                return x.to(device=self.device, dtype=self.tfloat)
            else:
                return torch.tensor(x, device=self.device, dtype=self.tfloat)

        def _to_complex_tensor_preserve(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=self.tcomplex)
            else:
                return torch.tensor(x, device=self.device, dtype=self.tcomplex)

        theta_batch = torch.stack([_to_float_tensor_preserve(src["theta"]) for src in sources_list], dim=0)
        pte_batch = torch.stack([_to_complex_tensor_preserve(src["pte"]) for src in sources_list], dim=0)
        ptm_batch = torch.stack([_to_complex_tensor_preserve(src["ptm"]) for src in sources_list], dim=0)

        # Initialize polarization tensors
        ate_batch = torch.zeros(n_sources, self.n_freqs, 3, dtype=self.tcomplex, device=self.device)
        atm_batch = torch.zeros(n_sources, self.n_freqs, 3, dtype=self.tcomplex, device=self.device)

        # Compute TE polarization for all sources uniformly (no data-dependent branching).
        # Ensure kinc is always 3D: (n_sources, n_freqs, 3)
        if kinc.dim() == 2:
            kinc_3d = kinc[None, :, :].expand(n_sources, -1, -1)
        else:
            kinc_3d = kinc.expand(n_sources, -1, -1)

        # Oblique: ate = cross(kinc, z_hat), normalized
        norm_vec = torch.tensor([0.0, 0.0, 1.0], dtype=self.tcomplex, device=self.device)
        norm_vec_expanded = norm_vec.reshape(1, 1, 3).expand_as(kinc_3d)
        ate_oblique = torch.cross(kinc_3d, norm_vec_expanded, dim=2)
        ate_oblique = ate_oblique / (torch.norm(ate_oblique, dim=2, keepdim=True) + 1e-10)

        # Normal: ate = [0, 1, 0]
        ate_normal = torch.zeros_like(ate_oblique)
        ate_normal[:, :, 1] = 1.0

        # Select per-source via torch.where (compile-friendly, no data-dependent branch).
        # theta_batch is (B,) for scalar-per-source or (B, F) for per-frequency.
        # Reduce to per-source mask (B,) then unsqueeze for (B, 1, 1) broadcasting.
        is_normal = torch.abs(theta_batch) < 1e-3
        if is_normal.dim() > 1:
            is_normal = is_normal.all(dim=-1)  # per-frequency -> per-source
        normal_mask = is_normal.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        ate_batch = torch.where(normal_mask, ate_normal, ate_oblique)

        kinc_for_atm = kinc_3d

        atm_batch = torch.cross(ate_batch, kinc_for_atm, dim=2)
        atm_norm = torch.norm(atm_batch, dim=2, keepdim=True)
        atm_batch = atm_batch / (atm_norm + 1e-10)

        # Create polarization vectors using broadcasting
        pte_vec = pte_batch[:, None, None] * ate_batch
        ptm_vec = ptm_batch[:, None, None] * atm_batch

        pol_vec = pte_vec + ptm_vec
        pol_vec_norm = torch.norm(pol_vec, dim=2, keepdim=True)
        pol_vec = pol_vec / (pol_vec_norm + 1e-10)

        # Create electric field source vectors
        delta = torch.zeros(
            size=(n_sources, self.n_freqs, n_harmonics_squared), dtype=self.tcomplex, device=self.device
        )
        delta[:, :, n_harmonics_squared // 2] = 1.0

        esrc = torch.zeros(
            size=(n_sources, self.n_freqs, 2 * n_harmonics_squared), dtype=self.tcomplex, device=self.device
        )
        esrc[:, :, :n_harmonics_squared] = pol_vec[:, :, 0].unsqueeze(-1) * delta
        esrc[:, :, n_harmonics_squared:] = pol_vec[:, :, 1].unsqueeze(-1) * delta

        # Remove source dimension for single source case
        if is_single:
            ate_batch = ate_batch.squeeze(0)
            atm_batch = atm_batch.squeeze(0)
            pol_vec = pol_vec.squeeze(0)
            esrc = esrc.squeeze(0)

        return {"ate": ate_batch, "atm": atm_batch, "pol_vec": pol_vec, "esrc": esrc}

    def _calculate_fields_and_efficiencies(
        self, smat_global, matrices, kx_0, ky_0, mat_v_ref=None, mat_v_trn=None, kinc=None, sources=None
    ):
        """Calculate fields and diffraction efficiencies -- vectorized over sources.

        All inputs are 4D ``(n_sources, n_freqs, ...)``. All outputs in the
        returned dict carry the same leading ``(n_sources, ...)`` dimension.

        Args:
            smat_global: Global scattering matrix dict, values ``(B, F, 2M, 2M)``
            matrices: Common-matrix dict, values ``(B, F, ...)``
            kx_0, ky_0: Wave vectors ``(B, F, H0, H1)``
            mat_v_ref: V matrix for reflection region ``(B, F, 2M, 2M)``
            mat_v_trn: V matrix for transmission region ``(B, F, 2M, 2M)``
            kinc: Incident wave vectors ``(B, F, 3)``. Falls back to ``self.kinc``.
            sources: List of source dicts (for batched polarization). Falls back to ``self.src``.

        Returns:
            dict: Fields and efficiencies with leading ``(B, ...)`` dimension.
        """
        if kinc is None:
            kinc = self.kinc
        n_sources = kinc.shape[0]

        n_harmonics_squared = self.harmonics[0] * self.harmonics[1]

        # Batched polarization: esrc shape (B, F, 2*H^2)
        polarization_data = self._calculate_polarization(sources=sources, kinc=kinc)
        esrc = polarization_data["esrc"]

        # W_ref = identity, shape (B, F, 2M, 2M)
        M2 = 2 * n_harmonics_squared
        mat_w_ref = self.ident_mat_k2.reshape(1, 1, M2, M2).expand(n_sources, self.n_freqs, -1, -1)
        mat_w_trn = mat_w_ref

        csrc = tsolve(mat_w_ref, esrc.unsqueeze(-1))  # (B, F, 2M, 1)

        # Reflected fields
        cref = smat_global.S11 @ csrc  # (B, F, 2M, 1)
        eref = mat_w_ref @ cref

        ref_field_x = eref[:, :, 0:n_harmonics_squared, :]
        ref_field_y = eref[:, :, n_harmonics_squared:M2, :]

        min_kz = getattr(self, "min_kinc_z", 1e-3)

        mat_kz_ref_protected = softplus_magnitude_floor(matrices["mat_kz_ref"], min_kz, beta=100)
        kz_ref_diag = to_diag_util(mat_kz_ref_protected, self.harmonics)
        ref_field_z = (
            -matrices["mat_kx_diag"] @ tsolve(kz_ref_diag, ref_field_x)
            - matrices["mat_ky_diag"] @ tsolve(kz_ref_diag, ref_field_y)
        )

        # Transmitted fields
        ctrn = smat_global.S21 @ csrc
        etrn = mat_w_trn @ ctrn
        trn_field_x = etrn[:, :, 0:n_harmonics_squared, :]
        trn_field_y = etrn[:, :, n_harmonics_squared:M2, :]

        mat_kz_trn_protected = softplus_magnitude_floor(matrices["mat_kz_trn"], min_kz, beta=100)
        kz_trn_diag = to_diag_util(mat_kz_trn_protected, self.harmonics)
        trn_field_z = (
            -matrices["mat_kx_diag"] @ tsolve(kz_trn_diag, trn_field_x)
            - matrices["mat_ky_diag"] @ tsolve(kz_trn_diag, trn_field_y)
        )

        # Diffraction efficiencies -- kinc shape (B, F, 3)
        kinc_z = kinc[:, :, 2]  # (B, F)
        min_kinc_z = getattr(self, "min_kinc_z", 1e-3)
        kinc_z_protected = softplus_floor(torch.real(kinc_z), min_kinc_z, beta=100)  # (B, F)

        def _field_intensity(fx, fy, fz):
            return (
                torch.real(fx) ** 2 + torch.imag(fx) ** 2
                + torch.real(fy) ** 2 + torch.imag(fy) ** 2
                + torch.real(fz) ** 2 + torch.imag(fz) ** 2
            )

        H0, H1 = self.harmonics
        # kinc_z_protected[:, :, None, None] broadcasts to (B, F, H^2, 1)
        ref_diff_efficiency = torch.reshape(
            torch.real(
                self.ur2 / self.ur1 * kz_ref_diag / kinc_z_protected[:, :, None, None]
            ) @ _field_intensity(ref_field_x, ref_field_y, ref_field_z),
            shape=(n_sources, self.n_freqs, H0, H1),
        ).transpose(dim0=-2, dim1=-1)

        trn_diff_efficiency = torch.reshape(
            torch.real(
                self.ur1 / self.ur2 * kz_trn_diag / kinc_z_protected[:, :, None, None]
            ) @ _field_intensity(trn_field_x, trn_field_y, trn_field_z),
            shape=(n_sources, self.n_freqs, H0, H1),
        ).transpose(dim0=-2, dim1=-1)

        total_ref_efficiency = torch.sum(ref_diff_efficiency, dim=(-1, -2))  # (B, F)
        total_trn_efficiency = torch.sum(trn_diff_efficiency, dim=(-1, -2))

        # Magnetic field coefficients
        if mat_v_ref is not None and mat_v_trn is not None:
            uref = -mat_v_ref @ cref
            ref_mag_x = uref[:, :, 0:n_harmonics_squared, :]
            ref_mag_y = uref[:, :, n_harmonics_squared:M2, :]
            ref_mag_z = (
                -matrices["mat_kx_diag"] @ tsolve(kz_ref_diag, ref_mag_x)
                - matrices["mat_ky_diag"] @ tsolve(kz_ref_diag, ref_mag_y)
            )

            utrn = mat_v_trn @ ctrn
            trn_mag_x = utrn[:, :, 0:n_harmonics_squared, :]
            trn_mag_y = utrn[:, :, n_harmonics_squared:M2, :]
            trn_mag_z = (
                -matrices["mat_kx_diag"] @ tsolve(kz_trn_diag, trn_mag_x)
                - matrices["mat_ky_diag"] @ tsolve(kz_trn_diag, trn_mag_y)
            )
        else:
            ref_mag_x = ref_mag_y = ref_mag_z = None
            trn_mag_x = trn_mag_y = trn_mag_z = None

        return {
            "ref_s_x": ref_field_x, "ref_s_y": ref_field_y, "ref_s_z": ref_field_z,
            "trn_s_x": trn_field_x, "trn_s_y": trn_field_y, "trn_s_z": trn_field_z,
            "ref_u_x": ref_mag_x, "ref_u_y": ref_mag_y, "ref_u_z": ref_mag_z,
            "trn_u_x": trn_mag_x, "trn_u_y": trn_mag_y, "trn_u_z": trn_mag_z,
            "ref_diff_efficiency": ref_diff_efficiency,
            "trn_diff_efficiency": trn_diff_efficiency,
            "total_ref_efficiency": total_ref_efficiency,
            "total_trn_efficiency": total_trn_efficiency,
        }
