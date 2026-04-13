"""Layer processing for electromagnetic solvers.

Handles per-layer scattering matrix computation, mask-based permittivity
updates, and Fast Fourier Factorization (FFF) tangent-field generation.
Extracted from ``solver.py`` to reduce module size.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from torch.linalg import inv as tinv
from torch.linalg import solve as tsolve

from .cell import CellType
from .materials import MaterialClass
from .numerics import safe_kz_reciprocal
from .utils import SMatrix, blockmat2x2, redhstar, to_diag_util

if TYPE_CHECKING:
    pass


class LayerProcessingMixin:
    """Mixin providing layer-processing methods for FourierBaseSolver."""

    def _process_layer(self, n_layer, matrices):
        """Unified layer processing for both single and batched sources.

        This method handles both homogeneous and non-homogeneous layers,
        automatically detecting whether input is single source or batched.
        It also respects the per-layer ``slice_count`` property by splitting
        the layer thickness into identical slices and repeatedly composing the
        resulting scattering matrices when ``slice_count > 1``.

        Args:
            n_layer: Layer index to process.
            matrices: Dictionary of matrices from ``setup_common_matrices``.
                     Can be either single source (3D) or batched (4D).

        Returns:
            dict: Scattering matrix for the layer with the same shape convention
                  as the input (3D for single sources, 4D for batched sources).
        """
        # Matrices are always 4D: (n_sources, n_freqs, ...)
        matrices_batched = matrices
        n_sources = matrices_batched["mat_kx_diag"].shape[0]

        if hasattr(self, "debug_tensorization") and self.debug_tensorization:
            print(f"[DEBUG] _process_layer: layer {n_layer}, n_sources={n_sources}")
            print(f"[DEBUG] mat_kx_diag shape: {matrices_batched['mat_kx_diag'].shape}")

        layer = self.layer_manager.layers[n_layer]
        slice_count = getattr(layer, "slice_count", 1)
        if isinstance(layer.thickness, torch.Tensor):
            layer_thickness_tensor = layer.thickness.to(self.device)
        else:
            layer_thickness_tensor = torch.tensor(
                layer.thickness, dtype=self.tfloat, device=self.device
            )
        layer_thickness_complex = layer_thickness_tensor.to(self.tcomplex)
        slice_thickness = layer_thickness_complex / slice_count  # Evenly split thickness across slices
        smat_layer = None

        # Handle non-homogeneous layers
        if not layer.is_homogeneous:
            # Extract needed matrices - now 4D: (n_sources, n_freqs, n_harmonics, n_harmonics)
            mat_kx_diag = matrices_batched["mat_kx_diag"]
            mat_ky_diag = matrices_batched["mat_ky_diag"]

            if layer.is_dispersive:
                toeplitz_er = layer.kermat
                # Expand to match batch dimension if needed
                if toeplitz_er.dim() == 3:
                    toeplitz_er = toeplitz_er.unsqueeze(0).expand(n_sources, -1, -1, -1)
            else:
                toeplitz_er = self.expand_dims(layer.kermat)
                if toeplitz_er.dim() == 3:
                    toeplitz_er = toeplitz_er.unsqueeze(0).expand(n_sources, -1, -1, -1)

            # assuming permeability always non-dispersive
            toeplitz_ur = self.expand_dims(layer.kurmat)
            if toeplitz_ur.dim() == 3:
                toeplitz_ur = toeplitz_ur.unsqueeze(0).expand(n_sources, -1, -1, -1)

            # Solve for all frequencies and sources (tsolve already supports 4D)
            solve_ter_mky = tsolve(toeplitz_er, mat_ky_diag)
            solve_ter_mkx = tsolve(toeplitz_er, mat_kx_diag)
            solve_tur_mky = tsolve(toeplitz_ur, mat_ky_diag)
            solve_tur_mkx = tsolve(toeplitz_ur, mat_kx_diag)

            # Create P matrix (blockmat2x2 already supports 4D)
            p_mat_i = blockmat2x2(
                [
                    [mat_kx_diag @ solve_ter_mky, toeplitz_ur - mat_kx_diag @ solve_ter_mkx],
                    [mat_ky_diag @ solve_ter_mky - toeplitz_ur, -mat_ky_diag @ solve_ter_mkx],
                ]
            )

            # Handle Fast Fourier Factorization (FFF)
            if self.is_use_FFF:
                if layer.mask_format is None:
                    raise ValueError(f"Layer {n_layer} is missing mask data required for FFF.")

                p_xx, p_yy, p_xy, p_yx = self._calculate_vector_field(
                    mask=layer.mask_format.squeeze().to(self.tfloat).to(self.device),
                    layer_index=n_layer,
                )
                reciprocal_toeplitz_er = self.layer_manager._gen_toeplitz2d(
                    torch.reciprocal(layer.ermat.to(self.device).to(self.tcomplex)),
                    nharmonic_1=self.harmonics[0],
                    nharmonic_2=self.harmonics[1],
                    method="FFT",
                )

                if reciprocal_toeplitz_er.dim() == 3:
                    reciprocal_toeplitz_er = reciprocal_toeplitz_er.unsqueeze(0).expand(n_sources, -1, -1, -1)

                delta_toeplitz_er = toeplitz_er - tinv(reciprocal_toeplitz_er)

                if p_xy.dim() == 3:
                    p_xy = p_xy.unsqueeze(0).expand(n_sources, -1, -1, -1)
                    p_yx = p_yx.unsqueeze(0).expand(n_sources, -1, -1, -1)
                    p_yy = p_yy.unsqueeze(0).expand(n_sources, -1, -1, -1)
                    p_xx = p_xx.unsqueeze(0).expand(n_sources, -1, -1, -1)

                # Liu 2012 eq.51 defines E2 in the [-Ey, Ex] field ordering,
                # but torchrdit's Q matrix uses the [Ex, Ey] ordering.  The
                # coordinate transformation swaps the diagonal P blocks:
                #   Q[0:n, n:] uses p_xx (not p_yy)
                #   Q[n:, 0:n] uses p_yy (not p_xx)
                q_mat_i = blockmat2x2(
                    [
                        [
                            mat_kx_diag @ solve_tur_mky - delta_toeplitz_er @ p_xy,
                            toeplitz_er - mat_kx_diag @ solve_tur_mkx - delta_toeplitz_er @ p_xx,
                        ],
                        [
                            mat_ky_diag @ solve_tur_mky - toeplitz_er + delta_toeplitz_er @ p_yy,
                            delta_toeplitz_er @ p_yx - mat_ky_diag @ solve_tur_mkx,
                        ],
                    ]
                )
            else:
                q_mat_i = blockmat2x2(
                    [
                        [mat_kx_diag @ solve_tur_mky, toeplitz_er - mat_kx_diag @ solve_tur_mkx],
                        [mat_ky_diag @ solve_tur_mky - toeplitz_er, -mat_ky_diag @ solve_tur_mkx],
                    ]
                )

            # Non-homogeneous layer -- algorithms accept arbitrary leading batch dims
            k_0_batched = self.k_0.unsqueeze(0).expand(n_sources, -1)  # (B, F)

            smat_layer = self._solve_nonhomo_layer(
                layer_thickness=slice_thickness,
                p_mat_i=p_mat_i,                       # (B, F, M, N)
                q_mat_i=q_mat_i,                       # (B, F, M, N)
                mat_w0=matrices_batched["mat_w0"],     # (B, F, M, N)
                mat_v0=matrices_batched["mat_v0"],     # (B, F, M, N)
                harmonics=self.harmonics,
                k_0=k_0_batched,                       # (B, F)
            )

        # Handle homogeneous layers
        else:
            # Debug print for tensor shapes
            if hasattr(self, "debug_tensorization") and self.debug_tensorization:
                print(f"[DEBUG] Processing homogeneous layer {n_layer}")
                print(f"[DEBUG] mat_kx shape: {matrices_batched['mat_kx'].shape}")

            if layer.is_dispersive:
                # Get material properties for all frequencies
                toeplitz_er = self._matlib[layer.material_name].er.detach().to(self.tcomplex).to(self.device)
                toeplitz_ur = self._matlib[layer.material_name].ur.detach().to(self.tcomplex).to(self.device)

                # Debug logging
                if hasattr(self, "debug_tensorization") and self.debug_tensorization:
                    print(
                        f"[DEBUG] Dispersive layer - toeplitz_er shape: {toeplitz_er.shape}, dim: {toeplitz_er.dim()}"
                    )
                    print(
                        f"[DEBUG] Dispersive layer - toeplitz_ur shape: {toeplitz_ur.shape}, dim: {toeplitz_ur.dim()}"
                    )

                # Handle scalar case (single wavelength)
                if toeplitz_er.dim() == 0:
                    toeplitz_er = toeplitz_er.unsqueeze(0)  # Make it 1D
                    toeplitz_ur = toeplitz_ur.unsqueeze(0)  # Make it 1D

                # Always expand to (n_sources, n_freqs) for uniform broadcasting
                toeplitz_er = toeplitz_er.unsqueeze(0).expand(n_sources, -1)
                toeplitz_ur = toeplitz_ur.unsqueeze(0).expand(n_sources, -1)

                toep_ur_er = toeplitz_ur * toeplitz_er
                conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                # kz for the layer: (n_sources, n_freqs, n_harmonics^2)
                mat_kz_i = torch.conj(
                    torch.sqrt(
                        conj_toep_ur_er[:, :, None]
                        - matrices_batched["mat_kx"] ** 2
                        - matrices_batched["mat_ky"] ** 2
                    )
                    + 0 * 1j
                ).to(self.tcomplex)

                # Differentiable kz protection (see numerics.py)
                inv_dmat_lam_i = safe_kz_reciprocal(mat_kz_i, dtype=self.tcomplex)

                mat_v_i = (
                    1
                    / toeplitz_ur[:, :, None]
                    * blockmat2x2(
                        [
                            [
                                to_diag_util(matrices_batched["mat_kx_ky"] * inv_dmat_lam_i, self.harmonics),
                                to_diag_util(
                                    (toep_ur_er[:, :, None] - matrices_batched["mat_kx_kx"]) * inv_dmat_lam_i,
                                    self.harmonics,
                                ),
                            ],
                            [
                                to_diag_util(
                                    (matrices_batched["mat_ky_ky"] - toep_ur_er[:, :, None]) * inv_dmat_lam_i,
                                    self.harmonics,
                                ),
                                -to_diag_util(matrices_batched["mat_ky_kx"] * inv_dmat_lam_i, self.harmonics),
                            ],
                        ]
                    )
                )
            else:
                # Get non-dispersive material properties
                toeplitz_er = self._matlib[layer.material_name].er.detach().clone().to(self.device)
                toeplitz_ur = self._matlib[layer.material_name].ur.detach().clone().to(self.device)

                # Calculate common values
                toep_ur_er = toeplitz_ur * toeplitz_er
                conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                mat_kz_i = torch.conj(
                    torch.sqrt(conj_toep_ur_er - matrices_batched["mat_kx"] ** 2 - matrices_batched["mat_ky"] ** 2)
                    + 0 * 1j
                ).to(self.tcomplex)

                # Differentiable kz protection (see numerics.py)
                inv_dmat_lam_i = safe_kz_reciprocal(mat_kz_i, dtype=self.tcomplex)
                mat_v_i = (
                    1
                    / toeplitz_ur
                    * blockmat2x2(
                        [
                            [
                                to_diag_util(matrices_batched["mat_kx_ky"] * inv_dmat_lam_i, self.harmonics),
                                to_diag_util((toep_ur_er - matrices_batched["mat_kx_kx"]) * inv_dmat_lam_i, self.harmonics),
                            ],
                            [
                                to_diag_util((matrices_batched["mat_ky_ky"] - toep_ur_er) * inv_dmat_lam_i, self.harmonics),
                                -to_diag_util(matrices_batched["mat_ky_kx"] * inv_dmat_lam_i, self.harmonics),
                            ],
                        ]
                    )
                )

            # Layer matrix: always (n_sources, n_freqs, n_harmonics^2)
            k_0_expanded = self.k_0.unsqueeze(0).unsqueeze(2)  # (1, n_freqs, 1)
            mat_x_i = -1j * mat_kz_i * k_0_expanded * slice_thickness
            mat_x_i_diag = torch.concat([mat_x_i, mat_x_i], dim=-1)

            # Debug shapes before to_diag_util
            if hasattr(self, "debug_tensorization") and self.debug_tensorization:
                print(f"[DEBUG] mat_x_i shape: {mat_x_i.shape}")
                print(f"[DEBUG] mat_x_i_diag shape: {mat_x_i_diag.shape}")
                print(f"[DEBUG] harmonics: {self.harmonics}")

            mat_x_i = to_diag_util(torch.exp(mat_x_i_diag), self.harmonics)

            # Calculate Layer Scattering Matrix
            atwi = matrices_batched["mat_w0"]
            atvi = tsolve(mat_v_i, matrices_batched["mat_v0"])
            mat_a_i = atwi + atvi
            mat_b_i = atwi - atvi

            solve_ai_xi = tsolve(mat_a_i, mat_x_i)

            mat_xi_bi = mat_x_i @ mat_b_i

            mat_d_i = mat_a_i - mat_xi_bi @ solve_ai_xi @ mat_b_i

            s11 = tsolve(mat_d_i, mat_xi_bi @ solve_ai_xi @ mat_a_i - mat_b_i)
            s12 = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
            smat_layer = SMatrix(S11=s11, S12=s12, S21=s12, S22=s11)

        if slice_count > 1:
            smat_combined = smat_layer
            for _ in range(slice_count - 1):
                smat_combined = redhstar(smat_combined, smat_layer, tcomplex=self.tcomplex)
            smat_layer = smat_combined

        return smat_layer

    def update_er_with_mask(
        self, mask: torch.Tensor, layer_index: int, bg_material: str = "air", method: str = "FFT"
    ) -> None:
        """Update the permittivity distribution in a layer using a binary mask.

        This method allows you to define the spatial distribution of materials in a layer
        using a binary mask. The mask defines regions where the foreground material (mask=1)
        and background material (mask=0) are located. This is particularly useful for
        defining complex patterns, gratings, or arbitrary shapes within a layer.

        Args:
            mask: Binary tensor representing the pattern mask.
            layer_index: Index of the layer to update.
            bg_material: Name of the background material to use where mask=0. Default is 'air'.
            method: Method for computing the Toeplitz matrix ('FFT' or 'Analytical').

        Returns:
            None
        """

        ndim1, ndim2 = mask.size()
        if (ndim1 != self.grids[0]) or (ndim2 != self.grids[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)

        er_bg = self._get_bg(layer_index=layer_index, param="er")

        if self.layer_manager.layers[layer_index].is_dispersive is False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        self.layer_manager.layers[layer_index].mask_format = mask_format.to(self.tfloat)

        # Store the background material in the layer for visualization
        if isinstance(bg_material, str):
            self.layer_manager.layers[layer_index].bg_material = bg_material
        elif hasattr(bg_material, "name"):
            self.layer_manager.layers[layer_index].bg_material = bg_material.name
        else:
            self.layer_manager.layers[layer_index].bg_material = str(bg_material)

        if isinstance(bg_material, str) and bg_material.lower() == "air":
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        elif isinstance(bg_material, MaterialClass) and bg_material.name.lower() == "air":
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            if self._matlib[bg_material].er.ndim == 0:
                self.layer_manager.layers[layer_index].ermat = (
                    self._matlib[bg_material].er * (1 - mask_format) + (er_bg - 1) * mask_format
                )
            else:
                self.layer_manager.layers[layer_index].ermat = (
                    self._matlib[bg_material].er[:, None, None] * (1 - mask_format) + (er_bg - 1) * mask_format
                )

        if method == "Analytical":
            if self.cell_type != CellType.Cartesian:
                print(f"method [{method}] does not support the cell type [{self.cell_type}], will use FFT instead.")
                method = "FFT"
        elif method != "Analytical" and method != "FFT":
            method = "FFT"

        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.harmonics[0], n_harmonic2=self.harmonics[1], param="er", method=method
        )

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        self.layer_manager.layers[layer_index].urmat = self._get_bg(layer_index=layer_index, param="ur")
        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.harmonics[0], n_harmonic2=self.harmonics[1], param="ur", method=method
        )

    def _vector_generator_cache_key(
        self,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        fourier_loss_weight: float,
        smoothness_loss_weight: float,
        steps: int,
    ) -> Tuple[
        Tuple[int, int],
        Tuple[float, ...],
        Tuple[float, ...],
        str,
        torch.dtype,
        float,
        float,
        int,
    ]:
        lattice_t1_vals = tuple(
            float(x) for x in self.lattice_t1.detach().cpu().view(-1).tolist()
        )
        lattice_t2_vals = tuple(
            float(x) for x in self.lattice_t2.detach().cpu().view(-1).tolist()
        )
        device_str = str(torch.device(device))
        return (
            tuple(int(k) for k in self.harmonics),
            lattice_t1_vals,
            lattice_t2_vals,
            device_str,
            dtype,
            float(fourier_loss_weight),
            float(smoothness_loss_weight),
            int(steps),
        )

    def _get_tangent_field_generator(
        self,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        fourier_loss_weight: float,
        smoothness_loss_weight: float,
        steps: int,
    ):
        key = self._vector_generator_cache_key(
            device=device,
            dtype=dtype,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
            steps=steps,
        )
        generator = self._vector_generator_cache.get(key)
        device_obj = torch.device(device)

        if generator is None:
            from . import vector as vector_module  # Local import to avoid circular deps

            generator = vector_module.TangentFieldGenerator(
                lattice_t1=self.lattice_t1.to(device=device_obj, dtype=dtype),
                lattice_t2=self.lattice_t2.to(device=device_obj, dtype=dtype),
                harmonics=tuple(int(k) for k in self.harmonics),
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )
            self._vector_generator_cache[key] = generator
        else:
            generator._expansion_manager.adapt_to(device=device_obj, dtype=dtype)
        return generator

    def _calculate_vector_field(self, mask: torch.Tensor, *, layer_index: Optional[int] = None):
        """Generate Fourier-factorization P-matrix components via tangent field."""
        debug_enabled = getattr(self, "debug_fff", False)
        if not debug_enabled:
            env_flag = os.getenv("TORCHRDIT_DEBUG_FFF")
            if env_flag:
                debug_enabled = env_flag.strip().lower() not in {"0", "false", "off"}
        start_time = time.perf_counter() if debug_enabled else None

        squeezed_mask = mask.squeeze()
        if squeezed_mask.dim() == 2:
            batched_mask = squeezed_mask.unsqueeze(0)
        elif squeezed_mask.dim() == 3:
            batched_mask = squeezed_mask
        else:
            raise ValueError(f"Expected mask to squeeze to 2D or 3D, got shape {mask.shape}")

        batched_mask = batched_mask.to(self.tfloat).to(self.device)

        scheme = getattr(self, "fff_vector_scheme", getattr(self, "vector_field_scheme", "POL"))
        fourier_loss_weight = getattr(self, "fff_fourier_weight", getattr(self, "vector_fourier_loss_weight", 1e-2))
        smoothness_loss_weight = getattr(
            self, "fff_smoothness_weight", getattr(self, "vector_smoothness_loss_weight", 1e-3)
        )
        steps = getattr(self, "fff_vector_steps", getattr(self, "vector_field_steps", 1))

        generator = self._get_tangent_field_generator(
            device=batched_mask.device,
            dtype=batched_mask.dtype,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
            steps=steps,
        )

        tx_components: List[torch.Tensor] = []
        ty_components: List[torch.Tensor] = []
        for mask_sample in batched_mask:
            tx_sample, ty_sample = generator.compute(
                mask=mask_sample,
                XO=self.XO,
                YO=self.YO,
                scheme=scheme,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )
            tx_components.append(tx_sample)
            ty_components.append(ty_sample)

        tx_stack = torch.stack(tx_components, dim=0)
        ty_stack = torch.stack(ty_components, dim=0)

        if tx_stack.shape[0] == 1:
            tx = tx_stack[0]
            ty = ty_stack[0]
        else:
            tx = tx_stack
            ty = ty_stack

        tx = tx.to(self.tcomplex)
        ty = ty.to(self.tcomplex)
        tx_conj = torch.conj(tx)
        ty_conj = torch.conj(ty)

        tx_mag_sq = (tx_conj * tx).real
        ty_mag_sq = (ty_conj * ty).real
        denom = torch.clamp(tx_mag_sq + ty_mag_sq, min=1e-12)
        inv_denom = denom.reciprocal()
        inv_denom_complex = inv_denom.to(dtype=self.tcomplex)

        weight_xx = (tx_mag_sq * inv_denom).to(dtype=self.tcomplex)
        weight_yy = (ty_mag_sq * inv_denom).to(dtype=self.tcomplex)
        weight_xy = (tx * ty_conj) * inv_denom_complex
        weight_yx = torch.conj(weight_xy)

        toeplitz_xx = self.layer_manager._gen_toeplitz2d(
            weight_xx, nharmonic_1=self.harmonics[0], nharmonic_2=self.harmonics[1], method="FFT"
        )
        toeplitz_yy = self.layer_manager._gen_toeplitz2d(
            weight_yy, nharmonic_1=self.harmonics[0], nharmonic_2=self.harmonics[1], method="FFT"
        )
        toeplitz_xy = self.layer_manager._gen_toeplitz2d(
            weight_xy, nharmonic_1=self.harmonics[0], nharmonic_2=self.harmonics[1], method="FFT"
        )
        toeplitz_yx = self.layer_manager._gen_toeplitz2d(
            weight_yx, nharmonic_1=self.harmonics[0], nharmonic_2=self.harmonics[1], method="FFT"
        )

        if debug_enabled:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            n_harmonics_sq = self.harmonics[0] * self.harmonics[1]
            bytes_per_complex = torch.tensor([], dtype=self.tcomplex).element_size()
            approx_mem_mb = (4 * n_harmonics_sq * n_harmonics_sq * bytes_per_complex) / (1024.0 * 1024.0)
            layer_info = f" layer {layer_index}" if layer_index is not None else ""
            print(
                f"[FFF DEBUG] calculate_vector_field{layer_info}: {duration_ms:.2f} ms, "
                f"P-matrix approx {approx_mem_mb:.2f} MB"
            )

        return toeplitz_xx, toeplitz_yy, toeplitz_xy, toeplitz_yx

    def get_vector_components(self, layer_index: int):
        if self.layer_manager.layers[layer_index].mask_format is not None:
            mask = self.layer_manager.layers[layer_index].mask_format.squeeze()
            if mask.dim() != 2:
                mask = mask.to(self.tfloat).to(self.device).squeeze()
            else:
                mask = mask.to(self.tfloat).to(self.device)

            scheme = getattr(self, "fff_vector_scheme", getattr(self, "vector_field_scheme", "POL"))
            fourier_loss_weight = getattr(
                self, "fff_fourier_weight", getattr(self, "vector_fourier_loss_weight", 1e-2)
            )
            smoothness_loss_weight = getattr(
                self, "fff_smoothness_weight", getattr(self, "vector_smoothness_loss_weight", 1e-3)
            )
            steps = getattr(self, "fff_vector_steps", getattr(self, "vector_field_steps", 1))

            generator = self._get_tangent_field_generator(
                device=mask.device,
                dtype=mask.dtype,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )

            tx, ty = generator.compute(
                mask=mask,
                XO=self.XO,
                YO=self.YO,
                scheme=scheme,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )

            return tx, ty
        else:
            return None, None

    def update_layer_thickness(self, layer_index: int, thickness: torch.Tensor):
        """Update the thickness of a specific layer in the structure.

        Args:
            layer_index (int): Index of the layer to modify.
            thickness (torch.Tensor): New thickness value for the layer.
        """
        self.layer_manager.update_layer_thickness(layer_index=layer_index, thickness=thickness)

    def _get_bg(self, layer_index: int, param="er") -> torch.Tensor:
        """Return the background matrix of the specified layer.

        Args:
            layer_index (int): layer_index
            param: 'er' or 'ur'

        Returns:
            torch.Tensor:
        """

        ret_mat = None

        if layer_index < self.layer_manager.nlayer:
            if self.layer_manager.layers[layer_index].is_dispersive is True:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == "er":
                    ret_mat = (
                        self._matlib[material_name]
                        .er.detach()
                        .clone()
                        .unsqueeze(1)
                        .unsqueeze(1)
                        .repeat(1, self.grids[0], self.grids[1])
                        .to(self.device)
                        .to(self.tcomplex)
                    )
                elif param == "ur":
                    param_val = self._matlib[material_name].ur.detach().clone()
                    ret_mat = param_val * torch.ones(
                        size=(self.grids[0], self.grids[1]), dtype=self.tcomplex, device=self.device
                    )
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

            else:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == "er":
                    param_val = self._matlib[material_name].er.detach().clone()
                elif param == "ur":
                    param_val = self._matlib[material_name].ur.detach().clone()
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

                ret_mat = param_val * torch.ones(
                    size=(self.grids[0], self.grids[1]), dtype=self.tcomplex, device=self.device
                )
        else:
            raise ValueError("The index exceeds the max layer number.")

        return ret_mat

    def expand_dims(self, mat: torch.Tensor) -> Optional[torch.Tensor]:
        """Expand the input matrix to standard output dimension without layer information:
            (n_freqs, n_harmonics_squared, n_harmonics_squared)

        Args:
            mat (torch.Tensor): input tensor matrix
        """

        ret = None

        if mat.ndim == 1 and mat.shape[0] == self.n_freqs:
            # The input tensor with dimension (n_freqs)
            ret = mat[:, None, None]
        elif mat.ndim == 2:
            # The input matrix with dimension (n_harmonics_squared, n_harmonics_squared)
            ret = mat.unsqueeze(0).repeat(self.n_freqs, 1, 1)
        else:
            raise RuntimeError("Not Listed in the Case")

        return ret
