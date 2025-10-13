"""Focused tests for ``MaterialClass.from_nk_data``.

Covers the physical conversion from (n, k) -> epsilon with the
expected negative-imaginary convention, plus permeability handling.
Redundant and overly specific cases are removed to avoid test noise.
"""

import pytest
import torch
import numpy as np
from torchrdit.materials import MaterialClass


class TestFromNKData:
    """Effective checks for the from_nk_data factory."""

    @pytest.mark.parametrize(
        "name,n,k",
        [
            ("absorbing", 1.5, 0.1),        # generic absorbing dielectric
            ("lossless", 2.0, 0.0),         # lossless (imag -> 0)
            ("edge_case", 1.0, 1.0),        # n == k -> Re(ε)=0
            ("metallic", 0.2, 3.0),         # typical metal-like values
        ],
    )
    def test_permittivity_calculation_and_convention(self, name: str, n: float, k: float) -> None:
        """Validate ε = (n^2 - k^2) + i(2nk) with negative-imag convention applied."""
        material = MaterialClass.from_nk_data(name, n=n, k=k)

        expected_real = n**2 - k**2
        expected_imag = -(2 * n * k)  # negative due to conjugation in initializer

        assert torch.is_complex(material.er)
        assert abs(material.er.real - expected_real) < 1e-6
        assert abs(material.er.imag - expected_imag) < 1e-6
        # Metals should show negative real part when k is large
        if k > n:
            assert material.er.real < 0

    def test_default_permeability(self) -> None:
        """Default μr should be 1.0 when not provided."""
        material = MaterialClass.from_nk_data("test", n=1.5, k=0.1)
        assert material.ur.item() == 1.0

    def test_custom_permeability(self) -> None:
        """Custom μr should be preserved."""
        material = MaterialClass.from_nk_data("test", n=1.5, k=0.1, permeability=2.5)
        assert material.ur.item() == 2.5

