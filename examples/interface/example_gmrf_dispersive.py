"""
# Example - GMRF with hexagonal unit cells with dispersive materials (interface solve)

This reproduces `torchrdit/examples/example_gmrf_dispersive.py`, but runs the
forward simulation through the regulated interface API:

```python
import torchrdit as tr
results = tr.simulate(spec)
```

The plotting parameters and numerical constants match the original script.
"""

from __future__ import annotations

import os
import numpy as np
import torch

import torchrdit as tr
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from torchrdit.viz import display_fitted_permittivity, plot_layer

import matplotlib.pyplot as plt


def main() -> None:
    # units, normalizing all units to 'um'
    um = 1
    nm = 1e-3 * um
    degrees = np.pi / 180

    # angles of incident waves
    theta = 0 * degrees
    phi = 0 * degrees

    # polarization
    pte = 1
    ptm = 0

    # refractive index
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    # dimensions of the cell
    a = 1150 * nm
    b = a * np.sqrt(3)

    # radius of the holes on the top layer
    r = 400 * nm

    # thickness of each layer
    h1 = torch.tensor(230 * nm, dtype=torch.float64)
    h2 = torch.tensor(345 * nm, dtype=torch.float64)

    # lattice vectors of the cell
    t1 = torch.tensor([[a / 2, -a * np.sqrt(3) / 2]], dtype=torch.float64)
    t2 = torch.tensor([[a / 2, a * np.sqrt(3) / 2]], dtype=torch.float64)

    # -------------------------------------------------------------------------
    # Plotting + material fitting section (matches original script parameters)
    # -------------------------------------------------------------------------
    torchrdit_sim = (
        get_solver_builder()
        .with_algorithm(Algorithm.RDIT)
        .with_precision(Precision.DOUBLE)
        .with_real_dimensions([512, 512])
        .with_k_dimensions([21, 21])
        .with_wavelengths(np.array([1540 * nm, 1550 * nm, 1560 * nm, 1570 * nm]))
        .with_length_unit("um")
        .with_lattice_vectors(t1, t2)
        .build()
    )

    material_sic = create_material(
        name="SiC",
        dielectric_dispersion=True,
        user_dielectric_file=os.path.join(os.path.dirname(__file__), "..", "Si_C-e.txt"),
        data_format="freq-eps",
        data_unit="thz",
    )
    material_sio2 = create_material(
        name="SiO2",
        dielectric_dispersion=True,
        user_dielectric_file=os.path.join(os.path.dirname(__file__), "..", "SiO2-e.txt"),
        data_format="freq-eps",
        data_unit="thz",
    )
    material_sin = create_material(name="SiN", permittivity=n_SiN**2)
    material_fs = create_material(name="FusedSilica", permittivity=n_fs**2)

    torchrdit_sim.add_materials(material_list=[material_sic, material_sio2, material_sin, material_fs])
    torchrdit_sim.update_trn_material(trn_material=material_fs)

    torchrdit_sim.add_layer(material_name=material_sic, thickness=h1, is_homogeneous=False, is_optimize=True)
    torchrdit_sim.add_layer(material_name=material_sin, thickness=h2, is_homogeneous=True, is_optimize=False)

    shape_generator = ShapeGenerator.from_solver(torchrdit_sim)
    c1 = shape_generator.generate_circle_mask(center=[0, b / 2], radius=r)
    c2 = shape_generator.generate_circle_mask(center=[0, -b / 2], radius=r)
    c3 = shape_generator.generate_circle_mask(center=[a / 2, 0], radius=r)
    c4 = shape_generator.generate_circle_mask(center=[-a / 2, 0], radius=r)

    mask = shape_generator.combine_masks(mask1=c1, mask2=c2, operation="union")
    mask = shape_generator.combine_masks(mask1=mask, mask2=c3, operation="union")
    mask = shape_generator.combine_masks(mask1=mask, mask2=c4, operation="union")
    mask = (1 - mask).to(torch.float64)
    layer_index = 0
    torchrdit_sim.update_er_with_mask(mask=mask, layer_index=layer_index)

    torchrdit_sim.get_layer_structure()

    fig, axes = plt.subplots()
    plot_layer(
        torchrdit_sim,
        layer_index=0,
        func="real",
        fig_ax=axes,
        cmap="BuGn",
        labels=("x (um)", "y (um)"),
        title="layer 0",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_layer_0.png")
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    display_fitted_permittivity(torchrdit_sim, fig_ax=axes)
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_fitted_dispersive.png")
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Interface simulation section (regulated)
    # -------------------------------------------------------------------------
    spec = {
        "solver": {
            "algorithm": "RDIT",
            "precision": "DOUBLE",
            "wavelengths": np.array([1540 * nm, 1550 * nm, 1560 * nm, 1570 * nm]),
            "length_unit": "um",
            "grids": [512, 512],
            "harmonics": [21, 21],
            "lattice_vectors": {"t1": [float(t1[0, 0]), float(t1[0, 1])], "t2": [float(t2[0, 0]), float(t2[0, 1])]},
            "trn_material": "FusedSilica",
        },
        "materials": {
            "air": {"permittivity": 1.0},
            "SiC": {
                "dielectric_dispersion": True,
                "dielectric_file": "../Si_C-e.txt",
                "data_format": "freq-eps",
                "data_unit": "thz",
            },
            "SiO2": {
                "dielectric_dispersion": True,
                "dielectric_file": "../SiO2-e.txt",
                "data_format": "freq-eps",
                "data_unit": "thz",
            },
            "SiN": {"permittivity": n_SiN**2},
            "FusedSilica": {"permittivity": n_fs**2},
        },
        "vars": {"$mask": mask},
        "layers": [
            {
                "material": "SiC",
                "thickness": float(h1.item()),
                "is_homogeneous": False,
                "is_optimize": True,
                "pattern": {
                    "bg_material": "air",
                    "method": "FFT",
                    "shapes": [{"name": "m", "type": "mask", "value": "$mask"}],
                    "layer_shape": "m",
                },
            },
            {"material": "SiN", "thickness": float(h2.item()), "is_homogeneous": True, "is_optimize": False},
        ],
        "sources": {"theta": theta, "phi": phi, "pte": pte, "ptm": ptm},
        "output": {"type": "torch"},
    }

    result = tr.simulate(spec)

    # Print the efficiency at each wavelength
    for i in range(len(spec["solver"]["wavelengths"])):
        print(
            f"The transmission efficiency at wavelength \t{spec['solver']['wavelengths'][i] * 1e3} nm is \t{result['efficiency']['transmission'][i] * 100}%"
        )
        print(
            f"The reflection efficiency at wavelength \t{spec['solver']['wavelengths'][i] * 1e3} nm is \t{result['efficiency']['reflection'][i] * 100}%"
        )

    print("\n===== Demonstrating SolverResults API with Dispersive Materials =====\n")

    # Example 1: Get zero-order field components
    print("Example 1: Getting zero-order field components")
    tx_field = result["field_fourier_coefficients"]["transmission"]["E"]["x"]
    kdim_x = tx_field.shape[1]
    kdim_y = tx_field.shape[2]
    ix = kdim_x // 2
    iy = kdim_y // 2
    tx = tx_field[:, ix, iy]
    ty = result["field_fourier_coefficients"]["transmission"]["E"]["y"][:, ix, iy]
    tz = result["field_fourier_coefficients"]["transmission"]["E"]["z"][:, ix, iy]

    rx = result["field_fourier_coefficients"]["reflection"]["E"]["x"][:, ix, iy]
    ry = result["field_fourier_coefficients"]["reflection"]["E"]["y"][:, ix, iy]
    rz = result["field_fourier_coefficients"]["reflection"]["E"]["z"][:, ix, iy]

    print("Zero-order transmission field amplitudes (first wavelength):")
    tx_amplitude = torch.abs(tx[0])
    ty_amplitude = torch.abs(ty[0])
    tz_amplitude = torch.abs(tz[0])
    print(f"  x-component amplitude: {tx_amplitude.item():.6f}")
    print(f"  y-component amplitude: {ty_amplitude.item():.6f}")
    print(f"  z-component amplitude: {tz_amplitude.item():.6f}")

    # Example 2: Analyzing phase for dispersive materials
    print("\nExample 2: Phase analysis across multiple wavelengths")
    tx_phase_rad = torch.angle(tx)
    ty_phase_rad = torch.angle(ty)

    rad_to_deg = 180.0 / np.pi

    print("Phase of transmission x-component across wavelengths (in degrees):")
    for i in range(len(spec["solver"]["wavelengths"])):
        wavelength = spec["solver"]["wavelengths"][i] * 1e3
        print(f"  Wavelength {wavelength:.1f} nm: {tx_phase_rad[i].item() * rad_to_deg:.2f}°")

    # Example 3: Phase differences vs wavelength
    print("\nExample 3: Phase differences between x and y components across wavelengths")
    tx_ty_phase_diff = tx_phase_rad - ty_phase_rad
    for i in range(len(spec["solver"]["wavelengths"])):
        diff_deg = tx_ty_phase_diff[i].item() * rad_to_deg
        while diff_deg > 180:
            diff_deg -= 360
        while diff_deg < -180:
            diff_deg += 360
        print(f"  Wavelength {spec['solver']['wavelengths'][i] * 1e3:.1f} nm: {diff_deg:.2f}°")

    # Example 4: Diffraction efficiency vs wavelength
    print("\nExample 4: Zero-order diffraction efficiency vs wavelength")
    zero_order_t = result["diffraction_efficiency"]["transmission"][:, ix, iy]
    zero_order_r = result["diffraction_efficiency"]["reflection"][:, ix, iy]

    print("Zero-order efficiencies for each wavelength:")
    for i in range(len(spec["solver"]["wavelengths"])):
        wavelength = spec["solver"]["wavelengths"][i] * 1e3
        print(f"  Wavelength {wavelength:.1f} nm:")
        print(f"    Transmission: {zero_order_t[i].item() * 100:.4f}%")
        print(f"    Reflection: {zero_order_r[i].item() * 100:.4f}%")
        print(f"    Sum: {(zero_order_t[i].item() + zero_order_r[i].item()) * 100:.4f}%")

    # Example 5: Analyzing propagating orders for each wavelength
    print("\nExample 5: Propagating diffraction orders vs wavelength")
    kzref = result["wavevectors"]["kz"]["reflection"]
    for i in range(len(spec["solver"]["wavelengths"])):
        wavelength = spec["solver"]["wavelengths"][i] * 1e3
        kz_reshaped = kzref[i].reshape(kdim_x, kdim_y)
        prop_orders = []
        for ix2 in range(kdim_x):
            for iy2 in range(kdim_y):
                if torch.abs(torch.imag(kz_reshaped[ix2, iy2])) < 1e-6:
                    prop_orders.append((ix2 - kdim_x // 2, iy2 - kdim_y // 2))
        print(f"  Wavelength {wavelength:.1f} nm: {len(prop_orders)} propagating orders")
        print(f"    Orders: {prop_orders}")

    # Example 6: Exploring scattering matrix for dispersive materials
    print("\nExample 6: Structure scattering matrix analysis")
    s11 = result["scattering_matrix"]["structure"]["S11"]
    s12 = result["scattering_matrix"]["structure"]["S12"]
    print(f"S11 shape: {s11.shape} - Shows matrix for each wavelength ({len(spec['solver']['wavelengths'])} wavelengths)")
    print("S-matrix values vary with wavelength due to material dispersion")

    # Example 7: Effect of wavelength on wave vectors
    print("\nExample 7: Wave vector analysis with dispersive materials")
    print("Wave vectors are affected by wavelength-dependent material properties:")
    print("kzref for different wavelengths:")
    for i in range(len(spec["solver"]["wavelengths"])):
        wavelength = spec["solver"]["wavelengths"][i] * 1e3
        kzref_val = kzref[i].reshape(kdim_x, kdim_y)[(kdim_x // 2, kdim_y // 2)]
        print(f"  Wavelength {wavelength:.1f} nm: kzref = {torch.abs(kzref_val).item():.6f}")

    print("\nEnergy conservation check for dispersive system:")
    for i in range(len(spec["solver"]["wavelengths"])):
        wavelength = spec["solver"]["wavelengths"][i] * 1e3
        total = result["efficiency"]["transmission"][i].item() + result["efficiency"]["reflection"][i].item()
        print(f"  Wavelength {wavelength:.1f} nm: {total * 100:.4f}% (T+R)")


if __name__ == "__main__":
    main()
