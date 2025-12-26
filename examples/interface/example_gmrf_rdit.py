"""
# Example - GMRF with hexagonal unit cells using R-DIT (interface solve)

This reproduces `torchrdit/examples/example_gmrf_rdit.py`, but runs the forward
simulation through the regulated interface API:

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
from torchrdit.viz import plot_cross_section, plot_layer

import matplotlib.pyplot as plt


def _get_diffraction_order_indices(field_x: torch.Tensor, order_x: int = 0, order_y: int = 0) -> tuple[int, int]:
    kdim_x = field_x.shape[1]
    kdim_y = field_x.shape[2]
    cx = kdim_x // 2
    cy = kdim_y // 2
    ix = cx + int(order_x)
    iy = cy + int(order_y)
    if ix < 0 or ix >= kdim_x or iy < 0 or iy >= kdim_y:
        raise ValueError(f"Diffraction order ({order_x}, {order_y}) is out of bounds")
    return (ix, iy)


def _get_all_diffraction_orders(field_x: torch.Tensor) -> list[tuple[int, int]]:
    kdim_x = field_x.shape[1]
    kdim_y = field_x.shape[2]
    cx = kdim_x // 2
    cy = kdim_y // 2
    return [(ix - cx, iy - cy) for ix in range(kdim_x) for iy in range(kdim_y)]


def _get_propagating_orders(kz_ref: torch.Tensor, *, field_x: torch.Tensor, wavelength_idx: int = 0) -> list[tuple[int, int]]:
    kdim_x = field_x.shape[1]
    kdim_y = field_x.shape[2]
    kz = kz_ref[wavelength_idx].reshape(kdim_x, kdim_y)
    cx = kdim_x // 2
    cy = kdim_y // 2
    orders: list[tuple[int, int]] = []
    for ix in range(kdim_x):
        for iy in range(kdim_y):
            if torch.abs(torch.imag(kz[ix, iy])) < 1e-6:
                orders.append((ix - cx, iy - cy))
    return orders


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
    h1 = torch.tensor(230 * nm, dtype=torch.float32)
    h2 = torch.tensor(345 * nm, dtype=torch.float32)

    # lattice vectors of the cell
    t1 = torch.tensor([a / 2, -a * np.sqrt(3) / 2], dtype=torch.float32)
    t2 = torch.tensor([a / 2, a * np.sqrt(3) / 2], dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Plotting section (matches original script parameters)
    # -------------------------------------------------------------------------
    material_sio = create_material(name="SiO", permittivity=n_SiO**2)
    material_sin = create_material(name="SiN", permittivity=n_SiN**2)
    material_fs = create_material(name="FusedSilica", permittivity=n_fs**2)

    builder = get_solver_builder()
    builder.with_algorithm(Algorithm.RDIT)
    builder.with_precision(Precision.DOUBLE)
    builder.with_real_dimensions([512, 512])
    builder.with_k_dimensions([9, 9])
    builder.with_wavelengths(np.array([1540 * nm, 1550 * nm, 1560 * nm, 1570 * nm]))
    builder.with_length_unit("um")
    builder.with_lattice_vectors(t1, t2)
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)
    builder.with_trn_material(material_fs)
    builder.add_layer({"material": "SiO", "thickness": h1.item(), "is_homogeneous": False, "is_optimize": True})
    builder.add_layer({"material": "SiN", "thickness": h2.item(), "is_homogeneous": True, "is_optimize": False})
    dev1 = builder.build()

    shapegen = ShapeGenerator.from_solver(dev1)
    c1 = shapegen.generate_circle_mask(center=[0, b / 2], radius=r)
    c2 = shapegen.generate_circle_mask(center=[0, -b / 2], radius=r)
    c3 = shapegen.generate_circle_mask(center=[a / 2, 0], radius=r)
    c4 = shapegen.generate_circle_mask(center=[-a / 2, 0], radius=r)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation="union")
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation="union")
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation="union")
    mask = (1 - mask).to(torch.float32)
    mask.requires_grad = True
    layer_index = 0
    dev1.update_er_with_mask(mask=mask, layer_index=layer_index)

    # plot the layer and save the figure
    fig, axes = plt.subplots(figsize=(5, 5))
    plot_layer(
        dev1,
        layer_index=layer_index,
        func="real",
        fig_ax=axes,
        cmap="BuGn",
        labels=("x (um)", "y (um)"),
        title=f"layer {layer_index}",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_layer_{layer_index}.png")
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

    # plot cross-sections of the layer stack structure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_cross_section(
        dev1,
        plane="xz",
        slice_position=0.0,
        fig_ax=ax1,
        title="XZ Cross-Section (y=0)\nHexagonal GMRF Structure",
    )
    plot_cross_section(
        dev1,
        plane="yz",
        slice_position=0.0,
        fig_ax=ax2,
        title="YZ Cross-Section (x=0)\nHexagonal GMRF Structure",
    )
    plt.tight_layout()
    cross_section_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_cross_sections.png")
    plt.savefig(cross_section_filename, dpi=300, bbox_inches="tight")
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
            "harmonics": [9, 9],
            "lattice_vectors": {"t1": [float(t1[0]), float(t1[1])], "t2": [float(t2[0]), float(t2[1])]},
            "trn_material": "FusedSilica",
        },
        "materials": {
            "air": {"permittivity": 1.0},
            "SiO": {"permittivity": n_SiO**2},
            "SiN": {"permittivity": n_SiN**2},
            "FusedSilica": {"permittivity": n_fs**2},
        },
        "vars": {"$mask": mask},
        "layers": [
            {
                "material": "SiO",
                "thickness": h1.item(),
                "is_homogeneous": False,
                "is_optimize": True,
                "pattern": {
                    "bg_material": "air",
                    "method": "FFT",
                    "shapes": [{"name": "m", "type": "mask", "value": "$mask"}],
                    "layer_shape": "m",
                },
            },
            {"material": "SiN", "thickness": h2.item(), "is_homogeneous": True, "is_optimize": False},
        ],
        "sources": {"theta": theta, "phi": phi, "pte": pte, "ptm": ptm},
        "output": {"type": "torch"},
    }

    data = tr.simulate(spec)
    lam0 = spec["solver"]["wavelengths"]

    # Print the Efficiency each wavelength
    for i in range(len(lam0)):
        print(f"The transmission efficiency at wavelength \t{lam0[i] * 1e3} nm is \t{data['efficiency']['transmission'][i] * 100}%")
        print(f"The reflection efficiency at wavelength \t{lam0[i] * 1e3} nm is \t{data['efficiency']['reflection'][i] * 100}%")

    print("\n===== Demonstrating SolverResults API capabilities =====\n")

    # Example 1: Get specific diffraction order indices
    print("Example 1: Getting indices for specific diffraction orders")
    try:
        tx_field = data["field_fourier_coefficients"]["transmission"]["E"]["x"]
        zero_order_idx = _get_diffraction_order_indices(tx_field, 0, 0)
        first_order_idx = _get_diffraction_order_indices(tx_field, 1, 0)
        print(f"Zero order (0,0) indices: {zero_order_idx}")
        print(f"First order (1,0) indices: {first_order_idx}")
    except ValueError as e:
        print(f"Error getting order indices: {e}")

    # Example 2: Get zero-order field components
    print("\nExample 2: Getting zero-order field components")
    ix, iy = _get_diffraction_order_indices(tx_field, 0, 0)
    tx = tx_field[:, ix, iy]
    ty = data["field_fourier_coefficients"]["transmission"]["E"]["y"][:, ix, iy]
    tz = data["field_fourier_coefficients"]["transmission"]["E"]["z"][:, ix, iy]
    print("Zero-order transmission field components (first wavelength):")
    print(f"  x-component: {tx[0].item():.6f}")
    print(f"  y-component: {ty[0].item():.6f}")
    print(f"  z-component: {tz[0].item():.6f}")

    rx = data["field_fourier_coefficients"]["reflection"]["E"]["x"][:, ix, iy]
    ry = data["field_fourier_coefficients"]["reflection"]["E"]["y"][:, ix, iy]
    rz = data["field_fourier_coefficients"]["reflection"]["E"]["z"][:, ix, iy]
    print("Zero-order reflection field components (first wavelength):")
    print(f"  x-component: {rx[0].item():.6f}")
    print(f"  y-component: {ry[0].item():.6f}")
    print(f"  z-component: {rz[0].item():.6f}")

    # Example 3: Get efficiency for specific diffraction orders
    print("\nExample 3: Getting efficiency for specific diffraction orders")
    zero_order_t = data["diffraction_efficiency"]["transmission"][:, ix, iy]
    zero_order_r = data["diffraction_efficiency"]["reflection"][:, ix, iy]
    print(f"Zero-order transmission efficiency across wavelengths: {zero_order_t.detach().numpy()}")
    print(f"Zero-order reflection efficiency across wavelengths: {zero_order_r.detach().numpy()}")

    # Try to get a higher-order if available in the simulation
    try:
        ix10, iy10 = _get_diffraction_order_indices(tx_field, 1, 0)
        first_order_t = data["diffraction_efficiency"]["transmission"][:, ix10, iy10]
        print(f"First-order (1,0) transmission efficiency: {first_order_t.detach().numpy()}")
    except ValueError as e:
        print(f"Could not get first-order efficiency: {e}")

    # Example 4: Get all available diffraction orders
    print("\nExample 4: Getting all available diffraction orders")
    all_orders = _get_all_diffraction_orders(tx_field)
    print(f"All available diffraction orders (total: {len(all_orders)}):")
    print(all_orders[:10])

    # Example 5: Get propagating orders for a specific wavelength
    print("\nExample 5: Getting propagating orders for the first wavelength")
    kzref = data["wavevectors"]["kz"]["reflection"]
    prop_orders = _get_propagating_orders(kzref, field_x=tx_field, wavelength_idx=0)
    print(f"Propagating orders for wavelength {lam0[0] * 1e3} nm (total: {len(prop_orders)}):")
    print(prop_orders)

    # Example 6: Accessing raw scattering matrix data
    print("\nExample 6: Accessing scattering matrix components")
    s11_shape = data["scattering_matrix"]["structure"]["S11"].shape
    s12_shape = data["scattering_matrix"]["structure"]["S12"].shape
    print(f"Structure scattering matrix S11 shape: {s11_shape}")
    print(f"Structure scattering matrix S12 shape: {s12_shape}")

    # Example 7: Accessing wave vector information
    print("\nExample 7: Accessing wave vector information")
    print(f"kx shape: {data['wavevectors']['kx'].shape}")
    print(f"ky shape: {data['wavevectors']['ky'].shape}")
    print(f"Incident wave vector (kinc) shape: {data['wavevectors']['incident'].shape}")
    print(f"Incident wave vector for first wavelength: {data['wavevectors']['incident'][0]}")

    # Example 8: Extracting phase information
    print("\nExample 8: Extracting phase information from field components")
    tx_phase_rad = torch.angle(tx)
    ty_phase_rad = torch.angle(ty)
    tz_phase_rad = torch.angle(tz)
    rx_phase_rad = torch.angle(rx)
    ry_phase_rad = torch.angle(ry)
    rz_phase_rad = torch.angle(rz)
    rad_to_deg = 180.0 / np.pi

    print("Phase of transmission field components (in degrees):")
    for i in range(len(lam0)):
        wavelength = lam0[i] * 1e3
        print(f"  Wavelength {wavelength:.1f} nm:")
        print(f"    x-component: {tx_phase_rad[i].item() * rad_to_deg:.2f}°")
        print(f"    y-component: {ty_phase_rad[i].item() * rad_to_deg:.2f}°")
        print(f"    z-component: {tz_phase_rad[i].item() * rad_to_deg:.2f}°")

    print("\nPhase of reflection field components (in degrees):")
    for i in range(len(lam0)):
        wavelength = lam0[i] * 1e3
        print(f"  Wavelength {wavelength:.1f} nm:")
        print(f"    x-component: {rx_phase_rad[i].item() * rad_to_deg:.2f}°")
        print(f"    y-component: {ry_phase_rad[i].item() * rad_to_deg:.2f}°")
        print(f"    z-component: {rz_phase_rad[i].item() * rad_to_deg:.2f}°")

    # Example 9: Phase difference between field components
    print("\nExample 9: Phase difference between field components")
    tx_ty_phase_diff = tx_phase_rad - ty_phase_rad
    print("Phase difference between x and y components of transmitted field (in degrees):")
    for i in range(len(lam0)):
        wavelength = lam0[i] * 1e3
        diff_deg = tx_ty_phase_diff[i].item() * rad_to_deg
        while diff_deg > 180:
            diff_deg -= 360
        while diff_deg < -180:
            diff_deg += 360
        print(f"  Wavelength {wavelength:.1f} nm: {diff_deg:.2f}°")

    # Example 10: Phase of off-axis diffraction orders (if available)
    print("\nExample 10: Phase of off-axis diffraction orders")
    try:
        idx_1_0 = _get_diffraction_order_indices(tx_field, 1, 0)
        tx_1_0 = tx_field[:, idx_1_0[0], idx_1_0[1]]
        ty_1_0 = data["field_fourier_coefficients"]["transmission"]["E"]["y"][:, idx_1_0[0], idx_1_0[1]]
        tx_1_0_phase = torch.angle(tx_1_0) * rad_to_deg
        ty_1_0_phase = torch.angle(ty_1_0) * rad_to_deg
        print("Phase of (1,0) order transmission field (first wavelength):")
        print(f"  x-component: {tx_1_0_phase[0].item():.2f}°")
        print(f"  y-component: {ty_1_0_phase[0].item():.2f}°")
    except ValueError as e:
        print(f"Could not analyze (1,0) order: {e}")


if __name__ == "__main__":
    main()
