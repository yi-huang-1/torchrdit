"""
# Example - Optimization of a guided-mode resonance filter (GMRF) (interface simulate/optimize)

This reproduces `torchrdit/examples/example_gmrf_variable_optimize.py`, but runs:
- forward simulation via `torchrdit.simulate(spec)`
- inverse optimization via `torchrdit.optimize(spec, objective=..., options=...)`

Numerical constants and plotting parameters match the original script.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

import torchrdit as tr


# Global constants
UM = 1
NM = 1e-3 * UM
DEGREES = np.pi / 180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gmrf_spec(*, radius, lams, rdit_orders: int, kdims: int, is_showinfo: bool):
    degrees = DEGREES

    theta = 0 * degrees
    phi = 0 * degrees
    pte = 1
    ptm = 0

    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * NM
    b = a * np.sqrt(3)

    h1 = torch.tensor(230 * NM, dtype=torch.float32, device=device)
    h2 = torch.tensor([345 * NM], dtype=torch.float32, device=device)

    t1 = torch.tensor([[a / 2, -a * np.sqrt(3) / 2]], dtype=torch.float32, device=device)
    t2 = torch.tensor([[a / 2, a * np.sqrt(3) / 2]], dtype=torch.float32, device=device)

    solid = torch.ones(512, 512, dtype=torch.float32, device="cpu")

    return {
        "solver": {
            "algorithm": "RDIT",
            "precision": "DOUBLE",
            "wavelengths": lams,
            "length_unit": "um",
            "grids": [512, 512],
            "harmonics": [kdims, kdims],
            "rdit_order": rdit_orders,
            "lattice_vectors": {"t1": [float(t1[0, 0].item()), float(t1[0, 1].item())], "t2": [float(t2[0, 0].item()), float(t2[0, 1].item())]},
            "device": str(device),
            "trn_material": "FusedSilica",
        },
        "materials": {
            "air": {"permittivity": 1.0},
            "SiO": {"permittivity": n_SiO**2},
            "SiN": {"permittivity": n_SiN**2},
            "FusedSilica": {"permittivity": n_fs**2},
        },
        "vars": {
            "$a": a,
            "$b": b,
            "$r": radius,
            "$solid": solid,
            "$a2": "$a/2",
            "$ma2": "-$a/2",
            "$b2": "$b/2",
            "$mb2": "-$b/2",
        },
        "layers": [
            {
                "material": "SiO",
                "thickness": h1,
                "is_homogeneous": False,
                "is_optimize": True,
                "pattern": {
                    "bg_material": "air",
                    "method": "FFT",
                    "shapes": [
                        {"name": "solid", "type": "mask", "value": "$solid"},
                        {"name": "c1", "type": "circle", "center": [0, "$b2"], "radius": "$r"},
                        {"name": "c2", "type": "circle", "center": [0, "$mb2"], "radius": "$r"},
                        {"name": "c3", "type": "circle", "center": ["$a2", 0], "radius": "$r"},
                        {"name": "c4", "type": "circle", "center": ["$ma2", 0], "radius": "$r"},
                        {"name": "holes12", "type": "op", "expr": "union(c1, c2)"},
                        {"name": "holes123", "type": "op", "expr": "union(holes12, c3)"},
                        {"name": "holes", "type": "op", "expr": "union(holes123, c4)"},
                        {"name": "mask", "type": "op", "expr": "subtract(solid, holes)"},
                    ],
                    "layer_shape": "mask",
                },
            },
            {
                "material": "SiN",
                "thickness": h2,
                "is_homogeneous": True,
                "is_optimize": False,
            },
        ],
        "sources": {"theta": theta, "phi": phi, "pte": pte, "ptm": ptm},
        "output": {"type": "torch"},
    }


def GMRF_simulator(radius, lams, rdit_orders=10, kdims=9, is_showinfo=False):
    """Interface-based forward simulation (returns structured result dict)."""
    spec = _gmrf_spec(radius=radius, lams=lams, rdit_orders=rdit_orders, kdims=kdims, is_showinfo=is_showinfo)
    return tr.simulate(spec)


def plot_spectrum(lamswp0, data_rdit):
    """
    Plot the transmission and reflection spectra.

    Args:
        lamswp0: Wavelength array
        data_rdit: Simulation results (structured dict from interface)

    Returns:
        tuple: Figure and axes objects
    """
    fig_size = (4, 3)
    markeverypoints = 4
    nlam = len(lamswp0)
    nref_gmrf_rdit = np.zeros(nlam)
    ntrn_gmrf_rdit = np.zeros(nlam)
    ncon_gmrf_rdit = np.zeros(nlam)

    for ilam, elem in enumerate(lamswp0):
        nref_gmrf_rdit[ilam] = data_rdit["efficiency"]["reflection"][ilam].detach().clone().cpu().numpy()
        ntrn_gmrf_rdit[ilam] = data_rdit["efficiency"]["transmission"][ilam].detach().clone().cpu().numpy()
        ncon_gmrf_rdit[ilam] = nref_gmrf_rdit[ilam] + ntrn_gmrf_rdit[ilam]

    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({"font.size": 10})
    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])

    ax_gmrf.plot(
        lamswp0,
        nref_gmrf_rdit,
        color="blue",
        marker="+",
        linestyle="-",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="Ref-RDIT",
    )
    ax_gmrf.plot(
        lamswp0,
        ntrn_gmrf_rdit,
        color="m",
        marker="x",
        linestyle="-",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="Trn-RDIT",
    )

    ax_gmrf.legend()
    ax_gmrf.set_xlabel("Wavelength (um)")
    ax_gmrf.set_ylabel("Transmission/Reflection Efficiency")
    ax_gmrf.set_ylim([0, 1.0])
    ax_gmrf.grid("on")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_spectrum.png")
    plt.savefig(output_filename, dpi=300)

    return fig_gmrf, ax_gmrf


def plot_spectrum_compare_opt(lamswp0, data_org, data_opt):
    """
    Plot and compare the transmission and reflection spectra before and after optimization.

    Args:
        lamswp0: Wavelength array
        data_org: Original simulation results (structured dict)
        data_opt: Optimized simulation results (structured dict)

    Returns:
        tuple: Figure and axes objects
    """
    fig_size = (6, 4)
    markeverypoints = 4
    nlam = len(lamswp0)
    nref_gmrf_org = np.zeros(nlam)
    ntrn_gmrf_org = np.zeros(nlam)
    ncon_gmrf_org = np.zeros(nlam)

    nref_gmrf_opt = np.zeros(nlam)
    ntrn_gmrf_opt = np.zeros(nlam)
    ncon_gmrf_opt = np.zeros(nlam)

    for ilam, elem in enumerate(lamswp0):
        nref_gmrf_org[ilam] = data_org["efficiency"]["reflection"][ilam].detach().clone().cpu().numpy()
        ntrn_gmrf_org[ilam] = data_org["efficiency"]["transmission"][ilam].detach().clone().cpu().numpy()
        ncon_gmrf_org[ilam] = nref_gmrf_org[ilam] + ntrn_gmrf_org[ilam]

        nref_gmrf_opt[ilam] = data_opt["efficiency"]["reflection"][ilam].detach().clone().cpu().numpy()
        ntrn_gmrf_opt[ilam] = data_opt["efficiency"]["transmission"][ilam].detach().clone().cpu().numpy()
        ncon_gmrf_opt[ilam] = nref_gmrf_opt[ilam] + ntrn_gmrf_opt[ilam]

    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({"font.size": 10})

    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])

    ax_gmrf.plot(
        lamswp0,
        nref_gmrf_org,
        color="red",
        marker="",
        linestyle="-.",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="R-Init",
    )
    ax_gmrf.plot(
        lamswp0,
        ntrn_gmrf_org,
        color="green",
        marker="",
        linestyle="-",
        linewidth=1,
        markersize=8,
        markevery=markeverypoints,
        label="T-Init",
    )

    ax_gmrf.plot(
        lamswp0,
        nref_gmrf_opt,
        color="blue",
        marker="x",
        linestyle="-",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="R-Opt",
    )
    ax_gmrf.plot(
        lamswp0,
        ntrn_gmrf_opt,
        color="m",
        marker="",
        linestyle="--",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="T-Opt",
    )

    ax_gmrf.legend(loc="center left", bbox_to_anchor=(0.6, 0.5), frameon=False)
    ax_gmrf.set_xlabel("Wavelength [um]")
    ax_gmrf.set_ylabel("T/R")
    ax_gmrf.set_ylim([0, 1.0])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_comparison.png")
    plt.savefig(output_filename, dpi=300)

    return fig_gmrf, ax_gmrf


def optimize_radius(lam_opt, initial_radius, num_epochs=10):
    """Optimize the radius via the interface optimizer."""
    r_opt = torch.tensor(initial_radius, device=device)
    r_opt.requires_grad = True

    spec = _gmrf_spec(radius=r_opt, lams=lam_opt, rdit_orders=10, kdims=9, is_showinfo=False)
    objective = "Results['s0']['efficiency']['transmission'][0] * 1e2"

    t1 = time.perf_counter()
    opt_out = tr.optimize(spec, objective=objective, options={"steps": num_epochs, "optimizer": {"name": "adam", "lr": 5e-3}})
    t2 = time.perf_counter()
    print(f"Optimization time = {(t2 - t1) * 1000:.2f} ms")

    return opt_out["vars"]["$r"]


def main():
    """Main function to demonstrate GMRF optimization."""
    print("Example 4: Optimization of a Guided-Mode Resonance Filter (GMRF)")
    print("=" * 70)

    print("\nPart 1: Basic GMRF simulation with gradient calculation")
    print("-" * 70)

    r0_rdit = torch.tensor(400 * NM, device=device)
    r0_rdit.requires_grad = True

    lam00 = np.array([1540 * NM])
    data_rdit = GMRF_simulator(r0_rdit, lam00, rdit_orders=10, kdims=15, is_showinfo=False)

    print(f"The T efficiency (R-DIT) is {data_rdit['efficiency']['transmission'][0].to('cpu') * 100:.2f}%")
    print(f"The R efficiency (R-DIT) is {data_rdit['efficiency']['reflection'][0].to('cpu') * 100:.2f}%")

    torch.sum(data_rdit["efficiency"]["transmission"][0]).backward()
    print(f"The derivative of transmission w.r.t. radius: {r0_rdit.grad}")

    print("\nPart 2: Spectrum calculation")
    print("-" * 70)

    r1_rdit = torch.tensor(400 * NM)
    r1_rdit.requires_grad = False

    nlam = 200
    lam1 = 1530 * NM
    lam2 = 1550 * NM
    lamswp_gmrf = np.linspace(lam1, lam2, nlam, endpoint=True)

    data_gmrfswp_rdit = GMRF_simulator(r1_rdit, lamswp_gmrf, rdit_orders=10, kdims=11)

    fig, ax = plot_spectrum(lamswp0=lamswp_gmrf, data_rdit=data_gmrfswp_rdit)
    plt.close(fig)
    print("Initial spectrum calculated and saved")

    print("\nPart 3: Optimization")
    print("-" * 70)

    lam_opt = np.array([1537 * NM])
    initial_radius = 400 * NM
    r_optimized = optimize_radius(lam_opt, initial_radius, num_epochs=10)

    print(f"The optimal radius is {r_optimized.item() * 1e3:.2f} nm")

    print("\nPart 4: Compare before and after optimization")
    print("-" * 70)

    data_optimized_rdit = GMRF_simulator(r_optimized.detach(), lamswp_gmrf, rdit_orders=10, kdims=9, is_showinfo=True)

    fig, ax = plot_spectrum_compare_opt(lamswp0=lamswp_gmrf, data_org=data_gmrfswp_rdit, data_opt=data_optimized_rdit)

    orig_idx = np.argmin(data_gmrfswp_rdit["efficiency"]["transmission"].detach().to("cpu").numpy())
    opt_idx = np.argmin(data_optimized_rdit["efficiency"]["transmission"].detach().to("cpu").numpy())

    print(f"Original resonance wavelength: {lamswp_gmrf[orig_idx]:.4f} um")
    print(f"Optimized resonance wavelength: {lamswp_gmrf[opt_idx]:.4f} um")
    print(f"Target wavelength: {lam_opt[0]:.4f} um")

    print("\nExample completed successfully!")
    print("Plots saved in the current directory.")


if __name__ == "__main__":
    main()

