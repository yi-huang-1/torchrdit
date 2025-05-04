"""
# Example - Optimization of a guided-mode resonance filter (GMRF)

This example demonstrates how to optimize a guided-mode resonance filter (GMRF) using 
differentiable RDIT algorithm in torchrdit. The device consists of a SiO hexagonal 
grating layer, a SiN waveguide layer, and a fused silica substrate.

The optimization goal is to shift the resonance peak to a target wavelength (1537 nm)
using gradient descent to adjust the radius of the hexagonal pattern.

The GMRF design is based on:
- A. A. Mehta, R. C. Rumpf, Z. A. Roth, and E. G. Johnson, "Guided mode resonance filter 
  as a spectrally selective feedback element in a double-cladding optical fiber laser," 
  IEEE Photonics Technology Letters, vol. 19, pp. 2030-2032, 12 2007.

Keywords:
    guided-mode resonance filter, GMRF, optimization, gradient descent, 
    guided mode resonance, guided mode resonance filter, R-DIT, builder,
    automatic differentiation, optimizer, scheduler
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from tqdm import trange
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator

# Global constants
UM = 1
NM = 1e-3 * UM
DEGREES = np.pi / 180


def GMRF_simulator(radius, lams, rdit_orders=10, kdims=9, is_showinfo=False):
    """
    Simulate the GMRF device with the given parameters.
    
    Args:
        radius: Radius of the holes in the hexagonal pattern
        lams: Array of wavelengths to simulate
        rdit_orders: Number of orders for the RDIT algorithm
        kdims: Dimensions of the k-space grid
        is_showinfo: Whether to show additional info during simulation
        
    Returns:
        SolverResults: Results object with transmission, reflection, and field data
    """
    # Setup units and angles
    degrees = DEGREES

    theta = 0 * degrees
    phi = 0 * degrees
    pte = 1
    ptm = 0

    # Device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * NM
    b = a * np.sqrt(3)

    h1 = torch.tensor(230 * NM, dtype=torch.float32)
    h2 = torch.tensor([345 * NM], dtype=torch.float32)

    t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float32)
    t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float32)

    # Create materials
    material_sio = create_material(name='SiO', permittivity=n_SiO**2)
    material_sin = create_material(name='SiN', permittivity=n_SiN**2)
    material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

    # Create and configure solver using Builder pattern
    builder = get_solver_builder()
    
    # Configure the builder with all necessary parameters
    builder.with_algorithm(Algorithm.RDIT)
    builder.with_precision(Precision.DOUBLE)
    builder.with_real_dimensions([512, 512])
    builder.with_k_dimensions([kdims, kdims])
    builder.with_wavelengths(lams)
    builder.with_length_unit('um')
    builder.with_lattice_vectors(t1, t2)
    
    # Add materials to builder
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)
    
    # Build the solver
    gmrf_sim = builder.build()
    
    gmrf_sim.set_rdit_order(rdit_orders)
    gmrf_sim.update_trn_material(trn_material=material_fs)

    # Add layers
    gmrf_sim.add_layer(material_name='SiO',
                  thickness=h1,
                  is_homogeneous=False,
                  is_optimize=True)

    gmrf_sim.add_layer(material_name='SiN',
                  thickness=h2,
                  is_homogeneous=True,
                  is_optimize=False)

    # Create source
    src = gmrf_sim.add_source(theta=theta,
                     phi=phi,
                     pte=pte,
                     ptm=ptm)

    # Create mask and update permittivity
    shapegen = ShapeGenerator.from_solver(gmrf_sim)
    c1 = shapegen.generate_circle_mask(center=[0, b/2], radius=radius)
    c2 = shapegen.generate_circle_mask(center=[0, -b/2], radius=radius)
    c3 = shapegen.generate_circle_mask(center=[a/2, 0], radius=radius)
    c4 = shapegen.generate_circle_mask(center=[-a/2, 0], radius=radius)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation='union')

    mask = (1 - mask)
    gmrf_sim.update_er_with_mask(mask=mask, layer_index=0)

    # Solve and return results
    if is_showinfo:
        gmrf_sim.get_layer_structure()
        
    data = gmrf_sim.solve(src)
    
    return data

def plot_spectrum(lamswp0, data_rdit):
    """
    Plot the transmission and reflection spectra.
    
    Args:
        lamswp0: Wavelength array
        data_rdit: Simulation results from RDIT
        
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
        nref_gmrf_rdit[ilam] = data_rdit.reflection[ilam].detach().clone()
        ntrn_gmrf_rdit[ilam] = data_rdit.transmission[ilam].detach().clone()
        ncon_gmrf_rdit[ilam] = nref_gmrf_rdit[ilam] + ntrn_gmrf_rdit[ilam]

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])

    ax_gmrf.plot(lamswp0, nref_gmrf_rdit, color='blue', marker='+', linestyle='-', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='Ref-RDIT')
    ax_gmrf.plot(lamswp0, ntrn_gmrf_rdit, color='m', marker='x', linestyle='-', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='Trn-RDIT')

    ax_gmrf.legend()
    ax_gmrf.set_xlabel('Wavelength (um)')
    ax_gmrf.set_ylabel('Transmission/Reflection Efficiency')
    ax_gmrf.set_ylim([0, 1.0])
    ax_gmrf.grid('on')
    
    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_spectrum.png")
    plt.savefig(output_filename, dpi=300)
    
    return fig_gmrf, ax_gmrf

def plot_spectrum_compare_opt(lamswp0, data_org, data_opt):
    """
    Plot and compare the transmission and reflection spectra before and after optimization.
    
    Args:
        lamswp0: Wavelength array
        data_org: Original simulation results
        data_opt: Optimized simulation results
        
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
        nref_gmrf_org[ilam] = data_org.reflection[ilam].detach().clone()
        ntrn_gmrf_org[ilam] = data_org.transmission[ilam].detach().clone()
        ncon_gmrf_org[ilam] = nref_gmrf_org[ilam] + ntrn_gmrf_org[ilam]

        nref_gmrf_opt[ilam] = data_opt.reflection[ilam].detach().clone()
        ntrn_gmrf_opt[ilam] = data_opt.transmission[ilam].detach().clone()
        ncon_gmrf_opt[ilam] = nref_gmrf_opt[ilam] + ntrn_gmrf_opt[ilam]

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    
    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])
    
    # Plot original data
    ax_gmrf.plot(lamswp0, nref_gmrf_org, color='red', marker='', linestyle='-.', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='R-Init')
    ax_gmrf.plot(lamswp0, ntrn_gmrf_org, color='green', marker='', linestyle='-', 
                 linewidth=1, markersize=8, markevery=markeverypoints, label='T-Init')
    
    # Plot optimized data
    ax_gmrf.plot(lamswp0, nref_gmrf_opt, color='blue', marker='x', linestyle='-', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='R-Opt')
    ax_gmrf.plot(lamswp0, ntrn_gmrf_opt, color='m', marker='', linestyle='--', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='T-Opt')

    ax_gmrf.legend(loc='center left', bbox_to_anchor=(0.6, 0.5), frameon=False)
    ax_gmrf.set_xlabel('Wavelength [um]')
    ax_gmrf.set_ylabel('T/R')
    ax_gmrf.set_ylim([0, 1.0])
    
    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_comparison.png")
    plt.savefig(output_filename, dpi=300)
    
    return fig_gmrf, ax_gmrf

def objective_GMRF(dev, src, radius):
    """
    Objective function for GMRF optimization.
    
    Args:
        dev: The solver device
        src: The source configuration
        radius: The radius of holes to optimize
        
    Returns:
        torch.Tensor: The transmission efficiency (to be minimized)
    """
    a = 1150 * NM
    b = a * np.sqrt(3)

    shapegen = ShapeGenerator.from_solver(dev)
    c1 = shapegen.generate_circle_mask(center=[0, b/2], radius=radius)
    c2 = shapegen.generate_circle_mask(center=[0, -b/2], radius=radius)
    c3 = shapegen.generate_circle_mask(center=[a/2, 0], radius=radius)
    c4 = shapegen.generate_circle_mask(center=[-a/2, 0], radius=radius)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation='union')

    mask = (1 - mask)
    dev.update_er_with_mask(mask=mask, layer_index=0)
    
    data = dev.solve(src)
    
    return data.transmission[0] * 1e2  # return transmission efficiency as FoM to be minimized

def setup_gmrf_solver(lam_opt):
    """
    Set up the GMRF solver for optimization.
    
    Args:
        lam_opt: Target wavelength for optimization
        
    Returns:
        tuple: The solver and source objects
    """
    # Device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * NM
    b = a * np.sqrt(3)

    h1 = torch.tensor(230 * NM, dtype=torch.float32)
    h2 = torch.tensor([345 * NM], dtype=torch.float32)

    t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float32)
    t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float32)

    material_sio = create_material(name='SiO', permittivity=n_SiO**2)
    material_sin = create_material(name='SiN', permittivity=n_SiN**2)
    material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

    r_dit_order = 10
    kdims = 9
        
    # Create and configure solver using Builder pattern
    builder = get_solver_builder()
    
    # Configure the builder with all necessary parameters
    builder.with_algorithm(Algorithm.RDIT)
    builder.with_precision(Precision.DOUBLE)
    builder.with_real_dimensions([512, 512])
    builder.with_k_dimensions([kdims, kdims])
    builder.with_wavelengths(lam_opt)
    builder.with_length_unit('um')
    builder.with_lattice_vectors(t1, t2)
    
    # Add materials to builder
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)
    
    # Build the solver
    gmrf_sim_rdit = builder.build()

    gmrf_sim_rdit.set_rdit_order(r_dit_order)
    gmrf_sim_rdit.update_trn_material(trn_material='FusedSilica')

    # Add layers
    gmrf_sim_rdit.add_layer(material_name='SiO',
                thickness=h1,
                is_homogeneous=False,
                is_optimize=True)

    gmrf_sim_rdit.add_layer(material_name='SiN',
                thickness=h2,
                is_homogeneous=True,
                is_optimize=False)

    # Create source
    src_rdit = gmrf_sim_rdit.add_source(theta=0 * DEGREES,
                    phi=0 * DEGREES,
                    pte=1,
                    ptm=0)
                    
    return gmrf_sim_rdit, src_rdit

def optimize_radius(gmrf_sim_rdit, src_rdit, initial_radius, num_epochs=10):
    """
    Optimize the radius of the GMRF holes using gradient descent.
    
    Args:
        gmrf_sim_rdit: The GMRF solver
        src_rdit: The source configuration
        initial_radius: Initial radius to start optimization
        num_epochs: Number of optimization epochs
        
    Returns:
        torch.Tensor: The optimized radius
    """
    r_opt = torch.tensor(initial_radius)
    r_opt.requires_grad = True

    # Learning rate
    lr_rate = 5e-3

    # Define the optimizer
    optimizer = torch.optim.Adam([r_opt], lr=lr_rate, eps=1e-2)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.9)

    # Optimization loop
    t1 = time.perf_counter()
    
    for epoch in trange(num_epochs):
        optimizer.zero_grad()
        loss = objective_GMRF(gmrf_sim_rdit, src_rdit, r_opt)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
    
    t2 = time.perf_counter()
    print(f"Optimization time = {(t2 - t1) * 1000:.2f} ms")
    
    return r_opt

def main():
    """Main function to demonstrate GMRF optimization."""
    print("Example 4: Optimization of a Guided-Mode Resonance Filter (GMRF)")
    print("=" * 70)
    
    # Part 1: Basic GMRF simulation with gradient calculation
    print("\nPart 1: Basic GMRF simulation with gradient calculation")
    print("-" * 70)
    
    # Setup initial radius with gradient tracking
    r0_rdit = torch.tensor(400 * NM)
    r0_rdit.requires_grad = True
    
    # Simulate at a single wavelength
    lam00 = np.array([1540 * NM])
    data_rdit = GMRF_simulator(r0_rdit, lam00, rdit_orders=10, kdims=15, is_showinfo=False)

    # Print efficiency and calculate gradient
    print(f"The T efficiency (R-DIT) is {data_rdit.transmission[0] * 100:.2f}%")
    print(f"The R efficiency (R-DIT) is {data_rdit.reflection[0] * 100:.2f}%")

    torch.sum(data_rdit.transmission[0]).backward()
    print(f"The derivative of transmission w.r.t. radius: {r0_rdit.grad}")
    
    # Part 2: Spectrum calculation
    print("\nPart 2: Spectrum calculation")
    print("-" * 70)
    
    r1_rdit = torch.tensor(400 * NM)
    r1_rdit.requires_grad = False

    # Calculate spectrum over wavelength range
    nlam = 200
    lam1 = 1530 * NM
    lam2 = 1550 * NM
    lamswp_gmrf = np.linspace(lam1, lam2, nlam, endpoint=True)

    data_gmrfswp_rdit = GMRF_simulator(r1_rdit, lamswp_gmrf, rdit_orders=10, kdims=11)
    
    # Plot initial spectrum
    fig, ax = plot_spectrum(lamswp0=lamswp_gmrf, data_rdit=data_gmrfswp_rdit)
    plt.close(fig)  # Close figure to avoid display in non-interactive mode
    print("Initial spectrum calculated and saved")
    
    # Part 3: Optimization
    print("\nPart 3: Optimization")
    print("-" * 70)
    
    # Target wavelength for optimization
    lam_opt = np.array([1537 * NM])
    
    # Setup solver for optimization
    gmrf_sim_rdit, src_rdit = setup_gmrf_solver(lam_opt)
    
    # Optimize radius
    initial_radius = 400 * NM
    r_optimized = optimize_radius(gmrf_sim_rdit, src_rdit, initial_radius)
    
    print(f"The optimal radius is {r_optimized.item() * 1e3:.2f} nm")
    
    # Part 4: Compare before and after optimization
    print("\nPart 4: Compare before and after optimization")
    print("-" * 70)
    
    # Simulate optimized design
    data_optimized_rdit = GMRF_simulator(
        r_optimized.detach(), lamswp_gmrf, rdit_orders=10, kdims=9, is_showinfo=True)
    
    # Plot comparison
    fig, ax = plot_spectrum_compare_opt(
        lamswp0=lamswp_gmrf, data_org=data_gmrfswp_rdit, data_opt=data_optimized_rdit)
    
    # Find the transmission dip for both the original and optimized designs
    orig_idx = np.argmin(data_gmrfswp_rdit.transmission.detach().numpy())
    opt_idx = np.argmin(data_optimized_rdit.transmission.detach().numpy())
    
    print(f"Original resonance wavelength: {lamswp_gmrf[orig_idx]:.4f} um")
    print(f"Optimized resonance wavelength: {lamswp_gmrf[opt_idx]:.4f} um")
    print(f"Target wavelength: {lam_opt[0]:.4f} um")
    
    print("\nExample completed successfully!")
    print("Plots saved in the current directory.")

if __name__ == "__main__":
    main()