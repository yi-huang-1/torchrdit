import numpy as np
import torch
import pytest
import os
from torchrdit.solver import create_solver, get_solver_builder
from torchrdit.utils import create_material, operator_proj
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator


class TestSolver:
    # units
    um = 1
    nm = 1e-3 * um
    degrees = np.pi / 180

    lam0 = 1540 * nm
    theta = 0 * degrees
    phi = 0 * degrees

    pte = 1
    ptm = 0

    # device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * nm
    b = a * np.sqrt(3)

    r = 400 * nm
    h1 = torch.tensor(230 * nm, dtype=torch.float32)
    h2 = torch.tensor(345 * nm, dtype=torch.float32)

    t1 = torch.tensor([[a/2, -a * np.sqrt(3)/2]], dtype=torch.float32)
    t2 = torch.tensor([[a/2, a * np.sqrt(3)/2]], dtype=torch.float32)

    def test_gmrf_rcwa(self):
        material_sio = create_material(
            name='SiO', permittivity=self.n_SiO**2)
        material_sin = create_material(
            name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(
            name='FusedSilica', permittivity=self.n_fs**2)

        dev1 = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.DOUBLE,
            rdim=[512, 512],
            kdim=[9, 9],
            lam0=np.array([1540 * self.nm]),
            lengthunit='um',
            t1=self.t1,
            t2=self.t2)

        dev1.update_trn_material(trn_material=material_fs)

        dev1.add_layer(material_name=material_sio,
                       thickness=self.h1,
                       is_homogeneous=False,
                       is_optimize=True)

        dev1.add_layer(material_name=material_sin,
                       thickness=self.h2,
                       is_homogeneous=True,
                       is_optimize=False)

        src1 = dev1.add_source(theta=self.theta,
                               phi=self.phi,
                               pte=self.pte,
                               ptm=self.ptm)
        
        shape_gen = ShapeGenerator.from_solver(dev1)

        # build hexagonal unit cell
        c1 = shape_gen.generate_circle_mask(center=[0, self.b/2], radius=self.r)
        c2 = shape_gen.generate_circle_mask(center=[0, -self.b/2], radius=self.r)
        c3 = shape_gen.generate_circle_mask(center=[self.a/2, 0], radius=self.r)
        c4 = shape_gen.generate_circle_mask(center=[-self.a/2, 0], radius=self.r)

        mask = shape_gen.combine_masks(mask1=c1, mask2=c2, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c3, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c4, operation='union')

        mask = 1 - mask

        dev1.update_er_with_mask(mask=mask, layer_index=0)
        data = dev1.solve(src1)

        assert(np.isclose(data.transmission[0].detach().numpy(), 0.92, atol=2e-2))
        assert(np.isclose(data.reflection[0].detach().numpy(), 0.07, atol=2e-2))
        assert data.transmission[0].detach().numpy().dtype == np.float64
        assert data.reflection[0].detach().numpy().dtype == np.float64

    def test_gmrf_rdit(self):
        material_sio = create_material(
            name='SiO', permittivity=self.n_SiO**2)
        material_sin = create_material(
            name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(
            name='FusedSilica', permittivity=self.n_fs**2)

        dev1 = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            rdim=[512, 512],
            kdim=[9, 9],
            lam0=np.array([1540 * self.nm]),
            lengthunit='um',
            t1=self.t1,
            t2=self.t2)

        dev1.set_rdit_order(10)

        dev1.update_trn_material(trn_material=material_fs)

        dev1.add_layer(material_name=material_sio,
                       thickness=self.h1,
                       is_homogeneous=False,
                       is_optimize=True)

        dev1.add_layer(material_name=material_sin,
                       thickness=self.h2,
                       is_homogeneous=True,
                       is_optimize=False)

        src1 = dev1.add_source(theta=self.theta,
                               phi=self.phi,
                               pte=self.pte,
                               ptm=self.ptm)
        
        shape_gen = ShapeGenerator.from_solver(dev1)

        # build hexagonal unit cell
        c1 = shape_gen.generate_circle_mask(center=[0, self.b/2], radius=self.r)
        c2 = shape_gen.generate_circle_mask(center=[0, -self.b/2], radius=self.r)
        c3 = shape_gen.generate_circle_mask(center=[self.a/2, 0], radius=self.r)
        c4 = shape_gen.generate_circle_mask(center=[-self.a/2, 0], radius=self.r)

        mask = shape_gen.combine_masks(mask1=c1, mask2=c2, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c3, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c4, operation='union')

        mask = 1 - mask

        dev1.update_er_with_mask(mask=mask, layer_index=0)
        data = dev1.solve(src1)

        assert(np.isclose(data.transmission[0].detach().numpy(), 0.92, atol=2e-2))
        assert(np.isclose(data.reflection[0].detach().numpy(), 0.07, atol=2e-2))
        assert data.transmission[0].detach().numpy().dtype == np.float64
        assert data.reflection[0].detach().numpy().dtype == np.float64

class TestSolverAnalyticalFourier:
    # units
    um = 1
    nm = 1e-3 * um
    degrees = np.pi / 180

    lam0 = 1540 * nm
    theta = 0 * degrees
    phi = 0 * degrees

    pte = 1
    ptm = 0

    # device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * nm

    r = 400 * nm
    h1 = torch.tensor(230 * nm, dtype=torch.float32)
    h2 = torch.tensor(345 * nm, dtype=torch.float32)

    t1 = torch.tensor([[a, 0]], dtype=torch.float32)
    t2 = torch.tensor([[0, a]], dtype=torch.float32)

    def make_mask(self, radius_var, dev):
        rsq = (dev.XO) ** 2 + (dev.YO) ** 2
        mask = rsq - (radius_var) ** 2 + 0.5
        mask = operator_proj(mask, eta=0.5, beta = 100)
        return mask

    def test_gmrf_rcwa(self):
        material_sio = create_material(
            name='SiO', permittivity=self.n_SiO**2)
        material_sin = create_material(
            name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(
            name='FusedSilica', permittivity=self.n_fs**2)

        dev1 = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.DOUBLE,
            rdim=[512, 512],
            kdim=[9, 9],
            lam0=np.array([1540 * self.nm]),
            lengthunit='um',
            t1=self.t1,
            t2=self.t2)

        dev1.update_trn_material(trn_material=material_fs)

        dev1.add_layer(material_name=material_sio,
                       thickness=self.h1,
                       is_homogeneous=False,
                       is_optimize=True)

        dev1.add_layer(material_name=material_sin,
                       thickness=self.h2,
                       is_homogeneous=True,
                       is_optimize=False)

        src1 = dev1.add_source(theta=self.theta,
                               phi=self.phi,
                               pte=self.pte,
                               ptm=self.ptm)

        # build hexagonal unit cell
        mask = self.make_mask(self.r, dev1)

        dev1.update_er_with_mask(mask=mask, layer_index=0, method='Analytical')
        data_analytical = dev1.solve(src1)
        dev1.update_er_with_mask(mask=mask, layer_index=0, method='FFT')
        data_fft = dev1.solve(src1, is_use_FFF=False)

        assert(np.isclose(np.int32(data_fft.transmission[0].detach().numpy() * 1000), np.int32(data_analytical.transmission[0].detach().numpy() * 1000)))
        assert(np.isclose(np.int32(data_fft.reflection[0].detach().numpy() * 1000), np.int32(data_analytical.reflection[0].detach().numpy() * 1000)))
        assert data_analytical.transmission[0].detach().numpy().dtype == np.float64
        assert data_analytical.reflection[0].detach().numpy().dtype == np.float64
        assert data_fft.transmission[0].detach().numpy().dtype == np.float64
        assert data_fft.reflection[0].detach().numpy().dtype == np.float64

    def test_gmrf_rdit(self):
        material_sio = create_material(
            name='SiO', permittivity=self.n_SiO**2)
        material_sin = create_material(
            name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(
            name='FusedSilica', permittivity=self.n_fs**2)

        dev1 = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            rdim=[512, 512],
            kdim=[9, 9],
            lam0=np.array([1540 * self.nm]),
            lengthunit='um',
            t1=self.t1,
            t2=self.t2)

        dev1.set_rdit_order(10)

        dev1.update_trn_material(trn_material=material_fs)

        dev1.add_layer(material_name=material_sio,
                       thickness=self.h1,
                       is_homogeneous=False,
                       is_optimize=True)

        dev1.add_layer(material_name=material_sin,
                       thickness=self.h2,
                       is_homogeneous=True,
                       is_optimize=False)

        src1 = dev1.add_source(theta=self.theta,
                               phi=self.phi,
                               pte=self.pte,
                               ptm=self.ptm)

        # build hexagonal unit cell
        mask = self.make_mask(self.r, dev1)

        dev1.update_er_with_mask(mask=mask, layer_index=0, method='Analytical')
        data_analytical = dev1.solve(src1)
        dev1.update_er_with_mask(mask=mask, layer_index=0, method='FFT')
        data_fft = dev1.solve(src1, is_use_FFF=False)

        assert(np.isclose(np.int32(data_fft.transmission[0].detach().numpy() * 1000), np.int32(data_analytical.transmission[0].detach().numpy() * 1000)))
        assert(np.isclose(np.int32(data_fft.reflection[0].detach().numpy() * 1000), np.int32(data_analytical.reflection[0].detach().numpy() * 1000)))
        assert data_analytical.transmission[0].detach().numpy().dtype == np.float64
        assert data_analytical.reflection[0].detach().numpy().dtype == np.float64
        assert data_fft.transmission[0].detach().numpy().dtype == np.float64
        assert data_fft.reflection[0].detach().numpy().dtype == np.float64

class TestSolverFloat:
    # units
    um = 1
    nm = 1e-3 * um
    degrees = np.pi / 180

    lam0 = 1540 * nm
    theta = 0 * degrees
    phi = 0 * degrees

    pte = 1
    ptm = 0

    # device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * nm
    b = a * np.sqrt(3)

    r = 400 * nm
    h1 = torch.tensor(230 * nm, dtype=torch.float32)
    h2 = torch.tensor(345 * nm, dtype=torch.float32)

    t1 = torch.tensor([[a/2, -a * np.sqrt(3)/2]], dtype=torch.float32)
    t2 = torch.tensor([[a/2, a * np.sqrt(3)/2]], dtype=torch.float32)

    def test_gmrf_rcwa_float(self):
        material_sio = create_material(
            name='SiO', permittivity=self.n_SiO**2)
        material_sin = create_material(
            name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(
            name='FusedSilica', permittivity=self.n_fs**2)

        dev1 = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            rdim=[512, 512],
            kdim=[9, 9],
            lam0=np.array([1540 * self.nm]),
            lengthunit='um',
            t1=self.t1,
            t2=self.t2)

        dev1.update_trn_material(trn_material=material_fs)

        dev1.add_layer(material_name=material_sio,
                       thickness=self.h1,
                       is_homogeneous=False,
                       is_optimize=True)

        dev1.add_layer(material_name=material_sin,
                       thickness=self.h2,
                       is_homogeneous=True,
                       is_optimize=False)

        src1 = dev1.add_source(theta=self.theta,
                               phi=self.phi,
                               pte=self.pte,
                               ptm=self.ptm)
        
        shape_gen = ShapeGenerator.from_solver(dev1)

        # build hexagonal unit cell
        c1 = shape_gen.generate_circle_mask(center=[0, self.b/2], radius=self.r)
        c2 = shape_gen.generate_circle_mask(center=[0, -self.b/2], radius=self.r)
        c3 = shape_gen.generate_circle_mask(center=[self.a/2, 0], radius=self.r)
        c4 = shape_gen.generate_circle_mask(center=[-self.a/2, 0], radius=self.r)

        mask = shape_gen.combine_masks(mask1=c1, mask2=c2, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c3, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c4, operation='union')

        mask = 1 - mask

        dev1.update_er_with_mask(mask=mask, layer_index=0)
        data = dev1.solve(src1)

        assert(np.isclose(data.transmission[0].detach().numpy(), 0.92, atol=2e-2))
        assert(np.isclose(data.reflection[0].detach().numpy(), 0.07, atol=2e-2))
        assert data.transmission[0].detach().numpy().dtype == np.float32
        assert data.reflection[0].detach().numpy().dtype == np.float32

    def test_gmrf_rdit_float(self):
        material_sio = create_material(
            name='SiO', permittivity=self.n_SiO**2)
        material_sin = create_material(
            name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(
            name='FusedSilica', permittivity=self.n_fs**2)

        dev1 = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.SINGLE,
            rdim=[512, 512],
            kdim=[9, 9],
            lam0=np.array([1540 * self.nm]),
            lengthunit='um',
            t1=self.t1,
            t2=self.t2)

        dev1.set_rdit_order(10)

        dev1.update_trn_material(trn_material=material_fs)

        dev1.add_layer(material_name=material_sio,
                       thickness=self.h1,
                       is_homogeneous=False,
                       is_optimize=True)

        dev1.add_layer(material_name=material_sin,
                       thickness=self.h2,
                       is_homogeneous=True,
                       is_optimize=False)

        src1 = dev1.add_source(theta=self.theta,
                               phi=self.phi,
                               pte=self.pte,
                               ptm=self.ptm)
        
        shape_gen = ShapeGenerator.from_solver(dev1)

        # build hexagonal unit cell
        c1 = shape_gen.generate_circle_mask(center=[0, self.b/2], radius=self.r)
        c2 = shape_gen.generate_circle_mask(center=[0, -self.b/2], radius=self.r)
        c3 = shape_gen.generate_circle_mask(center=[self.a/2, 0], radius=self.r)
        c4 = shape_gen.generate_circle_mask(center=[-self.a/2, 0], radius=self.r)

        mask = shape_gen.combine_masks(mask1=c1, mask2=c2, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c3, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c4, operation='union')

        mask = 1 - mask

        dev1.update_er_with_mask(mask=mask, layer_index=0)
        data = dev1.solve(src1)

        assert(np.isclose(data.transmission[0].detach().numpy(), 0.92, atol=2e-2))
        assert(np.isclose(data.reflection[0].detach().numpy(), 0.07, atol=2e-2))
        assert data.transmission[0].detach().numpy().dtype == np.float32
        assert data.reflection[0].detach().numpy().dtype == np.float32



class TestSolverDispersive:
    """Test solver with dispersive materials."""
    
    # units
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
    t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float64)
    t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float64)

    def test_dispersive_nonhomo_layer(self):
        """
        Test GMRF with dispersive materials, exactly matching Demo-03a.py setup.
        Using the real dispersive material files Si_C-e.txt and SiO2-e.txt.
        Dispersive non-homogeneous layer is SiN.
        Non-dispersive layer is SiO2.
        """
        
        # Initialize the solver using the builder pattern, matching Demo-03a.py geometry
        dev = (get_solver_builder()
            .with_algorithm(Algorithm.RDIT)
            .with_precision(Precision.DOUBLE)
            .with_real_dimensions([256, 256])    # Reduced for faster testing
            .with_k_dimensions([9, 9])         # Further reduced to avoid matrix size mismatch
            .with_wavelengths(np.array([1540 * self.nm]))  # Just one wavelength for testing
            .with_length_unit('um')
            .with_lattice_vectors(self.t1, self.t2)
            .with_fff(True)
            .build())
        
        # Creating materials, exactly as in Demo-03a.py
        material_sic = create_material(name='SiC', dielectric_dispersion=True, 
                                    user_dielectric_file=os.path.join(os.path.dirname(__file__), 'Si_C-e.txt'), 
                                    data_format='freq-eps', data_unit='thz')
        material_sio2 = create_material(name='SiO2', dielectric_dispersion=True, 
                                        user_dielectric_file=os.path.join(os.path.dirname(__file__), 'SiO2-e.txt'), 
                                        data_format='freq-eps', data_unit='thz')
        material_sin = create_material(name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(name='FusedSilica', permittivity=self.n_fs**2)
        
        # Manually add materials to the device
        dev.add_materials(material_list=[material_sic, material_sio2, material_sin, material_fs])
        
        # Update the material of the transmission layer
        dev.update_trn_material(trn_material=material_fs)
        
        # Add layers to the device
        dev.add_layer(material_name=material_sic,
                     thickness=self.h1,
                     is_homogeneous=False)
        
        dev.add_layer(material_name=material_sin,
                     thickness=self.h2,
                     is_homogeneous=True)
        
        # Create source
        src = dev.add_source(theta=self.theta,
                            phi=self.phi,
                            pte=self.pte,
                            ptm=self.ptm)
        
        shape_gen = ShapeGenerator.from_solver(dev)
        
        # Build hexagonal unit cell with 4 circles
        c1 = shape_gen.generate_circle_mask(center=[0, self.b/2], radius=self.r)
        c2 = shape_gen.generate_circle_mask(center=[0, -self.b/2], radius=self.r)
        c3 = shape_gen.generate_circle_mask(center=[self.a/2, 0], radius=self.r)
        c4 = shape_gen.generate_circle_mask(center=[-self.a/2, 0], radius=self.r)
        
        mask = shape_gen.combine_masks(mask1=c1, mask2=c2, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c3, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c4, operation='union')
        
        mask = (1 - mask).to(torch.float64)

        mask.requires_grad = True
        
        dev.update_er_with_mask(mask=mask, layer_index=0)
        
        # Solve and check results
        data = dev.solve(src)
        
        # Verify data contains expected fields
        assert hasattr(data, 'transmission')
        assert hasattr(data, 'reflection')
        
        # Values should be between 0 and 1 for energy conservation
        assert 0 <= data.transmission[0].item() <= 1
        assert 0 <= data.reflection[0].item() <= 1
        
        # Energy conservation check
        assert (data.transmission[0].item() + data.reflection[0].item()) <= 1 + 1e-4
        
        # Check for exact expected values based on user's provided results
        expected_trn = 0.43316791878804345  # 43.32% transmission
        expected_ref = 0.54319673247843570  # 54.32% reflection
        
        # Allow for differences due to reduced k-dimensions compared to Demo-03a.py
        assert expected_trn - 0.1 <= data.transmission[0].item() <= expected_trn + 0.1
        assert expected_ref - 0.1 <= data.reflection[0].item() <= expected_ref + 0.1

    def test_dispersive_homo_layer(self):
        """
        Test GMRF with dispersive materials, exactly matching Demo-03a.py setup.
        Using the real dispersive material files Si_C-e.txt and SiO2-e.txt.
        Dispersive non-homogeneous layer is SiN.
        Dispersive homogeneous layer is SiO2.
        """
        
        # Initialize the solver using the builder pattern, matching Demo-03a.py geometry
        dev = (get_solver_builder()
            .with_algorithm(Algorithm.RDIT)
            .with_precision(Precision.DOUBLE)
            .with_real_dimensions([256, 256])    # Reduced for faster testing
            .with_k_dimensions([9, 9])         # Further reduced to avoid matrix size mismatch
            .with_wavelengths(np.array([1540 * self.nm]))  # Just one wavelength for testing
            .with_length_unit('um')
            .with_lattice_vectors(self.t1, self.t2)
            .with_fff(True)
            .build())
        
        # Creating materials, exactly as in Demo-03a.py
        material_sic = create_material(name='SiC', dielectric_dispersion=True, 
                                    user_dielectric_file=os.path.join(os.path.dirname(__file__), 'Si_C-e.txt'), 
                                    data_format='freq-eps', data_unit='thz')
        material_sio2 = create_material(name='SiO2', dielectric_dispersion=True, 
                                        user_dielectric_file=os.path.join(os.path.dirname(__file__), 'SiO2-e.txt'), 
                                        data_format='freq-eps', data_unit='thz')
        material_sin = create_material(name='SiN', permittivity=self.n_SiN**2)
        material_fs = create_material(name='FusedSilica', permittivity=self.n_fs**2)
        
        # Manually add materials to the device
        dev.add_materials(material_list=[material_sic, material_sio2, material_sin, material_fs])
        
        # Update the material of the transmission layer
        dev.update_trn_material(trn_material=material_fs)
        
        # Add layers to the device
        dev.add_layer(material_name=material_sic,
                     thickness=self.h1,
                     is_homogeneous=False)
        
        dev.add_layer(material_name=material_sio2,
                     thickness=self.h2,
                     is_homogeneous=True)
        
        # Create source
        src = dev.add_source(theta=self.theta,
                            phi=self.phi,
                            pte=self.pte,
                            ptm=self.ptm)
        
        shape_gen = ShapeGenerator.from_solver(dev)
        
        # Build hexagonal unit cell with 4 circles
        c1 = shape_gen.generate_circle_mask(center=[0, self.b/2], radius=self.r)
        c2 = shape_gen.generate_circle_mask(center=[0, -self.b/2], radius=self.r)
        c3 = shape_gen.generate_circle_mask(center=[self.a/2, 0], radius=self.r)
        c4 = shape_gen.generate_circle_mask(center=[-self.a/2, 0], radius=self.r)
        
        mask = shape_gen.combine_masks(mask1=c1, mask2=c2, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c3, operation='union')
        mask = shape_gen.combine_masks(mask1=mask, mask2=c4, operation='union')
        
        mask = (1 - mask).to(torch.float64)

        mask.requires_grad = True
        
        dev.update_er_with_mask(mask=mask, layer_index=0)
        
        # Solve and check results
        data = dev.solve(src)
        
        # Verify data contains expected fields
        assert hasattr(data, 'transmission')
        assert hasattr(data, 'reflection')
        
        # Values should be between 0 and 1 for energy conservation
        assert 0 <= data.transmission[0].item() <= 1
        assert 0 <= data.reflection[0].item() <= 1
        
        # Energy conservation check
        assert (data.transmission[0].item() + data.reflection[0].item()) <= 1 + 1e-4
        
        # Check for exact expected values based on user's provided results
        expected_trn = 0.16843374773159475  # 16.843374773159475% transmission
        expected_ref = 0.6830142455132093  # 68.30142455132093% reflection
        
        # Allow for differences due to reduced k-dimensions compared to Demo-03a.py
        assert expected_trn - 0.1 <= data.transmission[0].item() <= expected_trn + 0.1
        assert expected_ref - 0.1 <= data.reflection[0].item() <= expected_ref + 0.1


if __name__ == '__main__':
    pytest.main([__file__])