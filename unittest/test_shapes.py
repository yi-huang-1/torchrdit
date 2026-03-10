import unittest
import torch
import numpy as np

from torchrdit.shapes import ShapeGenerator
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision


class TestShapeGenerator(unittest.TestCase):
    """Effective tests for ShapeGenerator geometry and gradients.

    Notes on prior tests removed:
    - Prior checks used loss = mask.sum(), which makes gradients wrt
      angle/center often zero by symmetry (area-invariant), so those
      tests didn’t validate meaningful differentiability.
    - Repeated combine_masks tests across many shape pairs were redundant.
    - `get_shape_generator_params` mapping tests duplicate solver coverage;
      we keep `from_solver` mapping checks which are sufficient here.
    """

    def setUp(self):
        # Use a modest grid for speed while retaining fidelity
        self.grids = (64, 64)
        x_grid = torch.linspace(-0.5, 0.5, self.grids[0])
        y_grid = torch.linspace(-0.5, 0.5, self.grids[1])
        X, Y = torch.meshgrid(x_grid, y_grid, indexing="xy")
        self.generator = ShapeGenerator(X, Y, self.grids)
        self.soft_edge = 0.01
        # Spatial weights to make loss sensitive to position/rotation
        self.weight = self.generator.XO + 2.0 * self.generator.YO

    def test_circle_mask_values_and_gradients(self):
        radius = torch.tensor(0.2, requires_grad=True)
        cx = torch.tensor(0.1, requires_grad=True)
        cy = torch.tensor(-0.05, requires_grad=True)

        m_soft = self.generator.generate_circle_mask(center=(cx, cy), radius=radius, soft_edge=self.soft_edge)
        m_hard = self.generator.generate_circle_mask(center=(cx.detach(), cy.detach()), radius=radius.detach(), soft_edge=0.0)

        # Value properties
        self.assertTrue((m_soft >= 0).all() and (m_soft <= 1).all())
        self.assertTrue(((m_hard == 0) | (m_hard == 1)).all())
        self.assertGreater(m_soft.mean().item(), 0.0)

        # Gradient wrt spatially sensitive loss
        loss = (m_soft * self.weight).sum()
        loss.backward()
        self.assertIsNotNone(radius.grad)
        self.assertIsNotNone(cx.grad)
        self.assertIsNotNone(cy.grad)
        self.assertGreater(radius.grad.abs().sum().item(), 0.0)
        self.assertGreater(cx.grad.abs().sum().item(), 0.0)
        self.assertGreater(cy.grad.abs().sum().item(), 0.0)

    def test_rectangle_mask_values_and_gradients(self):
        w = torch.tensor(0.3, requires_grad=True)
        h = torch.tensor(0.2, requires_grad=True)
        ang = torch.tensor(25.0, requires_grad=True)
        cx = torch.tensor(-0.1, requires_grad=True)
        cy = torch.tensor(0.08, requires_grad=True)

        m = self.generator.generate_rectangle_mask(center=(cx, cy), x_size=w, y_size=h, angle=ang, soft_edge=self.soft_edge)

        self.assertTrue((m >= 0).all() and (m <= 1).all())
        self.assertGreater(m.mean().item(), 0.0)

        loss = (m * self.weight).sum()
        loss.backward()
        for g in (w.grad, h.grad, ang.grad, cx.grad, cy.grad):
            self.assertIsNotNone(g)
            self.assertGreater(g.abs().sum().item(), 0.0)

    def test_polygon_mask_values_and_gradients(self):
        px = torch.tensor([0.0, 0.25, -0.2], requires_grad=True)
        py = torch.tensor([-0.2, 0.2, 0.25], requires_grad=True)
        ang = torch.tensor(10.0, requires_grad=True)
        cx = torch.tensor(0.05, requires_grad=True)
        cy = torch.tensor(0.05, requires_grad=True)
        pts = torch.stack([px, py], dim=1)

        m = self.generator.generate_polygon_mask(center=(cx, cy), polygon_points=pts, angle=ang, soft_edge=self.soft_edge)

        self.assertTrue((m >= 0).all() and (m <= 1).all())
        self.assertGreater(m.mean().item(), 0.0)

        loss = (m * self.weight).sum()
        loss.backward()
        for g in (px.grad, py.grad, ang.grad, cx.grad, cy.grad):
            self.assertIsNotNone(g)
            self.assertGreater(g.abs().sum().item(), 0.0)

    def test_combine_masks_operations_and_gradients(self):
        # Two different shapes
        r1 = torch.tensor(0.18, requires_grad=True)
        r2 = torch.tensor(0.22, requires_grad=True)
        cx1 = torch.tensor(-0.15, requires_grad=True)
        cy1 = torch.tensor(0.0, requires_grad=True)
        cx2 = torch.tensor(0.15, requires_grad=True)
        cy2 = torch.tensor(0.0, requires_grad=True)

        c1 = self.generator.generate_circle_mask(center=(cx1, cy1), radius=r1, soft_edge=self.soft_edge)
        c2 = self.generator.generate_circle_mask(center=(cx2, cy2), radius=r2, soft_edge=self.soft_edge)

        union = self.generator.combine_masks(c1, c2, operation="union")
        inter = self.generator.combine_masks(c1, c2, operation="intersection")
        diff = self.generator.combine_masks(c1, c2, operation="difference")
        sub = self.generator.combine_masks(c1, c2, operation="subtract")

        # Basic correctness relations
        self.assertGreaterEqual(union.sum().item(), c1.sum().item())
        self.assertGreaterEqual(union.sum().item(), c2.sum().item())
        self.assertLessEqual(inter.sum().item(), c1.sum().item())
        self.assertLessEqual(inter.sum().item(), c2.sum().item())
        self.assertGreaterEqual(diff.sum().item(), 0.0)
        self.assertLessEqual(sub.sum().item(), c1.sum().item())

        # Gradients through combinations
        loss = (union * self.weight).sum() + (inter * (self.weight**2)).sum() + diff.sum() + sub.sum()
        loss.backward()
        for g in (r1.grad, r2.grad, cx1.grad, cy1.grad, cx2.grad, cy2.grad):
            self.assertIsNotNone(g)
            self.assertGreater(g.abs().sum().item(), 0.0)

    def test_from_solver_mapping_cartesian_precisions(self):
        for precision, tfloat in ((Precision.SINGLE, torch.float32), (Precision.DOUBLE, torch.float64)):
            solver = create_solver(
                algorithm=Algorithm.RCWA,
                precision=precision,
                grids=[self.grids[0], self.grids[1]],
                harmonics=[3, 3],
                lam0=np.array([1.55]),
                t1=torch.tensor([[1.0, 0.0]]),
                t2=torch.tensor([[0.0, 1.0]])
            )
            sg = ShapeGenerator.from_solver(solver)
            self.assertEqual(sg.grids, tuple(solver.grids))
            self.assertTrue(torch.allclose(sg.XO, solver.XO))
            self.assertTrue(torch.allclose(sg.YO, solver.YO))
            self.assertTrue(torch.allclose(sg.lattice_t1, solver.lattice_t1))
            self.assertTrue(torch.allclose(sg.lattice_t2, solver.lattice_t2))
            self.assertEqual(sg.tfloat, tfloat)

    def test_from_solver_mapping_non_cartesian(self):
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            grids=[self.grids[0], self.grids[1]],
            harmonics=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.5]]),
            t2=torch.tensor([[0.0, 1.0]])
        )
        sg = ShapeGenerator.from_solver(solver)
        self.assertTrue(torch.allclose(sg.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(sg.lattice_t2, solver.lattice_t2))
        self.assertTrue(torch.allclose(sg.XO, solver.XO))
        self.assertTrue(torch.allclose(sg.YO, solver.YO))


class TestShapeGeneratorDirectSolverInit(unittest.TestCase):
    """Tests for ShapeGenerator(solver) direct initialization (Task 041)."""

    def _make_solver(self, algorithm=Algorithm.RCWA, precision=Precision.SINGLE,
                     grids=None, t1=None, t2=None):
        if grids is None:
            grids = [64, 64]
        if t1 is None:
            t1 = torch.tensor([[1.0, 0.0]])
        if t2 is None:
            t2 = torch.tensor([[0.0, 1.0]])
        return create_solver(
            algorithm=algorithm, precision=precision,
            grids=grids, harmonics=[3, 3],
            lam0=np.array([1.55]), t1=t1, t2=t2,
        )

    # ── Milestone 1: Solver-Based Init ──

    def test_init_from_solver_rcwa(self):
        solver = self._make_solver(algorithm=Algorithm.RCWA)
        sg = ShapeGenerator(solver)
        ref = ShapeGenerator.from_solver(solver)
        self.assertEqual(sg.grids, tuple(solver.grids))
        self.assertTrue(torch.allclose(sg.XO, solver.XO))
        self.assertTrue(torch.allclose(sg.YO, solver.YO))
        self.assertTrue(torch.allclose(sg.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(sg.lattice_t2, solver.lattice_t2))
        self.assertEqual(sg.tfloat, solver.tfloat)
        self.assertEqual(sg.tcomplex, solver.tcomplex)
        self.assertEqual(sg.tint, solver.tint)
        self.assertEqual(sg.nfloat, solver.nfloat)
        # Must be identical to from_solver result
        self.assertTrue(torch.allclose(sg.XO, ref.XO))
        self.assertTrue(torch.allclose(sg.YO, ref.YO))
        self.assertEqual(sg.grids, ref.grids)

    def test_init_from_solver_rdit(self):
        solver = self._make_solver(algorithm=Algorithm.RDIT)
        sg = ShapeGenerator(solver)
        self.assertEqual(sg.grids, tuple(solver.grids))
        self.assertTrue(torch.allclose(sg.XO, solver.XO))
        self.assertTrue(torch.allclose(sg.YO, solver.YO))
        self.assertEqual(sg.tfloat, solver.tfloat)

    def test_init_from_solver_non_cartesian(self):
        solver = self._make_solver(
            t1=torch.tensor([[1.0, 0.5]]),
            t2=torch.tensor([[0.0, 1.0]]),
        )
        sg = ShapeGenerator(solver)
        self.assertFalse(sg.is_cartesian)
        self.assertTrue(torch.allclose(sg.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(sg.lattice_t2, solver.lattice_t2))
        self.assertTrue(torch.allclose(sg.XO, solver.XO))
        self.assertTrue(torch.allclose(sg.YO, solver.YO))

    def test_init_from_solver_precisions(self):
        for precision, expected_tfloat, expected_tcomplex in (
            (Precision.SINGLE, torch.float32, torch.complex64),
            (Precision.DOUBLE, torch.float64, torch.complex128),
        ):
            solver = self._make_solver(precision=precision)
            sg = ShapeGenerator(solver)
            self.assertEqual(sg.tfloat, expected_tfloat)
            self.assertEqual(sg.tcomplex, expected_tcomplex)

    # ── Milestone 2: Backward Compatibility ──

    def test_init_direct_still_works(self):
        x = torch.linspace(-0.5, 0.5, 64)
        y = torch.linspace(-0.5, 0.5, 64)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sg = ShapeGenerator(X, Y, (64, 64))
        self.assertTrue(torch.allclose(sg.XO, X))
        self.assertTrue(torch.allclose(sg.YO, Y))
        self.assertEqual(sg.grids, (64, 64))
        self.assertTrue(sg.is_cartesian)

    def test_init_direct_with_lattice_still_works(self):
        x = torch.linspace(-0.5, 0.5, 64)
        y = torch.linspace(-0.5, 0.5, 64)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        t1 = torch.tensor([1.0, 0.5])
        t2 = torch.tensor([0.0, 1.0])
        sg = ShapeGenerator(X, Y, (64, 64), lattice_t1=t1, lattice_t2=t2)
        self.assertFalse(sg.is_cartesian)
        self.assertTrue(torch.allclose(sg.lattice_t1, t1))
        self.assertTrue(torch.allclose(sg.lattice_t2, t2))

    def test_init_kwargs_still_works(self):
        solver = self._make_solver()
        params = solver.get_shape_generator_params()
        sg = ShapeGenerator(**params)
        self.assertEqual(sg.grids, tuple(solver.grids))
        self.assertTrue(torch.allclose(sg.XO, solver.XO))
        self.assertTrue(torch.allclose(sg.YO, solver.YO))

    def test_invalid_first_arg_raises(self):
        with self.assertRaises(TypeError):
            ShapeGenerator("not a solver")
        with self.assertRaises(TypeError):
            ShapeGenerator(42)
        with self.assertRaises(TypeError):
            ShapeGenerator([1, 2, 3])
        with self.assertRaises(TypeError):
            ShapeGenerator({"XO": 1})  # dict with XO key but not solver-like

    # ── Milestone 3: Functional Equivalence ──

    def test_solver_init_masks_match_from_solver(self):
        solver = self._make_solver()
        sg_direct = ShapeGenerator(solver)
        sg_factory = ShapeGenerator.from_solver(solver)
        # Circle
        c1 = sg_direct.generate_circle_mask(center=(0.1, -0.1), radius=0.2)
        c2 = sg_factory.generate_circle_mask(center=(0.1, -0.1), radius=0.2)
        self.assertTrue(torch.allclose(c1, c2))
        # Rectangle
        r1 = sg_direct.generate_rectangle_mask(x_size=0.3, y_size=0.2, angle=30)
        r2 = sg_factory.generate_rectangle_mask(x_size=0.3, y_size=0.2, angle=30)
        self.assertTrue(torch.allclose(r1, r2))
        # Polygon
        pts = [(-0.1, -0.1), (0.1, -0.1), (0.0, 0.1)]
        p1 = sg_direct.generate_polygon_mask(pts)
        p2 = sg_factory.generate_polygon_mask(pts)
        self.assertTrue(torch.allclose(p1, p2))

    def test_solver_init_gradient_flow(self):
        solver = self._make_solver()
        sg = ShapeGenerator(solver)
        radius = torch.tensor(0.2, requires_grad=True)
        cx = torch.tensor(0.05, requires_grad=True)
        cy = torch.tensor(-0.05, requires_grad=True)
        mask = sg.generate_circle_mask(center=(cx, cy), radius=radius, soft_edge=0.01)
        weight = sg.XO + 2.0 * sg.YO
        loss = (mask * weight).sum()
        loss.backward()
        self.assertIsNotNone(radius.grad)
        self.assertGreater(radius.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(cx.grad)
        self.assertGreater(cx.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(cy.grad)
        self.assertGreater(cy.grad.abs().sum().item(), 0.0)

if __name__ == "__main__":
    unittest.main()
