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
        self.rdim = (64, 64)
        x_grid = torch.linspace(-0.5, 0.5, self.rdim[0])
        y_grid = torch.linspace(-0.5, 0.5, self.rdim[1])
        X, Y = torch.meshgrid(x_grid, y_grid, indexing="xy")
        self.generator = ShapeGenerator(X, Y, self.rdim)
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
                rdim=[self.rdim[0], self.rdim[1]],
                kdim=[3, 3],
                lam0=np.array([1.55]),
                t1=torch.tensor([[1.0, 0.0]]),
                t2=torch.tensor([[0.0, 1.0]])
            )
            sg = ShapeGenerator.from_solver(solver)
            self.assertEqual(sg.rdim, tuple(solver.rdim))
            self.assertTrue(torch.allclose(sg.XO, solver.XO))
            self.assertTrue(torch.allclose(sg.YO, solver.YO))
            self.assertTrue(torch.allclose(sg.lattice_t1, solver.lattice_t1))
            self.assertTrue(torch.allclose(sg.lattice_t2, solver.lattice_t2))
            self.assertEqual(sg.tfloat, tfloat)

    def test_from_solver_mapping_non_cartesian(self):
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            rdim=[self.rdim[0], self.rdim[1]],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.5]]),
            t2=torch.tensor([[0.0, 1.0]])
        )
        sg = ShapeGenerator.from_solver(solver)
        self.assertTrue(torch.allclose(sg.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(sg.lattice_t2, solver.lattice_t2))
        self.assertTrue(torch.allclose(sg.XO, solver.XO))
        self.assertTrue(torch.allclose(sg.YO, solver.YO))


if __name__ == "__main__":
    unittest.main()
