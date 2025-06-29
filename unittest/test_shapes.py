import unittest
import torch
import numpy as np
from torchrdit.shapes import ShapeGenerator
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision


class TestShapeGeneratorDifferentiable(unittest.TestCase):
    """Tests for the differentiable properties of the ShapeGenerator class.
    
    These tests verify that gradients flow correctly through the shape generation
    operations and remain differentiable through various shape operations.
    """
    
    def setUp(self):
        """Set up common test fixtures."""
        # Create a sample generator
        self.rdim = (256, 256)
        x_grid = torch.linspace(-0.5, 0.5, self.rdim[0])
        y_grid = torch.linspace(-0.5, 0.5, self.rdim[1])
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
        self.generator = ShapeGenerator(X, Y, self.rdim)
        self.soft_edge = 0.001

    def test_circle_mask_differentiable(self):
        """Test that circle mask parameters are differentiable."""
        # Setup differentiable parameters
        radius = torch.tensor(0.2, requires_grad=True)
        center_x = torch.tensor(0.0, requires_grad=True)
        center_y = torch.tensor(0.0, requires_grad=True)
        
        # Create a differentiable circle mask
        circle = self.generator.generate_circle_mask(
            center=(center_x, center_y), 
            radius=radius, 
            soft_edge=self.soft_edge
        )
        
        # Check if gradients flow through
        loss = circle.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        self.assertIsNotNone(radius.grad)
        self.assertIsNotNone(center_x.grad)
        self.assertIsNotNone(center_y.grad)
        
        self.assertFalse(torch.isnan(radius.grad).any())
        self.assertFalse(torch.isnan(center_x.grad).any())
        self.assertFalse(torch.isnan(center_y.grad).any())

    def test_rectangle_mask_differentiable(self):
        """Test that rectangle mask parameters are differentiable."""
        # Setup differentiable parameters
        width = torch.tensor(0.3, requires_grad=True)
        height = torch.tensor(0.2, requires_grad=True)
        angle = torch.tensor(45.0, requires_grad=True)
        center_x = torch.tensor(0.0, requires_grad=True)
        center_y = torch.tensor(0.0, requires_grad=True)
        
        # Create a differentiable rectangle mask
        rectangle = self.generator.generate_rectangle_mask(
            center=(center_x, center_y),
            width=width,
            height=height,
            angle=angle,
            soft_edge=self.soft_edge
        )
        
        # Check if gradients flow through
        loss = rectangle.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        self.assertIsNotNone(width.grad)
        self.assertIsNotNone(height.grad)
        self.assertIsNotNone(angle.grad)
        self.assertIsNotNone(center_x.grad)
        self.assertIsNotNone(center_y.grad)
        
        self.assertFalse(torch.isnan(width.grad).any())
        self.assertFalse(torch.isnan(height.grad).any())
        self.assertFalse(torch.isnan(angle.grad).any())
        self.assertFalse(torch.isnan(center_x.grad).any())
        self.assertFalse(torch.isnan(center_y.grad).any())

    def test_polygon_mask_differentiable(self):
        """Test that polygon mask parameters are differentiable."""
        # Setup differentiable parameters
        pts_x = torch.tensor([0.0, 0.3, -0.3], requires_grad=True)
        pts_y = torch.tensor([0.0, 0.3, 0.3], requires_grad=True)
        center_x = torch.tensor(0.0, requires_grad=True)
        center_y = torch.tensor(0.0, requires_grad=True)
        angle = torch.tensor(0.0, requires_grad=True)
        
        # Define polygon vertices
        polygon_points = torch.stack([pts_x, pts_y], dim=1)
        
        # Create a differentiable polygon mask
        polygon = self.generator.generate_polygon_mask(
            center=(center_x, center_y),
            polygon_points=polygon_points,
            angle=angle,
            invert=False,
            soft_edge=self.soft_edge
        )
        
        # Check if gradients flow through
        loss = polygon.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        self.assertIsNotNone(pts_x.grad)
        self.assertIsNotNone(pts_y.grad)
        self.assertIsNotNone(center_x.grad)
        self.assertIsNotNone(center_y.grad)
        self.assertIsNotNone(angle.grad)
        
        self.assertFalse(torch.isnan(pts_x.grad).any())
        self.assertFalse(torch.isnan(pts_y.grad).any())
        self.assertFalse(torch.isnan(center_x.grad).any())
        self.assertFalse(torch.isnan(center_y.grad).any())
        self.assertFalse(torch.isnan(angle.grad).any())

    def test_combine_masks_union_differentiable(self):
        """Test that union operation preserves differentiability."""
        # Setup differentiable parameters
        radius1 = torch.tensor(0.2, requires_grad=True)
        radius2 = torch.tensor(0.3, requires_grad=True)
        center_x1 = torch.tensor(-0.2, requires_grad=True)
        center_y1 = torch.tensor(0.0, requires_grad=True)
        center_x2 = torch.tensor(0.2, requires_grad=True)
        center_y2 = torch.tensor(0.0, requires_grad=True)
        
        # Create two differentiable circle masks
        circle1 = self.generator.generate_circle_mask(
            center=(center_x1, center_y1), 
            radius=radius1, 
            soft_edge=self.soft_edge
        )
        circle2 = self.generator.generate_circle_mask(
            center=(center_x2, center_y2), 
            radius=radius2, 
            soft_edge=self.soft_edge
        )
        
        # Combine the masks using the union operation
        union = self.generator.combine_masks(circle1, circle2, operation="union")
        
        # Check if gradients flow through
        loss = union.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        gradients = [radius1.grad, radius2.grad, center_x1.grad, center_y1.grad, 
                     center_x2.grad, center_y2.grad]
        
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(torch.isnan(grad).any())

    def test_combine_masks_intersection_differentiable(self):
        """Test that intersection operation preserves differentiability."""
        # Setup differentiable parameters
        radius1 = torch.tensor(0.2, requires_grad=True)
        radius2 = torch.tensor(0.3, requires_grad=True)
        center_x1 = torch.tensor(-0.2, requires_grad=True)
        center_y1 = torch.tensor(0.0, requires_grad=True)
        center_x2 = torch.tensor(0.2, requires_grad=True)
        center_y2 = torch.tensor(0.0, requires_grad=True)
        
        # Create two differentiable circle masks
        circle1 = self.generator.generate_circle_mask(
            center=(center_x1, center_y1), 
            radius=radius1, 
            soft_edge=self.soft_edge
        )
        circle2 = self.generator.generate_circle_mask(
            center=(center_x2, center_y2), 
            radius=radius2, 
            soft_edge=self.soft_edge
        )
        
        # Combine the masks using the intersection operation
        intersection = self.generator.combine_masks(circle1, circle2, operation="intersection")
        
        # Check if gradients flow through
        loss = intersection.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        gradients = [radius1.grad, radius2.grad, center_x1.grad, center_y1.grad, 
                     center_x2.grad, center_y2.grad]
        
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(torch.isnan(grad).any())

    def test_combine_masks_difference_differentiable(self):
        """Test that difference operation preserves differentiability."""
        # Setup differentiable parameters
        radius1 = torch.tensor(0.2, requires_grad=True)
        radius2 = torch.tensor(0.3, requires_grad=True)
        center_x1 = torch.tensor(-0.2, requires_grad=True)
        center_y1 = torch.tensor(0.0, requires_grad=True)
        center_x2 = torch.tensor(0.2, requires_grad=True)
        center_y2 = torch.tensor(0.0, requires_grad=True)
        
        # Create two differentiable circle masks
        circle1 = self.generator.generate_circle_mask(
            center=(center_x1, center_y1), 
            radius=radius1, 
            soft_edge=self.soft_edge
        )
        circle2 = self.generator.generate_circle_mask(
            center=(center_x2, center_y2), 
            radius=radius2, 
            soft_edge=self.soft_edge
        )
        
        # Combine the masks using the difference operation
        difference = self.generator.combine_masks(circle1, circle2, operation="difference")
        
        # Check if gradients flow through
        loss = difference.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        gradients = [radius1.grad, radius2.grad, center_x1.grad, center_y1.grad, 
                     center_x2.grad, center_y2.grad]
        
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(torch.isnan(grad).any())

    def test_combine_masks_rectangles_differentiable(self):
        """Test that rectangle combinations preserve differentiability."""
        # Setup differentiable parameters
        rec1_width = torch.tensor(0.4, requires_grad=True)
        rec1_height = torch.tensor(0.2, requires_grad=True)
        rec1_angle = torch.tensor(0.0, requires_grad=True)
        rec2_width = torch.tensor(0.2, requires_grad=True)
        rec2_height = torch.tensor(0.4, requires_grad=True)
        rec2_angle = torch.tensor(20.0, requires_grad=True)
        
        # Create two differentiable rectangle masks
        rectangle1 = self.generator.generate_rectangle_mask(
            center=(0.0, 0.0),
            width=rec1_width,
            height=rec1_height,
            angle=rec1_angle,
            soft_edge=self.soft_edge
        )
        rectangle2 = self.generator.generate_rectangle_mask(
            center=(0.0, 0.0),
            width=rec2_width,
            height=rec2_height,
            angle=rec2_angle,
            soft_edge=self.soft_edge
        )
        
        # Combine the masks using different operations
        union = self.generator.combine_masks(rectangle1, rectangle2, operation="union")
        intersection = self.generator.combine_masks(rectangle1, rectangle2, operation="intersection")
        
        # Check if gradients flow through
        loss = union.sum() + intersection.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        gradients = [rec1_width.grad, rec1_height.grad, rec1_angle.grad,
                     rec2_width.grad, rec2_height.grad, rec2_angle.grad]
        
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(torch.isnan(grad).any())

    def test_combine_masks_rectangle_polygon_differentiable(self):
        """Test that rectangle and polygon combinations preserve differentiability."""
        # Create differentiable parameters for rectangle
        rec_center_x = torch.tensor(0.0, requires_grad=True)
        rec_center_y = torch.tensor(0.0, requires_grad=True)
        rec_width = torch.tensor(0.2, requires_grad=True)
        rec_height = torch.tensor(0.4, requires_grad=True)
        rec_angle = torch.tensor(0.0, requires_grad=True)
        
        # Create differentiable parameters for polygon
        pts_x = torch.tensor([0.0, 0.2, -0.2], requires_grad=True)
        pts_y = torch.tensor([0.2, -0.2, -0.2], requires_grad=True)
        polygon_points = torch.stack([pts_x, pts_y], dim=1)
        polygon_center_x = torch.tensor(0.0, requires_grad=True)
        polygon_center_y = torch.tensor(0.0, requires_grad=True)
        polygon_angle = torch.tensor(0.0, requires_grad=True)
        
        # Create masks
        rectangle = self.generator.generate_rectangle_mask(
            center=(rec_center_x, rec_center_y),
            width=rec_width,
            height=rec_height,
            angle=rec_angle,
            soft_edge=self.soft_edge
        )
        polygon = self.generator.generate_polygon_mask(
            center=(polygon_center_x, polygon_center_y),
            polygon_points=polygon_points,
            angle=polygon_angle,
            invert=False,
            soft_edge=self.soft_edge
        )
        
        # Combine the masks using different operations
        union = self.generator.combine_masks(rectangle, polygon, operation="union")
        difference = self.generator.combine_masks(rectangle, polygon, operation="difference")
        
        # Check if gradients flow through
        loss = union.sum() + difference.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        gradients = [rec_center_x.grad, rec_center_y.grad, rec_width.grad, rec_height.grad, rec_angle.grad,
                     pts_x.grad, pts_y.grad, polygon_center_x.grad, polygon_center_y.grad, polygon_angle.grad]
        
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(torch.isnan(grad).any())

    def test_combine_masks_polygons_differentiable(self):
        """Test that polygon combinations preserve differentiability."""
        # Create differentiable parameters for two polygons
        pts1_x = torch.tensor([0.0, 0.2, -0.2], requires_grad=True)
        pts1_y = torch.tensor([0.2, -0.2, -0.2], requires_grad=True)
        pts2_x = torch.tensor([0.1, 0.3, -0.1], requires_grad=True)
        pts2_y = torch.tensor([0.0, 0.2, 0.2], requires_grad=True)
        
        polygon_points1 = torch.stack([pts1_x, pts1_y], dim=1)
        polygon_points2 = torch.stack([pts2_x, pts2_y], dim=1)
        
        polygon_center_x1 = torch.tensor(0.0, requires_grad=True)
        polygon_center_y1 = torch.tensor(0.0, requires_grad=True)
        polygon_angle1 = torch.tensor(0.0, requires_grad=True)
        polygon_center_x2 = torch.tensor(0.0, requires_grad=True)
        polygon_center_y2 = torch.tensor(0.0, requires_grad=True)
        polygon_angle2 = torch.tensor(0.0, requires_grad=True)
        
        # Create masks
        polygon1 = self.generator.generate_polygon_mask(
            center=(polygon_center_x1, polygon_center_y1),
            polygon_points=polygon_points1,
            angle=polygon_angle1,
            invert=False,
            soft_edge=self.soft_edge
        )
        polygon2 = self.generator.generate_polygon_mask(
            center=(polygon_center_x2, polygon_center_y2),
            polygon_points=polygon_points2,
            angle=polygon_angle2,
            invert=False,
            soft_edge=self.soft_edge
        )
        
        # Combine the masks using different operations
        union = self.generator.combine_masks(polygon1, polygon2, operation="union")
        intersection = self.generator.combine_masks(polygon1, polygon2, operation="intersection")
        difference = self.generator.combine_masks(polygon1, polygon2, operation="difference")
        
        # Check if gradients flow through
        loss = union.sum() + intersection.sum() + difference.sum()
        loss.backward()
        
        # Verify gradients exist and are not NaN
        gradients = [pts1_x.grad, pts1_y.grad, pts2_x.grad, pts2_y.grad,
                    polygon_center_x1.grad, polygon_center_y1.grad, polygon_angle1.grad,
                    polygon_center_x2.grad, polygon_center_y2.grad, polygon_angle2.grad]
        
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(torch.isnan(grad).any())


class TestShapeGeneratorFromSolver(unittest.TestCase):
    """Tests for creating ShapeGenerator from solver objects.
    
    These tests verify that ShapeGenerator objects can be correctly created
    from solver objects with different precision settings.
    """
    
    def test_from_solver_single_precision(self):
        """Test creating a ShapeGenerator from a single precision solver."""
        # Create a single precision RCWA solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            rdim=[256, 256],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]])
        )
        
        # Create a ShapeGenerator from the solver
        shape_gen = ShapeGenerator.from_solver(solver)
        
        # Verify that all properties were correctly transferred
        self.assertEqual(shape_gen.rdim, tuple(solver.rdim))
        self.assertTrue(torch.allclose(shape_gen.XO, solver.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, solver.YO))
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, solver.lattice_t2))
        
        # Check precision settings
        self.assertEqual(shape_gen.tcomplex, solver.tcomplex)
        self.assertEqual(shape_gen.tfloat, solver.tfloat)
        self.assertEqual(shape_gen.tint, solver.tint)
        self.assertEqual(shape_gen.nfloat, solver.nfloat)
        
        # Verify that precision is actually single
        self.assertEqual(solver.tfloat, torch.float32)
        self.assertEqual(shape_gen.tfloat, torch.float32)
        
    def test_from_solver_double_precision(self):
        """Test creating a ShapeGenerator from a double precision solver."""
        # Create a double precision RDIT solver
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            rdim=[256, 256],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]])
        )
        
        # Create a ShapeGenerator from the solver
        shape_gen = ShapeGenerator.from_solver(solver)
        
        # Verify that all properties were correctly transferred
        self.assertEqual(shape_gen.rdim, tuple(solver.rdim))
        self.assertTrue(torch.allclose(shape_gen.XO, solver.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, solver.YO))
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, solver.lattice_t2))
        
        # Check precision settings
        self.assertEqual(shape_gen.tcomplex, solver.tcomplex)
        self.assertEqual(shape_gen.tfloat, solver.tfloat)
        self.assertEqual(shape_gen.tint, solver.tint)
        self.assertEqual(shape_gen.nfloat, solver.nfloat)
        
        # Verify that precision is actually double
        self.assertEqual(solver.tfloat, torch.float64)
        self.assertEqual(shape_gen.tfloat, torch.float64)
        
    def test_from_solver_non_cartesian(self):
        """Test creating a ShapeGenerator from a solver with non-Cartesian lattice."""
        # Create a solver with non-Cartesian lattice vectors
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            rdim=[256, 256],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.5]]),  # Non-orthogonal lattice
            t2=torch.tensor([[0.0, 1.0]])
        )
        
        # Create a ShapeGenerator from the solver
        shape_gen = ShapeGenerator.from_solver(solver)
        
        # Verify that lattice vectors were correctly transferred
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, solver.lattice_t2))
        
        # Verify that coordinate grids were correctly transferred
        self.assertTrue(torch.allclose(shape_gen.XO, solver.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, solver.YO))
        
        # Verify that the lattice is indeed non-Cartesian
        self.assertNotEqual(solver.cell_type, "Cartesian")
        
    def test_get_shape_generator_params_single_precision(self):
        """Test creating a ShapeGenerator using parameters from a single precision solver."""
        # Create a single precision RCWA solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            rdim=[256, 256],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]])
        )
        
        # Get parameters dictionary from the solver
        params = solver.get_shape_generator_params()
        
        # Verify that all expected parameters are present
        expected_keys = ["XO", "YO", "rdim", "lattice_t1", "lattice_t2", 
                         "tcomplex", "tfloat", "tint", "nfloat"]
        for key in expected_keys:
            self.assertIn(key, params)
        
        # Create a ShapeGenerator using the parameters
        shape_gen = ShapeGenerator(**params)
        
        # Verify that all properties were correctly transferred
        self.assertEqual(shape_gen.rdim, tuple(solver.rdim))
        self.assertTrue(torch.allclose(shape_gen.XO, solver.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, solver.YO))
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, solver.lattice_t2))
        
        # Check precision settings
        self.assertEqual(shape_gen.tcomplex, solver.tcomplex)
        self.assertEqual(shape_gen.tfloat, solver.tfloat)
        self.assertEqual(shape_gen.tint, solver.tint)
        self.assertEqual(shape_gen.nfloat, solver.nfloat)
        
        # Verify that precision is actually single
        self.assertEqual(solver.tfloat, torch.float32)
        self.assertEqual(shape_gen.tfloat, torch.float32)
        self.assertEqual(params["tfloat"], torch.float32)
    
    def test_get_shape_generator_params_double_precision(self):
        """Test creating a ShapeGenerator using parameters from a double precision solver."""
        # Create a double precision RDIT solver
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            rdim=[256, 256],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]])
        )
        
        # Get parameters dictionary from the solver
        params = solver.get_shape_generator_params()
        
        # Verify that all expected parameters are present
        expected_keys = ["XO", "YO", "rdim", "lattice_t1", "lattice_t2", 
                         "tcomplex", "tfloat", "tint", "nfloat"]
        for key in expected_keys:
            self.assertIn(key, params)
        
        # Create a ShapeGenerator using the parameters
        shape_gen = ShapeGenerator(**params)
        
        # Verify that all properties were correctly transferred
        self.assertEqual(shape_gen.rdim, tuple(solver.rdim))
        self.assertTrue(torch.allclose(shape_gen.XO, solver.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, solver.YO))
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, solver.lattice_t2))
        
        # Check precision settings
        self.assertEqual(shape_gen.tcomplex, solver.tcomplex)
        self.assertEqual(shape_gen.tfloat, solver.tfloat)
        self.assertEqual(shape_gen.tint, solver.tint)
        self.assertEqual(shape_gen.nfloat, solver.nfloat)
        
        # Verify that precision is actually double
        self.assertEqual(solver.tfloat, torch.float64)
        self.assertEqual(shape_gen.tfloat, torch.float64)
        self.assertEqual(params["tfloat"], torch.float64)
        
    def test_get_shape_generator_params_non_cartesian(self):
        """Test creating a ShapeGenerator using parameters from a solver with non-Cartesian lattice."""
        # Create a solver with non-Cartesian lattice vectors
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE,
            rdim=[256, 256],
            kdim=[3, 3],
            lam0=np.array([1.55]),
            t1=torch.tensor([[1.0, 0.5]]),  # Non-orthogonal lattice
            t2=torch.tensor([[0.0, 1.0]])
        )
        
        # Get parameters dictionary from the solver
        params = solver.get_shape_generator_params()
        
        # Create a ShapeGenerator using the parameters
        shape_gen = ShapeGenerator(**params)
        
        # Verify that lattice vectors were correctly transferred
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, solver.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, solver.lattice_t2))
        
        # Verify that coordinate grids were correctly transferred
        self.assertTrue(torch.allclose(shape_gen.XO, solver.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, solver.YO))
        
        # Check that the parameters match between directly from solver and from params
        direct_gen = ShapeGenerator.from_solver(solver)
        self.assertTrue(torch.allclose(shape_gen.XO, direct_gen.XO))
        self.assertTrue(torch.allclose(shape_gen.YO, direct_gen.YO))
        self.assertTrue(torch.allclose(shape_gen.lattice_t1, direct_gen.lattice_t1))
        self.assertTrue(torch.allclose(shape_gen.lattice_t2, direct_gen.lattice_t2))
        self.assertEqual(shape_gen.tcomplex, direct_gen.tcomplex)
        self.assertEqual(shape_gen.tfloat, direct_gen.tfloat)


if __name__ == '__main__':
    unittest.main() 