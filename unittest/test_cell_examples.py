import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm
from torchrdit.cell import Cell3D, CellType


class TestCellDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in cell.py.
    
    This test suite ensures that the examples provided in the docstrings of
    cell.py work as expected. It tests examples for:
    1. CellType class
    2. ShapeGenerator class and its methods
    3. Cell3D class and its methods
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create basic coordinate grid for ShapeGenerator tests
        self.rdim = (256, 256)
        x_grid = torch.linspace(-0.5, 0.5, self.rdim[0])
        y_grid = torch.linspace(-0.5, 0.5, self.rdim[1])
        self.X, self.Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
        
        # Create materials for Cell3D tests
        self.air = create_material(name="air", permittivity=1.0)
        self.silicon = create_material(name="Si", permittivity=11.7)
        self.sio2 = create_material(name="sio2", permittivity=2.25)
    
    def test_celltype_example(self):
        """Test the CellType class examples."""
        # Example from CellType docstring
        cell = Cell3D()
        self.assertEqual(cell.cell_type, CellType.Cartesian)
        
        # Cell type is automatically determined from lattice vectors
        cell = Cell3D(t1=torch.tensor([[1.0, 0.5]]), t2=torch.tensor([[0.0, 1.0]]))
        self.assertEqual(cell.cell_type, CellType.Other)
    
    def test_cell3d_basic_example(self):
        """Test the basic Cell3D usage example."""
        # Example from Cell3D docstring
        import torch
        
        # Create a basic cell with default properties
        cell = Cell3D(rdim=[256, 256], kdim=[5, 5])
        
        # Add materials
        si = create_material(name="Si", permittivity=11.7)
        cell.add_materials([si])
        
        # Add a layer
        cell.add_layer(material_name="Si", thickness=torch.tensor(0.2))
        
        # Verify the cell was created correctly
        self.assertEqual(cell.rdim, [256, 256])
        self.assertEqual(cell.kdim, [5, 5])
        self.assertEqual(len(cell.layers), 1)
        self.assertEqual(cell.layers[0].thickness, 0.2)
        self.assertEqual(cell.layers[0].material_name, "Si")
    
    def test_cell3d_initialization(self):
        """Test the Cell3D initialization examples."""
        # Examples from Cell3D.__init__ docstring
        # Use the already imported module
        
        # Create an RDIT solver
        rdit_solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[1024, 1024],
            kdim=[7, 7]
        )
        
        # Verify solver was created correctly
        self.assertEqual(rdit_solver.rdim, [1024, 1024])
        self.assertEqual(rdit_solver.kdim, [7, 7])
        self.assertEqual(rdit_solver.algorithm.name, "R-DIT")
        
        # Create an RCWA solver with non-rectangular lattice
        rcwa_solver = create_solver(
            algorithm=Algorithm.RCWA,
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.5, 0.866]]),  # 30-degree lattice
            rdim=[512, 512],
            kdim=[5, 5]
        )
        
        # Verify solver was created correctly
        self.assertEqual(rcwa_solver.rdim, [512, 512])
        self.assertEqual(rcwa_solver.kdim, [5, 5])
        self.assertEqual(rcwa_solver.algorithm.name, "RCWA")
        self.assertEqual(rcwa_solver.cell_type, CellType.Other)  # Non-rectangular lattice
    
    def test_add_materials_example(self):
        """Test the add_materials method examples."""
        # Example from add_materials docstring
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        sio2 = create_material(name='sio2', permittivity=2.25)
        
        # Add multiple materials at once
        cell.add_materials([silicon, sio2])
        
        # Add a single material
        gold = create_material(name='gold', permittivity=complex(-10.0, 1.5))
        cell.add_materials([gold])
        
        # Verify materials were added
        self.assertIn('silicon', cell._matlib)
        self.assertIn('sio2', cell._matlib)
        self.assertIn('gold', cell._matlib)
        
        # Check material properties
        self.assertEqual(cell._matlib['silicon'].er, 11.7)
        self.assertEqual(cell._matlib['sio2'].er, 2.25)
        self.assertEqual(cell._matlib['gold'].er, np.conj(complex(-10.0, 1.5))) # automatically check and convert to negative convention
    
    def test_add_layer_examples(self):
        """Test the add_layer method examples."""
        # Example from add_layer docstring
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        sio2 = create_material(name='sio2', permittivity=2.25)
        cell.add_materials([silicon, sio2])
        
        # Add a homogeneous silicon layer with thickness 0.2 μm
        cell.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
        
        # Add a layer using a material object directly
        air = create_material(name='air2', permittivity=1.0)
        cell.add_layer(material_name=air, thickness=torch.tensor(0.1))
        
        # Add a patterned (non-homogeneous) layer
        cell.add_layer(
            material_name='silicon',
            thickness=torch.tensor(0.3),
            is_homogeneous=False
        )
        
        # Add a layer that will be optimized
        cell.add_layer(
            material_name='sio2',
            thickness=torch.tensor(0.15),
            is_optimize=True
        )
        
        # Verify layers were added correctly
        self.assertEqual(len(cell.layers), 4)
        
        # Check layer properties
        self.assertEqual(cell.layers[0].material_name, 'silicon')
        self.assertEqual(cell.layers[0].thickness, 0.2)
        self.assertTrue(cell.layers[0].is_homogeneous)
        
        self.assertEqual(cell.layers[1].material_name, 'air2')
        self.assertEqual(cell.layers[1].thickness, 0.1)
        
        self.assertEqual(cell.layers[2].material_name, 'silicon')
        self.assertEqual(cell.layers[2].thickness, 0.3)
        self.assertFalse(cell.layers[2].is_homogeneous)
        
        self.assertEqual(cell.layers[3].material_name, 'sio2')
        self.assertEqual(cell.layers[3].thickness, 0.15)
        self.assertTrue(cell.layers[3].is_optimize)
    
    def test_update_trn_material_example(self):
        """Test the update_trn_material method examples."""
        # Example from update_trn_material docstring
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        cell.add_materials([silicon])
        
        # Set transmission material by name
        cell.update_trn_material(trn_material='silicon')
        self.assertEqual(cell.layer_manager.trn_material_name, 'silicon')
        
        # Set transmission material by providing a material object
        water = create_material(name='water', permittivity=1.77)
        cell.update_trn_material(trn_material=water)
        self.assertEqual(cell.layer_manager.trn_material_name, 'water')
        self.assertIn('water', cell._matlib)
    
    def test_update_ref_material_example(self):
        """Test the update_ref_material method examples."""
        # Example from update_ref_material docstring
        cell = Cell3D()
        silicon = create_material(name='silicon', permittivity=11.7)
        cell.add_materials([silicon])
        
        # Set reflection material by name
        cell.update_ref_material(ref_material='silicon')
        self.assertEqual(cell.layer_manager.ref_material_name, 'silicon')
        
        # Set reflection material by providing a material object
        metal = create_material(name='silver', permittivity=complex(-15.0, 1.0))
        cell.update_ref_material(ref_material=metal)
        self.assertEqual(cell.layer_manager.ref_material_name, 'silver')
        self.assertIn('silver', cell._matlib)
    
    def test_get_layout_example(self):
        """Test the get_layout method example."""
        # Example from get_layout docstring
        cell = Cell3D()
        X, Y = cell.get_layout()
        
        # Verify return values
        self.assertEqual(X.shape, tuple(cell.rdim))
        self.assertEqual(Y.shape, tuple(cell.rdim))
        
        # Simple verification that these are coordinate grids
        self.assertAlmostEqual(X[0, 0].item(), -0.5, delta=0.01)
        self.assertAlmostEqual(X[-1, -1].item(), 0.5, delta=0.01)
        self.assertAlmostEqual(Y[0, 0].item(), -0.5, delta=0.01)
        self.assertAlmostEqual(Y[-1, -1].item(), 0.5, delta=0.01)
    
    def test_get_cell_type_example(self):
        """Test the get_cell_type method examples."""
        # Example from get_cell_type docstring
        cell1 = Cell3D(t1=torch.tensor([[1.0, 0.0]]), t2=torch.tensor([[0.0, 1.0]]))
        self.assertEqual(cell1.get_cell_type(), CellType.Cartesian)
        
        cell2 = Cell3D(t1=torch.tensor([[1.0, 0.2]]), t2=torch.tensor([[0.0, 1.0]]))
        self.assertEqual(cell2.get_cell_type(), CellType.Other)


if __name__ == '__main__':
    unittest.main() 