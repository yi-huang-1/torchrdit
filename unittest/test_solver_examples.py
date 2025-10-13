import unittest
import numpy as np
import torch

from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material


class TestSolverDocExamples(unittest.TestCase):
    """Lean doc-example tests that actually exercise the solver path.

    Ensures high-level examples in solver.py remain executable without duplicating
    broader API, builder, or observer tests already covered elsewhere.
    """

    def test_basic_doc_example_end_to_end(self):
        """Create solver, add materials/layer, add source, and solve end-to-end."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[32, 32],
            kdim=[3, 3],
            device='cpu',
        )

        # Materials and simple homogeneous layer
        air = create_material(name="Air", permittivity=1.0)
        si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([air, si])
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.2), is_homogeneous=True)

        # Source and solve (run the example fully)
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        result = solver.solve(source)

        # Minimal, meaningful assertions
        self.assertEqual(result.reflection.shape, (1,))
        self.assertEqual(result.transmission.shape, (1,))
        self.assertGreaterEqual(result.reflection[0].item(), 0.0)
        self.assertGreaterEqual(result.transmission[0].item(), 0.0)
        # Allow small slack for truncation/tolerance in tiny test config
        self.assertLessEqual(result.reflection[0].item() + result.transmission[0].item(), 1.01)


if __name__ == '__main__':
    unittest.main()

