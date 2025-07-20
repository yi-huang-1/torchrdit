import numpy as np
import torch
import os
import json

from torchrdit.solver import create_solver_from_config
from torchrdit.builder import flip_config
from torchrdit.constants import Algorithm, Precision

class TestTorchrditConfig:
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

    def test_create_solver_from_dict(self):
        """Test solver creation using a dictionary-based configuration."""
        config_dict = {
            "algorithm": "RCWA",
            "precision": "DOUBLE",
            "wavelengths": [1540e-3, 1550e-3, 1560e-3, 1570e-3],
            "length_unit": "um",
            "rdim": [512, 512],
            "kdim": [9, 9],
            "use_FFF": True,
            "lattice_vectors": {
                "t1": [0.5, -0.866],
                "t2": [0.5, 0.866]
            },
            "materials": {
                "SiO": {
                    "permittivity": 2.1
                },
                "SiN": {
                    "permittivity": 3.8
                },
                "FusedSilica": {
                    "permittivity": 2.28
                }
            },
            "layers": [
                {
                    "material": "SiO",
                    "thickness": 230e-3,
                    "is_homogeneous": False,
                    "is_optimize": True
                },
                {
                    "material": "SiN",
                    "thickness": 345e-3,
                    "is_homogeneous": True,
                    "is_optimize": False
                }
            ],
            "trn_material": "FusedSilica",
            "ref_material": "SiN"
        }

        solver = create_solver_from_config(config_dict)
        # Verify solver was created with correct configuration
        # Algorithm is an instance, not a type
        assert solver.algorithm.__class__.__name__ == 'RCWAAlgorithm'
        assert len(solver.lam0) == 4  # 4 wavelengths
        assert solver.rdim == [512, 512]
        assert solver.kdim == [9, 9]
        # Verify materials were added
        assert 'SiO' in solver._matlib
        assert 'SiN' in solver._matlib
        assert 'FusedSilica' in solver._matlib

    def test_create_solver_from_json(self, tmp_path):
        """Test solver creation using a JSON file-based configuration."""
        config_json = {
            "algorithm": "RCWA",
            "precision": "DOUBLE",
            "wavelengths": [1540e-3, 1550e-3, 1560e-3, 1570e-3],
            "length_unit": "um",
            "rdim": [512, 512],
            "kdim": [9, 9],
            "use_FFF": True,
            "lattice_vectors": {
                "t1": [0.5, -0.866],
                "t2": [0.5, 0.866]
            },
            "materials": {
                "SiO": {
                    "permittivity": 2.1
                },
                "SiN": {
                    "permittivity": 3.8
                },
                "FusedSilica": {
                    "permittivity": 2.28
                }
            },
            "layers": [
                {
                    "material": "SiO",
                    "thickness": 230e-3,
                    "is_homogeneous": False,
                    "is_optimize": True
                },
                {
                    "material": "SiN",
                    "thickness": 345e-3,
                    "is_homogeneous": True,
                    "is_optimize": False
                }
            ],
            "trn_material": "FusedSilica",
            "ref_material": "SiN"
        }

        # Save JSON to a temporary path
        config_path = tmp_path / "config_test.json"
        with open(config_path, "w") as json_file:
            json.dump(config_json, json_file, indent=4)

        solver = create_solver_from_config(str(config_path))
        # Verify solver was created with correct configuration from JSON
        # Algorithm is an instance, not a type
        assert solver.algorithm.__class__.__name__ == 'RCWAAlgorithm'
        assert len(solver.lam0) == 4  # 4 wavelengths
        assert solver.rdim == [512, 512]
        assert solver.kdim == [9, 9]
        # Verify materials were added
        assert 'SiO' in solver._matlib
        assert 'SiN' in solver._matlib
        assert 'FusedSilica' in solver._matlib

    def test_flip_config_both_materials(self):
        config = {
            "layers": [{"material": "SiO"}, {"material": "SiN"}],
            "trn_material": "FusedSilica",
            "ref_material": "SiN"
        }
        flipped = flip_config(config)
        assert flipped["layers"] == [{"material": "SiN"}, {"material": "SiO"}]
        assert flipped["trn_material"] == "SiN"
        assert flipped["ref_material"] == "FusedSilica"

    def test_flip_config_trn_only(self):
        config = {
            "layers": [{"material": "SiO"}, {"material": "SiN"}],
            "trn_material": "FusedSilica"
        }
        flipped = flip_config(config)
        assert flipped["layers"] == [{"material": "SiN"}, {"material": "SiO"}]
        assert flipped["ref_material"] == "FusedSilica"
        assert "trn_material" not in flipped

    def test_flip_config_ref_only(self):
        config = {
            "layers": [{"material": "SiO"}, {"material": "SiN"}],
            "ref_material": "SiN"
        }
        flipped = flip_config(config)
        assert flipped["layers"] == [{"material": "SiN"}, {"material": "SiO"}]
        assert flipped["trn_material"] == "SiN"
        assert "ref_material" not in flipped

    def test_flip_config_no_materials(self):
        config = {
            "layers": [{"material": "SiO"}, {"material": "SiN"}]
        }
        flipped = flip_config(config)
        assert flipped["layers"] == [{"material": "SiN"}, {"material": "SiO"}]
        assert "trn_material" not in flipped
        assert "ref_material" not in flipped