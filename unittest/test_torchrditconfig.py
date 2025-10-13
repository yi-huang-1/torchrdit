import json
import pytest

from torchrdit.solver import create_solver_from_config
from torchrdit.builder import flip_config

class TestTorchrditConfig:

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
        assert getattr(solver.algorithm, 'name', None) == 'RCWA'
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
        assert getattr(solver.algorithm, 'name', None) == 'RCWA'
        assert len(solver.lam0) == 4  # 4 wavelengths
        assert solver.rdim == [512, 512]
        assert solver.kdim == [9, 9]
        # Verify materials were added
        assert 'SiO' in solver._matlib
        assert 'SiN' in solver._matlib
        assert 'FusedSilica' in solver._matlib

    @pytest.mark.parametrize(
        "config, expected_layers, expected_trn, expected_ref",
        [
            (
                {"layers": [{"material": "SiO"}, {"material": "SiN"}], "trn_material": "FusedSilica", "ref_material": "SiN"},
                [{"material": "SiN"}, {"material": "SiO"}],
                "SiN",
                "FusedSilica",
            ),
            (
                {"layers": [{"material": "SiO"}, {"material": "SiN"}], "trn_material": "FusedSilica"},
                [{"material": "SiN"}, {"material": "SiO"}],
                None,
                "FusedSilica",
            ),
            (
                {"layers": [{"material": "SiO"}, {"material": "SiN"}], "ref_material": "SiN"},
                [{"material": "SiN"}, {"material": "SiO"}],
                "SiN",
                None,
            ),
            (
                {"layers": [{"material": "SiO"}, {"material": "SiN"}]},
                [{"material": "SiN"}, {"material": "SiO"}],
                None,
                None,
            ),
        ],
    )
    def test_flip_config_parametrized(self, config, expected_layers, expected_trn, expected_ref):
        flipped = flip_config(config)
        assert flipped["layers"] == expected_layers
        if expected_trn is None:
            assert "trn_material" not in flipped
        else:
            assert flipped["trn_material"] == expected_trn
        if expected_ref is None:
            assert "ref_material" not in flipped
        else:
            assert flipped["ref_material"] == expected_ref
