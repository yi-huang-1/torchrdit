"""Golden-value regression tests for the always-batched refactor (Task 052).

These tests pin the exact numeric output of the solver BEFORE the refactoring
so that every subsequent phase can verify it doesn't change the results.

Original reference values captured at commit 1b812a2 (Task 051 numerics module,
double precision, CPU).

Update history:
- Task 053 (item 7): θ=60° mixed-polarization golden updated after k-vector
  relaxation epsilon changed from fixed 1e-6 to eps_for_dtype(complex128)=1e-12.
  New values match Fresnel analytic solution to <1e-12 (vs. ~1e-6 previously)
  and satisfy energy conservation R+T=1 to machine precision.  Validated against
  closed-form Fresnel for Air(n=1)→Glass(n=1.5) at θ=60°, pte=ptm=0.5.
"""

import numpy as np
import pytest
import torch

from torchrdit.constants import Algorithm, Precision
from torchrdit.solver import create_solver
from torchrdit.utils import create_material


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interface_solver(algo, *, n2=2.25, precision=Precision.DOUBLE):
    """Air | Glass(n2) homogeneous slab, harmonics=[1,1]."""
    solver = create_solver(
        algorithm=algo, lam0=np.array([1.55]),
        grids=[64, 64], harmonics=[1, 1],
        device="cpu", precision=precision,
    )
    air = create_material(name="Air", permittivity=1.0)
    glass = create_material(name="Glass", permittivity=n2)
    solver.add_materials([air, glass])
    solver.update_ref_material("Air")
    solver.update_trn_material("Glass")
    solver.add_layer(material_name="Glass", thickness=0.5, is_homogeneous=True)
    return solver


def _make_grating_solver(algo):
    """Air | Si grating | Air, harmonics=[3,3]."""
    solver = create_solver(
        algorithm=algo, lam0=np.array([1.0]),
        grids=[128, 128], harmonics=[3, 3],
        device="cpu", precision=Precision.DOUBLE,
    )
    air = create_material(name="Air", permittivity=1.0)
    si = create_material(name="Si", permittivity=11.7)
    solver.add_materials([air, si])
    solver.update_ref_material("Air")
    solver.update_trn_material("Air")
    solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
    return solver


# Tolerance: tight enough to catch refactoring bugs, loose enough for
# platform/PyTorch-version variation.
ATOL = 1e-8
RTOL = 1e-6


# ---------------------------------------------------------------------------
# Golden values (pinned)
# ---------------------------------------------------------------------------

class TestNormalIncidenceGolden:
    """Normal incidence (θ=0) through Air|Glass slab, both algorithms."""

    GOLDEN_R = 3.999999760837617e-02
    GOLDEN_T = 9.599999997916238e-01

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_reflection(self, algo):
        solver = _make_interface_solver(algo)
        result = solver.solve(solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0))
        R = torch.sum(result.reflection).item()
        assert abs(R - self.GOLDEN_R) < ATOL, f"{algo.name}: R={R}"

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_transmission(self, algo):
        solver = _make_interface_solver(algo)
        result = solver.solve(solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0))
        T = torch.sum(result.transmission).item()
        assert abs(T - self.GOLDEN_T) < ATOL, f"{algo.name}: T={T}"


class TestObliqueIncidenceGolden:
    """Oblique incidence (θ=30°) through Air|Glass slab, TE and TM."""

    GOLDEN = {
        "TE": {"R": 5.779610209457898e-02, "T": 9.422038942383214e-01},
        "TM": {"R": 2.524914510304306e-02, "T": 9.747508531816750e-01},
    }

    @pytest.mark.parametrize("pol,pte,ptm", [("TE", 1.0, 0.0), ("TM", 0.0, 1.0)])
    def test_oblique_30deg(self, pol, pte, ptm):
        solver = _make_interface_solver(Algorithm.RCWA)
        theta = np.deg2rad(30.0)
        result = solver.solve(solver.add_source(theta=theta, phi=0, pte=pte, ptm=ptm))
        R = torch.sum(result.reflection).item()
        T = torch.sum(result.transmission).item()
        assert abs(R - self.GOLDEN[pol]["R"]) < ATOL, f"{pol}: R={R}"
        assert abs(T - self.GOLDEN[pol]["T"]) < ATOL, f"{pol}: T={T}"


class TestBatchedSourcesGolden:
    """Batched solve must match sequential AND pinned golden values."""

    GOLDEN = [
        {"R": 3.999999760837617e-02, "T": 9.599999997916238e-01},  # θ=0
        {"R": 5.779610209457898e-02, "T": 9.422038942383214e-01},  # θ=30°
        {"R": 8.918671280144222e-02, "T": 9.108132871969183e-01},  # θ=60° mixed
    ]

    SOURCES = [
        {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        {"theta": float(np.deg2rad(30.0)), "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        {"theta": float(np.deg2rad(60.0)), "phi": 0.0, "pte": 0.5, "ptm": 0.5},
    ]

    def test_batched_golden_values(self):
        solver = _make_interface_solver(Algorithm.RDIT)
        batched = solver.solve(self.SOURCES)
        for i, gold in enumerate(self.GOLDEN):
            R = torch.sum(batched[i].reflection).item()
            T = torch.sum(batched[i].transmission).item()
            assert abs(R - gold["R"]) < ATOL, f"src{i}: R={R}, expected={gold['R']}"
            assert abs(T - gold["T"]) < ATOL, f"src{i}: T={T}, expected={gold['T']}"

    def test_sequential_equals_batched(self):
        """Sequential and batched must produce identical results."""
        solver = _make_interface_solver(Algorithm.RDIT)
        batched = solver.solve(self.SOURCES)

        for i, src in enumerate(self.SOURCES):
            solver_seq = _make_interface_solver(Algorithm.RDIT)
            seq_result = solver_seq.solve(src)
            bR = torch.sum(batched[i].reflection).item()
            sR = torch.sum(seq_result.reflection).item()
            bT = torch.sum(batched[i].transmission).item()
            sT = torch.sum(seq_result.transmission).item()
            assert abs(bR - sR) < 1e-12, f"src{i}: batched R={bR} != seq R={sR}"
            assert abs(bT - sT) < 1e-12, f"src{i}: batched T={bT} != seq T={sT}"


class TestGratingGolden:
    """Non-homogeneous (grating) layer, both algorithms.

    Golden values recaptured at commit 716bdbe (P5) after fixing redhstar()
    to derive identity_mat dtype from the actual tensor (complex128) instead
    of defaulting to the tcomplex argument (complex64). The shift is ~2.8e-8,
    consistent with correcting a mixed-precision identity matrix.
    """

    GOLDEN_R = 7.021454152565152e-01
    GOLDEN_T = 2.978545845411916e-01

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_grating_golden(self, algo):
        solver = _make_grating_solver(algo)
        result = solver.solve(solver.add_source(theta=np.deg2rad(10), phi=0, pte=1.0, ptm=0.0))
        R = torch.sum(result.reflection).item()
        T = torch.sum(result.transmission).item()
        assert abs(R - self.GOLDEN_R) < ATOL, f"{algo.name}: R={R}"
        assert abs(T - self.GOLDEN_T) < ATOL, f"{algo.name}: T={T}"

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_grating_energy_conservation(self, algo):
        solver = _make_grating_solver(algo)
        result = solver.solve(solver.add_source(theta=np.deg2rad(10), phi=0, pte=1.0, ptm=0.0))
        R = torch.sum(result.reflection).item()
        T = torch.sum(result.transmission).item()
        assert np.isclose(R + T, 1.0, atol=1e-6), f"{algo.name}: R+T={R+T}"


class TestRawDataRoundTrip:
    """Verify raw_data schema and to_dict/from_dict round-trip survive vectorization."""

    from torchrdit.results import SolverResults

    def test_single_source_raw_data_keys(self):
        """Single-source raw_data must contain the full legacy schema."""
        solver = _make_interface_solver(Algorithm.RDIT)
        result = solver.solve(solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0))
        raw = result.raw_data
        for key in ("ref_s_x", "ref_s_y", "ref_s_z", "trn_s_x", "trn_s_y", "trn_s_z",
                     "rx", "ry", "rz", "tx", "ty", "tz",
                     "REF", "TRN", "RDE", "TDE", "smat_structure",
                     "kzref", "kztrn", "kinc", "kx", "ky"):
            assert key in raw, f"Missing key '{key}' in raw_data"

    def test_single_source_from_dict_roundtrip(self):
        """from_dict(result.raw_data) must reconstruct fields correctly."""
        solver = _make_interface_solver(Algorithm.RDIT)
        result = solver.solve(solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0))
        rebuilt = self.SolverResults.from_dict(result.raw_data)
        assert torch.allclose(rebuilt.reflection, result.reflection, atol=ATOL)
        assert torch.allclose(rebuilt.transmission, result.transmission, atol=ATOL)
        assert torch.allclose(rebuilt.reflection_field.x, result.reflection_field.x, atol=ATOL)
        assert torch.allclose(rebuilt.transmission_field.x, result.transmission_field.x, atol=ATOL)

    def test_batched_per_source_raw_data_keys(self):
        """Each per-source raw_data in a batched solve must have full schema."""
        solver = _make_interface_solver(Algorithm.RDIT)
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": float(np.deg2rad(30.0)), "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ]
        result = solver.solve(sources)
        for i in range(2):
            raw_i = result[i].raw_data
            for key in ("ref_s_x", "trn_s_x", "rx", "tx", "REF", "TRN", "smat_structure"):
                assert key in raw_i, f"Source {i}: missing key '{key}' in raw_data"

    def test_smatrix_dict_compat(self):
        """SMatrix must support dict-style access for backward compat."""
        solver = _make_interface_solver(Algorithm.RDIT)
        result = solver.solve(solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0))
        smat = result.raw_data["smat_structure"]
        # Dict-style access must work
        assert torch.equal(smat["S11"], smat.S11)
        assert "S11" in smat
        assert set(smat.keys()) == {"S11", "S12", "S21", "S22"}
