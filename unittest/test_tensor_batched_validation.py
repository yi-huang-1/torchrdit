"""Test to validate tensor-batched implementation against sequential processing."""

import pytest
import torch
import numpy as np

from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material


class TestBatchedValidation:
    """Tests to validate that batched processing matches sequential results."""
    
    def setup_solver(self, n_freqs=3, debug=False):
        """Create a test solver with basic configuration."""
        lam0 = np.linspace(1.5, 1.6, n_freqs)
        
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=lam0,
            rdim=[256, 256],  # Smaller for faster tests
            kdim=[3, 3],
            device="cpu",
            precision=Precision.SINGLE
        )
        
        # Set debug flag directly
        solver.debug_batching = debug
        
        # Add materials
        si = create_material(name="Si", permittivity=11.7)
        sio2 = create_material(name="SiO2", permittivity=2.25)
        solver.add_materials([si, sio2])
        
        # Add a simple layer structure
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.2))
        solver.add_layer(material_name="SiO2", thickness=torch.tensor(0.3))
        
        return solver
    
    def test_batched_matches_sequential_simple(self):
        """Test that batched results match sequential for simple case."""
        solver = self.setup_solver(n_freqs=2)
        
        # Define test sources
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},
        ]
        
        # Sequential processing
        sequential_results = []
        for src in sources:
            result = solver.solve(src)
            sequential_results.append(result)
        
        # Batched processing
        batched_results = solver.solve(sources)
        
        # Compare results
        rtol = 1e-5
        atol = 1e-6
        
        # Check reflection and transmission
        for i, src in enumerate(sources):
            # Reflection
            print(f"\nSource {i}:")
            print(f"  Batched reflection: {batched_results.reflection[i]}")
            print(f"  Sequential reflection: {sequential_results[i].reflection}")
            
            # NaN values are bugs - tests should fail if they appear
            assert not torch.isnan(batched_results.reflection[i]).any(), \
                f"NaN values found in batched reflection for source {i} - this is a bug!"
            assert not torch.isnan(sequential_results[i].reflection).any(), \
                f"NaN values found in sequential reflection for source {i} - this is a bug!"
            
            torch.testing.assert_close(
                batched_results.reflection[i],
                sequential_results[i].reflection,
                rtol=rtol,
                atol=atol,
                msg=f"Reflection mismatch for source {i}"
            )
            
            # Transmission
            assert not torch.isnan(batched_results.transmission[i]).any(), \
                f"NaN values found in batched transmission for source {i} - this is a bug!"
            assert not torch.isnan(sequential_results[i].transmission).any(), \
                f"NaN values found in sequential transmission for source {i} - this is a bug!"
            
            torch.testing.assert_close(
                batched_results.transmission[i],
                sequential_results[i].transmission,
                rtol=rtol,
                atol=atol,
                msg=f"Transmission mismatch for source {i}"
            )
            
            # Check energy conservation
            R_batch = batched_results.reflection[i].sum().item()
            T_batch = batched_results.transmission[i].sum().item()
            R_seq = sequential_results[i].reflection.sum().item()
            T_seq = sequential_results[i].transmission.sum().item()
            
            assert abs((R_batch + T_batch) - (R_seq + T_seq)) < 1e-4, \
                f"Energy conservation mismatch for source {i}"
    
    def test_batched_matches_sequential_complex(self):
        """Test with more complex layer structure and multiple wavelengths."""
        solver = self.setup_solver(n_freqs=5)
        
        # Add more layers
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.1))
        solver.add_layer(material_name="SiO2", thickness=torch.tensor(0.15))
        
        # Define test sources with various angles
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},     # Normal incidence
            {"theta": np.pi/6, "phi": 0.0, "pte": 0.7, "ptm": 0.3}, # 30 degrees
            {"theta": np.pi/4, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5}, # 45 degrees
            {"theta": np.pi/3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0}, # 60 degrees
        ]
        
        # Sequential processing
        sequential_results = []
        for src in sources:
            result = solver.solve(src)
            sequential_results.append(result)
        
        # Batched processing
        batched_results = solver.solve(sources)
        
        # Compare field components
        rtol = 1e-5
        atol = 1e-6
        
        for i, src in enumerate(sources):
            # NaN values are bugs - fields should never contain NaN
            assert not torch.isnan(batched_results.Erx[i]).any(), \
                f"NaN values found in batched Erx for source {i} - this is a bug!"
            assert not torch.isnan(sequential_results[i].reflection_field.x).any(), \
                f"NaN values found in sequential Erx for source {i} - this is a bug!"
            
            assert not torch.isnan(batched_results.Ery[i]).any(), \
                f"NaN values found in batched Ery for source {i} - this is a bug!"
            assert not torch.isnan(sequential_results[i].reflection_field.y).any(), \
                f"NaN values found in sequential Ery for source {i} - this is a bug!"
            
            assert not torch.isnan(batched_results.Etx[i]).any(), \
                f"NaN values found in batched Etx for source {i} - this is a bug!"
            assert not torch.isnan(sequential_results[i].transmission_field.x).any(), \
                f"NaN values found in sequential Etx for source {i} - this is a bug!"
            
            # Compare reflected fields
            torch.testing.assert_close(
                batched_results.Erx[i],
                sequential_results[i].reflection_field.x,
                rtol=rtol,
                atol=atol,
                msg=f"Erx mismatch for source {i}"
            )
            torch.testing.assert_close(
                batched_results.Ery[i],
                sequential_results[i].reflection_field.y,
                rtol=rtol,
                atol=atol,
                msg=f"Ery mismatch for source {i}"
            )
            
            # Compare transmitted fields
            torch.testing.assert_close(
                batched_results.Etx[i],
                sequential_results[i].transmission_field.x,
                rtol=rtol,
                atol=atol,
                msg=f"Etx mismatch for source {i}"
            )
    
    def test_single_source_batched(self):
        """Test that single source works with batched implementation."""
        solver = self.setup_solver(n_freqs=3)
        
        # Single source as list
        sources = [{"theta": 0.15, "phi": 0.5, "pte": 0.8, "ptm": 0.2}]
        
        # Process as batch
        batched_result = solver.solve(sources)
        
        # Process as single
        single_result = solver.solve(sources[0])
        
        # Results should match exactly
        torch.testing.assert_close(
            batched_result.reflection[0],
            single_result.reflection,
            rtol=1e-6,
            atol=1e-8
        )
        
        torch.testing.assert_close(
            batched_result.transmission[0],
            single_result.transmission,
            rtol=1e-6,
            atol=1e-8
        )
    
    def test_debug_output(self, capsys):
        """Test that debug output works correctly."""
        solver = self.setup_solver(n_freqs=2, debug=True)
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # This should produce debug output
        _result = solver.solve(sources)
        
        # Check that debug output was produced
        captured = capsys.readouterr()
        assert "[DEBUG]" in captured.out, "Debug output not found"
    
    def test_gradient_flow_batched(self):
        """Test that gradients flow correctly through batched implementation."""
        solver = self.setup_solver(n_freqs=2)
        
        # Create a patterned layer with gradient tracking
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.3), is_homogeneous=False)
        
        # Create a simple pattern manually
        pattern = torch.zeros(solver.rdim[0], solver.rdim[1], dtype=torch.float32)
        # Create a square in the center
        center_x, center_y = solver.rdim[0] // 2, solver.rdim[1] // 2
        size = min(solver.rdim[0], solver.rdim[1]) // 4
        pattern[center_x-size:center_x+size, center_y-size:center_y+size] = 1.0
        pattern = pattern.requires_grad_(True)
        
        solver.update_er_with_mask(mask=pattern, layer_index=2)
        
        # Multiple sources
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": 0.0, "pte": 0.5, "ptm": 0.5},
        ]
        
        # Solve and compute loss
        result = solver.solve(sources)
        
        # Maximize average transmission
        loss = -result.transmission.mean()
        loss.backward()
        
        # Check that pattern has gradients
        assert pattern.grad is not None, "No gradients for pattern"
        assert not torch.allclose(pattern.grad, torch.zeros_like(pattern.grad)), \
            "Pattern gradients are all zero"
    
    def test_performance_improvement(self):
        """Test that batched processing is faster than sequential."""
        import time
        
        solver = self.setup_solver(n_freqs=3)
        
        # Create multiple sources
        n_sources = 10
        sources = []
        for i in range(n_sources):
            theta = i * 0.05
            sources.append({
                "theta": theta,
                "phi": 0.0,
                "pte": 1.0,
                "ptm": 0.0
            })
        
        # Time sequential processing
        start_seq = time.time()
        sequential_results = []
        for src in sources:
            result = solver.solve(src)
            sequential_results.append(result)
        time_seq = time.time() - start_seq
        
        # Time batched processing
        start_batch = time.time()
        _batched_results = solver.solve(sources)
        time_batch = time.time() - start_batch
        
        print(f"\nSequential time: {time_seq:.3f}s")
        print(f"Batched time: {time_batch:.3f}s")
        print(f"Speedup: {time_seq/time_batch:.2f}x")
        
        # For phase 1 (sequential in loop), we don't expect huge speedup
        # But it should at least not be slower
        assert time_batch <= time_seq * 1.1, "Batched processing is slower than sequential"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])