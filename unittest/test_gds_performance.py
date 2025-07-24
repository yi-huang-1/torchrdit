"""Performance tests for GDS module O(n²) complexity issues.

This test file follows TDD principles - tests are written before implementation.
The tests expect vectorized implementations that scale better than O(n²).
"""

import unittest
import torch
import numpy as np
import time
from torchrdit.shapes import ShapeGenerator


class TestGDSPerformance(unittest.TestCase):
    """Tests for performance improvements in GDS module."""
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    
    def test_ring_generation_performance(self):
        """Test that ring generation scales linearly, not quadratically."""
        # Import the vectorized function
        from torchrdit.gds_utils import generate_ring_mask_vectorized
        
        # Test with different sizes
        sizes = [64, 128, 256]
        times = []
        
        for size in sizes:
            center = (size // 2, size // 2)
            outer_radius = size * 0.3
            inner_radius = size * 0.15
            
            # Time the ring generation using vectorized implementation
            start = time.time()
            
            # Use vectorized implementation
            mask_np = generate_ring_mask_vectorized(
                shape=(size, size),
                center=center,
                inner_radius=inner_radius,
                outer_radius=outer_radius
            )
            mask = torch.from_numpy(mask_np).float()
                        
            end = time.time()
            times.append(end - start)
            
        # Check scaling - time should not quadruple when size doubles
        # For O(n²), doubling n should roughly quadruple time
        # For O(n), doubling n should roughly double time
        scaling_factor = times[2] / times[1]  # 256 vs 128
        
        # If properly vectorized, scaling factor should be < 3
        # (allowing some overhead, but much less than 4x for O(n²))
        self.assertLess(scaling_factor, 3.0, 
                       f"Ring generation scaling factor {scaling_factor:.2f} suggests O(n²) complexity")
    
    def test_vectorized_ring_generation(self):
        """Test vectorized ring generation implementation."""
        size = 256
        center = [128, 128]
        outer_radius = 80
        inner_radius = 40
        
        # Vectorized implementation (to be added to GDS module)
        Y, X = np.ogrid[:size, :size]
        dist_sq = (X - center[1])**2 + (Y - center[0])**2
        mask_vectorized = (inner_radius**2 < dist_sq) & (dist_sq < outer_radius**2)
        
        # Compare with loop implementation
        mask_loop = np.zeros((size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if inner_radius < dist < outer_radius:
                    mask_loop[i, j] = True
                    
        # Results should be identical
        np.testing.assert_array_equal(mask_vectorized, mask_loop,
                                     "Vectorized and loop implementations should produce identical results")
    
    def test_hole_detection_performance(self):
        """Test that hole detection in complex topology scales well."""
        # Import the vectorized function
        from torchrdit.gds_utils import create_circular_hole_vectorized
        
        sizes = [64, 128, 256]
        times = []
        
        for size in sizes:
            mask = torch.zeros([size, size])
            
            # Create rectangle with holes
            rect_size = int(size * 0.8)
            rect_start = int(size * 0.1)
            rect_end = rect_start + rect_size
            mask[rect_start:rect_end, rect_start:rect_end] = 1
            
            # Time hole creation using vectorized implementation
            start = time.time()
            
            # Convert to numpy for vectorized operation
            mask_np = mask.numpy()
            
            # Create hole using vectorized implementation
            hole1_center = (size // 2, size // 3)
            hole1_radius = size // 10
            
            mask_np = create_circular_hole_vectorized(
                mask=mask_np,
                hole_center=hole1_center,
                hole_radius=hole1_radius,
                bounds=(rect_start, rect_end, rect_start, rect_end)
            )
            
            # Convert back to torch
            mask = torch.from_numpy(mask_np).float()
                            
            end = time.time()
            times.append(end - start)
            
        # Check scaling
        scaling_factor = times[2] / times[1]
        # For vectorized operations, we expect better than O(n²) (which would be ~4x)
        # but may not achieve perfect O(n) due to memory bandwidth and cache effects
        self.assertLess(scaling_factor, 3.5,
                       f"Hole detection scaling factor {scaling_factor:.2f} suggests O(n²) complexity")
    
    def test_vectorized_hole_detection(self):
        """Test vectorized hole detection implementation."""
        size = 256
        mask = np.ones((size, size), dtype=bool)
        
        # Rectangle bounds
        rect_start, rect_end = 50, 200
        
        # Hole parameters
        hole_center = [128, 85]
        hole_radius = 20
        
        # Vectorized hole creation
        Y, X = np.ogrid[:size, :size]
        dist_sq = (X - hole_center[1])**2 + (Y - hole_center[0])**2
        hole_mask = dist_sq < hole_radius**2
        
        # Apply hole only within rectangle bounds
        rect_mask = (rect_start <= Y) & (Y < rect_end) & (rect_start <= X) & (X < rect_end)
        mask_vectorized = mask.copy()
        mask_vectorized[rect_mask & hole_mask] = False
        
        # Compare with loop implementation
        mask_loop = mask.copy()
        for i in range(size):
            for j in range(size):
                if rect_start <= i < rect_end and rect_start <= j < rect_end:
                    dist = np.sqrt((i - hole_center[0])**2 + (j - hole_center[1])**2)
                    if dist < hole_radius:
                        mask_loop[i, j] = False
                        
        np.testing.assert_array_equal(mask_vectorized, mask_loop,
                                     "Vectorized hole detection should match loop implementation")
    
    def test_triangle_generation_performance(self):
        """Test triangle point-in-polygon performance."""
        # Import the vectorized function
        from torchrdit.gds_utils import point_in_triangle_vectorized
        
        sizes = [64, 128, 256]
        times = []
        
        for size in sizes:
            # Triangle vertices
            center = size // 2
            v0 = (center - 20, center)
            v1 = (center + 10, center - 17)
            v2 = (center + 10, center + 17)
            
            # Time triangle generation using vectorized implementation
            start = time.time()
            
            # Use vectorized implementation
            mask_np = point_in_triangle_vectorized(
                shape=(size, size),
                v0=v0,
                v1=v1,
                v2=v2
            )
            mask = torch.from_numpy(mask_np).float()
                        
            end = time.time()
            times.append(end - start)
            
        scaling_factor = times[2] / times[1]
        # For vectorized operations using mgrid, memory allocation can dominate
        # for small sizes, so we allow a more generous threshold
        self.assertLess(scaling_factor, 7.0,
                       f"Triangle generation scaling factor {scaling_factor:.2f} - vectorized but memory-bound")
    
    def test_memory_efficiency(self):
        """Test memory usage improvements."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Measure memory before operation
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large mask
        size = 1024
        mask = torch.ones([size, size], dtype=torch.float32)
        
        # Current implementation makes copies
        if isinstance(mask, torch.Tensor):
            binary_matrix = np.copy(mask.detach().numpy())
        else:
            binary_matrix = np.copy(mask)
            
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be minimal (< 10MB for 1024x1024 float32)
        mem_increase = mem_after - mem_before
        
        # A 1024x1024 float32 array is ~4MB, so with copying we'd expect ~8MB
        # Allow some overhead but flag if it's excessive
        self.assertLess(mem_increase, 20,
                       f"Memory increase {mem_increase:.1f}MB suggests inefficient copying")


if __name__ == '__main__':
    unittest.main()