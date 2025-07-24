"""Utility functions for efficient GDS operations.

This module provides vectorized implementations of common mask generation
operations to replace O(n²) loop-based implementations.
"""

import numpy as np
import torch
from typing import Tuple, Union, List


def generate_ring_mask_vectorized(
    shape: Tuple[int, int], 
    center: Tuple[float, float], 
    inner_radius: float, 
    outer_radius: float
) -> np.ndarray:
    """Generate ring mask using vectorized operations.
    
    Args:
        shape: Shape of the mask (height, width)
        center: Center coordinates (row, col)
        inner_radius: Inner radius of the ring
        outer_radius: Outer radius of the ring
        
    Returns:
        Binary mask with ring shape
    """
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_sq = (X - center[1])**2 + (Y - center[0])**2
    return (inner_radius**2 < dist_sq) & (dist_sq < outer_radius**2)


def create_circular_hole_vectorized(
    mask: np.ndarray,
    hole_center: Tuple[float, float],
    hole_radius: float,
    bounds: Tuple[int, int, int, int] = None
) -> np.ndarray:
    """Create a circular hole in mask using vectorized operations.
    
    Args:
        mask: Input mask to modify
        hole_center: Center coordinates (row, col) of the hole
        hole_radius: Radius of the hole
        bounds: Optional (row_start, row_end, col_start, col_end) to limit hole region
        
    Returns:
        Modified mask with hole
    """
    Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
    dist_sq = (X - hole_center[1])**2 + (Y - hole_center[0])**2
    hole_mask = dist_sq < hole_radius**2
    
    if bounds is not None:
        row_start, row_end, col_start, col_end = bounds
        bounds_mask = (row_start <= Y) & (Y < row_end) & (col_start <= X) & (X < col_end)
        hole_mask = hole_mask & bounds_mask
    
    result = mask.copy()
    result[hole_mask] = 0
    return result


def point_in_triangle_vectorized(
    shape: Tuple[int, int],
    v0: Tuple[float, float],
    v1: Tuple[float, float], 
    v2: Tuple[float, float]
) -> np.ndarray:
    """Generate triangle mask using vectorized operations.
    
    Uses barycentric coordinates for efficient point-in-triangle test.
    
    Args:
        shape: Shape of the mask (height, width)
        v0, v1, v2: Triangle vertices as (row, col) tuples
        
    Returns:
        Binary mask with triangle shape
    """
    Y, X = np.mgrid[:shape[0], :shape[1]]
    
    # Compute vectors
    v0v1 = (v1[0] - v0[0], v1[1] - v0[1])
    v0v2 = (v2[0] - v0[0], v2[1] - v0[1])
    
    # Compute dot products
    dot00 = v0v2[0] * v0v2[0] + v0v2[1] * v0v2[1]
    dot01 = v0v2[0] * v0v1[0] + v0v2[1] * v0v1[1]
    dot11 = v0v1[0] * v0v1[0] + v0v1[1] * v0v1[1]
    
    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    
    # For each point
    v0p_x = Y - v0[0]
    v0p_y = X - v0[1]
    
    dot02 = v0v2[0] * v0p_x + v0v2[1] * v0p_y
    dot12 = v0v1[0] * v0p_x + v0v1[1] * v0p_y
    
    # Barycentric coordinates
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is in triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def create_multiple_holes_vectorized(
    mask: np.ndarray,
    holes: List[dict]
) -> np.ndarray:
    """Create multiple holes in mask efficiently.
    
    Args:
        mask: Input mask
        holes: List of hole specifications, each with:
            - 'center': (row, col) tuple
            - 'radius': float
            - 'bounds': optional (row_start, row_end, col_start, col_end)
            
    Returns:
        Modified mask with all holes
    """
    result = mask.copy()
    
    # Create meshgrid once
    Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
    
    for hole in holes:
        center = hole['center']
        radius = hole['radius']
        bounds = hole.get('bounds', None)
        
        # Calculate distance for this hole
        dist_sq = (X - center[1])**2 + (Y - center[0])**2
        hole_mask = dist_sq < radius**2
        
        if bounds is not None:
            row_start, row_end, col_start, col_end = bounds
            bounds_mask = (row_start <= Y) & (Y < row_end) & (col_start <= X) & (X < col_end)
            hole_mask = hole_mask & bounds_mask
        
        result[hole_mask] = 0
        
    return result


def convert_mask_to_torch(mask: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert mask to torch tensor if needed."""
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask).float()
    return mask.float()


def convert_mask_to_numpy(mask: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert mask to numpy array without unnecessary copying."""
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return np.asarray(mask)  # Returns view if possible