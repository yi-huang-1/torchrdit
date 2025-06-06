import torch
from typing import Tuple


class ShapeGenerator:
    """Class to generate binary shape masks for photonic structures with lattice vector support.
    
    This class provides methods to create various shapes (circles, rectangles, polygons) 
    as binary masks for use in photonic simulations. It supports both Cartesian and 
    non-Cartesian coordinate systems through lattice vectors.
    
    Attributes:
        XO (torch.Tensor): Tensor containing the x-coordinates of each point in the grid.
        YO (torch.Tensor): Tensor containing the y-coordinates of each point in the grid.
        rdim (Tuple[int, int]): Dimensions of the real-space grid as (height, width).
        lattice_t1 (torch.Tensor): First lattice vector, defaults to [1,0] if not provided.
        lattice_t2 (torch.Tensor): Second lattice vector, defaults to [0,1] if not provided.
        tcomplex (torch.dtype): Complex tensor data type.
        tfloat (torch.dtype): Float tensor data type.
        tint (torch.dtype): Integer tensor data type.
        nfloat (torch.dtype): Number tensor data type for calculations.
        is_cartesian (bool): Flag indicating if the coordinate system is Cartesian.
    
    Examples:
    ```python
    import torch
    from torchrdit.shapes import ShapeGenerator
    # Create coordinate grids
    x = torch.linspace(-1, 1, 128)
    y = torch.linspace(-1, 1, 128)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # Initialize shape generator
    sg = ShapeGenerator(X, Y, (128, 128))
    # Generate a circle mask
    circle = sg.generate_circle_mask(center=(0.0, 0.0), radius=0.5)
    ```
    
    Keywords:
        shape, mask, photonics, circle, rectangle, polygon, lattice, binary mask
    """
    
    def __init__(self, XO: torch.Tensor, YO: torch.Tensor, rdim: Tuple[int, int], 
                 lattice_t1=None, lattice_t2=None, tcomplex=torch.complex128, tfloat=torch.float64, tint=torch.int64, nfloat=torch.float64):
        """Initialize a shape generator with coordinate grids and lattice vectors.
        
        Creates a new ShapeGenerator instance with the provided coordinate grids and 
        optional lattice vectors for non-Cartesian coordinate systems.
        
        Args:
            XO (torch.Tensor): Tensor containing the x-coordinates of each point in the grid.
            YO (torch.Tensor): Tensor containing the y-coordinates of each point in the grid.
            rdim (Tuple[int, int]): Dimensions of the real-space grid as (height, width).
            lattice_t1 (torch.Tensor, optional): First lattice vector. Defaults to [1,0].
            lattice_t2 (torch.Tensor, optional): Second lattice vector. Defaults to [0,1].
            tcomplex (torch.dtype, optional): Complex tensor data type. Defaults to torch.complex128.
            tfloat (torch.dtype, optional): Float tensor data type. Defaults to torch.float64.
            tint (torch.dtype, optional): Integer tensor data type. Defaults to torch.int64.
            nfloat (torch.dtype, optional): Number type for calculations. Defaults to torch.float64.
        
        Raises:
            AssertionError: If XO and YO are not torch.Tensor objects.
            
        Examples:
        ```python
        import torch
        from torchrdit.shapes import ShapeGenerator
        # Create coordinate grids
        x = torch.linspace(-1, 1, 128)
        y = torch.linspace(-1, 1, 128)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        # Initialize shape generator with default Cartesian coordinates
        sg = ShapeGenerator(X, Y, (128, 128))
        ```
            
        Keywords:
            initialization, lattice, coordinates, grid, tensor types
        """
        assert isinstance(XO, torch.Tensor) and isinstance(YO, torch.Tensor), "XO and YO must be torch.Tensor"

        self.tcomplex = tcomplex
        self.tfloat = tfloat
        self.tint = tint
        self.nfloat = nfloat
        self.rdim = rdim
        
        # Store the original real-space coordinates
        self.XO = XO
        self.YO = YO
        
        # Define lattice vectors (defaults to Cartesian)
        device = XO.device
        if lattice_t1 is None:
            self.lattice_t1 = torch.tensor([1.0, 0.0], device=device, dtype=self.tfloat)
        else:
            self.lattice_t1 = lattice_t1
            
        if lattice_t2 is None:
            self.lattice_t2 = torch.tensor([0.0, 1.0], device=device, dtype=self.tfloat)
        else:
            self.lattice_t2 = lattice_t2
        
        # Check if we're using non-Cartesian coordinates
        self.is_cartesian = torch.allclose(
            torch.tensor([[self.lattice_t1[0], self.lattice_t2[0]], 
                         [self.lattice_t1[1], self.lattice_t2[1]]]), 
            torch.eye(2, device=device, dtype=self.tfloat)
        )

    @classmethod
    def from_solver(cls, solver):
        """Create a ShapeGenerator from a solver.
        
        This factory method creates a ShapeGenerator instance using the coordinate grids
        and lattice vectors from an existing solver object, ensuring consistency between
        the solver and shape generator.
        
        Args:
            solver: A solver object containing coordinate grids and lattice vectors.
            
        Returns:
            ShapeGenerator: A new ShapeGenerator initialized with the solver's parameters.
            
        Examples:
        ```python
        from torchrdit.solver import create_solver
        from torchrdit.constants import Algorithm
        import torch
        from torchrdit.shapes import ShapeGenerator
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[1024, 1024],
            kdim=[7, 7]
        )
        shape_gen = ShapeGenerator.from_solver(solver)
        # Now shape_gen uses the same coordinate system as solver
        circle_mask = shape_gen.generate_circle_mask(radius=0.3)
        ```
            
        Keywords:
            factory, solver, initialization, coordinate system
        """
        return cls(XO=solver.XO, YO=solver.YO, rdim=tuple(solver.rdim),
                lattice_t1=solver.lattice_t1, lattice_t2=solver.lattice_t2,
                tcomplex=solver.tcomplex, tfloat=solver.tfloat, tint=solver.tint, nfloat=solver.nfloat)
    
    def generate_circle_mask(self, center=None, radius=0.1, soft_edge=0.001):
        """Generate a mask for a circle in Cartesian coordinates.
        
        Creates a binary mask representing a circle with the specified center and radius.
        The mask can have hard or soft edges based on the soft_edge parameter.
        
        Args:
            center (Tuple[float, float], optional): Center coordinates (x, y) of the circle.
                Defaults to (0.0, 0.0).
            radius (float, optional): Radius of the circle. Defaults to 0.1.
            soft_edge (float, optional): Width of the soft transition at the edge.
                Use 0 for a binary hard edge. Defaults to 0.001.
                
        Returns:
            torch.Tensor: A tensor of shape rdim containing the circle mask.
            Values are 1.0 inside the circle and 0.0 outside,
            with a smooth transition at the edge if soft_edge > 0.
            
        Examples:
        ```python
        import torch
        from torchrdit.shapes import ShapeGenerator
        # Create coordinate grids
        x = torch.linspace(-1, 1, 128)
        y = torch.linspace(-1, 1, 128)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        sg = ShapeGenerator(X, Y, (128, 128))
        # Generate a circle mask with hard edges
        hard_circle = sg.generate_circle_mask(center=(0.2, -0.3), radius=0.4, soft_edge=0)
        # Generate a circle mask with soft edges
        soft_circle = sg.generate_circle_mask(center=(0.2, -0.3), radius=0.4, soft_edge=0.02)
        ```
            
        Keywords:
            circle, mask, binary mask, shape generation, photonics
        """
        device = self.XO.device
        mask = torch.zeros(self.rdim, dtype=self.tfloat, device=device)
        
        if center is None:
            center = (0.0, 0.0)
        
        # Compute distance in real Cartesian coordinates
        distance = torch.sqrt(
            (self.XO - center[0])**2 + 
            (self.YO - center[1])**2
        )
        
        # Generate mask
        if soft_edge > 0:
            mask = 0.5 - 0.5 * torch.tanh((distance - radius) / soft_edge)
        else:
            mask = torch.where(distance <= radius, 
                              torch.ones_like(distance, dtype=self.tfloat), 
                              torch.zeros_like(distance, dtype=self.tfloat))
        
        return mask
    
    def generate_rectangle_mask(self, center=(0.0, 0.0), width=0.2, height=0.2, angle=0.0, soft_edge=0.001):
        """Generate a mask for a rectangle in Cartesian coordinates.
        
        Creates a binary mask representing a rectangle with the specified center,
        dimensions, and orientation. The mask can have hard or soft edges.
        
        Args:
            center (Tuple[float, float], optional): Center coordinates (x, y) of the rectangle.
                Defaults to (0.0, 0.0).
            width (float, optional): Width of the rectangle. Defaults to 0.2.
            height (float, optional): Height of the rectangle. Defaults to 0.2.
            angle (float, optional): Rotation angle in degrees. Defaults to 0.0.
            soft_edge (float, optional): Width of the soft transition at the edge.
                Use 0 for a binary hard edge. Defaults to 0.001.
                
        Returns:
            torch.Tensor: A tensor of shape rdim containing the rectangle mask.
            Values are 1.0 inside the rectangle and 0.0 outside,
            with a smooth transition at the edge if soft_edge > 0.
            
        Examples:
        ```python
        import torch
        from torchrdit.shapes import ShapeGenerator
        # Create coordinate grids
        x = torch.linspace(-1, 1, 128)
        y = torch.linspace(-1, 1, 128)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        sg = ShapeGenerator(X, Y, (128, 128))
        # Generate a rectangle mask
        rect = sg.generate_rectangle_mask(width=0.5, height=0.3, angle=45)
        # Generate a square mask
        square = sg.generate_rectangle_mask(width=0.4, height=0.4, angle=0)
        ```
            
        Keywords:
            rectangle, square, mask, binary mask, shape generation, photonics, rotation
        """
        device = self.XO.device
        mask = torch.zeros(self.rdim, dtype=self.tfloat, device=device)
        
        # Convert angle to radians
        if not isinstance(angle, torch.Tensor):
            angle_rad = torch.tensor(angle, device=device, dtype=self.tfloat) * torch.pi / 180.0
        else:
            angle_rad = angle * torch.pi / 180.0
        
        # Get coordinates relative to center
        x_centered = self.XO - center[0]
        y_centered = self.YO - center[1]
        
        # Rotate coordinates in the opposite direction
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)
        x_rotated = x_centered * cos_theta + y_centered * sin_theta
        y_rotated = -x_centered * sin_theta + y_centered * cos_theta
        
        # Calculate half dimensions
        half_width = width / 2.0
        half_height = height / 2.0
        
        # Calculate distances to edges
        left_dist = x_rotated + half_width
        right_dist = half_width - x_rotated
        bottom_dist = y_rotated + half_height
        top_dist = half_height - y_rotated
        
        # Create mask
        if soft_edge > 0:
            # Smooth edges
            left_mask = torch.sigmoid(left_dist / soft_edge)
            right_mask = torch.sigmoid(right_dist / soft_edge)
            bottom_mask = torch.sigmoid(bottom_dist / soft_edge)
            top_mask = torch.sigmoid(top_dist / soft_edge)
            mask = left_mask * right_mask * bottom_mask * top_mask
        else:
            # Hard edges
            x_mask = torch.where((-half_width <= x_rotated) & (x_rotated <= half_width),
                                torch.ones_like(x_rotated, dtype=self.tfloat),
                                torch.zeros_like(x_rotated, dtype=self.tfloat))
            
            y_mask = torch.where((-half_height <= y_rotated) & (y_rotated <= half_height),
                                torch.ones_like(y_rotated, dtype=self.tfloat),
                                torch.zeros_like(y_rotated, dtype=self.tfloat))
            
            mask = x_mask * y_mask
        
        return mask
    
    def generate_polygon_mask(self, polygon_points, center=None, angle=None, invert=False, soft_edge=0.001):
        """Generate a mask for a polygon in Cartesian coordinates.
        
        Creates a binary mask representing an arbitrary polygon defined by its vertices.
        The polygon can be rotated, translated, and have soft or hard edges.
        
        Args:
            polygon_points (List[Tuple] or torch.Tensor): List of (x, y) coordinates 
                defining the polygon vertices, or a tensor of shape (n, 2) where n is 
                the number of vertices.
            center (Tuple[float, float], optional): Center coordinates (x, y) for the polygon.
                If provided, the polygon will be translated to this position. Defaults to None.
            angle (float, optional): Rotation angle in degrees. If provided, the polygon
                will be rotated around its center. Defaults to None.
            invert (bool, optional): If True, inverts the mask (0s inside, 1s outside).
                Defaults to False.
            soft_edge (float, optional): Width of the soft transition at the edge.
                Use 0 for a binary hard edge. Defaults to 0.001.
                
        Returns:
            torch.Tensor: A tensor of shape rdim containing the polygon mask.
            Values are 1.0 inside the polygon and 0.0 outside (unless inverted),
            with a smooth transition at the edge if soft_edge > 0.
            
        Examples:
        ```python
        import torch
        from torchrdit.shapes import ShapeGenerator
        # Create coordinate grids
        x = torch.linspace(-1, 1, 128)
        y = torch.linspace(-1, 1, 128)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        sg = ShapeGenerator(X, Y, (128, 128))
        # Generate a triangle mask
        triangle_points = [(-0.2, -0.2), (0.2, -0.2), (0.0, 0.2)]
        triangle = sg.generate_polygon_mask(triangle_points)
        # Generate a hexagon mask
        import numpy as np
        n = 6  # hexagon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 0.3
        hexagon_points = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
        hexagon = sg.generate_polygon_mask(hexagon_points, center=(0.1, 0.1), angle=30)
        ```
            
        Keywords:
            polygon, mask, binary mask, shape generation, photonics, arbitrary shape
        """
        device = self.XO.device
        
        # Handle inputs
        if isinstance(polygon_points, torch.Tensor):
            polygon = polygon_points.clone()
        else:
            polygon = torch.tensor(polygon_points, dtype=self.tfloat, device=device)
        
        # Apply rotation if specified
        if angle is not None:
            if not isinstance(angle, torch.Tensor):
                angle_rad = torch.tensor(angle, device=device, dtype=self.tfloat) * torch.pi / 180.0
            else:
                angle_rad = angle * torch.pi / 180.0
            
            cos_theta = torch.cos(angle_rad)
            sin_theta = torch.sin(angle_rad)
            
            x_orig, y_orig = polygon[:, 0], polygon[:, 1]
            x_rotated = x_orig * cos_theta - y_orig * sin_theta
            y_rotated = x_orig * sin_theta + y_orig * cos_theta
            
            polygon = torch.stack([x_rotated, y_rotated], dim=1)
        
        # Apply translation if center is provided
        if center is not None:
            if not isinstance(center[0], torch.Tensor):
                center_x = torch.tensor(center[0], device=device, dtype=self.tfloat)
            else:
                center_x = center[0]
                
            if not isinstance(center[1], torch.Tensor):
                center_y = torch.tensor(center[1], device=device, dtype=self.tfloat)
            else:
                center_y = center[1]
            
            polygon[:, 0] = polygon[:, 0] + center_x
            polygon[:, 1] = polygon[:, 1] + center_y
        
        # Create inside/outside mask using winding number algorithm
        winding_number = self._calculate_winding_number(self.XO, self.YO, polygon)
        inside_mask = torch.where(
            torch.abs(winding_number) > 0.5,
            torch.ones_like(self.XO, dtype=self.tfloat),
            torch.zeros_like(self.XO, dtype=self.tfloat)
        )
        
        # Calculate edge distances for smoothing if needed
        if soft_edge > 0:
            min_dist = torch.ones_like(self.XO) * 1e6
            
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                
                edge = p2 - p1
                edge_length_squared = torch.sum(edge**2) + 1e-10
                
                vec_x = self.XO - p1[0]
                vec_y = self.YO - p1[1]
                
                t = torch.clamp(
                    (vec_x * edge[0] + vec_y * edge[1]) / edge_length_squared,
                    0.0, 1.0
                )
                
                closest_x = p1[0] + t * edge[0]
                closest_y = p1[1] + t * edge[1]
                
                dist = torch.sqrt(
                    (self.XO - closest_x)**2 + 
                    (self.YO - closest_y)**2 + 
                    1e-10
                )
                
                min_dist = torch.minimum(min_dist, dist)
            
            # Create a smooth edge effect
            edge_effect = torch.sigmoid(-(min_dist / soft_edge) + 5.0)
            
            # Combine the inside mask with edge effect
            # Smooth transition only at the boundary
            boundary = self._detect_boundary(inside_mask)
            final_mask = torch.where(
                boundary > 0.5,
                edge_effect,
                inside_mask
            )
        else:
            final_mask = inside_mask
        
        if invert:
            final_mask = 1.0 - final_mask
            
        return final_mask
    
    def _calculate_winding_number(self, X, Y, polygon):
        """Calculate the winding number for each point relative to the polygon.
        
        Implements the winding number algorithm to determine whether points are inside
        or outside a polygon. For each point, it calculates the total angle subtended by
        the polygon, which will be ±2π for points inside and 0 for points outside.
        
        Args:
            X (torch.Tensor): Tensor of x-coordinates.
            Y (torch.Tensor): Tensor of y-coordinates.
            polygon (torch.Tensor): Tensor of shape (n, 2) with polygon vertices.
            
        Returns:
            torch.Tensor: Tensor of same shape as X and Y with winding numbers.
            
        Note:
            This is a private helper method used by generate_polygon_mask.
            
        Keywords:
            winding number, polygon, point-in-polygon, computational geometry
        """
        winding = torch.zeros_like(X, dtype=self.tfloat)
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            # Vectors from each grid point to the vertices
            vec_x1 = X - p1[0]
            vec_y1 = Y - p1[1]
            vec_x2 = X - p2[0]
            vec_y2 = Y - p2[1]
            
            # Calculate angles
            angle1 = torch.atan2(vec_y1, vec_x1)
            angle2 = torch.atan2(vec_y2, vec_x2)
            
            # Angle difference, handling the wrap-around at ±π
            delta = angle2 - angle1
            delta = torch.where(delta > torch.pi, delta - 2*torch.pi, delta)
            delta = torch.where(delta < -torch.pi, delta + 2*torch.pi, delta)
            
            # Accumulate winding number
            winding = winding + delta
        
        # Normalize by 2π to get the number of winds around each point
        return winding / (2 * torch.pi)
    
    def _detect_boundary(self, mask):
        """Detect the boundary of a mask.
        
        Identifies the boundary pixels of a binary mask using a combination of
        dilation and erosion operations.
        
        Args:
            mask (torch.Tensor): Binary mask tensor.
            
        Returns:
            torch.Tensor: Tensor of same shape as mask with 1.0 at boundary pixels
            and 0.0 elsewhere.
            
        Note:
            This is a private helper method used by generate_polygon_mask for
            creating smooth boundaries.
            
        Keywords:
            boundary detection, edge detection, morphological operations, dilation, erosion
        """
        # Use a simple dilation/erosion approach
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Apply pooling operations for dilation and erosion
        dilated = torch.nn.functional.max_pool2d(
            mask_expanded, kernel_size=3, stride=1, padding=1
        ).to(self.tfloat)
        
        eroded = -torch.nn.functional.max_pool2d(
            -mask_expanded, kernel_size=3, stride=1, padding=1
        ).to(self.tfloat)
        
        # The boundary is where dilated and eroded differ
        boundary = torch.abs(dilated - eroded).squeeze(0).squeeze(0).to(self.tfloat)
        
        return boundary
    
    def combine_masks(self, mask1, mask2, operation="union"):
        """Combine two masks using a specified operation.
        
        Performs boolean operations on two binary masks to create complex shapes.
        Supported operations include union, intersection, difference, and subtraction.
        
        Args:
            mask1 (torch.Tensor): First binary mask tensor.
            mask2 (torch.Tensor): Second binary mask tensor.
            operation (str, optional): The operation to perform. Options are:
                - "union": Logical OR (max) of the masks
                - "intersection": Logical AND (min) of the masks
                - "difference": Absolute difference between masks
                - "subtract": Remove mask2 from mask1
                Defaults to "union".
                
        Returns:
            torch.Tensor: The combined mask resulting from the specified operation.
            
        Raises:
            ValueError: If an invalid operation is specified.
            
        Examples:
        ```python
        import torch
        from torchrdit.shapes import ShapeGenerator
        # Create coordinate grids
        x = torch.linspace(-1, 1, 128)
        y = torch.linspace(-1, 1, 128)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        sg = ShapeGenerator(X, Y, (128, 128))
        # Create two circular masks
        circle1 = sg.generate_circle_mask(center=(-0.1, 0), radius=0.3)
        circle2 = sg.generate_circle_mask(center=(0.1, 0), radius=0.3)
        # Combine masks using different operations
        union = sg.combine_masks(circle1, circle2, operation="union")
        intersection = sg.combine_masks(circle1, circle2, operation="intersection")
        difference = sg.combine_masks(circle1, circle2, operation="difference")
        circle1_minus_circle2 = sg.combine_masks(circle1, circle2, operation="subtract")
        ```
            
        Keywords:
            mask combination, boolean operations, union, intersection, difference, compound shape
        """
        if operation == "union":
            return torch.maximum(mask1, mask2)
        elif operation == "intersection":
            return torch.minimum(mask1, mask2)
        elif operation == "difference":
            return torch.abs(mask1 - mask2)
        elif operation == "subtract":
            return mask1 * (1 - mask2)
        else:
            raise ValueError("Invalid operation. Choose from 'union', 'intersection', 'difference', or 'subtract'.")
