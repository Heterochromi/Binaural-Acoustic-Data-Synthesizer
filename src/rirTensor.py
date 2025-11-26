import torch
from typing import Tuple, Optional, Literal, Union
import sofar as sf
from torch_geometric.transforms import Delaunay


class RIRTensor:
    """
    GPU-optimized HRIR tensor wrapper with vectorized batch interpolation.

    Provides easy access to HRIRs by azimuth/elevation angles while maintaining
    PyTorch tensor performance with full GPU acceleration.
    """

    def __init__(self, data: torch.Tensor, source_positions: torch.Tensor = None, device: str = 'cpu'):
        """
        Initialize HRIR tensor wrapper.

        Args:
            data: Tensor of shape (num_positions, 2, tap_length) where:
                - num_positions: number of spatial measurement points
                - 2: left and right channels
                - tap_length: impulse response length
            source_positions: Tensor of shape (num_positions, 3) with [azimuth, elevation, distance]
                             If None, will try to read from SOFA metadata
            device: Device to store tensors on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.data = data.to(self.device)  # Full tensor (positions, channels, taps)
        self.num_positions = data.shape[0]
        self.num_channels = data.shape[1]
        self.tap_length = data.shape[2]
        self.dtype = self.data.dtype

        # Store spatial positions
        if source_positions is not None:
            self.source_positions = source_positions.to(self.device, dtype=self.dtype)
        else:
            self.source_positions = None

        # Build spatial index for fast lookup
        self._build_spatial_index()

        # Build Delaunay triangulation for 3D interpolation
        self._build_delaunay_triangulation()

    def _build_spatial_index(self):
        """Build a spatial index for fast azimuth/elevation lookup."""
        if self.source_positions is None:
            self.spatial_index = None
            self.azimuths = None
            self.elevations = None
            self.available_angles = None
            return

        # Store azimuth and elevation separately for easier queries
        self.azimuths = self.source_positions[:, 0]  # Shape: (num_positions,)
        self.elevations = self.source_positions[:, 1]  # Shape: (num_positions,)

        # Create available angles tensor for vectorized operations
        self.available_angles = torch.stack([self.azimuths, self.elevations], dim=1)  # (num_positions, 2)

    def _build_delaunay_triangulation(self):
        """Build Delaunay triangulation for 3D spherical interpolation."""
        if self.source_positions is None or self.available_angles is None:
            self.triangles = None
            self.delaunay_points = None
            return

        # Convert spherical coordinates to 2D for triangulation
        # Handle angle wrapping by using coordinates in [-180, 180] range
        azimuths_wrapped = torch.where(
            self.azimuths > 180,
            self.azimuths - 360,
            self.azimuths
        )

        # Create 2D points for Delaunay: [azimuth, elevation]
        points_2d = torch.stack([azimuths_wrapped, self.elevations], dim=1)  # (num_positions, 2)
        self.delaunay_points = points_2d

        # Use torch_geometric's Delaunay transform
        # Create a dummy data object with pos attribute
        class DummyData:
            def __init__(self, pos):
                self.pos = pos

        try:
            data = DummyData(pos=points_2d)
            delaunay_transform = Delaunay()
            data_with_faces = delaunay_transform(data)

            # Store the triangulation (face indices)
            if hasattr(data_with_faces, 'face'):
                self.triangles = data_with_faces.face.t()  # Shape: (num_triangles, 3)
            else:
                # Fallback if no triangulation possible
                self.triangles = None
        except Exception as e:
            print(f"Warning: Could not build Delaunay triangulation: {e}")
            self.triangles = None

    @classmethod
    def from_sofa(cls, sofa_path: str, device: Literal['cpu', 'cuda'] = 'cpu') -> 'RIRTensor':
        """Load HRIR from SOFA file."""
        sofa_obj = sf.read_sofa(sofa_path)

        # Extract IR data
        if hasattr(sofa_obj, 'Data_IR'):
            ir_data = sofa_obj.Data_IR
        elif hasattr(sofa_obj, 'data'):
            ir_data = sofa_obj.data
        else:
            raise ValueError("Cannot find IR data in SOFA object")

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(ir_data)

        # Extract source positions if available
        source_positions = None
        if hasattr(sofa_obj, 'SourcePosition'):
            source_positions = torch.from_numpy(sofa_obj.SourcePosition).float()

        return cls(tensor, source_positions, device=device)

    def _spherical_to_cartesian_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Convert azimuth and elevation to cartesian coordinates (vectorized).

        Args:
            azimuth: Tensor of shape (batch_size,) in degrees
            elevation: Tensor of shape (batch_size,) in degrees

        Returns:
            Tensor of shape (batch_size, 3) with [x, y, z]
        """
        azimuth_rad = torch.deg2rad(azimuth)
        elevation_rad = torch.deg2rad(elevation)

        x = torch.cos(azimuth_rad) * torch.cos(elevation_rad)
        y = torch.sin(azimuth_rad) * torch.cos(elevation_rad)
        z = torch.sin(elevation_rad)

        return torch.stack([x, y, z], dim=-1)

    def _cartesian_to_spherical_batch(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Convert cartesian coordinates to azimuth and elevation (vectorized).

        Args:
            xyz: Tensor of shape (batch_size, 3) with [x, y, z]

        Returns:
            Tensor of shape (batch_size, 2) with [azimuth, elevation] in degrees
        """
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        azimuth = torch.rad2deg(torch.atan2(y, x))
        elevation = torch.rad2deg(torch.atan2(z, torch.sqrt(x**2 + y**2)))

        azimuth = (azimuth + 360) % 360

        return torch.stack([azimuth, elevation], dim=-1)

    def _get_angle_distance_batch(self, azimuth1: torch.Tensor, elevation1: torch.Tensor,
                                   azimuth2: torch.Tensor, elevation2: torch.Tensor) -> torch.Tensor:
        """
        Calculate angular distance between two directions using cartesian conversion (vectorized).

        Args:
            azimuth1, elevation1: Query angles, shape (batch_size,)
            azimuth2, elevation2: Reference angles, shape (batch_size,) or (num_positions,)

        Returns:
            Distances, shape depends on broadcasting
        """
        point1 = self._spherical_to_cartesian_batch(azimuth1, elevation1)
        point2 = self._spherical_to_cartesian_batch(azimuth2, elevation2)
        return torch.norm(point1 - point2, dim=-1)

    def _find_nearest_direction_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Find the index of the nearest spatial direction for each query (vectorized).

        Args:
            azimuth: Tensor of shape (batch_size,)
            elevation: Tensor of shape (batch_size,)

        Returns:
            Indices of shape (batch_size,)
        """
        if self.source_positions is None:
            raise ValueError("No spatial position data available")

        # Normalize angles
        azimuth = azimuth % 360
        elevation = torch.clamp(elevation, -90, 90)

        # Expand dimensions for broadcasting: (batch_size, 1) vs (num_positions,)
        azimuth_expanded = azimuth.unsqueeze(1)  # (batch_size, 1)
        elevation_expanded = elevation.unsqueeze(1)  # (batch_size, 1)

        available_az = self.azimuths.unsqueeze(0)  # (1, num_positions)
        available_el = self.elevations.unsqueeze(0)  # (1, num_positions)

        # Calculate distances: (batch_size, num_positions)
        distances = self._get_angle_distance_batch(
            azimuth_expanded, elevation_expanded,
            available_az, available_el
        )

        # Return index of nearest position for each query
        return torch.argmin(distances, dim=1)  # (batch_size,)

    def _angle_exists_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor,
                           tolerance: float = 0.001) -> torch.Tensor:
        """
        Check if exact angles exist in dataset (vectorized).

        Args:
            azimuth: Tensor of shape (batch_size,)
            elevation: Tensor of shape (batch_size,)
            tolerance: Angular tolerance in degrees

        Returns:
            Boolean tensor of shape (batch_size,)
        """
        azimuth = azimuth % 360

        # Expand for broadcasting
        azimuth_expanded = azimuth.unsqueeze(1)  # (batch_size, 1)
        elevation_expanded = elevation.unsqueeze(1)  # (batch_size, 1)

        available_az = self.azimuths.unsqueeze(0)  # (1, num_positions)
        available_el = self.elevations.unsqueeze(0)  # (1, num_positions)

        # Check if any available angle matches within tolerance
        az_match = torch.abs(available_az - azimuth_expanded) < tolerance
        el_match = torch.abs(available_el - elevation_expanded) < tolerance

        matches = az_match & el_match
        return torch.any(matches, dim=1)  # (batch_size,)

    def _get_angle_index_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor,
                               tolerance: float = 0.001) -> torch.Tensor:
        """
        Get the index for specific angles (vectorized).

        Args:
            azimuth: Tensor of shape (batch_size,)
            elevation: Tensor of shape (batch_size,)
            tolerance: Angular tolerance in degrees

        Returns:
            Indices tensor of shape (batch_size,)
        """
        azimuth = azimuth % 360

        # Expand for broadcasting
        azimuth_expanded = azimuth.unsqueeze(1)  # (batch_size, 1)
        elevation_expanded = elevation.unsqueeze(1)  # (batch_size, 1)

        available_az = self.azimuths.unsqueeze(0)  # (1, num_positions)
        available_el = self.elevations.unsqueeze(0)  # (1, num_positions)

        # Find matches
        az_match = torch.abs(available_az - azimuth_expanded) < tolerance
        el_match = torch.abs(available_el - elevation_expanded) < tolerance
        matches = az_match & el_match

        # Get first matching index for each query
        indices = torch.argmax(matches.float(), dim=1)

        return indices

    def _get_hrir_at_index_batch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get HRIR at specific indices (vectorized).

        Args:
            idx: Tensor of shape (batch_size,)

        Returns:
            Tuple of (left, right) each of shape (batch_size, tap_length)
        """
        return self.data[idx, 0], self.data[idx, 1]

    def _find_containing_triangles_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Find which Delaunay triangle contains each query point (vectorized).

        Args:
            azimuth: Tensor of shape (batch_size,) in degrees
            elevation: Tensor of shape (batch_size,) in degrees

        Returns:
            Tensor of shape (batch_size, 3) with indices of triangle vertices
        """
        if self.triangles is None:
            # Fallback to planar neighbors
            return self._get_planar_neighbours_batch(azimuth, elevation)

        batch_size = azimuth.shape[0]

        # Wrap azimuths to [-180, 180] to match delaunay_points
        azimuth_wrapped = torch.where(
            azimuth > 180,
            azimuth - 360,
            azimuth
        )

        # Query points in 2D: (batch_size, 2)
        query_points = torch.stack([azimuth_wrapped, elevation], dim=1)

        # For each query point, find the triangle that contains it
        # Use barycentric coordinates to check containment
        result_indices = []

        for i in range(batch_size):
            query = query_points[i]  # (2,)

            # Get all triangle vertices
            triangle_verts = self.delaunay_points[self.triangles]  # (num_triangles, 3, 2)

            # Compute barycentric coordinates for all triangles
            # For triangle with vertices v0, v1, v2 and point p:
            # Barycentric coords (w0, w1, w2) where p = w0*v0 + w1*v1 + w2*v2
            v0 = triangle_verts[:, 0, :]  # (num_triangles, 2)
            v1 = triangle_verts[:, 1, :]  # (num_triangles, 2)
            v2 = triangle_verts[:, 2, :]  # (num_triangles, 2)

            # Compute barycentric coordinates
            v0_to_p = query - v0
            v0_to_v1 = v1 - v0
            v0_to_v2 = v2 - v0

            # Cross products (in 2D, this is the z-component of 3D cross)
            denom = v0_to_v1[:, 0] * v0_to_v2[:, 1] - v0_to_v1[:, 1] * v0_to_v2[:, 0]

            # Avoid division by zero
            valid_triangles = torch.abs(denom) > 1e-8

            w1 = torch.zeros_like(denom)
            w2 = torch.zeros_like(denom)

            w1[valid_triangles] = (v0_to_p[valid_triangles, 0] * v0_to_v2[valid_triangles, 1] -
                                   v0_to_p[valid_triangles, 1] * v0_to_v2[valid_triangles, 0]) / denom[valid_triangles]
            w2[valid_triangles] = (v0_to_v1[valid_triangles, 0] * v0_to_p[valid_triangles, 1] -
                                   v0_to_v1[valid_triangles, 1] * v0_to_p[valid_triangles, 0]) / denom[valid_triangles]
            w0 = 1.0 - w1 - w2

            # Check if point is inside triangle (all barycentric coords in [0,1])
            inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01) & valid_triangles

            if inside.any():
                # Use the first containing triangle
                triangle_idx = torch.where(inside)[0][0]
                result_indices.append(self.triangles[triangle_idx])
            else:
                # Fallback: find closest triangle by distance to centroid
                centroids = (v0 + v1 + v2) / 3.0
                distances = torch.norm(centroids - query, dim=1)
                closest_triangle_idx = torch.argmin(distances)
                result_indices.append(self.triangles[closest_triangle_idx])

        # Stack results: (batch_size, 3)
        result = torch.stack(result_indices, dim=0)
        return result[:, 0], result[:, 1], result[:, 2]

    def _get_planar_neighbours_batchrir_pathh(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get nearest planar azimuth neighbours on the closest elevation plane (vectorized).
        Fallback method when Delaunay triangulation is not available.

        Args:
            azimuth: Tensor of shape (batch_size,)
            elevation: Tensor of shape (batch_size,)

        Returns:
            Three tensors of indices, each of shape (batch_size,)
        """
        batch_size = azimuth.shape[0]
        azimuth = (azimuth + 360) % 360

        # Find nearest elevation for each query
        # Shape: (batch_size, num_positions)
        el_diff = torch.abs(self.elevations.unsqueeze(0) - elevation.unsqueeze(1))
        nearest_el_idx = torch.argmin(el_diff, dim=1)  # (batch_size,)
        nearest_el = self.elevations[nearest_el_idx]  # (batch_size,)

        # For simplicity, return nearest 3 points on that elevation plane
        # Create mask for each batch element
        indices_list = []

        for i in range(batch_size):
            # Find all points at this elevation
            el_mask = torch.abs(self.elevations - nearest_el[i]) < 0.001
            planar_indices = torch.where(el_mask)[0]
            planar_azimuths = self.azimuths[planar_indices]

            # Calculate azimuth differences (handle wrapping)
            az_diff = planar_azimuths - azimuth[i]
            az_diff = torch.where(az_diff > 180, az_diff - 360, az_diff)
            az_diff = torch.where(az_diff < -180, az_diff + 360, az_diff)

            # Find closest points
            abs_diff = torch.abs(az_diff)
            sorted_indices = torch.argsort(abs_diff)

            # Take 3 nearest (or duplicate if fewer available)
            if len(sorted_indices) >= 3:
                idx1, idx2, idx3 = planar_indices[sorted_indices[:3]]
            elif len(sorted_indices) == 2:
                idx1, idx2 = planar_indices[sorted_indices[:2]]
                idx3 = idx2
            else:
                idx1 = idx2 = idx3 = planar_indices[sorted_indices[0]]

            indices_list.append([idx1, idx2, idx3])

        indices_tensor = torch.tensor(indices_list, device=self.device)  # (batch_size, 3)
        return indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]

    def angle_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor,
                   mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto",
                   distance_threshold: float = 0.035) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get HRIR for multiple directions with interpolation (fully vectorized).

        Args:
            azimuth: Tensor of shape (batch_size,) with azimuth angles in degrees (0 to 360)
            elevation: Tensor of shape (batch_size,) with elevation angles in degrees (-90 to 90)
            mode: Interpolation mode:
                - "auto": Automatically choose best interpolation
                - "nearest": Use nearest available angle
                - "two_point": Use 2-point interpolation
                - "three_point": Use 3-point interpolation
            distance_threshold: Threshold for using nearest neighbor vs interpolation

        Returns:
            Tuple of (left_channel, right_channel) tensors of shape (batch_size, tap_length)
        """
        if self.source_positions is None:
            raise ValueError("No spatial position data available")

        batch_size = azimuth.shape[0]

        # Ensure tensors are on correct device
        azimuth = azimuth.to(self.device , dtype=self.dtype)
        elevation = elevation.to(self.device , dtype=self.dtype)

        # Normalize angles
        azimuth = (azimuth + 360) % 360
        elevation = torch.clamp(elevation, -90, 90)

        # Initialize output tensors
        left_output = torch.zeros(batch_size, self.tap_length, device=self.device , dtype=self.dtype)
        right_output = torch.zeros(batch_size, self.tap_length, device=self.device , dtype=self.dtype)

        if mode == "nearest":
            # Simple nearest neighbor for all
            nearest_idx = self._find_nearest_direction_batch(azimuth, elevation)
            left_output, right_output = self._get_hrir_at_index_batch(nearest_idx)
            return left_output, right_output

        # Check for exact matches
        exact_matches = self._angle_exists_batch(azimuth, elevation)

        if torch.all(exact_matches):
            # All angles exist exactly
            indices = self._get_angle_index_batch(azimuth, elevation)
            return self._get_hrir_at_index_batch(indices)

        # Find nearest for all queries
        nearest_idx = self._find_nearest_direction_batch(azimuth, elevation)
        nearest_az = self.azimuths[nearest_idx]
        nearest_el = self.elevations[nearest_idx]
        nearest_dist = self._get_angle_distance_batch(azimuth, elevation, nearest_az, nearest_el)

        # Determine which need interpolation
        if mode == "auto":
            needs_interp = (~exact_matches) & (nearest_dist >= distance_threshold)
        else:
            needs_interp = ~exact_matches

        # Handle exact matches and close neighbors
        use_nearest = (~needs_interp) | exact_matches
        if torch.any(use_nearest):
            idx_nearest = torch.where(use_nearest)[0]
            nearest_for_output = nearest_idx[idx_nearest]
            left_nearest, right_nearest = self._get_hrir_at_index_batch(nearest_for_output)
            left_output[idx_nearest] = left_nearest
            right_output[idx_nearest] = right_nearest

        # Handle interpolation
        if torch.any(needs_interp):
            interp_mask = needs_interp
            az_interp = azimuth[interp_mask]
            el_interp = elevation[interp_mask]

            # Get three nearest neighbors for interpolation
            # Use Delaunay triangulation if available, otherwise fall back to planar
            if self.triangles is not None:
                idx1, idx2, idx3 = self._find_containing_triangles_batch(az_interp, el_interp)
            else:
                idx1, idx2, idx3 = self._get_planar_neighbours_batch(az_interp, el_interp)

            # Get angles for the three points
            az1, el1 = self.azimuths[idx1], self.elevations[idx1]
            az2, el2 = self.azimuths[idx2], self.elevations[idx2]
            az3, el3 = self.azimuths[idx3], self.elevations[idx3]

            # Calculate distances
            dist1 = self._get_angle_distance_batch(az_interp, el_interp, az1, el1)
            dist2 = self._get_angle_distance_batch(az_interp, el_interp, az2, el2)
            dist3 = self._get_angle_distance_batch(az_interp, el_interp, az3, el3)

            # Calculate inverse distance weights
            eps = 1e-6
            inv_dist1 = 1.0 / (dist1 + eps)
            inv_dist2 = 1.0 / (dist2 + eps)
            inv_dist3 = 1.0 / (dist3 + eps)

            if mode == "two_point":
                # Use only two closest points
                sum_inv = inv_dist1 + inv_dist2
                w1 = inv_dist1 / sum_inv
                w2 = inv_dist2 / sum_inv

                left1, right1 = self.data[idx1, 0], self.data[idx1, 1]
                left2, right2 = self.data[idx2, 0], self.data[idx2, 1]

                left_interp = w1.unsqueeze(1) * left1 + w2.unsqueeze(1) * left2
                right_interp = w1.unsqueeze(1) * right1 + w2.unsqueeze(1) * right2

            else:  # three_point or auto
                # Use all three points
                sum_inv = inv_dist1 + inv_dist2 + inv_dist3
                w1 = inv_dist1 / sum_inv
                w2 = inv_dist2 / sum_inv
                w3 = inv_dist3 / sum_inv

                left1, right1 = self.data[idx1, 0], self.data[idx1, 1]
                left2, right2 = self.data[idx2, 0], self.data[idx2, 1]
                left3, right3 = self.data[idx3, 0], self.data[idx3, 1]

                left_interp = (w1.unsqueeze(1) * left1 +
                              w2.unsqueeze(1) * left2 +
                              w3.unsqueeze(1) * left3)
                right_interp = (w1.unsqueeze(1) * right1 +
                               w2.unsqueeze(1) * right2 +
                               w3.unsqueeze(1) * right3)

            # Store interpolated results
            interp_indices = torch.where(interp_mask)[0]
            left_output[interp_indices] = left_interp
            right_output[interp_indices] = right_interp

        return left_output, right_output

    def angle(self, azimuth: float, elevation: float,
              mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get HRIR for a single direction (convenience wrapper for batch method).

        Args:
            azimuth: Azimuth angle in degrees (0 to 360)
            elevation: Elevation angle in degrees (-90 to 90)
            mode: Interpolation mode

        Returns:
            Tuple of (left_channel, right_channel) tensors of shape (tap_length,)
        """
        azimuth_tensor = torch.tensor([azimuth], device=self.device , dtype=self.dtype)
        elevation_tensor = torch.tensor([elevation], device=self.device , dtype=self.dtype)

        left, right = self.angle_batch(azimuth_tensor, elevation_tensor, mode=mode)

        return left[0], right[0]

    @property
    def left(self) -> torch.Tensor:
        """Get all left channel HRIRs."""
        return self.data[:, 0, :]

    @property
    def right(self) -> torch.Tensor:
        """Get all right channel HRIRs."""
        return self.data[:, 1, :]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get shape of HRIR tensor."""
        return self.data.shape

    def to(self, device: Union[str, torch.device]) -> 'RIRTensor':
        """Move all tensors to specified device."""
        self.device = torch.device(device)
        self.data = self.data.to(self.device , dtype=self.dtype)
        if self.source_positions is not None:
            self.source_positions = self.source_positions.to(self.device , dtype=self.dtype)
            self._build_spatial_index()
            self._build_delaunay_triangulation()
        return self

    def __repr__(self) -> str:
        return f"RIRTensor(positions={self.num_positions}, channels={self.num_channels}, taps={self.tap_length}, device={self.device})"


class HRIRChannel:
    """Wrapper for single channel HRIR access."""

    def __init__(self, parent: RIRTensor, channel_idx: int):
        """
        Initialize channel wrapper.

        Args:
            parent: Parent RIRTensor object
            channel_idx: Channel index (0 for left, 1 for right)
        """
        self.parent = parent
        self.channel_idx = channel_idx

    def angle(self, azimuth: float, elevation: float,
              mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto") -> torch.Tensor:
        """Get HRIR for a specific direction for this channel."""
        left, right = self.parent.angle(azimuth, elevation, mode=mode)
        return left if self.channel_idx == 0 else right

    def angle_batch(self, azimuth: torch.Tensor, elevation: torch.Tensor,
                   mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto") -> torch.Tensor:
        """Get HRIR for multiple directions for this channel."""
        left, right = self.parent.angle_batch(azimuth, elevation, mode=mode)
        return left if self.channel_idx == 0 else right

    def __repr__(self) -> str:
        channel_name = "left" if self.channel_idx == 0 else "right"
        return f"HRIRChannel({channel_name}, {self.parent})"
