from typing import Literal, Tuple, Union

import sofar as sf
import torch
from torch_geometric.transforms import Delaunay


class RIRTensor:
    """
    GPU-optimized HRIR tensor wrapper with vectorized batch interpolation.

    Provides easy access to HRIRs by azimuth/elevation angles while maintaining
    PyTorch tensor performance with full GPU acceleration.
    """

    def __init__(
        self,
        data: torch.Tensor,
        source_positions: torch.Tensor = None,
        device: torch.device = torch.device("cpu"),
    ):
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
            self.cartesian_positions = None
            return

        # Store azimuth and elevation separately for easier queries
        self.azimuths = self.source_positions[:, 0]  # Shape: (num_positions,)
        self.elevations = self.source_positions[:, 1]  # Shape: (num_positions,)

        # Create available angles tensor for vectorized operations
        self.available_angles = torch.stack(
            [self.azimuths, self.elevations], dim=1
        )  # (num_positions, 2)

        # Precompute Cartesian coordinates for source positions (normalized)
        self.cartesian_positions = self._spherical_to_cartesian_batch(
            self.azimuths, self.elevations
        )

    def _build_delaunay_triangulation(self):
        """Build Delaunay triangulation for 3D spherical interpolation."""
        if self.source_positions is None or self.available_angles is None:
            self.triangles = None
            self.delaunay_points = None
            return

        # Convert spherical coordinates to 2D for triangulation
        # Handle angle wrapping by using coordinates in [-180, 180] range
        azimuths_wrapped = torch.where(
            self.azimuths > 180, self.azimuths - 360, self.azimuths
        )

        # Create 2D points for Delaunay: [azimuth, elevation]
        points_2d = torch.stack(
            [azimuths_wrapped, self.elevations], dim=1
        )  # (num_positions, 2)
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
            if hasattr(data_with_faces, "face"):
                self.triangles = data_with_faces.face.t()  # Shape: (num_triangles, 3)

                # Build vertex-to-triangle adjacency list for optimized search
                # Use CPU for construction to avoid complex scatter logic
                triangles_cpu = self.triangles.cpu()
                num_vertices = self.num_positions
                vertex_to_tri = [[] for _ in range(num_vertices)]

                for i in range(self.triangles.shape[0]):
                    for j in range(3):
                        v = triangles_cpu[i, j].item()
                        vertex_to_tri[v].append(i)

                max_degree = max(len(l) for l in vertex_to_tri) if vertex_to_tri else 0

                # Create padded tensor
                self.vertex_triangles = torch.full(
                    (num_vertices, max_degree),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )

                for i, l in enumerate(vertex_to_tri):
                    if l:
                        self.vertex_triangles[i, : len(l)] = torch.tensor(
                            l, dtype=torch.long, device=self.device
                        )
            else:
                # Fallback if no triangulation possible
                self.triangles = None
                self.vertex_triangles = None
        except Exception as e:
            print(f"Warning: Could not build Delaunay triangulation: {e}")
            self.triangles = None

    @classmethod
    def from_sofa(
        cls, sofa_path: str, device: Literal["cpu", "cuda"] = "cpu"
    ) -> "RIRTensor":
        """Load HRIR from SOFA file."""
        sofa_obj = sf.read_sofa(sofa_path)

        # Extract IR data
        if hasattr(sofa_obj, "Data_IR"):
            ir_data = sofa_obj.Data_IR
        elif hasattr(sofa_obj, "data"):
            ir_data = sofa_obj.data
        else:
            raise ValueError("Cannot find IR data in SOFA object")

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(ir_data)

        # Extract source positions if available
        source_positions = None
        if hasattr(sofa_obj, "SourcePosition"):
            source_positions = torch.from_numpy(sofa_obj.SourcePosition).float()

        return cls(tensor, source_positions, device=device)

    def _spherical_to_cartesian_batch(
        self, azimuth: torch.Tensor, elevation: torch.Tensor
    ) -> torch.Tensor:
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

    def _find_nearest_direction_batch(
        self, azimuth: torch.Tensor, elevation: torch.Tensor
    ) -> torch.Tensor:
        """
        Find the index of the nearest spatial direction for each query.

        Args:
            azimuth: Tensor of shape (batch_size,)
            elevation: Tensor of shape (batch_size,)

        Returns:
            Indices of shape (batch_size,)
        """
        if self.source_positions is None:
            raise ValueError("No spatial position data available")

        # Normalize angles (clone to avoid modifying input, then use in-place)
        azimuth = azimuth.clone().remainder_(360)
        elevation = elevation.clone().clamp_(-90, 90)

        # Convert query points to cartesian coordinates
        query_xyz = self._spherical_to_cartesian_batch(
            azimuth, elevation
        )  # (batch_size, 3)

        # Transpose cartesian positions for matrix multiplication: (3, num_positions)
        available_xyz_t = self.cartesian_positions.t()

        # Calculate dot products: (batch_size, num_positions)
        # Since vectors are unit length, dot product is cosine similarity
        # Max dot product corresponds to min distance (and max similarity)
        dot_products = torch.matmul(query_xyz, available_xyz_t)

        # Get nearest index
        result_indices = torch.argmax(dot_products, dim=1)

        return result_indices

    def _get_hrir_at_index_batch(
        self, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get HRIR at specific indices (vectorized).

        Args:
            idx: Tensor of shape (batch_size,)

        Returns:
            Tuple of (left, right) each of shape (batch_size, tap_length)
        """
        return self.data[idx, 0], self.data[idx, 1]

    def _find_containing_triangles_batch(
        self, azimuth: torch.Tensor, elevation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find which Delaunay triangle contains each query point using optimized adjacency search.

        Args:
            azimuth: Tensor of shape (batch_size,) in degrees
            elevation: Tensor of shape (batch_size,) in degrees

        Returns:
            Tuple of three tensors, each of shape (batch_size,) with indices of triangle vertices
        """
        if self.triangles is None:
            # Fallback to planar neighbors
            return self._get_planar_neighbours_batch(azimuth, elevation)

        batch_size = azimuth.shape[0]

        # Wrap azimuths to [-180, 180] to match delaunay_points
        azimuth_wrapped = azimuth.clone()
        azimuth_wrapped[azimuth_wrapped > 180] -= 360

        # Query points in 2D: (batch_size, 2)
        query_points = torch.stack([azimuth_wrapped, elevation], dim=1)

        # 1. Find nearest vertex for each query point
        # This is fast now due to dot-product optimization
        nearest_idx = self._find_nearest_direction_batch(azimuth, elevation)

        # 2. Get candidate triangles connected to the nearest vertex
        # Shape: (batch_size, max_degree)
        candidate_tri_indices = self.vertex_triangles[nearest_idx]

        # Handle padding (-1)
        mask = candidate_tri_indices != -1
        # Use index 0 for padding, will be masked out later
        safe_indices = torch.where(
            mask, candidate_tri_indices, torch.zeros_like(candidate_tri_indices)
        )

        # 3. Gather triangle vertices for candidates
        # self.triangles: (num_triangles, 3)
        # candidate_triangles: (batch_size, max_degree, 3)
        candidate_triangles = self.triangles[safe_indices]

        # 4. Gather 2D coordinates of these vertices
        # self.delaunay_points: (num_positions, 2)
        # We need to gather from delaunay_points using candidate_triangles indices
        # Flatten to use simple indexing
        flat_verts_idx = candidate_triangles.view(-1)
        flat_coords = self.delaunay_points[flat_verts_idx]
        # Reshape back: (batch_size, max_degree, 3, 2)
        tri_coords = flat_coords.view(batch_size, -1, 3, 2)

        # Extract vertices
        v0 = tri_coords[:, :, 0, :]  # (batch, max_degree, 2)
        v1 = tri_coords[:, :, 1, :]
        v2 = tri_coords[:, :, 2, :]

        # 5. Compute Barycentric coordinates
        # Query points need to be expanded: (batch, 1, 2)
        p = query_points.unsqueeze(1)

        v0_to_v1 = v1 - v0
        v0_to_v2 = v2 - v0
        v0_to_p = p - v0

        denom = (
            v0_to_v1[..., 0] * v0_to_v2[..., 1] - v0_to_v1[..., 1] * v0_to_v2[..., 0]
        )

        # Avoid division by zero
        denom_safe = denom.clone()
        denom_safe[torch.abs(denom) <= 1e-8] = 1.0

        cross_p_v2 = (
            v0_to_p[..., 0] * v0_to_v2[..., 1] - v0_to_p[..., 1] * v0_to_v2[..., 0]
        )
        cross_v1_p = (
            v0_to_v1[..., 0] * v0_to_p[..., 1] - v0_to_v1[..., 1] * v0_to_p[..., 0]
        )

        w1 = cross_p_v2.div_(denom_safe)
        w2 = cross_v1_p.div_(denom_safe)
        w0 = 1.0 - w1 - w2

        # 6. Check if point is inside triangle
        tolerance = 0.01
        inside = (w0 >= -tolerance) & (w1 >= -tolerance) & (w2 >= -tolerance)

        # Apply mask (ignore padded triangles)
        inside = inside & mask

        # 7. Select the containing triangle
        has_match = inside.any(dim=1)
        match_idx_local = inside.float().argmax(
            dim=1
        )  # Index into candidates (0..max_degree)

        # Gather the actual triangle index
        match_tri_idx = candidate_tri_indices.gather(
            1, match_idx_local.unsqueeze(1)
        ).squeeze(1)

        # Fallback: if no containing triangle found (rare), use the first candidate
        # (which is connected to the nearest vertex)
        final_tri_idx = torch.where(
            has_match, match_tri_idx, candidate_tri_indices[:, 0]
        )

        # Retrieve vertices for the selected triangles
        result_triangles = self.triangles[final_tri_idx]

        return result_triangles[:, 0], result_triangles[:, 1], result_triangles[:, 2]

    def _get_planar_neighbours_batch(
        self, azimuth: torch.Tensor, elevation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get nearest planar azimuth neighbours on the closest elevation plane.
        Fallback method when Delaunay triangulation is not available.

        Args:
            azimuth: Tensor of shape (batch_size,)
            elevation: Tensor of shape (batch_size,)

        Returns:
            Three tensors of indices, each of shape (batch_size,)
        """
        azimuth = azimuth.clone().add_(360).remainder_(360)

        # Find nearest elevation for each query
        el_diff = (self.elevations.unsqueeze(0) - elevation.unsqueeze(1)).abs_()
        nearest_el_idx = torch.argmin(el_diff, dim=1)
        nearest_el = self.elevations[nearest_el_idx]

        # Calculate azimuth differences with wrapping
        az_diff = self.azimuths.unsqueeze(0) - azimuth.unsqueeze(1)
        # Wrap to [-180, 180] range in-place
        az_diff[az_diff > 180] -= 360
        az_diff[az_diff < -180] += 360
        az_diff.abs_()

        # Create elevation mask
        el_tolerance = 0.1
        el_mask = (
            torch.abs(self.elevations.unsqueeze(0) - nearest_el.unsqueeze(1))
            < el_tolerance
        )

        # Set distance to infinity for positions not on the elevation plane (in-place)
        az_diff[~el_mask] = float("inf")

        # Get indices of 3 smallest distances
        _, top3_idx = torch.topk(az_diff, k=3, dim=1, largest=False)

        return top3_idx[:, 0], top3_idx[:, 1], top3_idx[:, 2]

    def angle_batch(
        self,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto",
        distance_threshold: float = 0.035,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Ensure tensors are on correct device and normalize angles (clone to avoid modifying input)
        azimuth = azimuth.to(self.device, dtype=self.dtype).clone()
        elevation = elevation.to(self.device, dtype=self.dtype).clone()

        # Normalize angles in-place
        azimuth.add_(360).remainder_(360)
        elevation.clamp_(-90, 90)

        # Initialize output tensors
        left_output = torch.zeros(
            batch_size, self.tap_length, device=self.device, dtype=self.dtype
        )
        right_output = torch.zeros(
            batch_size, self.tap_length, device=self.device, dtype=self.dtype
        )

        # Precompute query Cartesian coordinates
        query_xyz = self._spherical_to_cartesian_batch(azimuth, elevation)

        # Find nearest for all queries (used for nearest mode, exact match check, and fallback)
        nearest_idx = self._find_nearest_direction_batch(azimuth, elevation)

        if mode == "nearest":
            left_output, right_output = self._get_hrir_at_index_batch(nearest_idx)
            return left_output, right_output

        # Check for exact matches using the nearest neighbors we just found
        nearest_az = self.azimuths[nearest_idx]
        nearest_el = self.elevations[nearest_idx]

        # Check differences (handling wrapping for azimuth)
        az_diff = (azimuth - nearest_az).abs_()
        torch.minimum(az_diff, 360 - az_diff, out=az_diff)
        el_diff = (elevation - nearest_el).abs_()

        tolerance = 0.001
        exact_matches = (az_diff < tolerance) & (el_diff < tolerance)

        if torch.all(exact_matches):
            # All angles exist exactly
            return self._get_hrir_at_index_batch(nearest_idx)

        # Calculate distance to nearest neighbor
        nearest_xyz = self.cartesian_positions[nearest_idx]
        nearest_dist = torch.norm(query_xyz - nearest_xyz, dim=-1)

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
            left_nearest, right_nearest = self._get_hrir_at_index_batch(
                nearest_for_output
            )
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
                idx1, idx2, idx3 = self._find_containing_triangles_batch(
                    az_interp, el_interp
                )
            else:
                idx1, idx2, idx3 = self._get_planar_neighbours_batch(
                    az_interp, el_interp
                )

            # Get Cartesian coordinates for the three points
            p1 = self.cartesian_positions[idx1]
            p2 = self.cartesian_positions[idx2]
            p3 = self.cartesian_positions[idx3]

            # Get Cartesian coordinates for query points
            p_interp = query_xyz[interp_mask]

            # Calculate distances
            dist1 = torch.norm(p_interp - p1, dim=-1)
            dist2 = torch.norm(p_interp - p2, dim=-1)
            dist3 = torch.norm(p_interp - p3, dim=-1)

            # Calculate inverse distance weights (in-place)
            eps = 1e-6
            dist1.add_(eps).reciprocal_()  # inv_dist1 = 1.0 / (dist1 + eps)
            dist2.add_(eps).reciprocal_()  # inv_dist2 = 1.0 / (dist2 + eps)
            dist3.add_(eps).reciprocal_()  # inv_dist3 = 1.0 / (dist3 + eps)
            inv_dist1, inv_dist2, inv_dist3 = dist1, dist2, dist3

            if mode == "two_point":
                # Use only two closest points
                sum_inv = inv_dist1 + inv_dist2
                w1 = inv_dist1.div_(sum_inv)
                w2 = inv_dist2.div_(sum_inv)

                left1, right1 = self.data[idx1, 0], self.data[idx1, 1]
                left2, right2 = self.data[idx2, 0], self.data[idx2, 1]

                left_interp = w1.unsqueeze(1) * left1 + w2.unsqueeze(1) * left2
                right_interp = w1.unsqueeze(1) * right1 + w2.unsqueeze(1) * right2

            else:  # three_point or auto
                # Use all three points
                sum_inv = inv_dist1 + inv_dist2 + inv_dist3
                w1 = inv_dist1.div_(sum_inv)
                w2 = inv_dist2.div_(sum_inv)
                w3 = inv_dist3.div_(sum_inv)

                left1, right1 = self.data[idx1, 0], self.data[idx1, 1]
                left2, right2 = self.data[idx2, 0], self.data[idx2, 1]
                left3, right3 = self.data[idx3, 0], self.data[idx3, 1]

                # Compute weighted sum using in-place operations
                w1_expanded = w1.unsqueeze(1)
                w2_expanded = w2.unsqueeze(1)
                w3_expanded = w3.unsqueeze(1)

                left_interp = left1.mul(w1_expanded)
                left_interp.addcmul_(left2, w2_expanded)
                left_interp.addcmul_(left3, w3_expanded)

                right_interp = right1.mul(w1_expanded)
                right_interp.addcmul_(right2, w2_expanded)
                right_interp.addcmul_(right3, w3_expanded)

            # Store interpolated results
            interp_indices = torch.where(interp_mask)[0]
            left_output[interp_indices] = left_interp
            right_output[interp_indices] = right_interp

        return left_output, right_output

    def angle(
        self,
        azimuth: float,
        elevation: float,
        mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get HRIR for a single direction (convenience wrapper for batch method).

        Args:
            azimuth: Azimuth angle in degrees (0 to 360)
            elevation: Elevation angle in degrees (-90 to 90)
            mode: Interpolation mode

        Returns:
            Tuple of (left_channel, right_channel) tensors of shape (tap_length,)
        """
        azimuth_tensor = torch.tensor([azimuth], device=self.device, dtype=self.dtype)
        elevation_tensor = torch.tensor(
            [elevation], device=self.device, dtype=self.dtype
        )

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

    def to(self, device: Union[str, torch.device]) -> "RIRTensor":
        """Move all tensors to specified device."""
        self.device = torch.device(device)
        self.data = self.data.to(self.device, dtype=self.dtype)
        if self.source_positions is not None:
            self.source_positions = self.source_positions.to(
                self.device, dtype=self.dtype
            )
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

    def angle(
        self,
        azimuth: float,
        elevation: float,
        mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto",
    ) -> torch.Tensor:
        """Get HRIR for a specific direction for this channel."""
        left, right = self.parent.angle(azimuth, elevation, mode=mode)
        return left if self.channel_idx == 0 else right

    def angle_batch(
        self,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto",
    ) -> torch.Tensor:
        """Get HRIR for multiple directions for this channel."""
        left, right = self.parent.angle_batch(azimuth, elevation, mode=mode)
        return left if self.channel_idx == 0 else right

    def __repr__(self) -> str:
        channel_name = "left" if self.channel_idx == 0 else "right"
        return f"HRIRChannel({channel_name}, {self.parent})"
