"""
Batched version of FramRIR implementation for GPU-accelerated binaural room impulse response generation.

Based on the original FramRIR implementation from Tencent AI Lab by Rongzhi Gu, Yi Luo.

My Implmentation will use the same framework as FramRIR for genreating reflection directions but instead i will make a binaural room impulse response.

GITHUB: https://github.com/tencent-ailab/FRA-RIR/blob/main/FRAM_RIR.py

CITATION: Luo, Y., & Gu, R. (2024, April). Fast random approximation of multi-channel room impulse response. In 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW) (pp. 449-454). IEEE.
"""

from typing import Tuple

import torch
from torch import Tensor
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample

from .rirTensor import RIRTensor


@torch.no_grad()
def dist_first_order_reflection_batch(
    src_pos: Tensor, mic_pos: Tensor, room_dim: Tensor
) -> Tensor:
    """
    Generate first order reflection directions for a given source and microphone position within a room.
    Note: this is actually the same method used in the original ISM(Image source method) creating a room of perfect mirrors to trace reflections,
    FRAM-RIR actually skips this and uses a heuristic probability approach(FASTER) to create reflections,
    but i will simply calculate the first order reflections and stop there just to be able to get an accurate initial Time Delay.

    Args:
        src_pos (Tensor): Source positions in the room, shape (B, 3).
        mic_pos (Tensor): Microphone positions in the room, shape (B, 3).
        room_dim (Tensor): Room dimensions, shape (B, 3).

    Returns:
        Tensor: Minimum first-order reflection distances, shape (B,).
    """
    # Batched version: src_pos, mic_pos, room_dim all have shape (B, 3)

    image_x0 = torch.stack([-src_pos[:, 0], src_pos[:, 1], src_pos[:, 2]], dim=-1)
    image_xR = torch.stack(
        [2 * room_dim[:, 0] - src_pos[:, 0], src_pos[:, 1], src_pos[:, 2]], dim=-1
    )

    image_y0 = torch.stack([src_pos[:, 0], -src_pos[:, 1], src_pos[:, 2]], dim=-1)
    image_yR = torch.stack(
        [src_pos[:, 0], 2 * room_dim[:, 1] - src_pos[:, 1], src_pos[:, 2]], dim=-1
    )

    image_z0 = torch.stack([src_pos[:, 0], src_pos[:, 1], -src_pos[:, 2]], dim=-1)
    image_zR = torch.stack(
        [src_pos[:, 0], src_pos[:, 1], 2 * room_dim[:, 2] - src_pos[:, 2]], dim=-1
    )

    images_all = torch.stack(
        [image_x0, image_xR, image_y0, image_yR, image_z0, image_zR], dim=1
    )  # (B, 6, 3)

    diff = images_all - mic_pos.unsqueeze(1)  # (B, 6, 3)

    dists = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-8)  # (B, 6)

    min_reflection_dist = torch.min(dists, dim=1).values  # (B,)

    return min_reflection_dist


@torch.no_grad()
def batch_fram_brir(
    target_sr: int,
    t60: Tensor,
    h_rir: RIRTensor,
    hrir_sr: int = 96000,
    mic_pos: Tensor = None,
    room_dim: Tensor = None,
    src_pos: Tensor = None,
    n_reflection: Tensor = None,
    a: float = -2.0,
    b: float = 2.0,
    tau: float = 0.25,
    device: torch.device = torch.device("cpu"),
    reflection_chunk_size: int = 100,
) -> Tensor:
    """
    Batched binaural room impulse response generation using the FRAM-RIR algorithm.

    This function generates reverb-only BRIRs for multiple configurations in parallel
    using fully vectorized operations suitable for GPU acceleration.

    Args:
        target_sr (int): Target sample rate for the output BRIR.
        t60 (Tensor): Reverberation times in seconds, shape (B,).
        h_rir (RIRTensor): The head-related impulse response class that generates HRIRs.
        hrir_sr (int): Sample rate of the HRIR data. Default: 96000.
        mic_pos (Tensor): Microphone/receiver positions, shape (B, 3).
        room_dim (Tensor): Room dimensions, shape (B, 3).
        src_pos (Tensor): Sound source positions, shape (B, 3).
        n_reflection (Tensor): Range of reflection counts per batch, shape (B, 2) where
                               [:, 0] is min and [:, 1] is max reflections.
        a (float): Minimum of random perturbation. Default: -2.0.
        b (float): Maximum of random perturbation. Default: 2.0.
        tau (float): Time constant for exponential decay (distance shrinkage factor). Default: 0.25.
        device (torch.device): Device to use. Default: cpu.
        reflection_chunk_size (int): Number of reflections to process at once per batch item.
            Controls peak memory usage. Lower = less memory, more iterations. Default: 100.

    Returns:
        Tensor: Batched 2-channel reverb-only BRIRs, shape (B, 2, rir_length).
    """
    # Set defaults
    if mic_pos is None:
        mic_pos = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    if room_dim is None:
        room_dim = torch.tensor([[4.0, 4.0, 4.0]], device=device)
    if src_pos is None:
        src_pos = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    if n_reflection is None:
        n_reflection = torch.tensor([[100, 700]], device=device)

    if (
        mic_pos.shape[0] != room_dim.shape[0]
        or mic_pos.shape[0] != src_pos.shape[0]
        or mic_pos.shape[0] != n_reflection.shape[0]
        or mic_pos.shape[0] != t60.shape[0]
    ):
        raise ValueError(
            "mic_pos, room_dim, src_pos, t60, and n_reflection must have the same batch size"
        )

    # Move tensors to device
    mic_pos = mic_pos.to(device).float()
    src_pos = src_pos.to(device).float()
    room_dim = room_dim.to(device).float()
    t60 = t60.to(device).float()
    n_reflection = n_reflection.to(device)

    B = t60.shape[0]

    downsampler = Resample(orig_freq=hrir_sr, new_freq=target_sr).to(device)

    if target_sr > hrir_sr:
        print(
            "Warning: Target sample rate is higher than HRIR sample rate, this can cause worse timing accuracy"
        )

    if hrir_sr == 96000:
        hrir_len = 512
    else:
        hrir_len = 256

    # Randomly sample number of reflections for each batch element
    # n_reflection: (B, 2) where [:, 0] is low, [:, 1] is high
    n_ref_low = n_reflection[:, 0]  # (B,)
    n_ref_high = n_reflection[:, 1]  # (B,)

    # Sample uniformly in [low, high) for each batch element
    rand_vals = torch.rand(B, device=device)
    image_counts = (n_ref_low + rand_vals * (n_ref_high - n_ref_low)).long()  # (B,)

    # Use maximum image count for uniform tensor operations
    max_image_count = image_counts.max().item()

    # Geometric environment setup
    # volume_to_surface_area_ratio for each batch
    # V/S = 1 / (2 * (1/L + 1/W + 1/H))
    inv_sum = 1.0 / room_dim[:, 0] + 1.0 / room_dim[:, 1] + 1.0 / room_dim[:, 2]  # (B,)
    volume_to_surface_area_ratio = 1.0 / (2 * inv_sum)  # (B,)

    eps = torch.finfo(torch.float32).eps
    velocity = 343.0

    # Direct distance from mic to source for each batch
    direct_dist = torch.sqrt((mic_pos - src_pos).pow(2).sum(dim=-1) + eps)  # (B,)

    # Reflection coefficient
    reflect_coef = torch.sqrt(
        1 - (1 - torch.exp(-0.16 * volume_to_surface_area_ratio / t60)).pow(2)
    )  # (B,)

    # Maximum reflection order
    reflect_max = (torch.log10(velocity * t60) - 3) / torch.log10(
        reflect_coef + eps
    )  # (B,)

    # First order reflection distances
    first_reflection_dist = dist_first_order_reflection_batch(
        src_pos, mic_pos, room_dim
    )  # (B,)

    # Safety check for shortest path
    shortest_path_safe_check = torch.maximum(
        first_reflection_dist, direct_dist + 0.001
    )  # (B,)

    min_dist_ratio = shortest_path_safe_check / direct_dist  # (B,)

    # Maximum RIR length at high sample rate (use max t60 for tensor sizing)
    max_t60 = t60.max().item()
    max_rir_length_high = int(hrir_sr * max_t60)

    # For each batch, we need to sample distances
    # Create distance ranges and probabilities for each batch element
    # We'll use the maximum possible length and mask invalid samples

    # dist_range end values: velocity * t60 / direct_dist - 1 for each batch
    dist_range_end = velocity * t60 / direct_dist - 1  # (B,)

    # RIR lengths for each batch element
    rir_length_high = (hrir_sr * t60).long()  # (B,)

    # Create output BRIR tensor
    brir_high = torch.zeros(B, 2, max_rir_length_high, device=device)

    # Time offsets for scatter (reuse across chunks)
    time_offsets = torch.arange(hrir_len, device=device)

    # ===== CHUNKED PROCESSING =====
    # Process reflections in chunks to limit peak memory
    # Instead of allocating (B, max_image_count, hrir_len) tensors (~14GB for 6 HRIRs),
    # we process (B, chunk_size, hrir_len) at a time (~800MB per chunk)

    for chunk_start in range(0, max_image_count, reflection_chunk_size):
        chunk_end = min(chunk_start + reflection_chunk_size, max_image_count)
        chunk_size = chunk_end - chunk_start

        # Create reflection indices for this chunk
        reflection_indices = (
            torch.arange(chunk_start, chunk_end, device=device)
            .unsqueeze(0)
            .expand(B, chunk_size)
        )
        valid_reflection_mask = reflection_indices < image_counts.unsqueeze(1)

        # Sample distances for this chunk
        u = torch.rand(B, chunk_size, device=device)
        normalized_dist_samples = u.pow(1.0 / 3.0)
        dist_nearest_ratio = min_dist_ratio.unsqueeze(1) + normalized_dist_samples * (
            dist_range_end - min_dist_ratio
        ).unsqueeze(1)

        # Sample random directions for this chunk
        azm = torch.empty(B, chunk_size, device=device).uniform_(-torch.pi, torch.pi)
        ele = torch.empty(B, chunk_size, device=device).uniform_(
            -torch.pi / 2, torch.pi / 2
        )

        # Compute unit vectors
        unit_3d = torch.stack(
            [
                torch.sin(ele) * torch.cos(azm),
                torch.cos(ele),
                torch.sin(ele) * torch.sin(azm),
            ],
            dim=-1,
        )

        # Compute image positions
        image2nearest_dis = dist_nearest_ratio * direct_dist.unsqueeze(1)
        image_position = (
            mic_pos.unsqueeze(1) + image2nearest_dis.unsqueeze(-1) * unit_3d
        )

        # Compute distances from mic to image positions
        dist = torch.sqrt(
            (mic_pos.unsqueeze(1) - image_position).pow(2).sum(dim=-1) + eps
        )

        # Compute gain decays
        reflect_ratio = (dist / (velocity * t60.unsqueeze(1))) * (
            reflect_max.unsqueeze(1) - 1
        ) + 1

        # Random perturbation
        reflect_pertub = torch.empty(B, chunk_size, device=device).uniform_(
            a, b
        ) * dist_nearest_ratio.pow(tau)

        reflect_ratio = torch.maximum(
            reflect_ratio + reflect_pertub,
            torch.ones(B, chunk_size, device=device),
        )

        # Gains
        gains = reflect_coef.unsqueeze(1).pow(reflect_ratio) / dist

        # Compute time delays
        path_diff = dist - direct_dist.unsqueeze(1)
        delays = torch.ceil(path_diff * hrir_sr / velocity).long()

        # Valid mask for this chunk
        valid_delay_mask = (delays + hrir_len) < rir_length_high.unsqueeze(1)
        valid_mask = valid_reflection_mask & valid_delay_mask

        # Compute direction of arrival for HRIR lookup
        vec_mic_to_img = image_position - mic_pos.unsqueeze(1)
        radius = torch.sqrt(vec_mic_to_img.pow(2).sum(dim=-1) + eps)
        ux = vec_mic_to_img[..., 0] / radius
        uy = vec_mic_to_img[..., 1] / radius
        uz = vec_mic_to_img[..., 2] / radius

        azm_of_arrival = torch.atan2(-ux, uz)
        ele_of_arrival = torch.asin(torch.clamp(uy, -1.0, 1.0))

        azm_degree = torch.rad2deg(azm_of_arrival).view(-1)
        ele_degree = torch.rad2deg(ele_of_arrival).view(-1)

        left_hrirs, right_hrirs = h_rir.angle_batch(azm_degree, ele_degree)
        left_hrirs = left_hrirs.view(B, chunk_size, hrir_len)
        right_hrirs = right_hrirs.view(B, chunk_size, hrir_len)

        # Apply gains and validity mask
        weighted_left = (
            left_hrirs * gains.unsqueeze(-1) * valid_mask.unsqueeze(-1).float()
        )
        weighted_right = (
            right_hrirs * gains.unsqueeze(-1) * valid_mask.unsqueeze(-1).float()
        )

        # Scatter-add for this chunk
        target_indices = delays.unsqueeze(-1) + time_offsets.unsqueeze(0).unsqueeze(0)
        valid_indices_mask = (target_indices >= 0) & (
            target_indices < max_rir_length_high
        )
        valid_indices_mask = valid_indices_mask & valid_mask.unsqueeze(-1)
        target_indices = torch.clamp(target_indices, 0, max_rir_length_high - 1)

        batch_idx = (
            torch.arange(B, device=device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(B, chunk_size, hrir_len)
        )

        flat_batch_idx = batch_idx.reshape(-1)
        flat_target_idx = target_indices.reshape(-1)
        flat_left = (weighted_left * valid_indices_mask.float()).reshape(-1)
        flat_right = (weighted_right * valid_indices_mask.float()).reshape(-1)

        linear_idx = flat_batch_idx * max_rir_length_high + flat_target_idx

        brir_left_flat = brir_high[:, 0, :].reshape(-1)
        brir_right_flat = brir_high[:, 1, :].reshape(-1)
        brir_left_flat.index_add_(0, linear_idx, flat_left.float())
        brir_right_flat.index_add_(0, linear_idx, flat_right.float())
        brir_high[:, 0, :] = brir_left_flat.view(B, max_rir_length_high)
        brir_high[:, 1, :] = brir_right_flat.view(B, max_rir_length_high)

        # Memory is freed at end of loop iteration

    # Apply highpass filter to each batch element
    # highpass_biquad expects (..., time) and applies along last dimension
    # brir_high = highpass_biquad(brir_high, hrir_sr, 80.0)

    # Downsample to target sample rate
    # Resample expects (..., time) format
    brir_final = downsampler(brir_high)

    valid_after_dry = (target_sr * t60).long()
    return brir_final, valid_after_dry


# original non CHUNKED


@torch.no_grad()
def batch_fram_brir_none_chunk(
    target_sr: int,
    t60: Tensor,
    h_rir: RIRTensor,
    hrir_sr: int = 96000,
    mic_pos: Tensor = None,
    room_dim: Tensor = None,
    src_pos: Tensor = None,
    n_reflection: Tensor = None,
    a: float = -2.0,
    b: float = 2.0,
    tau: float = 0.25,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Batched binaural room impulse response generation using the FRAM-RIR algorithm.

    This function generates reverb-only BRIRs for multiple configurations in parallel
    using fully vectorized operations suitable for GPU acceleration.

    Args:
        target_sr (int): Target sample rate for the output BRIR.
        t60 (Tensor): Reverberation times in seconds, shape (B,).
        h_rir (RIRTensor): The head-related impulse response class that generates HRIRs.
        hrir_sr (int): Sample rate of the HRIR data. Default: 96000.
        mic_pos (Tensor): Microphone/receiver positions, shape (B, 3).
        room_dim (Tensor): Room dimensions, shape (B, 3).
        src_pos (Tensor): Sound source positions, shape (B, 3).
        n_reflection (Tensor): Range of reflection counts per batch, shape (B, 2) where
                               [:, 0] is min and [:, 1] is max reflections.
        a (float): Minimum of random perturbation. Default: -2.0.
        b (float): Maximum of random perturbation. Default: 2.0.
        tau (float): Time constant for exponential decay (distance shrinkage factor). Default: 0.25.
        device (torch.device): Device to use. Default: cpu.

    Returns:
        Tensor: Batched 2-channel reverb-only BRIRs, shape (B, 2, rir_length).
    """
    # Set defaults
    if mic_pos is None:
        mic_pos = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    if room_dim is None:
        room_dim = torch.tensor([[4.0, 4.0, 4.0]], device=device)
    if src_pos is None:
        src_pos = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    if n_reflection is None:
        n_reflection = torch.tensor([[100, 700]], device=device)

    if (
        mic_pos.shape[0] != room_dim.shape[0]
        or mic_pos.shape[0] != src_pos.shape[0]
        or mic_pos.shape[0] != n_reflection.shape[0]
        or mic_pos.shape[0] != t60.shape[0]
    ):
        raise ValueError(
            "mic_pos, room_dim, src_pos, t60, and n_reflection must have the same batch size"
        )

    # Move tensors to device
    mic_pos = mic_pos.to(device).float()
    src_pos = src_pos.to(device).float()
    room_dim = room_dim.to(device).float()
    t60 = t60.to(device).float()
    n_reflection = n_reflection.to(device)

    B = t60.shape[0]

    downsampler = Resample(orig_freq=hrir_sr, new_freq=target_sr).to(device)

    if target_sr > hrir_sr:
        print(
            "Warning: Target sample rate is higher than HRIR sample rate, this can cause worse timing accuracy"
        )

    if hrir_sr == 96000:
        hrir_len = 512
    else:
        hrir_len = 256

    n_ref_low = n_reflection[:, 0]  # (B,)
    n_ref_high = n_reflection[:, 1]  # (B,)

    rand_vals = torch.rand(B, device=device)
    image_counts = (n_ref_low + rand_vals * (n_ref_high - n_ref_low)).long()  # (B,)

    max_image_count = image_counts.max().item()

    inv_sum = 1.0 / room_dim[:, 0] + 1.0 / room_dim[:, 1] + 1.0 / room_dim[:, 2]  # (B,)
    volume_to_surface_area_ratio = 1.0 / (2 * inv_sum)  # (B,)

    eps = torch.finfo(torch.float32).eps
    velocity = 343.0

    direct_dist = torch.sqrt((mic_pos - src_pos).pow(2).sum(dim=-1) + eps)  # (B,)

    reflect_coef = torch.sqrt(
        1 - (1 - torch.exp(-0.16 * volume_to_surface_area_ratio / t60)).pow(2)
    )  # (B,)

    reflect_max = (torch.log10(velocity * t60) - 3) / torch.log10(
        reflect_coef + eps
    )  # (B,)

    first_reflection_dist = dist_first_order_reflection_batch(
        src_pos, mic_pos, room_dim
    )  # (B,)

    shortest_path_safe_check = torch.maximum(
        first_reflection_dist, direct_dist + 0.001
    )  # (B,)

    min_dist_ratio = shortest_path_safe_check / direct_dist  # (B,)

    max_t60 = t60.max().item()
    max_rir_length_high = int(hrir_sr * max_t60)

    dist_range_end = velocity * t60 / direct_dist - 1  # (B,)

    reflection_indices = (
        torch.arange(max_image_count, device=device)
        .unsqueeze(0)
        .expand(B, max_image_count)
    )
    valid_reflection_mask = reflection_indices < image_counts.unsqueeze(
        1
    )  # (B, max_image_count)

    u = torch.rand(B, max_image_count, device=device)
    normalized_dist_samples = u.pow(
        1.0 / 3.0
    )  # inverse of cubic CDF gives quadratic PDF

    dist_nearest_ratio = min_dist_ratio.unsqueeze(1) + normalized_dist_samples * (
        dist_range_end - min_dist_ratio
    ).unsqueeze(1)  # (B, max_image_count)

    azm = torch.empty(B, max_image_count, device=device).uniform_(
        -torch.pi, torch.pi
    )  # (B, max_image_count)
    ele = torch.empty(B, max_image_count, device=device).uniform_(
        -torch.pi / 2, torch.pi / 2
    )  # (B, max_image_count)

    # Compute unit vectors for each direction
    # unit_3d: (B, max_image_count, 3)
    unit_3d = torch.stack(
        [
            torch.sin(ele) * torch.cos(azm),
            torch.cos(ele),
            torch.sin(ele) * torch.sin(azm),
        ],
        dim=-1,
    )  # (B, max_image_count, 3)

    image2nearest_dis = dist_nearest_ratio * direct_dist.unsqueeze(1)

    image_position = mic_pos.unsqueeze(1) + image2nearest_dis.unsqueeze(-1) * unit_3d

    dist = torch.sqrt((mic_pos.unsqueeze(1) - image_position).pow(2).sum(dim=-1) + eps)

    reflect_ratio = (dist / (velocity * t60.unsqueeze(1))) * (
        reflect_max.unsqueeze(1) - 1
    ) + 1

    reflect_pertub = torch.empty(B, max_image_count, device=device).uniform_(
        a, b
    ) * dist_nearest_ratio.pow(tau)

    reflect_ratio = torch.maximum(
        reflect_ratio + reflect_pertub,
        torch.ones(B, max_image_count, device=device),
    )

    gains = reflect_coef.unsqueeze(1).pow(reflect_ratio) / dist

    path_diff = dist - direct_dist.unsqueeze(1)  # (B, max_image_count)
    delays = torch.ceil(path_diff * hrir_sr / velocity).long()  # (B, max_image_count)

    rir_length_high = (hrir_sr * t60).long()  # (B,)

    valid_delay_mask = (delays + hrir_len) < rir_length_high.unsqueeze(
        1
    )  # (B, max_image_count)
    valid_mask = valid_reflection_mask & valid_delay_mask  # (B, max_image_count)

    vec_mic_to_img = image_position - mic_pos.unsqueeze(1)

    radius = torch.sqrt(vec_mic_to_img.pow(2).sum(dim=-1) + eps)  # (B, max_image_count)
    ux = vec_mic_to_img[..., 0] / radius
    uy = vec_mic_to_img[..., 1] / radius
    uz = vec_mic_to_img[..., 2] / radius

    azm_of_arrival = torch.atan2(-ux, uz)  # (B, max_image_count)
    ele_of_arrival = torch.asin(torch.clamp(uy, -1.0, 1.0))  # (B, max_image_count)

    azm_degree = torch.rad2deg(azm_of_arrival)  # (B, max_image_count)
    ele_degree = torch.rad2deg(ele_of_arrival)  # (B, max_image_count)

    # Flatten for batch HRIR lookup
    flat_azm = azm_degree.view(-1)  # (B * max_image_count,)
    flat_ele = ele_degree.view(-1)  # (B * max_image_count,)

    left_hrirs, right_hrirs = h_rir.angle_batch(flat_azm, flat_ele)

    left_hrirs = left_hrirs.view(
        B, max_image_count, hrir_len
    )  # (B, max_image_count, hrir_len)
    right_hrirs = right_hrirs.view(B, max_image_count, hrir_len)

    weighted_left = left_hrirs * gains.unsqueeze(-1)
    weighted_right = right_hrirs * gains.unsqueeze(-1)

    weighted_left = weighted_left * valid_mask.unsqueeze(-1).float()
    weighted_right = weighted_right * valid_mask.unsqueeze(-1).float()

    brir_high = torch.zeros(B, 2, max_rir_length_high, device=device)

    time_offsets = torch.arange(hrir_len, device=device)

    target_indices = delays.unsqueeze(-1) + time_offsets.unsqueeze(0).unsqueeze(0)

    valid_indices_mask = (target_indices >= 0) & (target_indices < max_rir_length_high)
    valid_indices_mask = valid_indices_mask & valid_mask.unsqueeze(-1)

    target_indices = torch.clamp(target_indices, 0, max_rir_length_high - 1)

    batch_idx = (
        torch.arange(B, device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(B, max_image_count, hrir_len)
    )

    flat_batch_idx = batch_idx.reshape(-1)  # (B * max_image_count * hrir_len,)
    flat_target_idx = target_indices.reshape(-1)  # (B * max_image_count * hrir_len,)
    flat_left = (weighted_left * valid_indices_mask.float()).reshape(-1)
    flat_right = (weighted_right * valid_indices_mask.float()).reshape(-1)

    linear_idx_left = flat_batch_idx * max_rir_length_high + flat_target_idx
    linear_idx_right = flat_batch_idx * max_rir_length_high + flat_target_idx

    brir_left_flat = brir_high[:, 0, :].reshape(-1)  # (B * max_rir_length_high,)
    brir_right_flat = brir_high[:, 1, :].reshape(-1)

    brir_left_flat.index_add_(0, linear_idx_left, flat_left.float())
    brir_right_flat.index_add_(0, linear_idx_right, flat_right.float())

    brir_high[:, 0, :] = brir_left_flat.view(B, max_rir_length_high)
    brir_high[:, 1, :] = brir_right_flat.view(B, max_rir_length_high)

    brir_high = highpass_biquad(brir_high, hrir_sr, 80.0)

    brir_final = downsampler(brir_high)

    valid_after_dry = (target_sr * t60).long()
    return brir_final, valid_after_dry
