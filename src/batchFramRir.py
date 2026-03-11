"""
Batched version of FramRIR implementation for GPU-accelerated binaural room impulse response generation.

Based on the original FramRIR implementation from Tencent AI Lab by Rongzhi Gu, Yi Luo.

My Implmentation will use the same framework as FramRIR for genreating reflection directions but instead i will make a binaural room impulse response.

GITHUB: https://github.com/tencent-ailab/FRA-RIR/blob/main/FRAM_RIR.py

CITATION: Luo, Y., & Gu, R. (2024, April). Fast random approximation of multi-channel room impulse response. In 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW) (pp. 449-454). IEEE.
"""

import torch
from torch import Tensor
from torchaudio.functional import allpass_biquad, highpass_biquad, lowpass_biquad
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
def compute_late_transition(room_dim, velocity=343.0):
    """
    Computes the transition zone for early to late reverberation based on
    the echo density thresholds which show us when the late reverb statistically emerges highlighted in Abel & Huang (2006) that then get plugged into
    Kuttruff, Heinrich's formula for the average rate of refections received in a point of a real rectangular or ISM.

    original formula: 4π * c^3 * t^2 / V = average rate of received reflections.
    we solve for time instead.
    """
    # Calculate volume from room dimensions
    V = room_dim[:, 0] * room_dim[:, 1] * room_dim[:, 2]

    # The constant denominator for the expanding sound sphere: 4π * c^3
    denominator = 4.0 * torch.pi * (velocity**3)

    # --- Onset of Transition ---
    # Reflections start to overlap significantly, but aren't fully Gaussian.
    # The paper notes a statistical field begins forming around 2000 echoes/s.
    rho_onset = 2000
    t_onset = torch.sqrt(rho_onset * V / denominator)

    # --- Complete Late Field ---
    # The threshold where echoes exceed ~4000/s and can be treated as
    # statistically independent (Gaussian noise). The texture becomes "sandy".
    rho_late = 4000.0
    t_late = torch.sqrt(rho_late * V / denominator)

    # Clamp to practical minimum/maximum times in seconds to prevent
    # mathematical extremes in very small or massive virtual spaces.
    t_onset = t_onset.clamp(min=0.010, max=0.080)
    t_late = t_late.clamp(min=0.025, max=0.150)

    # Ensure the late time is strictly after the onset
    t_late = torch.max(t_late, t_onset * 1.3)

    return t_onset, t_late


# ALLPASS_CENTER_FREQS = [250.0, 700.0, 1800.0, 4500.0, 9000.0, 16000.0]


# def apply_allpass_decorrelation(
#     hrirs: Tensor,
#     blend: Tensor,
#     sr: int,
#     center_freqs: list[float] = ALLPASS_CENTER_FREQS,
#     q_max: float = 50.0,
#     q_min: float = 0.2,
# ) -> Tensor:
#     """
#     Apply cascaded allpass decorrelation to HRIRs using torchaudio.

#     Args:
#         hrirs: (B, chunk_size, hrir_len) — the HRIR waveforms
#         blend: (B, chunk_size) — decorrelation strength, 0=identity, 1=full
#         sr: sample rate of the HRIRs
#         center_freqs: list of center frequencies for allpass sections
#         q_max: Q at blend=0 (near identity — very narrow phase shift)
#         q_min: Q at blend=1 (broadband phase shift — maximum decorrelation)

#     Returns:
#         Tensor: decorrelated HRIRs, same shape as input
#     """
#     B, chunk_size, hrir_len = hrirs.shape

#     # Flatten to (B*chunk_size, hrir_len) for allpass_biquad
#     flat_hrirs = hrirs.reshape(-1, hrir_len)

#     n_levels = 8  # 8 discrete decorrelation levels (0 = identity, 7 = max)
#     blend_flat = blend.reshape(-1)  # (B*chunk_size,)

#     # Quantize blend to levels
#     level_indices = (blend_flat * (n_levels - 1)).round().long().clamp(0, n_levels - 1)

#     output = flat_hrirs.clone()

#     for level in range(n_levels):
#         mask = level_indices == level
#         if not mask.any():
#             continue

#         # Skip level 0 entirely — these are early reflections, no filtering
#         if level == 0:
#             continue

#         level_blend = level / (n_levels - 1)  # 0.0 to 1.0
#         level_q = q_max - (q_max - q_min) * (level_blend**2)

#         subset = flat_hrirs[mask]  # (N_subset, hrir_len)

#         # Cascade allpass sections at different frequencies
#         for freq in center_freqs:
#             # Skip if frequency is above Nyquist
#             if freq >= sr / 2:
#                 continue
#             subset = allpass_biquad(subset, sr, freq, level_q)

#         output[mask] = subset

#     return output.reshape(B, chunk_size, hrir_len)


def compute_reflection_lowpass_freq(
    dist: Tensor,
    t60: Tensor,
    velocity: float = 343.0,
    f_ref: float = 12000.0,
    air_absorption_coeff: float = 0.02,
) -> Tensor:
    """
    Compute per-reflection lowpass cutoff frequency based on distance traveled.

    Models combined air absorption and surface absorption frequency dependence.

    Air absorption in dB/m ≈ air_absorption_coeff * f² / 1e6
    This means high frequencies lose energy faster with distance.

    We model this as a lowpass filter whose cutoff drops with distance:
        f_cutoff = f_ref * exp(-alpha * dist)

    where alpha controls how fast HF rolls off with distance.

    Args:
        dist: (B, chunk_size) — total path length of each reflection in meters
        t60: (B,) — reverberation time
        velocity: speed of sound
        f_ref: starting cutoff frequency for zero-distance reflection
        air_absorption_coeff: controls HF decay rate with distance

    Returns:
        cutoff_freqs: (B, chunk_size) — lowpass cutoff frequency per reflection in Hz
    """
    # Scale absorption by room liveness — lively rooms (high T60) have
    # less absorptive surfaces, so frequency-dependent decay is slower
    # Dry rooms (low T60) have more absorption, faster HF decay
    absorption_rate = air_absorption_coeff / t60.unsqueeze(1)  # (B, chunk_size)

    cutoff = f_ref * torch.exp(-absorption_rate * dist)

    # Clamp to reasonable range
    cutoff = cutoff.clamp(min=200.0, max=f_ref)

    return cutoff


def apply_freq_dependent_decay(
    hrirs: Tensor,
    cutoff_freqs: Tensor,
    sr: int,
) -> Tensor:
    """
    Apply per-reflection lowpass filtering via frequency-domain multiplication.
    Fully vectorized, no loops, no discretization.

    Uses a first-order Butterworth magnitude response:
        |H(f)|² = 1 / (1 + (f / f_c)²)

    Args:
        hrirs: (B, chunk_size, hrir_len)
        cutoff_freqs: (B, chunk_size) in Hz
        sr: sample rate

    Returns:
        filtered HRIRs, same shape
    """
    B, chunk_size, L = hrirs.shape
    n_bins = L // 2 + 1

    # Frequency axis in Hz: (n_bins,)
    freqs = torch.linspace(0, sr / 2, n_bins, device=hrirs.device)  # (n_bins,)

    # Reshape for broadcasting: freqs (1, 1, n_bins), cutoffs (B, chunk_size, 1)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    fc = cutoff_freqs.unsqueeze(-1)

    # First-order Butterworth magnitude response
    H_mag = 1.0 / torch.sqrt(1.0 + (freqs / fc).pow(2))

    # Apply in frequency domain
    spec = torch.fft.rfft(hrirs, dim=-1)  # (B, chunk_size, n_bins)
    filtered = torch.fft.irfft(spec * H_mag, n=L, dim=-1)

    return filtered


# t_mix ≈ sqrt(V) / c
# V is volume and c is speed of sound which is 343 m/s
@torch.no_grad()
def batch_fram_brir(
    target_sr: int,
    # t60: Tensor,
    h_rir: RIRTensor,
    hrir_sr: int = 96000,
    mic_pos: Tensor = None,
    room_dim: Tensor = None,
    src_pos: Tensor = None,
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

    if mic_pos.shape[0] != room_dim.shape[0] or mic_pos.shape[0] != src_pos.shape[0]:
        raise ValueError("mic_pos, room_dim, src_pos must have the same batch size")

    # Move tensors to device
    mic_pos = mic_pos.to(device).float()
    src_pos = src_pos.to(device).float()
    room_dim = room_dim.to(device).float()

    B = mic_pos.shape[0]

    oversample = 1
    working_sr = hrir_sr * oversample

    downsampler = Resample(orig_freq=working_sr, new_freq=target_sr).to(device)

    hrir_upsampler = Resample(orig_freq=hrir_sr, new_freq=working_sr).to(device)

    if target_sr > hrir_sr:
        print(
            "Warning: Target sample rate is higher than HRIR sample rate, this can cause worse timing accuracy"
        )

    if hrir_sr == 96000:
        hrir_len = 512
    else:
        hrir_len = 256

    hrir_len_up = hrir_len * oversample

    t60 = torch.empty(B, device=device).uniform_(1.5, 1.5)
    # density varies ±15% for room character diversity
    density = torch.empty(B, device=device).uniform_(2000, 10000)
    image_counts = (density * t60).int()  # (B,)a
    print("Image counts per batch item:", image_counts)
    print("t60", t60)
    print("Density", density)

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
    max_rir_length_high = int(working_sr * max_t60)

    # For each batch, we need to sample distances
    # Create distance ranges and probabilities for each batch element
    # We'll use the maximum possible length and mask invalid samples

    # dist_range end values: velocity * t60 / direct_dist - 1 for each batch
    dist_range_end = velocity * t60 / direct_dist - 1  # (B,)

    # RIR lengths for each batch element
    rir_length_high = (working_sr * t60).long()  # (B,)

    # Create output BRIR tensor
    brir_high = torch.zeros(B, 2, max_rir_length_high, device=device)

    # Time offsets for scatter (reuse across chunks)
    time_offsets = torch.arange(hrir_len_up, device=device)

    # ===== CHUNKED PROCESSING =====
    # Process reflections in chunks to limit peak memory
    # t_onset, t_complete = compute_late_transition(room_dim, velocity=velocity)

    # onset_samples = (t_onset * working_sr).long()  # (B,)
    # complete_samples = (t_complete * working_sr).long()

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
        normalized_dist_samples = u.pow(
            1.0 / 3.0
        )  # inverse of cubic CDF gives quadratic PDF
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
        gains = reflect_coef.unsqueeze(1).pow(reflect_ratio) / dist * 2

        # Compute time delays
        path_diff = dist - direct_dist.unsqueeze(1)
        delays = torch.ceil(path_diff * working_sr / velocity).long()

        # Valid mask for this chunk
        valid_delay_mask = (delays + hrir_len_up) < rir_length_high.unsqueeze(1)
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
        left_hrirs = left_hrirs.to(dtype=torch.float32).view(B, chunk_size, hrir_len)
        right_hrirs = right_hrirs.to(dtype=torch.float32).view(B, chunk_size, hrir_len)

        left_hrirs = hrir_upsampler(left_hrirs)
        right_hrirs = hrir_upsampler(right_hrirs)

        # samples_past_onset = (delays - onset_samples.unsqueeze(1)).float()
        # transition_width = (complete_samples - onset_samples).unsqueeze(1).float()
        # t_normalized = (samples_past_onset / transition_width).clamp(0.0, 1.0)
        # blend = t_normalized.pow(2)

        # is_late = blend >= 1.0
        # is_early = ~is_late
        #
        cutoff_freqs = compute_reflection_lowpass_freq(dist, t60, velocity)

        left_hrirs = apply_freq_dependent_decay(left_hrirs, cutoff_freqs, working_sr)
        right_hrirs = apply_freq_dependent_decay(right_hrirs, cutoff_freqs, working_sr)

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
            .expand(B, chunk_size, hrir_len_up)
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
    brir_high = highpass_biquad(brir_high, working_sr, 80.0)
    # Downsample to target sample rate
    # Resample expects (..., time) format
    brir_final = downsampler(brir_high)

    valid_after_dry = (target_sr * t60).long()
    return brir_final, valid_after_dry
