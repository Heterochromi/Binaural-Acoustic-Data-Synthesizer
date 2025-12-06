"""
Original FramRIR implementation from tencent AI lab by Rongzhi Gu, Yi Luo is refrenced below.
My Implmentation will use the same framework as FramRIR for genreating reflection directions but instead i will make a binaural room impulse response.

GITHUB: https://github.com/tencent-ailab/FRA-RIR/blob/main/FRAM_RIR.py

CITATION: Luo, Y., & Gu, R. (2024, April). Fast random approximation of multi-channel room impulse response. In 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW) (pp. 449-454). IEEE.
"""

from typing import Literal, Tuple

import torch
from torch import Tensor
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample

from .rirTensor import RIRTensor


def fram_brir(
    target_sr: int,
    t60: float,
    h_rir: RIRTensor,
    hrir_sr: int = 96000,
    mic_pos: Tensor = torch.tensor([1, 1, 1]),
    room_dim: Tensor = torch.tensor([4, 4, 4]),
    src_pos: Tensor = torch.tensor([1, 1, 1]),
    n_reflection: Tuple[int, int] = (100, 500),
    a: float = -2.0,
    b: float = 2.0,
    tau: float = 0.25,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    all source and microphone pick up patterns will technically be OMNI at first but then changed when mixed HRIR to match direction.
    this function is time invariant therefor it can not work with multi source scenarios with different timings,
    a simple soution is to get the reverb only IR or (BRIR) then apply it to the audio signal and mix it with a HRIR of the dry source.

    Args:
        t60 (float): The reverberation time in seconds.
        h_rir (RIRTensor): The head-related impulse class that generates the HRIRs for the reflections.
        mic_pos (Tensor): The position of the microphone/receiver.
        room_dim (Tensor): The dimensions of the room.
        src_pos (Tensor): The position of the sound source.
        n_reflection (Tuple[int, int]): Range to sample from a number of reflections.
        a (float): Minimum of the random perturbation.
        b (float): Maximum of the random perturbation.
        tau (float): The time constant for the exponential decay(distance shrinkage factor).
        device (torch.device): The device to use.

        Returns:
            Tensor: 2 channel reverb only BRIR (reverb tail).

    """
    mic_pos = mic_pos.to(device)
    src_pos = src_pos.to(device)
    room_dim = room_dim.to(device)
    downsampler = Resample(orig_freq=hrir_sr, new_freq=target_sr).to(device)

    if target_sr > hrir_sr:
        print(
            "Warning: Target sample rate is lower than HRIR sample rate, this can cause worse timing accuracy"
        )

    if hrir_sr == 96000:
        hrir_len = 512
    else:
        hrir_len = 256

    # randomly sample number of reflections
    image_count = torch.randint(
        low=n_reflection[0], high=n_reflection[1], size=(1,), device=device
    ).item()

    #  geometric environment set up
    volume_to_surface_area_ratio = 1.0 / (
        2 * (1.0 / room_dim[0] + 1.0 / room_dim[1] + 1.0 / room_dim[2])
    )
    eps = torch.finfo(torch.float32).eps
    velocity = 343.0

    direct_dist = torch.sqrt((mic_pos - src_pos).pow(2).sum(dim=-1) + eps)
    t60_tensor = torch.tensor(t60, device=device).float()

    reflect_coef = torch.sqrt(
        (1 - (1 - torch.exp(-0.16 * volume_to_surface_area_ratio / t60_tensor)).pow(2))
    )
    reflect_max = (torch.log10(torch.tensor(velocity * t60)) - 3) / torch.log10(
        reflect_coef
    )

    # Distances
    dist_range = torch.linspace(
        1.0, velocity * t60 / direct_dist - 1, int(hrir_sr * t60), device=device
    )
    dist_prob = torch.linspace(0, 1.0, int(hrir_sr * t60), device=device)
    dist_prob /= dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(num_samples=image_count, replacement=True)
    dist_nearest_ratio = dist_range[dist_select_idx]

    # Directions
    azm = torch.empty(image_count, device=device).uniform_(-torch.pi, torch.pi)
    ele = torch.empty(image_count, device=device).uniform_(-torch.pi / 2, torch.pi / 2)

    unit_3d = torch.stack(
        [
            torch.sin(ele) * torch.cos(azm),
            torch.sin(ele) * torch.sin(azm),
            torch.cos(ele),
        ],
        -1,
    )

    image2nearest_dis = dist_nearest_ratio * direct_dist
    image_position = mic_pos.unsqueeze(0) + image2nearest_dis.unsqueeze(-1) * unit_3d
    dist = torch.sqrt((mic_pos.unsqueeze(0) - image_position).pow(2).sum(-1) + eps)

    # Gain decays
    reflect_ratio = (dist / (velocity * t60_tensor)) * (reflect_max - 1) + 1
    reflect_pertub = torch.empty(image_count, device=device).uniform_(
        a, b
    ) * dist_nearest_ratio.pow(tau)
    reflect_ratio = torch.maximum(
        reflect_ratio + reflect_pertub, torch.ones(image_count, device=device)
    )
    gains = reflect_coef.pow(reflect_ratio) / dist

    # Time delays
    path_diff = dist - direct_dist
    delays = torch.ceil(path_diff * hrir_sr / velocity).long()

    rir_length_high = int(hrir_sr * t60_tensor)

    valid_mask = (delays + hrir_len) < rir_length_high

    delays = delays[valid_mask]
    gains = gains[valid_mask]

    valid_image_pos = image_position[valid_mask]
    vec_mic_to_img = valid_image_pos - mic_pos.unsqueeze(0)

    # azm_valid = azm[valid_mask]
    # ele_valid = ele[valid_mask]

    # HRIR Generation

    radius = torch.sqrt(vec_mic_to_img.pow(2).sum(dim=-1) + eps)
    ux = vec_mic_to_img[..., 0] / radius
    uy = vec_mic_to_img[..., 1] / radius
    uz = vec_mic_to_img[..., 2] / radius

    azm_of_arrival = torch.atan2(uy, ux)
    ele_of_arrival = torch.asin(uz)

    azm_degree = torch.rad2deg(azm_of_arrival)
    ele_degree = torch.rad2deg(ele_of_arrival)

    left_hrirs, right_hrirs = h_rir.angle_batch(azm_degree, ele_degree)

    time_offsets = torch.arange(hrir_len, device=device).unsqueeze(0)

    target_indices_matrix = delays.unsqueeze(1) + time_offsets
    flat_indices = target_indices_matrix.view(-1)

    weighted_left = left_hrirs * gains.unsqueeze(1)
    weighted_right = right_hrirs * gains.unsqueeze(1)

    flat_left = weighted_left.view(-1)
    flat_right = weighted_right.view(-1)

    brir_high = torch.zeros((2, rir_length_high), device=device)
    brir_high[0].index_add_(0, flat_indices, flat_left.float())
    brir_high[1].index_add_(0, flat_indices, flat_right.float())

    # Apply highpass filter to remove low-frequency noise and then downsample to target sample rate, just like original FRAM
    brir_high = highpass_biquad(brir_high, hrir_sr, 80.0)
    brir_final = downsampler(brir_high)

    return brir_final


"""!
Author: Rongzhi Gu, Yi Luo
Copyright: Tencent AI Lab
"""

'''
import numpy as np
import torch
from torchaudio.transforms import Resample
from torchaudio.functional import highpass_biquad


def calc_cos(orientation_rad):
    """
    cos_theta: tensor, [azimuth, elevation] with shape [..., 2]
    return: [..., 3], a=-2.0, b=2.0, tau=0.25,
    """
    return torch.stack([torch.cos(orientation_rad[...,0]*torch.sin(orientation_rad[...,1])),
                        torch.sin(orientation_rad[...,0]*torch.sin(orientation_rad[...,1])),
                        torch.cos(orientation_rad[...,1])], -1)


def freq_invariant_decay_func(cos_theta, pattern='cardioid'):
    """
    cos_theta: tensor
    Return:
    amplitude: tensor with same shape as cos_theta
    """

    if pattern == 'cardioid':
        return 0.5 + 0.5 * cos_theta

    elif pattern == 'omni':
        return torch.ones_like(cos_theta)

    elif pattern == 'bidirectional':
        return cos_theta

    elif pattern == 'hyper_cardioid':
        return 0.25 + 0.75 * cos_theta

    elif pattern == 'sub_cardioid':
        return 0.75 + 0.25 * cos_theta

    elif pattern == 'half_omni':
        c = torch.clamp(cos_theta, 0)
        c[c > 0] = 1.0
        return c
    else:
        raise NotImplementedError

def freq_invariant_src_decay_func(mic_pos, src_pos, src_orientation_rad, pattern='cardioid'):
    """
    mic_pos: [n_mic, 3] (tensor)
    src_pos: [n_src, 3] (tensor)
    src_orientation_rad: [n_src, 2] (tensor), elevation, azimuth

    Return:
    amplitude: [n_mic, n_src, n_image]
    """
    # Steering vector of source(s)
    orV_src = calc_cos(src_orientation_rad).unsqueeze(0)  # [nsrc, 3]

    # receiver to src vector
    rcv_to_src_vec = mic_pos.unsqueeze(1) - src_pos.unsqueeze(0) # [n_mic, n_src, 3]

    cos_theta = (rcv_to_src_vec * orV_src).sum(-1)  # [n_mic, n_src]
    cos_theta /= torch.sqrt(rcv_to_src_vec.pow(2).sum(-1))
    cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))

    return freq_invariant_decay_func(cos_theta, pattern)

def freq_invariant_mic_decay_func(mic_pos, img_pos, mic_orientation_rad, pattern='cardioid'):
    """
    mic_pos: [n_mic, 3] (tensor)
    img_pos: [n_src, n_image, 3] (tensor)
    mic_orientation_rad: [n_mic, 2] (tensor), azimuth, elevation

    Return:
    amplitude: [n_mic, n_src, n_image]
    """
    # Steering vector of source(s)
    orV_src = calc_cos(mic_orientation_rad)  # [nmic, 3]
    orV_src = orV_src.view(-1,1,1,3)  # [n_mic, 1, 1, 3]

    # image to receiver vector
    # [1, n_src, n_image, 3] - [n_mic, 1, 1, 3] => [n_mic, n_src, n_image, 3]
    img_to_rcv_vec = img_pos.unsqueeze(0) - mic_pos.unsqueeze(1).unsqueeze(1)

    cos_theta = (img_to_rcv_vec * orV_src).sum(-1)  # [n_mic, n_src, n_image]
    cos_theta /= torch.sqrt(img_to_rcv_vec.pow(2).sum(-1))
    cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))

    return freq_invariant_decay_func(cos_theta, pattern)

def FRAM_RIR(mic_pos, sr, t60, room_dim, src_pos,
             num_src=1, direct_range=(-6, 50),
             n_image=(1024, 4097), a=-2.0, b=2.0, tau=0.25,
             src_pattern='omni', src_orientation_rad=None,
             mic_pattern='omni', mic_orientation_rad=None,
            ):
    """Fast Random Appoximation of Multi-channel Room Impulse Response (FRAM-RIR)
    """

    # sample image
    image = np.random.choice(range(n_image[0], n_image[1]))

    R = torch.tensor(1. / (2 * (1./room_dim[0]+1./room_dim[1] + 1./room_dim[2])))

    eps = np.finfo(np.float16).eps
    mic_position = torch.from_numpy(mic_pos)
    src_position = torch.from_numpy(src_pos)  # [nsource, 3]
    n_mic = mic_position.shape[0]
    num_src = src_position.shape[0]

    # [nmic, nsource]
    direct_dist = ((mic_position.unsqueeze(1) - src_position.unsqueeze(0)).pow(2).sum(-1) + 1e-3).sqrt()
    # [nsource]
    nearest_dist, nearest_mic_idx = direct_dist.min(0)
    # [nsource, 3]
    nearest_mic_position = mic_position[nearest_mic_idx]

    ns = n_mic * num_src
    ratio = 64
    sample_sr = sr*ratio
    velocity = 340.
    t60 = torch.tensor(t60)

    direct_idx = torch.ceil(direct_dist * sample_sr / velocity).long().view(ns,)
    rir_length = int(np.ceil(sample_sr * t60))

    resample1 = Resample(sample_sr, sample_sr//int(np.sqrt(ratio)))
    resample2 = Resample(sample_sr//int(np.sqrt(ratio)), sr)

    reflect_coef = (1 - (1 - torch.exp(-0.16*R/t60)).pow(2)).sqrt()
    dist_range = [torch.linspace(1., velocity*t60/nearest_dist[i]-1, rir_length) for i in range(num_src)]

    dist_prob = torch.linspace(0., 1., rir_length)
    dist_prob /= dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(num_samples=int(image*num_src), replacement=True).view(num_src, image)

    dist_nearest_ratio = torch.stack(
        [dist_range[i][dist_select_idx[i]] for i in range(num_src)], 0)

    # apply different dist ratios to mirophones
    azm = torch.FloatTensor(num_src, image).uniform_(-np.pi, np.pi)
    ele = torch.FloatTensor(num_src, image).uniform_(-np.pi/2, np.pi/2)
    # [nsource, nimage, 3]
    unit_3d = torch.stack([torch.sin(ele) * torch.cos(azm), torch.sin(ele) * torch.sin(azm), torch.cos(ele)], -1)
    # [nsource] x [nsource, T] x [nsource, nimage, 3] => [nsource, nimage, 3]
    image2nearest_dist = nearest_dist.view(-1, 1, 1) * dist_nearest_ratio.unsqueeze(-1)
    image_position = nearest_mic_position.unsqueeze(1) + image2nearest_dist * unit_3d
    # [nmic, nsource, nimage]
    dist = ((mic_position.view(-1, 1, 1, 3) - image_position.unsqueeze(0)).pow(2).sum(-1) + 1e-3).sqrt()

    # reflection perturbation
    reflect_max = (torch.log10(velocity*t60) - 3) / torch.log10(reflect_coef)
    reflect_ratio = (dist / (velocity*t60)) * (reflect_max.view(1, -1, 1) - 1) + 1
    reflect_pertub = torch.FloatTensor(num_src, image).uniform_(a, b) * dist_nearest_ratio.pow(tau)
    reflect_ratio = torch.maximum(reflect_ratio + reflect_pertub.unsqueeze(0), torch.ones(1))

    # [nmic, nsource, 1 + nimage]
    dist = torch.cat([direct_dist.unsqueeze(2), dist], 2)
    reflect_ratio = torch.cat([torch.zeros(n_mic, num_src, 1), reflect_ratio], 2)

    delta_idx = torch.minimum(torch.ceil(dist * sample_sr / velocity), torch.ones(1)*rir_length-1).long().view(ns, -1)
    delta_decay = reflect_coef.pow(reflect_ratio) / dist

    #################################
    # source orientation simulation #
    #################################
    if src_pattern != 'omni':
        # randomly sample each image's relative orientation with respect to the original source
        # equivalent to a random decay corresponds to the source's orientation pattern decay
        img_orientation_rad = torch.FloatTensor(num_src, image, 2).uniform_(-np.pi, np.pi)
        img_cos_theta = torch.cos(img_orientation_rad[...,0]) * torch.cos(img_orientation_rad[...,1])   # [nsource, nimage]
        img_orientation_decay = freq_invariant_decay_func(img_cos_theta, pattern=src_pattern)  # [nsource, nimage]

        # direct path orientation should use the provided parameter
        if src_orientation_rad is None:
            # assume random orientation if not given
            src_orientation_azi = torch.FloatTensor(num_src).uniform_(-np.pi, np.pi)
            src_orientation_ele = torch.FloatTensor(num_src).uniform_(-np.pi, np.pi)
            src_orientation_rad = torch.stack([src_orientation_azi, src_orientation_ele], -1)
        else:
            src_orientation_rad = torch.from_numpy(src_orientation_rad) # [nsource, 2]

        src_orientation_decay = freq_invariant_src_decay_func(mic_position, src_position,
                                                            src_orientation_rad, pattern=src_pattern)  # [nmic, nsource]
        # apply decay
        delta_decay[:,:,0] *= src_orientation_decay
        delta_decay[:,:,1:] *= img_orientation_decay.unsqueeze(0)

    if mic_pattern != 'omni':
        # mic orientation simulation #
        # when not given, assume that all mics facing up (positive z axis)
        if mic_orientation_rad is None:
            mic_orientation_rad = torch.stack([torch.zeros(n_mic), torch.zeros(n_mic)], -1)  # [nmic, 2]
        else:
            mic_orientation_rad = torch.from_numpy(mic_orientation_rad)
        all_src_img_pos = torch.cat((src_position.unsqueeze(1), image_position), 1) # [nsource, nimage+1, 3]
        mic_orientation_decay = freq_invariant_mic_decay_func(mic_position, all_src_img_pos, mic_orientation_rad, pattern=mic_pattern)  # [nmic, nsource, nimage+1]
        # apply decay
        delta_decay *= mic_orientation_decay

    rir = torch.zeros(ns, rir_length)
    delta_decay = delta_decay.view(ns, -1)
    for i in range(ns):
        remainder_idx = delta_idx[i]
        valid_mask = np.ones(len(remainder_idx))
        while np.sum(valid_mask) > 0:
            valid_remainder_idx, unique_remainder_idx = np.unique(remainder_idx, return_index=True)
            rir[i][valid_remainder_idx] += delta_decay[i][unique_remainder_idx] * valid_mask[unique_remainder_idx]
            valid_mask[unique_remainder_idx] = 0
            remainder_idx[unique_remainder_idx] = 0

    direct_mask = torch.zeros(ns, rir_length).float()

    for i in range(ns):
        direct_mask[i, max(direct_idx[i]+sample_sr*direct_range[0]//1000, 0):
                    min(direct_idx[i]+sample_sr*direct_range[1]//1000, rir_length)] = 1.

    rir_direct = rir * direct_mask

    all_rir = torch.stack([rir, rir_direct], 1).view(ns*2, -1)
    rir_downsample = resample1(all_rir)
    rir_hp = highpass_biquad(rir_downsample, sample_sr // int(np.sqrt(ratio)), 80.)
    rir = resample2(rir_hp).float().view(n_mic, num_src, 2, -1)

    return rir[:, :, 0].data.numpy(), rir[:, :, 1].data.numpy()
'''
