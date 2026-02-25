from typing import Literal

import torch

# Material Type,Center Freq. (fc​) Range,Total Width Range (Hz),Dip Depth Range,Physical Characteristics
# "Heavy/Masonry (Concrete, Brick, Cinderblock)",100 Hz – 300 Hz,100 Hz – 500 Hz,5 dB – 10 dB,Shallow and narrow in linear Hz. High mass and high internal damping kill the dip quickly.
# "Wood/Timber (Plywood, OSB, Solid Doors)","1,000 Hz – 2,000 Hz","800 Hz – 2,500 Hz",10 dB – 15 dB,Wood is stiff but light. The dip sits right in the lower midrange.
# "Standard Walls (Drywall, Gypsum, Plaster)","2,500 Hz – 3,200 Hz","1,500 Hz – 3,500 Hz",8 dB – 15 dB,"The classic residential wall. Less rigid, pushing the dip higher up into the sensitive hearing range."
# "Rigid/Thin (Glass, Sheet Metal, Acrylic)","1,000 Hz – 4,000 Hz","1,000 Hz – 4,000 Hz",12 dB – 20 dB,"Severe, sharp, and deep. Low internal damping means the material rings like a bell at the coincidence frequency."
herustic_random_occlusion_params = {
    "Heavy": {
        "crit_freq_hz": [100, 300],
        "attenuation_dip_strength_db": [5, 10],
        "crit_width_hz": [100, 500],
    },
    "Wood": {
        "crit_freq_hz": [1000, 2000],
        "attenuation_dip_strength_db": [10, 15],
        "crit_width_hz": [800, 2500],
    },
    "Standard": {
        "crit_freq_hz": [2500, 3200],
        "attenuation_dip_strength_db": [8, 15],
        "crit_width_hz": [1500, 3500],
    },
    "Rigid": {
        "crit_freq_hz": [1000, 4000],
        "attenuation_dip_strength_db": [12, 20],
        "crit_width_hz": [1000, 4000],
    },
}


@torch.no_grad()
def apply_occlusion_frequency_domain(
    waveforms: torch.Tensor,  # [B, T]
    sample_rate: int,
    herustic_occlusion_type: Literal[
        "Heavy", "Wood", "Standard", "Rigid", "Random", None
    ] = None,
    same_wall_across_batch: bool = True,
    crit_freq_hz: float = 4000.0,
    crit_width_hz: float = 1000.0,
    attenuation_dip_strength_db: float = 6.0,
    probability: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    """
    Simple frequency-domain (single panel sound transmission loss) occlusion filter using direct mask multiplication.

    Args:
        waveforms (torch.Tensor): Input audio waveforms [B, T].
        sample_rate (int): Sample rate of the audio signal.
        max_attenuation_db (float): Maximum attenuation in decibels at nyquist.
        crit_freq_hz (float): Critical frequency in Hz for the dip (move down for thicker wall or denser material, move up for thinner wall or lighter material).
        crit_width_hz (float): Width of the critical frequency dip in Hz.
        attenuation_dip_strength_db (float): Strength of the dip at critical frequency in dB.
        probability (float): Probability of applying occlusion per waveform.
        device (torch.device): Device to run the computation on.

    Returns:
        torch.Tensor: Filtered audio waveforms [B, T].
        torch.Tensor: Mask indicating which waveforms were occluded [B].
    """
    if probability < 0 or probability > 1:
        raise ValueError("probability must be between 0 and 1")

    if herustic_occlusion_type is not None:
        print(
            f"occlusion type is set, therefor crit_freq_hz,crit_width_hz, attenuation_dip_strength_db will be ignored and set according to the occlusion type of {herustic_occlusion_type}"
        )
    else:
        print(
            "same_wall_across_batch will always be true because occlusion type is None"
        )
        same_wall_across_batch = True

    waveforms = waveforms.to(device)
    original_waveforms = waveforms.clone()
    batch_size = waveforms.shape[0]
    n_samples = waveforms.shape[1]

    n_fft = n_samples

    apply_mask = (torch.rand(batch_size, device=device) < probability).float()

    # Sample occlusion parameters from heuristic ranges if applicable
    if herustic_occlusion_type is not None:
        material_types = list(herustic_random_occlusion_params.keys())

        if herustic_occlusion_type == "Random":
            if same_wall_across_batch:
                chosen_idx = torch.randint(len(material_types), (1,)).item()
                chosen = material_types[chosen_idx]
                p = herustic_random_occlusion_params[chosen]
                crit_freq_hz = (
                    torch.empty(1, device=device).uniform_(*p["crit_freq_hz"]).item()
                )
                crit_width_hz = (
                    torch.empty(1, device=device).uniform_(*p["crit_width_hz"]).item()
                )
                attenuation_dip_strength_db = (
                    torch.empty(1, device=device)
                    .uniform_(*p["attenuation_dip_strength_db"])
                    .item()
                )
            else:
                type_indices = torch.randint(len(material_types), (batch_size,))
                crit_freq_hz_list = []
                crit_width_hz_list = []
                atten_list = []
                for i in range(batch_size):
                    chosen = material_types[type_indices[i].item()]
                    p = herustic_random_occlusion_params[chosen]
                    crit_freq_hz_list.append(
                        torch.empty(1).uniform_(*p["crit_freq_hz"]).item()
                    )
                    crit_width_hz_list.append(
                        torch.empty(1).uniform_(*p["crit_width_hz"]).item()
                    )
                    atten_list.append(
                        torch.empty(1)
                        .uniform_(*p["attenuation_dip_strength_db"])
                        .item()
                    )
                crit_freq_hz = torch.tensor(crit_freq_hz_list, device=device).unsqueeze(
                    1
                )  # [B, 1]
                crit_width_hz = torch.tensor(
                    crit_width_hz_list, device=device
                ).unsqueeze(1)  # [B, 1]
                attenuation_dip_strength_db = torch.tensor(
                    atten_list, device=device
                ).unsqueeze(1)  # [B, 1]
        else:
            p = herustic_random_occlusion_params[herustic_occlusion_type]
            if same_wall_across_batch:
                crit_freq_hz = (
                    torch.empty(1, device=device).uniform_(*p["crit_freq_hz"]).item()
                )
                crit_width_hz = (
                    torch.empty(1, device=device).uniform_(*p["crit_width_hz"]).item()
                )
                attenuation_dip_strength_db = (
                    torch.empty(1, device=device)
                    .uniform_(*p["attenuation_dip_strength_db"])
                    .item()
                )
            else:
                crit_freq_hz = torch.empty(batch_size, 1, device=device).uniform_(
                    *p["crit_freq_hz"]
                )  # [B, 1]
                crit_width_hz = torch.empty(batch_size, 1, device=device).uniform_(
                    *p["crit_width_hz"]
                )  # [B, 1]
                attenuation_dip_strength_db = torch.empty(
                    batch_size, 1, device=device
                ).uniform_(*p["attenuation_dip_strength_db"])  # [B, 1]

    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device).unsqueeze(
        0
    )  # [1, F]

    # 1. Low frequency mask (0 to crit_freq_hz): 6dB per octave

    low_fc = crit_freq_hz  # Use crit_freq as the reference point
    low_freq_mask = low_fc / (low_fc + freqs)

    # 2. Critical frequency mask (gaussian dip at crit_freq_hz)
    sigma = crit_width_hz / 2.355  # FWHM to sigma
    gaussian = torch.exp(-0.5 * ((freqs - crit_freq_hz) / sigma) ** 2)
    attenuation_dip = 10.0 ** (-attenuation_dip_strength_db / 20.0)
    crit_freq_mask = 1 - gaussian * (attenuation_dip - 1)

    # 3. high frequency mask (after crit_freq_hz): 9dB per octave
    high_fc = crit_freq_hz  # Transition point at crit_freq
    high_freq_rolloff = (high_fc / (high_fc + freqs)) ** 1.5

    # Blend masks: use low_freq below crit, high_freq above crit
    transition_width = crit_width_hz
    transition = torch.sigmoid((freqs - crit_freq_hz) / (transition_width / 4))

    # Combine low and high frequency masks with smooth transition
    freq_response_mask = (
        1 - transition
    ) * low_freq_mask + transition * high_freq_rolloff

    # 4. Combined frequency mask (apply crit dip on top)
    freq_mask = freq_response_mask * crit_freq_mask  # [n_fft // 2 + 1]

    # 5. Apply mask in frequency domain
    # FFT of waveforms
    waveforms_fft = torch.fft.rfft(waveforms, n=n_fft)  # [B, n_fft // 2 + 1]

    # Multiply by frequency mask (freq_mask is [1, F] or [B, F], broadcasts with [B, F])
    filtered_fft = waveforms_fft * freq_mask  # [B, n_fft // 2 + 1]

    # Inverse FFT back to time domain
    filtered_waveforms = torch.fft.irfft(filtered_fft, n=n_fft)  # [B, n_fft]

    # Trim to original length
    filtered_waveforms = filtered_waveforms[:, :n_samples]

    # Blend between original and filtered based on apply_mask
    apply_mask_expanded = apply_mask.unsqueeze(-1)  # [B, 1]
    output = (
        apply_mask_expanded * filtered_waveforms
        + (1 - apply_mask_expanded) * original_waveforms
    )

    return output, apply_mask
