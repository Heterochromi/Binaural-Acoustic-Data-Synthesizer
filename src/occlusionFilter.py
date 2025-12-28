import torch
import torchaudio.functional as F_audio


@torch.no_grad()
def apply_occlusion(
    waveforms: torch.Tensor,  # [B, T]
    sample_rate: int,
    base_attenuation_db: float = 10.0,
    max_attenuation_db: float = 40.0,
    crit_freq_hz: float = 1500.0,
    crit_width_hz: float = 1000.0,
    attenuation_dip_strength_db: float = 5.0,
    probability: float = 1,
    device: torch.device = torch.device("cpu"),
):
    """
    This is a simple audio occlusion filter that tries to mimic real world single panel sound transmission loss.

    Args:
        waveforms (torch.Tensor): Input audio waveforms.
        sample_rate (int): Sample rate of the audio signal.
        base_attenuation_db (float): Base attenuation in decibels.
        max_attenuation_db (float): Maximum attenuation in decibels.
        crit_freq_hz (float): Critical frequency in Hz.
        crit_width_hz (float): Critical width in Hz.
        attenuation_dip_strength_db (float): Attenuation dip strength in decibels.
        probability (float): Probability of applying occlusion per waveform.
        device (torch.device): Device to run the computation on.

    Returns:
        torch.Tensor: Filtered audio waveform.
        torch.Tensor: Mask indicating which waveforms were occluded.
    """
    if probability < 0 or probability > 1:
        raise ValueError("probability must be between 0 and 1")

    waveforms = waveforms.to(device)
    original_waveforms = waveforms.clone()

    # Create mask for which waveforms to apply occlusion to [B, 1]
    batch_size = waveforms.shape[0] if waveforms.dim() > 1 else 1
    apply_mask = (
        (torch.rand(batch_size, device=device) < probability).float().unsqueeze(-1)
    )
    base_gain = 10.0 ** (-base_attenuation_db / 20.0)
    waveforms = waveforms * base_gain

    nyquist = sample_rate / 2

    target_gain = 10.0 ** (-max_attenuation_db / 20.0)
    fc = nyquist / torch.sqrt(torch.tensor((1 / target_gain**2) - 1))

    waveforms = F_audio.lowpass_biquad(
        waveforms,
        sample_rate,
        cutoff_freq=fc,
        Q=0.5,
    )

    Q_factor = crit_freq_hz / crit_width_hz
    waveforms = F_audio.equalizer_biquad(
        waveforms,
        sample_rate,
        center_freq=crit_freq_hz,
        gain=attenuation_dip_strength_db,
        Q=Q_factor,
    )

    # Blend between original and processed based on mask
    waveforms = apply_mask * waveforms + (1 - apply_mask) * original_waveforms

    return waveforms, apply_mask.squeeze(1)


@torch.no_grad()
def apply_occlusion_frequency_domain(
    waveforms: torch.Tensor,  # [B, T]
    sample_rate: int,
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

    waveforms = waveforms.to(device)
    original_waveforms = waveforms.clone()
    batch_size = waveforms.shape[0]
    n_samples = waveforms.shape[1]

    n_fft = n_samples

    apply_mask = (torch.rand(batch_size, device=device) < probability).float()

    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device)

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

    # Multiply by frequency mask
    filtered_fft = waveforms_fft * freq_mask.unsqueeze(0)  # [B, n_fft // 2 + 1]

    # Inverse FFT
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
