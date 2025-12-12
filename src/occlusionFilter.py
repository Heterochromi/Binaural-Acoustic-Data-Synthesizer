import torch
import torchaudio
import torchaudio.functional as F_audio


def apply_occlusion(
    waveform: torch.Tensor,
    sample_rate: int,
    base_attenuation_db: float = 15.0,
    max_attenuation_db: float = 40.0,
    crit_freq_hz: float = 1500.0,
    crit_width_hz: float = 1000.0,
    attenuation_dip_strength_db: float = 15.0,
):
    """
    This is a simple audio occlusion filter that tries to mimic real world single panel sound transmission loss.

    Args:
        waveform (torch.Tensor): Input audio waveform.
        sample_rate (int): Sample rate of the audio signal.
        base_attenuation_db (float): Base attenuation in decibels.
        max_attenuation_db (float): Maximum attenuation in decibels.
        crit_freq_hz (float): Critical frequency in Hz.
        crit_width_hz (float): Critical width in Hz.
        attenuation_dip_strength_db (float): Attenuation dip strength in decibels.

    Returns:
        torch.Tensor: Filtered audio waveform.
    """

    if base_attenuation_db * 2 < max_attenuation_db:
        ValueError(
            "for a realistic range max attenuation must be at least twice the base attenuation"
        )

    base_gain = 10.0 ** (-base_attenuation_db / 20.0)
    waveform = waveform * base_gain

    nyquist = sample_rate / 2

    target_gain = 10.0 ** (-max_attenuation_db / 20.0)
    fc = nyquist / torch.sqrt(torch.tensor((1 / target_gain**2) - 1))

    waveform = F_audio.lowpass_biquad(
        waveform,
        sample_rate,
        cutoff_freq=fc,
        Q=0.5,
    )

    Q_factor = crit_freq_hz / crit_width_hz
    waveform = F_audio.equalizer_biquad(
        waveform,
        sample_rate,
        center_freq=crit_freq_hz,
        gain=attenuation_dip_strength_db,
        Q=Q_factor,
    )

    return waveform


def batch_occlusion(
    waveforms,
    sample_rate,
    max_attenuation_db,
    base_attenuation_db,
    crit_freq_hz,
    crit_width_hz,
    dip_strength_db,
):
    torch.vmap(apply_occlusion, in_dims=(0, None, 0, 0, 0, 0, 0))(
        waveforms,
        sample_rate,
        max_attenuation_db,
        base_attenuation_db,
        crit_freq_hz,
        crit_width_hz,
        dip_strength_db,
    )


# def apply_occlusion(
#     sample_rate: int = 44100,
#     max_attenuation_db: float = 40.0,
#     base_attenuation_db: float = 20.0,
#     crit_freq_hz: float = 1500.0,
#     crit_width_hz: float = 1000.0,
#     dip_strength_db: float = 10.0,
#     n_fft: int = 2048,
#     ir_len: int = 513,
# ):
#     """
#     This is a simple audio occlusion filter that tries to mimic real world single panel sound transmission loss.

#     Args:
#         sample_rate (int): Sample rate of the audio signal.
#         max_attenuation_db (float): Maximum attenuation in dB.
#         base_attenuation_db (float): Base attenuation in dB.
#         crit_freq_hz (float): Critical frequency in Hz.
#         crit_width_hz (float): Critical width in Hz.
#         dip_strength_db (float): Dip strength in dB.
#         n_fft (int): Number of FFT bins.

#     Returns:
#         torch.Tensor: Filter curve in dB.
#     """
#     # Frequency grid
#     freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

#     # 1. Linear Attenuation
#     nyquist = sample_rate / 2
#     slope = -max_attenuation_db / nyquist
#     base_db = slope * freqs - base_attenuation_db

#     # gaussian dip
#     sigma = crit_width_hz / 2.355
#     gauss = torch.exp(-0.5 * ((freqs - crit_freq_hz) / sigma) ** 2)
#     drip_db = gauss * dip_strength_db
#     db = base_db + drip_db

#     final_db = torch.clamp(db, max=0.0)

#     # dB to Magnitude
#     mag = 10.0 ** (final_db / 20.0)

#     # Create Minimum Phase or Linear Phase IR (Same as before)
#     H = torch.zeros(n_fft, dtype=torch.complex64)
#     H[: n_fft // 2 + 1] = mag * torch.exp(1j * torch.zeros_like(mag))
#     H[n_fft // 2 + 1 :] = torch.conj(torch.flip(H[1 : n_fft // 2], dims=[0]))

#     h = torch.fft.ifft(H).real
#     h = torch.roll(h, shifts=-(ir_len // 2))
#     h = h[:ir_len]

#     # Optional windowing to smooth edges
#     window = torch.hann_window(ir_len)
#     h = h * window

#     return final_db


if "__main__" == __name__:
    import torch

    max_attenuation_db = 25
    base_attenuation_db = 10
    crit_freq_hz = 1000
    crit_width_hz = 1000.0
    attenuation_dip_strength_db = 5.0

    waveform, sr = torchaudio.load("waveforms/hegrenade_detonate_02.wav")
    waveform = waveform / waveform.abs().max()
    occluded_waveforms = apply_occlusion(
        waveform,
        sr,
        max_attenuation_db=max_attenuation_db,
        base_attenuation_db=base_attenuation_db,
        crit_freq_hz=crit_freq_hz,
        crit_width_hz=crit_width_hz,
        attenuation_dip_strength_db=attenuation_dip_strength_db,
    )
    torchaudio.save("output.wav", occluded_waveforms, sr)
