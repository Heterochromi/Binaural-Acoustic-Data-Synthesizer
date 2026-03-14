import os
from typing import Iterator, Literal

import torch
import torchaudio
from torchaudio.transforms import Resample

from .rirTensor import RIRTensor


class BatchedHRIR:
    def __init__(
        self,
        sample_rate: int,
        interpolation_mode: Literal[
            "auto", "nearest", "two_point", "three_point"
        ] = "auto",
        verbose: bool = False,
        batch_size: int = 32,
        sofa_path: str = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            sample_rate: Sample rate
            interpolation_mode: method to estimate angel that does not exactly exist in sadie
            verbose: Enable detailed output
            batch_size: Size of batches to process
            sofa_path: Path to the SOFA file
            device: Device to use for processing
        """
        self.sample_rate = sample_rate
        self.interpolation_mode = interpolation_mode
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device

        if self.sample_rate > 96000:
            raise ValueError("Sample rate must be less than or equal to 96 kHz")

        self.sofa_path = sofa_path
        self.sampler = Resample(orig_freq=96000, new_freq=self.sample_rate).to(device)
        self.hrirTensor: RIRTensor = RIRTensor.from_sofa(self.sofa_path, device=device)

    def render_random_angles_hrir(self, waveforms: torch.Tensor):
        """
        Render HRIRs at random angles using FFT convolution.

        Args:
            waveforms: Input audio tensor of shape [B, Time]

        Returns:
            Tuple of (convolved audio [B, 2, Time], angles [B, 2])
        """
        # waveforms shape: [B, Channels, Time]

        azmiuth = torch.empty(len(waveforms), device=self.device)
        azmiuth.uniform_(-180, 180)
        elevation = torch.empty(len(waveforms), device=self.device)
        elevation.uniform_(-90, 90)

        tupled_azimuth_elevation = torch.stack([azmiuth, elevation], dim=1)
        left_hrir, right_hrir = self.hrirTensor.angle_batch(azmiuth, elevation)

        # Convert HRIRs to match waveform dtype
        left_hrir = left_hrir.to(dtype=waveforms.dtype)
        right_hrir = right_hrir.to(dtype=waveforms.dtype)
        left_hrir = self.sampler(left_hrir)
        right_hrir = self.sampler(right_hrir)

        convolved_left = torchaudio.functional.fftconvolve(
            waveforms, left_hrir, mode="full"
        )

        convolved_right = torchaudio.functional.fftconvolve(
            waveforms, right_hrir, mode="full"
        )

        convolved = torch.stack([convolved_left, convolved_right], dim=1)

        return convolved, tupled_azimuth_elevation

    def render_controlled_angel_hrir(
        self,
        waveforms: torch.Tensor,
        azmiuth: torch.Tensor,
        elevation: torch.Tensor,
    ):
        """
        Render HRIRs at controlled angles using FFT convolution.

        Args:
            waveforms: Input audio tensor of shape [B, Time]
            azmiuth: Azimuth angles in degrees [B]
            elevation: Elevation angles in degrees [B]

        Returns:
            Convolved audio tensor of shape [B, 2, Time]
        """
        batch_size = waveforms.shape[0]
        if batch_size != len(azmiuth) and batch_size != len(elevation):
            raise ValueError(
                "Batch size mismatch , waveforms length must match azmiuth and elevation [waveforms , azmiuth , elevation]"
            )

        left_hrir, right_hrir = self.hrirTensor.angle_batch(azmiuth, elevation)
        left_hrir = left_hrir.to(dtype=waveforms.dtype)
        right_hrir = right_hrir.to(dtype=waveforms.dtype)
        left_hrir = self.sampler(left_hrir)
        right_hrir = self.sampler(right_hrir)

        convolved_left = torchaudio.functional.fftconvolve(
            waveforms, left_hrir, mode="full"
        )
        convolved_right = torchaudio.functional.fftconvolve(
            waveforms, right_hrir, mode="full"
        )

        convolved = torch.stack([convolved_left, convolved_right], dim=1)

        return convolved
