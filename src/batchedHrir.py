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
        subject_id: Literal[
            "D1",
            "D2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H7",
            "H8",
            "H9",
            "H10",
            "H11",
            "H12",
            "H13",
            "H14",
            "H15",
            "H16",
            "H17",
            "H18",
            "H19",
            "H20",
        ] = "D2",
        interpolation_mode: Literal[
            "auto", "nearest", "two_point", "three_point"
        ] = "auto",
        verbose: bool = False,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            subject_ids: Subject HRIR to use
            sample_rate: Sample rate
            interpolation_mode: method to estimate angel that does not exactly exist in sadie
            verbose: Enable detailed output
            batch_size: Size of batches to process
        """
        sadie_path = "sadie/Database-Master_V2-1"
        hrir_path_slug = "_HRIR_SOFA"
        hrir_slug_96k = "_96K_24bit_512tap_FIR_SOFA.sofa"  # Slug for the 96 kHz 24-bit 512-tap FIR SOFA file
        self.subject_id = subject_id
        self.sample_rate = sample_rate
        self.interpolation_mode = interpolation_mode
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device

        if self.sample_rate > 96000:
            raise ValueError("Sample rate must be less than or equal to 96 kHz")

        self.hrir_path = os.path.join(
            sadie_path,
            self.subject_id,
            f"{self.subject_id}{hrir_path_slug}",
            f"{self.subject_id}{hrir_slug_96k}",
        )
        self.sampler = Resample(orig_freq=96000, new_freq=self.sample_rate).to(device)
        self.hrirTensor: RIRTensor = RIRTensor.from_sofa(self.hrir_path, device=device)

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
