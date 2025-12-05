import os
from typing import Iterator, Literal

import torch
import torchaudio

# import numpy as np
# import sofar as sf
from rirTensor import RIRTensor


class BatchedHRIR:
    def __init__(
        self,
        sample_rate: Literal[44100, 48000, 96000],
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
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """
        Args:
            subject_ids: Subject HRIR to use
            sample_rate: Sample rate (44100, 48000, or 96000)
            interpolation_mode: method to estimate angel that does not exactly exist in sadie
            verbose: Enable detailed output
            batch_size: Size of batches to process
        """
        sadie_path = "sadie/Database-Master_V2-1"
        hrir_path_slug = "_HRIR_SOFA"
        hrir_slug_44k = "_44K_16bit_256tap_FIR_SOFA.sofa"  # Slug for the 44.1 kHz 16-bit 256-tap FIR SOFA file
        hrir_slug_48k = "_48K_24bit_256tap_FIR_SOFA.sofa"  # Slug for the 48 kHz 24-bit 256-tap FIR SOFA file
        hrir_slug_96k = "_96K_24bit_512tap_FIR_SOFA.sofa"  # Slug for the 96 kHz 24-bit 512-tap FIR SOFA file
        self.subject_id = subject_id
        self.sample_rate = sample_rate
        self.interpolation_mode = interpolation_mode
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device

        # example fulle end result path "sadie/Database-Master_V2-1/D2/D2_HRIR_SOFA/D2_44K_16bit_256tap_FIR_SOFA.sofa"
        if self.sample_rate == 44100:
            hrir_slug = hrir_slug_44k
        elif self.sample_rate == 48000:
            hrir_slug = hrir_slug_48k
        elif self.sample_rate == 96000:
            hrir_slug = hrir_slug_96k
        else:
            raise ValueError("Unsupported sample rate")

        self.hrir_path = os.path.join(
            sadie_path,
            self.subject_id,
            f"{self.subject_id}{hrir_path_slug}",
            f"{self.subject_id}{hrir_slug}",
        )
        self.hrirTensor: RIRTensor = RIRTensor.from_sofa(self.hrir_path, device=device)

    def batch_load_directory(
        self, directory: str
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yields batches of WAV files from a directory.

        Args:
            directory: Path to directory containing WAV files

        Yields:
            Tuple of:
                - batch_tensor: Tensor of shape [B, Time] where all audio is padded to the
                  maximum length found in the current batch
                - original_lengths: Tensor of shape [B] containing original length (in samples) of each audio file
        """
        wav_files = []

        # Get all .wav files from directory
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(".wav"):
                wav_files.append(filename)

        if not wav_files:
            raise ValueError(f"No WAV files found in directory: {directory}")

        total_files = len(wav_files)
        if self.verbose:
            print(
                f"Found {total_files} WAV files in {directory}. Processing in batches of {self.batch_size}."
            )

        for i in range(0, total_files, self.batch_size):
            batch_files = wav_files[i : i + self.batch_size]
            waveforms = []
            original_lengths = []
            max_length = 0

            # Load files in current batch and find maximum length
            for filename in batch_files:
                filepath = os.path.join(directory, filename)
                waveform, sr = torchaudio.load(filepath)
                waveform = waveform.to(self.device)

                # Resample if necessary
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono if stereo or multi-channel
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Remove channel dimension
                waveform = waveform.squeeze(0)

                # Store original length before padding
                original_lengths.append(waveform.shape[-1])
                waveforms.append(waveform)
                max_length = max(max_length, waveform.shape[-1])

            if self.verbose:
                # print(len(waveforms))
                print(
                    f"Batch {i // self.batch_size + 1}: Loaded {len(waveforms)} files. Max length: {max_length}"
                )

            # Pad all waveforms to max length of the batch
            padded_waveforms = []
            for waveform in waveforms:
                current_length = waveform.shape[-1]
                if current_length < max_length:
                    # Pad on the right (end) with zeros
                    padding = (0, max_length - current_length)
                    waveform = torch.nn.functional.pad(waveform, padding)
                padded_waveforms.append(waveform)

            # Stack all waveforms into batch tensor [B, Time]
            batch_tensor = torch.stack(padded_waveforms, dim=0)

            # Convert original lengths to tensor
            original_lengths_tensor = torch.tensor(original_lengths, dtype=torch.int32)

            yield batch_tensor, original_lengths_tensor

    def render_random_angles_hrir(
        self, waveforms: torch.Tensor, mode: Literal["full", "same", "valid"] = "full"
    ):
        """
        Render HRIRs at random angles using FFT convolution.

        Args:
            waveforms: Input audio tensor of shape [B, Time]
            mode: Convolution mode
                - 'full': Returns full convolution (signal_len + kernel_len - 1)
                - 'same': Returns output same length as input signal (signal_len)
                - 'valid': Returns only valid portion (signal_len - kernel_len + 1)

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



        # Determine output length
        signal_len = waveforms.shape[-1]
        kernel_len = left_hrir.shape[-1]
        output_len = signal_len + kernel_len - 1

        fft_len = 2 ** (output_len - 1).bit_length()

        # FFT of waveforms and HRIRs
        waveforms_fft = torch.fft.rfft(waveforms, n=fft_len, dim=-1)
        left_hrir_fft = torch.fft.rfft(left_hrir, n=fft_len, dim=-1)
        right_hrir_fft = torch.fft.rfft(right_hrir, n=fft_len, dim=-1)

        # Multiply in frequency domain (element-wise per batch)
        convolved_left_fft = waveforms_fft * left_hrir_fft
        convolved_right_fft = waveforms_fft * right_hrir_fft

        # IFFT back to time domain
        convolved_left = torch.fft.irfft(convolved_left_fft, n=fft_len, dim=-1)
        convolved_right = torch.fft.irfft(convolved_right_fft, n=fft_len, dim=-1)

        # Trim based on mode
        if mode == "full":
            # Full convolution: signal_len + kernel_len - 1
            convolved_left = convolved_left[..., :output_len]
            convolved_right = convolved_right[..., :output_len]
        elif mode == "same":
            # Same as input signal length
            convolved_left = convolved_left[..., :signal_len]
            convolved_right = convolved_right[..., :signal_len]
        elif mode == "valid":
            # Only the valid portion without zero-padding effects
            valid_len = signal_len - kernel_len + 1
            if valid_len < 1:
                raise ValueError(
                    f"For 'valid' mode, signal length ({signal_len}) must be >= kernel length ({kernel_len})"
                )
            start_idx = kernel_len - 1
            convolved_left = convolved_left[..., start_idx : start_idx + valid_len]
            convolved_right = convolved_right[..., start_idx : start_idx + valid_len]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'full', 'same', or 'valid'"
            )

        convolved = torch.stack([convolved_left, convolved_right], dim=1)

        return convolved, tupled_azimuth_elevation

    # get

    def render_controlled_angel_hrir(
        self,
        waveforms: torch.Tensor,
        azmiuth: torch.Tensor,
        elevation: torch.Tensor,
        mode: Literal["full", "same", "valid"] = "full",
    ):
        """
        Render HRIRs at controlled angles using FFT convolution.

        Args:
            waveforms: Input audio tensor of shape [B, Time]
            azmiuth: Azimuth angles in degrees [B]
            elevation: Elevation angles in degrees [B]
            mode: Convolution mode
                - 'full': Returns full convolution (signal_len + kernel_len - 1)
                - 'same': Returns output same length as input signal (signal_len)
                - 'valid': Returns only valid portion (signal_len - kernel_len + 1)

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

        # Determine output length
        signal_len = waveforms.shape[-1]
        kernel_len = left_hrir.shape[-1]
        output_len = signal_len + kernel_len - 1

        fft_len = 2 ** (output_len - 1).bit_length()

        # FFT of waveforms and HRIRs
        waveforms_fft = torch.fft.rfft(waveforms, n=fft_len, dim=-1)
        left_hrir_fft = torch.fft.rfft(left_hrir, n=fft_len, dim=-1)
        right_hrir_fft = torch.fft.rfft(right_hrir, n=fft_len, dim=-1)

        # Multiply in frequency domain (element-wise per batch)
        convolved_left_fft = waveforms_fft * left_hrir_fft
        convolved_right_fft = waveforms_fft * right_hrir_fft

        # IFFT back to time domain
        convolved_left = torch.fft.irfft(convolved_left_fft, n=fft_len, dim=-1)
        convolved_right = torch.fft.irfft(convolved_right_fft, n=fft_len, dim=-1)

        # Trim based on mode
        if mode == "full":
            # Full convolution: signal_len + kernel_len - 1
            convolved_left = convolved_left[..., :output_len]
            convolved_right = convolved_right[..., :output_len]
        elif mode == "same":
            # Same as input signal length
            convolved_left = convolved_left[..., :signal_len]
            convolved_right = convolved_right[..., :signal_len]
        elif mode == "valid":
            # Only the valid portion without zero-padding effects
            valid_len = signal_len - kernel_len + 1
            if valid_len < 1:
                raise ValueError(
                    f"For 'valid' mode, signal length ({signal_len}) must be >= kernel length ({kernel_len})"
                )
            start_idx = kernel_len - 1
            convolved_left = convolved_left[..., start_idx : start_idx + valid_len]
            convolved_right = convolved_right[..., start_idx : start_idx + valid_len]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'full', 'same', or 'valid'"
            )

        convolved = torch.stack([convolved_left, convolved_right], dim=1)

        return convolved
