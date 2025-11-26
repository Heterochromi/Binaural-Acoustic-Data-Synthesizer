import torch
import torchaudio
from typing import Literal
import os
import numpy as np
import sofar as sf
from rirTensor import RIRTensor


class BatchedHRIR:
    def __init__(self,
                 sample_rate: Literal[44100, 48000, 96000],
                 subject_id: Literal["D1", "D2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20"] = "D2",
                 interpolation_mode: Literal["auto", "nearest", "two_point", "three_point"] = "auto",
                 verbose: bool = False,
                 batch_size: int = 32,
                 device: Literal['cpu', 'cuda'] = 'cpu'):
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

        self.hrir_path = os.path.join(sadie_path, self.subject_id, f"{self.subject_id}{hrir_path_slug}", f"{self.subject_id}{hrir_slug}")
        self.hrirTensor : RIRTensor = RIRTensor.from_sofa(self.hrir_path , device= device)

    def _batch_load_directory(self, directory : str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load all WAV files from a directory and return padded tensor with original lengths.

        Args:
            directory: Path to directory containing WAV files

        Returns:
            Tuple of:
                - batch_tensor: Tensor of shape [B, Time] where all audio is padded to the
                  maximum length found in the directory
                - original_lengths: Tensor of shape [B] containing original length (in samples) of each audio file
        """
        wav_files = []
        waveforms = []
        original_lengths = []
        max_length = 0

        # Get all .wav files from directory
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith('.wav'):
                wav_files.append(filename)

        if not wav_files:
            raise ValueError(f"No WAV files found in directory: {directory}")

        # Load all files and find maximum length
        for filename in wav_files:
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
            print(f"Loaded {len(waveforms)} WAV files from {directory}")
            print(f"Maximum audio length: {max_length} samples")

        # Pad all waveforms to max length
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

        return batch_tensor, original_lengths_tensor


    def render_random_angles_HRIR(self, wavDirectory : str):
        # waveforms shape: [B, Channels, Time]
        waveforms, lengths = self._batch_load_directory(wavDirectory)

        azmiuth = torch.empty(len(waveforms), device=self.device)
        azmiuth.uniform_(-180, 180)
        elevation = torch.empty(len(waveforms), device=self.device)
        elevation.uniform_(-90, 90)

        tupled_azimuth_elevation = torch.stack([azmiuth, elevation], dim=1)
        left_hrir, right_hrir = self.hrirTensor.angle_batch(azmiuth, elevation)

        # Convert HRIRs to match waveform dtype
        left_hrir = left_hrir.to(dtype=waveforms.dtype)
        right_hrir = right_hrir.to(dtype=waveforms.dtype)

        batch_size = waveforms.shape[0]

        # Reshape inputs for group convolution
        # waveforms: [Batch, Time] -> [1, Batch, Time]
        # hrir: [Batch, Kernel] -> [Batch, 1, Kernel]

        # We use padding to maintain roughly the same length
        # padding = left_hrir.shape[-1] // 2

        convolved_left = torch.nn.functional.conv1d(
            waveforms.unsqueeze(0),
            left_hrir.unsqueeze(1),
            groups=batch_size,
        ).squeeze(0)

        convolved_right = torch.nn.functional.conv1d(
            waveforms.unsqueeze(0),
            right_hrir.unsqueeze(1),
            groups=batch_size,
        ).squeeze(0)

        # Ensure output length matches input length
        if convolved_left.shape[-1] > waveforms.shape[-1]:
            print("Output length exceeds input length")
            convolved_left = convolved_left[..., :waveforms.shape[-1]]
            convolved_right = convolved_right[..., :waveforms.shape[-1]]

        convolved = torch.stack([convolved_left, convolved_right], dim=1)

        return convolved, lengths, tupled_azimuth_elevation











    def controlled_angel_hrir(self ,wavTensor : torch.Tensor , azmiuth : torch.Tensor, elevation : torch.Tensor):
        rirs = self.hrirTensor.angle_batch(azmiuth, elevation)
