from pathlib import Path
from typing import List, Union

import torch
import torchaudio

from src.binauralSynth import BinauralSynth


def load_waveforms(
    files: Union[List[str], List[Path]],
    target_sample_rate: int = None,
    mono: bool = True,
    max_length: int = None,
    pad_to_longest: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Load multiple audio files and return them as a batched tensor of shape [B, T].

    Args:
        files: List of file paths (strings or Path objects) to audio files
        target_sample_rate: If provided, resample all audio to this sample rate
        mono: If True, convert stereo to mono by averaging channels
        max_length: If provided, truncate or pad all waveforms to this length.
                    If None and pad_to_longest is True, pad to longest file length.
        pad_to_longest: If True and max_length is None, pad all waveforms to match
                        the longest waveform in the batch

    Returns:
        waveforms: Tensor of shape [B, T] where B is batch size and T is time
        sample_rate: The sample rate of the loaded audio

    Example:
        >>> files = ["audio1.wav", "audio2.wav", "audio3.wav"]
        >>> waveforms, sr = load_waveforms(files, target_sample_rate=44100, pad_to_longest=True)
        >>> print(waveforms.shape)  # [3, T]
    """
    if not files:
        raise ValueError("files list cannot be empty")

    loaded_waveforms = []
    sample_rates = []

    for file_path in files:
        # Load audio file
        waveform, sr = torchaudio.load(file_path)

        # Convert to mono if requested
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform / waveform.abs().max()

        # Resample if target sample rate is specified
        if target_sample_rate is not None and sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
            sr = target_sample_rate

        # Remove channel dimension to get shape [T]
        waveform = waveform.squeeze(0)
        # Normalize waveforms

        loaded_waveforms.append(waveform)
        sample_rates.append(sr)

    # Check that all sample rates are the same
    if len(set(sample_rates)) > 1:
        raise ValueError(
            f"All audio files must have the same sample rate. Got: {set(sample_rates)}. "
            "Use target_sample_rate parameter to resample."
        )

    # If pad_to_longest is True and max_length is not specified, use longest waveform length
    if max_length is None and pad_to_longest:
        max_length = max(waveform.shape[0] for waveform in loaded_waveforms)

    # Pad or truncate to max_length if specified
    if max_length is not None:
        processed_waveforms = []
        for waveform in loaded_waveforms:
            if waveform.shape[0] > max_length:
                # Truncate
                waveform = waveform[:max_length]
            elif waveform.shape[0] < max_length:
                # Pad with zeros
                padding = max_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            processed_waveforms.append(waveform)
        loaded_waveforms = processed_waveforms

    # Stack all waveforms to create [B, T] tensor
    waveforms_batch = torch.stack(loaded_waveforms, dim=0)

    return waveforms_batch, sample_rates[0]


if __name__ == "__main__":
    label_names = ["AK47", "DOOR", "FIRE", "MOLOTOV"]
    sample_rate = 44100
    subject_id = "D2"
    verbose = True
    batch_size = 4
    waveforms_files = [
        "waveforms/ak47_01.wav",
        "waveforms/door_plastic_full_close_02.wav",
        "waveforms/fire_loop_1.wav",
        "waveforms/molotov_extinguish.wav",
    ]

    binaural_synth = BinauralSynth(
        label_names=label_names,
        sample_total_length=10,
        sample_rate=sample_rate,
        subject_id=subject_id,
        verbose=verbose,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )

    # Example waveforms and labels

    labels = ["AK47", "DOOR", "FIRE", "MOLOTOV"]

    for i in range(10):
        print(f"Generating sample {i + 1}/10")
        final_waveform, event_bounds, label_onehot, random_offsets = (
            binaural_synth.single_sample_auralize(waveforms_files, labels)
        )
        final_waveform = final_waveform.squeeze(0)

        torchaudio.save(f"test_audio/combined{i}.wav", final_waveform, sample_rate)

    # final_waveform.to("cpu")

    # for label, waveform in zip(labels, stuff):
    #     print(f"Label: {label}, Waveform Shape: {waveform.shape}")
    #     waveform = waveform.to("cpu")
    #     torchaudio.save(f"test_audio/{label}.wav", waveform, sample_rate)

    # waveforms, label_onehot = binaural_synth.encode_waveforms(waveforms, labels)

    # binaural_synth.single_sample_auralize(labels)
