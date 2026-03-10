from dataclasses import dataclass
from typing import List, Optional

import torch
import torchaudio

from src.batchFramRir import batch_fram_brir

from .batchedHrir import BatchedHRIR

# from .framRir import fram_brir
from .occlusionFilter import apply_occlusion_frequency_domain
from .rirTensor import RIRTensor


class BinauralSynth:
    def __init__(
        self,
        label_names: List[str],
        sample_total_length: int = 2,
        sample_rate: int = 44100,
        subject_id: str = "D2",
        verbose: bool = True,
        max_events_per_batch: int = 10,
        max_intance_of_class_per_frame: int = 3,
        frame_length_ms: float = 40,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        self.sample_rate = sample_rate
        self.subject_id = subject_id
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device
        self.label_names = label_names
        self.sample_length = sample_rate * sample_total_length
        self.hrirTensor: RIRTensor = RIRTensor.from_sofa(
            "sadie/Database-Master_V2-1/D2/D2_HRIR_SOFA/D2_96K_24bit_512tap_FIR_SOFA.sofa",
            device=device,
        )
        self.hrir_kernel_len = int(
            self.hrirTensor.kernel_size * self.sample_rate / 96000
        )
        self.batchHrir = BatchedHRIR(
            sample_rate=self.sample_rate,
            subject_id="D2",
            device=self.device,
        )
        self.max_events_per_batch = max_events_per_batch
        self.max_intance_of_class_per_frame = max_intance_of_class_per_frame
        self.frame_length_ms = frame_length_ms
        self.frame_length_samples = int(self.sample_rate * self.frame_length_ms / 1000)
        self._waveform_cache: dict[str, torch.Tensor] = {}

    def _load_waveforms(self, file_paths: List[str]):
        waveforms = []
        for file_path in file_paths:
            if file_path in self._waveform_cache:
                waveforms.append(self._waveform_cache[file_path].clone())
                continue
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            # Convert stereo (or multi-channel) to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)  # [T]
            # Normalize
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
            self._waveform_cache[file_path] = waveform
            waveforms.append(waveform.clone())

        # Record original lengths before padding
        lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)  # [B]
        # Pad all waveforms to the longest length
        max_len = lengths.max().item()
        padded = []
        for w in waveforms:
            pad_amount = max_len - w.shape[0]
            if pad_amount > 0:
                w = torch.nn.functional.pad(
                    w, (0, pad_amount), mode="constant", value=0
                )
            padded.append(w)
        waveforms = torch.stack(padded, dim=0)  # [B, T]
        waveforms = waveforms.to(self.device)
        lengths = lengths.to(self.device)
        return waveforms, lengths

    @torch.no_grad()
    def single_sample_auralize(
        self,
        waveform_paths: List[str],
        labels: List[str],
        occlusion_probability: Optional[float] = 0.3,
    ):
        waveforms, original_length = self._load_waveforms(waveform_paths)
        print("original_length", original_length)
        if len(waveforms) != len(labels):
            raise ValueError("Number of waveform paths must match number of labels")
        label_len = len(labels)

        # crit_freq_hz = (
        #     torch.empty(1, dtype=torch.float32).uniform_(300.0, 4000.0).to(self.device)
        # )
        # crit_width_hz = (
        #     torch.empty(1, dtype=torch.float32).uniform_(800, 1600.0).to(self.device)
        # )
        # attenuation_dip_strength_db = (
        #     torch.empty(1, dtype=torch.float32).uniform_(5.0, 15.0).to(self.device)
        # )

        occluded_waveforms, occlusion_mask = apply_occlusion_frequency_domain(
            waveforms,
            sample_rate=self.sample_rate,
            herustic_occlusion_type="Random",
            same_wall_across_batch=False,
            # crit_freq_hz=crit_freq_hz.item(),
            # crit_width_hz=crit_width_hz.item(),
            # attenuation_dip_strength_db=attenuation_dip_strength_db.item(),
            probability=occlusion_probability,
            device=self.device,
        )

        room_dim_xz = (
            torch.empty(1, dtype=torch.float32).uniform_(3, 15).to(self.device)
        )
        room_dim_y = torch.empty(1, dtype=torch.float32).uniform_(2, 4).to(self.device)
        room_dim = torch.cat([room_dim_xz, room_dim_y, room_dim_xz]).to(self.device)
        print(f"Room dimensions (x, y, z): {room_dim.cpu().numpy()} meters")
        src_pos = torch.empty(label_len, 3, dtype=torch.float32).uniform_(0, 1).to(
            self.device
        ) * room_dim.unsqueeze(0)

        dist_to_low = src_pos  # distances to x=0, y=0, z=0 walls
        dist_to_high = (
            room_dim.unsqueeze(0) - src_pos
        )  # distances to x=max, y=max, z=max
        all_distances = torch.cat([dist_to_low, dist_to_high], dim=1)  # Shape: (N, 6)

        # Find which wall is closest for each source
        closest_wall_idx = torch.argmin(all_distances, dim=1)  # Shape: (N,)

        # Determine dimension (0=x, 1=y, 2=z) and whether it's the high wall
        dim_idx = closest_wall_idx % 3
        is_high = closest_wall_idx >= 3

        # Create snapped positions at the closest wall
        snapped_pos = src_pos.clone()
        batch_idx = torch.arange(label_len, device=self.device)
        target_values = torch.where(
            is_high, room_dim[dim_idx], torch.zeros(label_len, device=self.device)
        )
        snapped_pos[batch_idx, dim_idx] = target_values

        # Apply only to occluded samples using the mask
        occlusion_mask_expanded = occlusion_mask.unsqueeze(1).bool()  # Shape: (N, 1)
        src_pos = torch.where(occlusion_mask_expanded, snapped_pos, src_pos)

        mic_pos = torch.empty(1, 3, dtype=torch.float32).uniform_(0, 1).to(self.device)
        mic_pos = mic_pos.repeat(label_len, 1) * room_dim.unsqueeze(0)

        relative_pos = src_pos - mic_pos

        x = relative_pos[:, 0]
        y = relative_pos[:, 1]
        z = relative_pos[:, 2]

        azm = torch.atan2(-x, z)
        ele = torch.atan2(y, torch.sqrt(x**2 + z**2))

        azm_degree = torch.rad2deg(azm)
        ele_degree = torch.rad2deg(ele)

        hrirs = self.batchHrir.render_controlled_angel_hrir(
            occluded_waveforms, azm_degree, ele_degree
        )

        # apply distance attenuation to the direct sound
        direct_dist = torch.clamp(
            torch.sqrt((mic_pos - src_pos).pow(2).sum(dim=-1) + 1e-6), min=1.0
        )
        gain = 1 / direct_dist

        hrirs = hrirs * gain.unsqueeze(-1).unsqueeze(-1)

        # note for later, in order to make reverb more realistic we need to scale n_reflections with t_60,
        # because a long t60 and low n_reflections will make it sound like  the same sound played many time over when its reverb,
        # while many n_reflections in a short t_60 will play so many reflections at the same time to the point where it will over power the original sound and hide it,
        # making localization impossible.
        # done for now, by scaling t60 inside the fram function depending on a random logical density of reflections.

        # t60 = (
        #     torch.empty(src_pos.shape[0], dtype=torch.float32)
        #     .uniform_(0.3, 0.3)
        #     .to(self.device)
        # )
        # n_reflections = torch.randint(
        #     300, 4000, (src_pos.shape[0], 2), dtype=torch.int32
        # ).to(self.device)

        room_dim_expanded = room_dim.unsqueeze(0).expand(mic_pos.shape[0], -1)
        reverb, valid_after_dry = batch_fram_brir(
            target_sr=self.sample_rate,
            hrir_sr=96000,
            h_rir=self.hrirTensor,
            mic_pos=mic_pos,
            src_pos=src_pos,
            room_dim=room_dim_expanded,
            reflection_chunk_size=100,
            device=self.device,
        )
        reverb_left = reverb[:, 0]
        reverb_right = reverb[:, 1]

        left_reverb = torchaudio.functional.fftconvolve(
            occluded_waveforms, reverb_left, mode="full"
        )
        right_reverb = torchaudio.functional.fftconvolve(
            occluded_waveforms, reverb_right, mode="full"
        )
        final_wet_reverb = torch.stack([left_reverb, right_reverb], dim=1)

        # Pad hrirs on the right to match final_wet_reverb shape
        pad_amount = final_wet_reverb.shape[-1] - hrirs.shape[-1]
        hrirs_padded = torch.nn.functional.pad(
            hrirs, (0, pad_amount), mode="constant", value=0
        )

        # Combine directional dry sound + wet reverb
        combined_samples = hrirs_padded + final_wet_reverb

        n_events = combined_samples.shape[0]

        # --- Compute per-event valid lengths ---
        #

        # Per-event: how long the meaningful convolved signal actually is
        # fftconvolve(original, reverb_brir, mode="full") -> original_length + brir_valid_len - 1
        valid_total_len = (
            original_length + valid_after_dry - 1
        )  # (B,) wet end relative to event start
        dry_len = (
            original_length + self.hrir_kernel_len - 1
        )  # (B,) dry end relative to event start

        # Trim tensor to the longest valid event (remove guaranteed-zero tail)
        max_valid = valid_total_len.max().item()
        combined_samples = combined_samples[:, :, :max_valid]
        print(f"Combined samples shape after trimming: {combined_samples.shape}")

        max_offsets = (self.sample_length - valid_total_len).clamp(min=0)  # (B,)
        max_offsets = torch.minimum(
            max_offsets,
            torch.tensor(int(self.sample_length * 0.8), device=self.device).expand(
                n_events
            ),
        )
        random_offsets = (
            torch.rand(n_events, device=self.device) * max_offsets
        ).long()  # (B,)

        final_waveform = torch.zeros(1, 2, self.sample_length, device=self.device)

        # Vectorized placement: build index tensor for scatter_add
        # For each event i, samples 0..valid_total_len[i]-1 go to offset[i]..offset[i]+valid_total_len[i]-1
        trimmed_len = combined_samples.shape[-1]  # max_valid after trim
        sample_idx = torch.arange(trimmed_len, device=self.device).unsqueeze(
            0
        )  # (1, max_valid)
        target_idx = sample_idx + random_offsets.unsqueeze(1)  # (B, max_valid)

        # Mask: only place samples that are (a) within this event's valid range AND (b) fit in final waveform
        place_mask = (sample_idx < valid_total_len.unsqueeze(1)) & (
            target_idx < self.sample_length
        )
        # Clamp target indices for scatter (invalid ones will be masked to 0 contribution)
        target_idx = target_idx.clamp(0, self.sample_length - 1)  # (B, max_valid)

        # Expand for 2 channels: target_idx -> (B, 1, max_valid) -> (B, 2, max_valid)
        target_idx_2ch = target_idx.unsqueeze(1).expand(-1, 2, -1)
        place_mask_2ch = place_mask.unsqueeze(1).expand(-1, 2, -1)

        # Zero out invalid positions in combined_samples, then scatter_add
        masked_samples = combined_samples * place_mask_2ch
        # scatter_add into a (B, 2, sample_length) then sum across events
        per_event = torch.zeros(n_events, 2, self.sample_length, device=self.device)
        per_event.scatter_add_(2, target_idx_2ch, masked_samples)
        final_waveform = per_event.sum(dim=0, keepdim=True)  # (1, 2, sample_length)

        dry_end = (
            random_offsets + dry_len
        )  # (B,) this is where the dry sound ends after the random offset
        wet_end = (
            random_offsets + valid_total_len
        )  # (B,) this is where the wet sound ends after the random offset

        # Metadata tensor: [B, 2] -> columns are [dry_end, wet_end]
        event_bounds = torch.stack([dry_end, wet_end], dim=1)  # (B, 2)
        print(f"Event bounds (dry_end, wet_end): {event_bounds.cpu().numpy()}")
        print(f"Final waveform shape: {final_waveform.shape}")

        return final_waveform, event_bounds, random_offsets
