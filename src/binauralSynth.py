from dataclasses import dataclass
from typing import List, Optional

import torch
import torchaudio

from src.batchFramRir import batch_fram_brir

from .batchedHrir import BatchedHRIR

# from .framRir import fram_brir
from .occlusionFilter import apply_occlusion_frequency_domain
from .rirTensor import RIRTensor
from .smartRandomizedPlacement import SmartRandomizedPlacement


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
        sofa_path: str = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.sample_rate = sample_rate
        self.subject_id = subject_id
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device
        self.label_names = label_names
        self.sample_length = int(sample_rate * sample_total_length)
        self.hrirTensor: RIRTensor = RIRTensor.from_sofa(
            sofa_path,
            device=self.device,
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
        self.frame_length_samples = int(
            self.sample_rate * (self.frame_length_ms / 1000)
        )
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
            waveform = waveform.to(self.device)
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
        occlusion_probability: Optional[float] = 0.5,
        reverb_probability: Optional[float] = 0.7,
    ):
        waveforms, original_length = self._load_waveforms(waveform_paths)
        if self.verbose:
            print("original_length", original_length)
        if len(waveforms) != len(labels):
            raise ValueError("Number of waveform paths must match number of labels")
        label_len = len(labels)

        occluded_waveforms, occlusion_mask = apply_occlusion_frequency_domain(
            waveforms,
            sample_rate=self.sample_rate,
            herustic_occlusion_type="Random",
            same_wall_across_batch=False,
            probability=occlusion_probability,
            device=self.device,
        )

        room_dim_xz = (
            torch.empty(1, dtype=torch.float32).uniform_(3, 15).to(self.device)
        )
        room_dim_y = torch.empty(1, dtype=torch.float32).uniform_(2, 4).to(self.device)
        room_dim = torch.cat([room_dim_xz, room_dim_y, room_dim_xz]).to(self.device)
        if self.verbose:
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

        if (torch.rand(1) < reverb_probability).item():
            room_dim_expanded = room_dim.unsqueeze(0).expand(mic_pos.shape[0], -1)
            reverb, valid_after_dry = batch_fram_brir(
                target_sr=self.sample_rate,
                hrir_sr=96000,
                h_rir=self.hrirTensor,
                mic_pos=mic_pos,
                src_pos=src_pos,
                room_dim=room_dim_expanded,
                reflection_chunk_size=120,
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

            # fftconvolve(original, reverb_brir, mode="full") -> original_length + brir_valid_len - 1
            valid_total_len = (
                original_length + valid_after_dry - 1
            )  # (B,) wet end relative to event start
        else:
            combined_samples = hrirs
            valid_total_len = original_length + self.hrir_kernel_len - 1

        n_events = combined_samples.shape[0]

        dry_len = (
            original_length + self.hrir_kernel_len - 1
        )  # (B,) dry end relative to event start

        # Trim tensor to the longest valid event (remove guaranteed-zero tail)
        max_valid = valid_total_len.max().item()
        combined_samples = combined_samples[:, :, :max_valid]
        if self.verbose:
            print(f"Combined samples shape after trimming: {combined_samples.shape}")

        # Build a mapping from event index -> placement start; track which were skipped
        offset_list = [0] * label_len
        skipped_mask = torch.zeros(label_len, dtype=torch.bool, device=self.device)
        placement_controller = SmartRandomizedPlacement(
            sample_total_length=self.sample_length,
            sample_rate=self.sample_rate,
            frame_ms=self.frame_length_ms,
            max_per_frame=self.max_intance_of_class_per_frame,
        )
        for i in range(label_len):
            result = placement_controller.try_insert_sound(labels[i], dry_len[i].item())
            if result is not None:
                offset_list[i] = result[-1]["start"]
                result[-1]["position"] = relative_pos[i].tolist()
                result[-1]["reverb_end"] = offset_list[i] + valid_total_len[i].item()
                result[-1]["gain"] = gain[i].item()
            else:
                skipped_mask[i] = True  # flag so we zero out this event's contribution

        # Zero out combined_samples for skipped events so they contribute nothing to the mix
        combined_samples[skipped_mask] = 0.0

        random_offsets = torch.tensor(offset_list, dtype=torch.long, device=self.device)

        final_waveform = torch.zeros(1, 2, self.sample_length, device=self.device)

        # Vectorized placement: build index tensor for scatter_add
        # For each event i, samples 0..valid_total_len[i]-1 go to offset[i]..offset[i]+valid_total_len[i]-1
        trimmed_len = combined_samples.shape[-1]  # max_valid after trim
        sample_idx = torch.arange(trimmed_len, device=self.device).unsqueeze(
            0
        )  # (1, max_valid)
        target_idx = sample_idx + random_offsets.unsqueeze(1)  # (B, max_valid)

        # Mask: only place samples that are (a) within this event's valid range AND (b) fit in final waveform
        place_mask = sample_idx < valid_total_len.unsqueeze(1)

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

        # Metadata tensor: [B, 2] -> columns are [dry_end, wet_end]
        # event_bounds = torch.stack([dry_end, wet_end], dim=1)  # (B, 2)
        if self.verbose:
            print(f"Final waveform shape: {final_waveform.shape}")

        return final_waveform, placement_controller.placements


    @torch.no_grad()
    def batch_auralize(
        self,
        waveform_paths_batch: List[List[str]],
        labels_batch: List[List[str]],
        continuous_background_noise_paths: Optional[List[str]] = None,
        intermittent_background_noise_paths: Optional[List[str]] = None,
        occlusion_probability: Optional[float] = 0.4,
        reverb_probability: Optional[float] = 0.7,
        continuous_background_noise_probability: Optional[float] = 0.5,
        intermittent_background_noise_probability: Optional[float] = 0.5,
    ):
        """
        Render a batch of binaural audio samples in parallel on the GPU.

        Each sample in the batch can have a different number of sound events (ragged).
        All events are flattened, processed through the GPU pipeline together, then
        scattered back into per-sample output waveforms.

        Args:
            waveform_paths_batch: List of lists of waveform file paths, one inner list per sample.
            labels_batch: List of lists of labels, one inner list per sample (parallel to waveform_paths_batch).
            occlusion_probability: Probability of applying occlusion per event.
            reverb_probability: Probability of applying reverb per sample (one coin flip per room).

        Returns:
            final_waveforms: Tensor of shape (n_samples, 2, sample_length).
            all_placements: List of placement lists, one per sample.
        """
        n_samples = len(waveform_paths_batch)
        if n_samples != len(labels_batch):
            raise ValueError(
                "waveform_paths_batch and labels_batch must have the same length"
            )

        events_per_sample = [len(paths) for paths in waveform_paths_batch]
        total_events = sum(events_per_sample)

        # Build a mapping: for each event, which sample does it belong to?
        sample_of_event = torch.zeros(
            total_events, dtype=torch.long, device=self.device
        )
        idx = 0
        for s, n_ev in enumerate(events_per_sample):
            sample_of_event[idx : idx + n_ev] = s
            idx += n_ev

        # Flatten all waveform paths and labels
        flat_paths = [p for paths in waveform_paths_batch for p in paths]
        flat_labels = [lbl for labels in labels_batch for lbl in labels]

        # ── 1. Load all waveforms at once ──────────────────────────────────
        waveforms, original_length = self._load_waveforms(flat_paths)
        # waveforms: (E, T), original_length: (E,)

        if waveforms.shape[0] != len(flat_labels):
            raise ValueError("Number of waveform paths must match number of labels")

        # ── 2. Occlusion ───────────────────────────────────────────────────
        occluded_waveforms, occlusion_mask = apply_occlusion_frequency_domain(
            waveforms,
            sample_rate=self.sample_rate,
            herustic_occlusion_type="Random",
            same_wall_across_batch=False,
            probability=occlusion_probability,
            device=self.device,
        )

        # ── 3. Room geometry: one room per sample, expanded to per-event ──
        room_dim_xz = (
            torch.empty(n_samples, dtype=torch.float32).uniform_(3, 15).to(self.device)
        )
        room_dim_y = (
            torch.empty(n_samples, dtype=torch.float32).uniform_(2, 4).to(self.device)
        )
        # room_dims: (n_samples, 3)
        room_dims = torch.stack([room_dim_xz, room_dim_y, room_dim_xz], dim=1)

        # Expand to per-event: (E, 3)
        room_dim_per_event = room_dims[sample_of_event]

        # ── 4. Source positions (per-event) ────────────────────────────────
        src_pos = (
            torch.empty(total_events, 3, dtype=torch.float32)
            .uniform_(0, 1)
            .to(self.device)
            * room_dim_per_event
        )

        # Snap occluded sources to the nearest wall
        dist_to_low = src_pos
        dist_to_high = room_dim_per_event - src_pos
        all_distances = torch.cat([dist_to_low, dist_to_high], dim=1)  # (E, 6)
        closest_wall_idx = torch.argmin(all_distances, dim=1)  # (E,)
        dim_idx = closest_wall_idx % 3
        is_high = closest_wall_idx >= 3

        snapped_pos = src_pos.clone()
        batch_idx = torch.arange(total_events, device=self.device)
        target_values = torch.where(
            is_high,
            room_dim_per_event[batch_idx, dim_idx],
            torch.zeros(total_events, device=self.device),
        )
        snapped_pos[batch_idx, dim_idx] = target_values

        occlusion_mask_expanded = occlusion_mask.unsqueeze(1).bool()  # (E, 1)
        src_pos = torch.where(occlusion_mask_expanded, snapped_pos, src_pos)

        # ── 5. Mic positions: one per sample, expanded to per-event ───────
        mic_pos_per_sample = (
            torch.empty(n_samples, 3, dtype=torch.float32)
            .uniform_(0, 1)
            .to(self.device)
            * room_dims
        )
        mic_pos = mic_pos_per_sample[sample_of_event]  # (E, 3)

        # ── 6. Azimuth / elevation ────────────────────────────────────────
        relative_pos = src_pos - mic_pos
        x = relative_pos[:, 0]
        y = relative_pos[:, 1]
        z = relative_pos[:, 2]

        azm = torch.atan2(-x, z)
        ele = torch.atan2(y, torch.sqrt(x**2 + z**2))

        azm_degree = torch.rad2deg(azm)
        ele_degree = torch.rad2deg(ele)

        # ── 7. HRTF convolution ────────────────────────────────────────────
        hrirs = self.batchHrir.render_controlled_angel_hrir(
            occluded_waveforms, azm_degree, ele_degree
        )
        # hrirs: (E, 2, T_conv)

        # ── 8. Distance attenuation ───────────────────────────────────────
        direct_dist = torch.clamp(
            torch.sqrt((mic_pos - src_pos).pow(2).sum(dim=-1) + 1e-6), min=1.0
        )
        gain = 1.0 / direct_dist  # (E,)
        hrirs = hrirs * gain.unsqueeze(-1).unsqueeze(-1)

        # ── 9. Reverb (per-sample coin flip) ──────────────────────────────
        reverb_coin = torch.rand(n_samples) < reverb_probability  # (n_samples,)
        has_any_reverb = reverb_coin.any().item()

        if has_any_reverb:
            # Build mask of events that belong to reverb-enabled samples
            reverb_event_mask = reverb_coin.to(self.device)[
                sample_of_event
            ]  # (E,) bool

            reverb_indices = torch.where(reverb_event_mask)[0]
            n_reverb = reverb_indices.shape[0]

            if n_reverb > 0:
                room_dim_reverb = room_dim_per_event[reverb_indices]  # (n_reverb, 3)
                mic_pos_reverb = mic_pos[reverb_indices]
                src_pos_reverb = src_pos[reverb_indices]
                occluded_reverb = occluded_waveforms[reverb_indices]

                reverb_brir, valid_after_dry_reverb = batch_fram_brir(
                    target_sr=self.sample_rate,
                    hrir_sr=96000,
                    h_rir=self.hrirTensor,
                    mic_pos=mic_pos_reverb,
                    src_pos=src_pos_reverb,
                    room_dim=room_dim_reverb,
                    reflection_chunk_size=120,
                    device=self.device,
                )
                # reverb_brir: (n_reverb, 2, rir_len)
                reverb_left = reverb_brir[:, 0]
                reverb_right = reverb_brir[:, 1]

                left_reverb = torchaudio.functional.fftconvolve(
                    occluded_reverb, reverb_left, mode="full"
                )
                right_reverb = torchaudio.functional.fftconvolve(
                    occluded_reverb, reverb_right, mode="full"
                )
                wet_reverb = torch.stack([left_reverb, right_reverb], dim=1)
                # wet_reverb: (n_reverb, 2, T_wet)

                # Pad hrirs for reverb events to match wet length, combine
                hrirs_reverb = hrirs[reverb_indices]
                pad_amount = wet_reverb.shape[-1] - hrirs_reverb.shape[-1]
                if pad_amount > 0:
                    hrirs_reverb = torch.nn.functional.pad(
                        hrirs_reverb, (0, pad_amount), mode="constant", value=0
                    )
                elif pad_amount < 0:
                    wet_reverb = torch.nn.functional.pad(
                        wet_reverb, (0, -pad_amount), mode="constant", value=0
                    )

                combined_reverb = hrirs_reverb + wet_reverb  # (n_reverb, 2, T_combined)

        # ── 10. Assemble combined_samples and valid_total_len for all events ──
        # Start with dry-only for all events
        valid_total_len = (original_length + self.hrir_kernel_len - 1).clone()
        combined_samples = hrirs  # (E, 2, T_hrir)

        if has_any_reverb and n_reverb > 0:
            # Replace reverb events in combined_samples
            # First, ensure combined_samples is wide enough
            max_combined_len = max(
                combined_samples.shape[-1], combined_reverb.shape[-1]
            )
            if combined_samples.shape[-1] < max_combined_len:
                combined_samples = torch.nn.functional.pad(
                    combined_samples,
                    (0, max_combined_len - combined_samples.shape[-1]),
                    mode="constant",
                    value=0,
                )
            if combined_reverb.shape[-1] < max_combined_len:
                combined_reverb = torch.nn.functional.pad(
                    combined_reverb,
                    (0, max_combined_len - combined_reverb.shape[-1]),
                    mode="constant",
                    value=0,
                )

            combined_samples[reverb_indices] = combined_reverb

            # Update valid_total_len for reverb events
            orig_reverb = original_length[reverb_indices]
            valid_total_len[reverb_indices] = orig_reverb + valid_after_dry_reverb - 1

        # ── 11. Per-event dry length ──────────────────────────────────────
        dry_len = original_length + self.hrir_kernel_len - 1  # (E,)

        # Trim to the longest valid event across the entire batch
        max_valid = valid_total_len.max().item()
        combined_samples = combined_samples[:, :, :max_valid]

        # ── 12. Placement (sequential per sample) ─────────────────────────
        offset_list = torch.zeros(total_events, dtype=torch.long, device=self.device)
        skipped_mask = torch.zeros(total_events, dtype=torch.bool, device=self.device)
        all_placements: List[list] = []

        ev_idx = 0
        for s in range(n_samples):
            n_ev = events_per_sample[s]
            placement_controller = SmartRandomizedPlacement(
                sample_total_length=self.sample_length,
                sample_rate=self.sample_rate,
                frame_ms=self.frame_length_ms,
                max_per_frame=self.max_intance_of_class_per_frame,
            )
            for j in range(n_ev):
                e = ev_idx + j
                label = flat_labels[e]
                result = placement_controller.try_insert_sound(label, dry_len[e].item())
                if result is not None:
                    offset_list[e] = result[-1]["start"]
                    result[-1]["position"] = relative_pos[e].tolist()
                    result[-1]["reverb_end"] = (
                        offset_list[e].item() + valid_total_len[e].item()
                    )
                    result[-1]["gain"] = gain[e].item()
                    result[-1]["occluded"] = occlusion_mask[e].item()
                else:
                    skipped_mask[e] = True
            all_placements.append(placement_controller.placements)
            ev_idx += n_ev

        # Zero out skipped events
        combined_samples[skipped_mask] = 0.0

        # ── 13. Vectorized scatter-add into per-sample output waveforms ───
        trimmed_len = combined_samples.shape[-1]
        sample_idx = torch.arange(trimmed_len, device=self.device).unsqueeze(
            0
        )  # (1, T)
        target_idx = sample_idx + offset_list.unsqueeze(1)  # (E, T)

        # Mask: within valid range
        place_mask = sample_idx < valid_total_len.unsqueeze(1)  # (E, T)

        # Clamp for scatter
        target_idx = target_idx.clamp(0, self.sample_length - 1)

        # Expand for 2 channels
        target_idx_2ch = target_idx.unsqueeze(1).expand(-1, 2, -1)  # (E, 2, T)
        place_mask_2ch = place_mask.unsqueeze(1).expand(-1, 2, -1)  # (E, 2, T)

        masked_samples = combined_samples * place_mask_2ch

        # scatter_add per event into a (E, 2, sample_length) buffer, then reduce per sample
        per_event = torch.zeros(total_events, 2, self.sample_length, device=self.device)
        per_event.scatter_add_(2, target_idx_2ch, masked_samples)

        # Sum events belonging to the same sample using index_add along dim=0
        final_waveforms = torch.zeros(
            n_samples, 2, self.sample_length, device=self.device
        )
        # Expand sample_of_event for (E, 2, sample_length) -> use a loop-free scatter
        # index shape for scatter_add_ on dim 0: same shape as per_event
        sample_idx_expanded = (
            sample_of_event.unsqueeze(1).unsqueeze(2).expand_as(per_event)
        )  # (E, 2, sample_length)
        final_waveforms.scatter_add_(0, sample_idx_expanded, per_event)

        # apply continuous background noise if provided
        if continuous_background_noise_paths is not None:
            final_waveforms = self._apply_continuous_background_noise(
                final_waveforms, continuous_background_noise_paths,continuous_background_noise_probability
            )

        if intermittent_background_noise_paths is not None:
            final_waveforms = self._apply_intermittent_background_noise(
                final_waveforms, intermittent_background_noise_paths,intermittent_background_noise_probability
            )

        if self.verbose:
            print(f"Batch auralize: {n_samples} samples, {total_events} total events")
            print(f"Final waveforms shape: {final_waveforms.shape}")

        return final_waveforms, all_placements
    def _apply_continuous_background_noise(self, waveforms: torch.Tensor, backgroundNoise: List[str], continuous_background_probability: float = 0.5):
        """
        Mix continuous background noise into auralized waveforms.

        Args:
            waveforms: Auralized waveforms of shape (N, 2, sample_length).
            backgroundNoise: List of file paths to background noise clips.

        Returns:
            Mixed waveforms of shape (N, 2, sample_length).
        """
        background_waveforms, lengths = self._load_waveforms(backgroundNoise)
        # background_waveforms: (B, T),  lengths: (B,)

        n_samples = waveforms.shape[0]
        n_bg = background_waveforms.shape[0]
        max_bg_len = background_waveforms.shape[1]
        apply_mask = (torch.rand(n_samples, device=self.device) < continuous_background_probability).float()


        # 1. Randomly select one background clip per sample
        bg_indices = torch.randint(0, n_bg, (n_samples,), device=self.device)  # (N,)
        selected_bg = background_waveforms[bg_indices]       # (N, T_padded)
        selected_lengths = lengths[bg_indices]                # (N,)

        # 2. Tile each selected clip so it covers at least target_len,
        #    then gather a random contiguous window of size target_len.
        #
        #    We tile the padded waveforms enough times, then use a per-sample
        #    validity mask derived from the true length to wrap correctly.
        #
        #    Strategy: build an index tensor where each position maps back into
        #    the valid portion using modulo, then apply a random offset.

        # Random start offset per sample, in [0, selected_length)
        # (the modulo wrapping handles overflow, so any offset is fine)
        max_starts = selected_lengths.clamp(min=1)  # avoid div-by-zero  (N,)
        random_offsets = (torch.rand(n_samples, device=self.device) * max_starts.float()).long()  # (N,)

        # Build a (N, target_len) index tensor that wraps around each clip's valid length
        sample_positions = torch.arange(self.sample_length, device=self.device).unsqueeze(0)  # (1, target_len)
        offsets = random_offsets.unsqueeze(1)  # (N, 1)
        # Modulo by each clip's true length to wrap around for short clips,
        # and to pick a random window for long clips
        gather_indices = (sample_positions + offsets) % selected_lengths.unsqueeze(1)  # (N, target_len)

        # Clamp for safety
        gather_indices = gather_indices.clamp(0, max_bg_len - 1)

        # Gather the background segments
        bg_segments = torch.gather(selected_bg, 1, gather_indices)  # (N, target_len)

        # 3. random target SNR in dB for each sample in the batch
        signal_rms = waveforms.flatten(1).pow(2).mean(dim=1).sqrt() + 1e-8
        noise_rms = bg_segments.pow(2).mean(dim=1).sqrt() + 1e-8

        min_snr_db, max_snr_db = 4.0, 15.0

        target_snr_db = min_snr_db + (max_snr_db - min_snr_db) * torch.rand(n_samples, device=self.device)

        snr_linear_target = 10.0 ** (target_snr_db / 20.0)
        noise_gains = (signal_rms / noise_rms) / snr_linear_target

        # Apply gain to background segments
        bg_segments = bg_segments * noise_gains.unsqueeze(1)          # (N, target_len)
        bg_stereo = bg_segments.unsqueeze(1).expand(-1, 2, -1)  # (N, 2, target_len)
        bg_stereo = bg_stereo * apply_mask.unsqueeze(1).unsqueeze(2)  # zero out for samples not getting background

        # 4. Mix
        return waveforms + bg_stereo
    def _apply_intermittent_background_noise(self, waveforms: torch.Tensor, backgroundNoise: List[str], intermittent_background_probability: float = 0.5, clip_count_range: tuple = (1, 5)):
        """
        Mix intermittent background noise into auralized waveforms.

        Args:
            waveforms: Auralized waveforms of shape (N, 2, sample_length).
            backgroundNoise: List of file paths to background noise clips.
            clip_count_range: (min, max) number of background clips to layer per sample.
            intermittent_background_probability: Probability of applying background noise to each sample.

        Returns:
            Mixed waveforms of shape (N, 2, sample_length).
        """
        background_waveforms, lengths = self._load_waveforms(backgroundNoise)

        # Discard clips that are >= sample_length
        keep = lengths < self.sample_length
        background_waveforms = background_waveforms[keep]
        lengths = lengths[keep]

        if keep.sum() < keep.shape[0]:
            print(f"Warning: Discarded {keep.shape[0] - keep.sum().item()} background clips that were too long")

        n_samples = waveforms.shape[0]
        n_bg = background_waveforms.shape[0]
        max_bg_len = background_waveforms.shape[1]
        apply_mask = (torch.rand(n_samples, device=self.device) < intermittent_background_probability).float()

        min_clips, max_clips = clip_count_range
        # Random number of clips per sample: (N,)
        n_clips_per_sample = torch.randint(min_clips, max_clips + 1, (n_samples,), device=self.device)  # (N,)
        max_clip_count = max_clips  # upper bound for the padded dimension

        # Select max_clip_count clips per sample, mask out the excess ones later
        # bg_indices: (N, max_clip_count)
        bg_indices = torch.randint(0, n_bg, (n_samples, max_clip_count), device=self.device)
        selected_bg = background_waveforms[bg_indices.flatten()].view(n_samples, max_clip_count, max_bg_len)  # (N, K, T_padded)
        selected_lengths = lengths[bg_indices.flatten()].view(n_samples, max_clip_count)  # (N, K)

        # Validity mask: which clip slots are actually used  (N, K)
        clip_slot_indices = torch.arange(max_clip_count, device=self.device).unsqueeze(0)  # (1, K)
        clip_valid_mask = clip_slot_indices < n_clips_per_sample.unsqueeze(1)  # (N, K)

        # Random placement offset within sample_length for each clip
        # Each clip starts at a random position; it occupies [start, start + clip_length)
        random_starts = (torch.rand(n_samples, max_clip_count, device=self.device) * self.sample_length).long()  # (N, K)

        # Build the mixed background buffer: (N, target_len)
        bg_buffer = torch.zeros(n_samples, self.sample_length, device=self.device)

        # For each clip slot, scatter the clip into the buffer at its random start
        for k in range(max_clip_count):
            clip_data = selected_bg[:, k, :]       # (N, T_padded)
            clip_len = selected_lengths[:, k]       # (N,)
            start = random_starts[:, k]             # (N,)
            valid = clip_valid_mask[:, k]            # (N,)

            # Build time indices for this clip: (N, max_bg_len)
            t = torch.arange(max_bg_len, device=self.device).unsqueeze(0)  # (1, T_padded)
            target_positions = start.unsqueeze(1) + t  # (N, T_padded)

            # Mask: within clip's true length AND within sample_length
            in_clip = t < clip_len.unsqueeze(1)                    # (N, T_padded)
            in_bounds = target_positions < self.sample_length              # (N, T_padded)
            slot_valid = valid.unsqueeze(1)                        # (N, 1)
            place_mask = in_clip & in_bounds & slot_valid          # (N, T_padded)

            # Clamp positions for scatter safety
            target_positions = target_positions.clamp(0, self.sample_length - 1)

            # Scatter-add the clip samples into the buffer
            bg_buffer.scatter_add_(1, target_positions, clip_data * place_mask.float())

        # SNR-based gain scaling
        signal_rms = waveforms.flatten(1).pow(2).mean(dim=1).sqrt() + 1e-8  # (N,)
        noise_rms = bg_buffer.pow(2).mean(dim=1).sqrt() + 1e-8              # (N,)

        min_snr_db, max_snr_db = 4.0, 15.0
        target_snr_db = min_snr_db + (max_snr_db - min_snr_db) * torch.rand(n_samples, device=self.device)
        snr_linear_target = 10.0 ** (target_snr_db / 20.0)
        noise_gains = (signal_rms / noise_rms) / snr_linear_target  # (N,)

        # Apply gain and broadcast to stereo
        bg_buffer = bg_buffer * noise_gains.unsqueeze(1)                    # (N, target_len)
        bg_stereo = bg_buffer.unsqueeze(1).expand(-1, 2, -1)                # (N, 2, target_len)
        bg_stereo = bg_stereo * apply_mask.unsqueeze(1).unsqueeze(2)        # zero out samples not selected

        # Mix
        return waveforms + bg_stereo
