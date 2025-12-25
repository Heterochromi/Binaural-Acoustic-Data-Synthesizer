from dataclasses import dataclass
from typing import List

import torch
import torchaudio

from src.batchFramRir import batch_fram_brir

from .batchedHrir import BatchedHRIR

# from .framRir import fram_brir
from .occlusionFilter import apply_occlusion
from .rirTensor import RIRTensor


@dataclass
class BinauralSynthConfig:
    features: torch.Tensor  # [batch, channel, time]
    positions: torch.Tensor  # Shape: (batch, num_sounds, 3) for x,y,z
    labels: torch.Tensor  # Shape: (batch, num_sounds,num_labels) where the corresponding label will be 1 for that specific sound (one hot encoded)
    timestamps: torch.Tensor  # Shape: (batch, num_sounds,)

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    def to(self, device):
        """Move all tensors to specified device"""
        return BinauralSynthConfig(
            features=self.features.to(device),
            positions=self.positions.to(device),
            labels=self.labels.to(device),
            timestamps=self.timestamps.to(device),
        )

    def pin_memory(self):
        """Pin memory for faster GPU transfer"""
        return BinauralSynthConfig(
            features=self.features.pin_memory(),
            positions=self.positions.pin_memory(),
            labels=self.labels.pin_memory(),
            timestamps=self.timestamps.pin_memory(),
        )

    def __repr__(self):
        return (
            f"BinauralSynthBatch("
            f"batch_size={self.batch_size}, "
            f"features={tuple(self.features.shape)}, "
            f"positions={tuple(self.positions.shape)}, "
            f"labels={tuple(self.labels.shape)}, "
            f"timestamps={tuple(self.timestamps.shape)})"
        )


class BinauralSynth:
    def __init__(
        self,
        label_names: List[str],
        sample_total_length: int = 2,
        sample_rate: int = 44100,
        subject_id: str = "D2",
        verbose: bool = True,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        self.sample_rate = sample_rate
        self.subject_id = subject_id
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device
        self.label_names = label_names
        self.label2id = {label: idx for idx, label in enumerate(label_names)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.sample_length = sample_rate * sample_total_length
        self.hrirTensor: RIRTensor = RIRTensor.from_sofa(
            "sadie/Database-Master_V2-1/D2/D2_HRIR_SOFA/D2_96K_24bit_512tap_FIR_SOFA.sofa",
            device=device,
        )
        self.batchHrir = BatchedHRIR(
            sample_rate=self.sample_rate,
            subject_id="D2",
            device=self.device,
        )

    def _encode_waveforms(
        self,
        waveforms: torch.Tensor,
        labels: List[str],
    ):
        """
        Encode waveforms and labels into one-hot encoded tensors.

        Args:
            waveforms (torch.Tensor): Waveforms to encode.
            labels (List[str]): Labels corresponding to the waveforms.

        Returns:
             torch.Tensor: with the waveforms and their corresponding encoded labels.
        """
        if len(waveforms) != len(labels):
            raise ValueError("Waveforms and labels must have the same length")
        label_indices = torch.tensor(
            [self.label2id[label] for label in labels], dtype=torch.long
        )
        label_onehot = torch.nn.functional.one_hot(
            label_indices, num_classes=len(self.label_names)
        ).float()

        waveforms = waveforms.to(self.device)
        label_onehot = label_onehot.to(self.device)
        return waveforms, label_onehot

    # def _batch_reverb(
    #     self,
    #     waveforms: torch.Tensor,
    # ):

    def decode_label_onehot(
        self,
        label_onehot: torch.Tensor,
    ):
        label_indices = torch.argmax(label_onehot, dim=1)
        labels = [self.id2label[idx] for idx in label_indices.tolist()]
        return labels

    def single_sample_auralize(self, waveforms: torch.Tensor, labels: List[str]):
        waveforms, label_onehot = self._encode_waveforms(waveforms, labels)
        label_len = label_onehot.shape[0]
        # samples = torch.zeros(label_len, self.sample_length).to(self.device)

        base_attenuation_db = (
            torch.empty(1, dtype=torch.float32).uniform_(8.0, 15.0).to(self.device)
        )
        max_attenuation_db = base_attenuation_db * (
            torch.empty(1, dtype=torch.float32).uniform_(2, 4).to(self.device)
        )
        crit_freq_hz = (
            torch.empty(1, dtype=torch.float32).uniform_(300.0, 4000.0).to(self.device)
        )
        crit_width_hz = (
            torch.empty(1, dtype=torch.float32).uniform_(800, 1600.0).to(self.device)
        )
        attenuation_dip_strength_db = (
            torch.empty(1, dtype=torch.float32).uniform_(5.0, 15.0).to(self.device)
        )

        occluded_waveforms, occlusion_mask = apply_occlusion(
            waveforms,
            sample_rate=self.sample_rate,
            base_attenuation_db=base_attenuation_db.item(),
            max_attenuation_db=max_attenuation_db.item(),
            crit_freq_hz=crit_freq_hz.item(),
            crit_width_hz=crit_width_hz.item(),
            attenuation_dip_strength_db=attenuation_dip_strength_db.item(),
            probability=0.0,
            device=self.device,
        )

        room_dim_xz = (
            torch.empty(1, dtype=torch.float32).uniform_(6, 10).to(self.device)
        )
        room_dim_y = torch.empty(1, dtype=torch.float32).uniform_(2, 4).to(self.device)
        room_dim = torch.cat([room_dim_xz, room_dim_y, room_dim_xz]).to(self.device)
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

        mic_pos = torch.empty(label_len, 3, dtype=torch.float32).uniform_(0, 1).to(
            self.device
        ) * room_dim.unsqueeze(0)

        relative_pos = src_pos - mic_pos

        x = relative_pos[:, 0]
        y = relative_pos[:, 1]
        z = relative_pos[:, 2]

        azm = torch.atan2(-x, z)
        ele = torch.atan2(y, torch.sqrt(x**2 + z**2))

        azm_degree = torch.rad2deg(azm)
        ele_degree = torch.rad2deg(ele)

        hrirs = self.batchHrir.render_controlled_angel_hrir(
            occluded_waveforms, azm_degree, ele_degree, mode="full"
        )

        # apply distance attenuation to the direct sound
        direct_dist = torch.sqrt((mic_pos - src_pos).pow(2).sum(dim=-1) + 1e-6)
        gain = 1 / direct_dist

        hrirs = hrirs * gain.unsqueeze(-1).unsqueeze(-1)

        t60 = (
            torch.empty(src_pos.shape[0], dtype=torch.float32)
            .uniform_(0.3, 0.3)
            .to(self.device)
        )
        n_reflections = torch.randint(
            300, 600, (src_pos.shape[0], 2), dtype=torch.int32
        ).to(self.device)

        room_dim_expanded = room_dim.unsqueeze(0).expand(mic_pos.shape[0], -1)
        reverb, valid_after_dry = batch_fram_brir(
            target_sr=self.sample_rate,
            hrir_sr=96000,
            h_rir=self.hrirTensor,
            t60=t60,
            mic_pos=mic_pos,
            n_reflection=n_reflections,
            src_pos=src_pos,
            room_dim=room_dim_expanded,
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
        final_output = hrirs_padded + final_wet_reverb

        print(final_output.shape)

        return final_output
