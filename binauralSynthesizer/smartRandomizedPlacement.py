import torch


class SmartRandomizedPlacement:
    def __init__(self, sample_total_length, sample_rate, frame_ms, max_per_frame=3):
        self.sr = sample_rate
        self.total_samples = sample_total_length
        self.frame_len_samples = int((frame_ms / 1000) * sample_rate)

        # Calculate total frames using ceiling division
        self.n_frames = (
            self.total_samples + self.frame_len_samples - 1
        ) // self.frame_len_samples

        # State: { "class_name": torch.Tensor([count_per_frame]) }
        self.state = {}
        self.max_per_frame = max_per_frame

        # To store the final placements: (class, start_sample, duration)
        self.placements = []

    def _get_affected_frames(self, start_sample, duration_samples):
        """Calculates exactly which frames a sound spans."""
        first_f = start_sample // self.frame_len_samples
        last_f = (start_sample + duration_samples - 1) // self.frame_len_samples
        return range(first_f, last_f + 1)

    def try_insert_sound(self, sound_class, duration_samples):
        """
        Finds a random valid start time.
        Returns (start_sample) if successful, None if impossible.
        """
        if sound_class not in self.state:
            self.state[sound_class] = torch.zeros(self.n_frames, dtype=torch.int32)

        # 1. Identify which frames for this class are FULL
        full_frames = torch.where(self.state[sound_class] >= self.max_per_frame)[0]

        # 2. Create a mask of all possible start samples
        # (Subtract duration so the sound doesn't fall off the end of the track)
        valid_starts_mask = torch.ones(
            self.total_samples - duration_samples, dtype=torch.bool
        )

        if len(full_frames) > 0:
            # Compute all block_start/block_end values at once — shape: (num_full_frames,)
            f_starts = full_frames * self.frame_len_samples
            f_ends = (full_frames + 1) * self.frame_len_samples

            N = valid_starts_mask.shape[0]
            block_starts = (f_starts - duration_samples + 1).clamp(min=0, max=N)
            block_ends = f_ends.clamp(max=N)

            # Skip frames whose forbidden range is empty after clamping — this
            # happens with long sounds where every possible start position already
            # places the sound past a given full frame, so that frame is unreachable.
            valid_frame_mask = block_starts < block_ends
            if valid_frame_mask.any():
                bs = block_starts[valid_frame_mask].unsqueeze(
                    1
                )  # (num_valid_frames, 1)
                be = block_ends[valid_frame_mask].unsqueeze(1)  # (num_valid_frames, 1)
                indices = torch.arange(N).unsqueeze(0)  # (1, N)
                in_block = (indices >= bs) & (indices < be)  # (num_valid_frames, N)

                # Any sample forbidden by at least one full frame is masked out
                valid_starts_mask &= ~in_block.any(dim=0)

        # 3. Pick a random sample from the remaining True values
        possible_indices = torch.where(valid_starts_mask)[0]

        if len(possible_indices) == 0:
            print(
                f"No valid placement for class '{sound_class}' with duration {duration_samples} samples ;therefor skipping."
            )
            return  # No room left

        chosen_start = possible_indices[
            torch.randint(len(possible_indices), (1,))
        ].item()

        # 4. ASSIGNMENT: Update the state
        affected_frames = self._get_affected_frames(chosen_start, duration_samples)
        for f in affected_frames:
            self.state[sound_class][f] += 1

        self.placements.append(
            {
                "class": sound_class,
                "start": chosen_start,
                "end": chosen_start + duration_samples,
                "frames": affected_frames,
            }
        )

        return self.placements
