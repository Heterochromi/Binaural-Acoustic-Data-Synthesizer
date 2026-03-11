import time

import torch

from src.binauralSynth import BinauralSynth


def run_stress_test(num_sounds=1024, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 44100
    duration_sec = 2
    T = sample_rate * duration_sec

    # Use a fixed pool of labels for the synth to avoid string memory overhead
    label_pool = [str(i) for i in range(batch_size)]

    print(f"--- Stress Test: {num_sounds} sounds total, Batch Size: {batch_size} ---")

    synth = BinauralSynth(
        label_names=label_pool,
        sample_total_length=duration_sec,
        sample_rate=sample_rate,
        subject_id="D2",
        verbose=False,
        batch_size=batch_size,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    # Process in chunks to avoid OOM
    for i in range(0, num_sounds, batch_size):
        current_batch_size = min(batch_size, num_sounds - i)

        # Generate waveforms only for the current batch
        batch_waveforms = torch.randn(current_batch_size, T, device=device)
        batch_labels = label_pool[:current_batch_size]

        # Perform auralization
        _ = synth.single_sample_auralize(batch_waveforms, batch_labels)

        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"Processed {i}/{num_sounds} sounds...")

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    sounds_per_min = (num_sounds / total_time) * 60

    print("\n" + "=" * 40)
    print("DIAGNOSTIC RESULTS")
    print("=" * 40)
    print(f"Total Sounds:        {num_sounds}")
    print(f"Total Time:          {total_time:.4f}s")
    print(f"Sounds / Minute:     {sounds_per_min:.2f}")

    if device.type == "cuda":
        max_vram = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Max VRAM Usage:      {max_vram:.2f} MB")
    print("=" * 40)


if __name__ == "__main__":
    # Adjust these to stress test your specific hardware
    TOTAL_SOUNDS = 3560
    BATCH_SIZE = 712
    run_stress_test(num_sounds=TOTAL_SOUNDS, batch_size=BATCH_SIZE)
