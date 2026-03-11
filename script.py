import torch
import torchaudio

from src.binauralSynth import BinauralSynth

if __name__ == "__main__":
    label_names = ["AK47", "DOOR", "FIRE", "MOLOTOV"]
    sample_rate = 44100
    subject_id = "D2"
    verbose = True
    batch_size = 2
    waveforms_files = ["/home/baraa/Desktop/test_audio/ME.wav", "waveforms/speech.wav"]

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

    labels = ["AK47", "AK47"]

    for i in range(10):
        print(f"Generating sample {i + 1}/10")
        final_waveform, info = binaural_synth.single_sample_auralize(
            waveforms_files, labels
        )
        final_waveform = final_waveform.squeeze(0)
        for p in info:
            print(f"{p}")

        torchaudio.save(f"test_audio/combined{i}.wav", final_waveform, sample_rate)

    # final_waveform.to("cpu")

    # for label, waveform in zip(labels, stuff):
    #     print(f"Label: {label}, Waveform Shape: {waveform.shape}")
    #     waveform = waveform.to("cpu")
    #     torchaudio.save(f"test_audio/{label}.wav", waveform, sample_rate)

    # waveforms, label_onehot = binaural_synth.encode_waveforms(waveforms, labels)

    # binaural_synth.single_sample_auralize(labels)
