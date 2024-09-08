import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import panns_inference
from panns_inference import AudioTagging, labels

ckpt_path = "checkpoints/koko.pth"
device = 'cuda' # 'cuda' | 'cpu'

def process_audio_and_plot(audio_path, segment_duration=1):
    """Proses audio dalam potongan waktu tertentu dan tampilkan hasil keseluruhan.
    
    Args:
      audio_path: Jalur ke file audio
      segment_duration: Durasi setiap potongan audio (detik)
    """
    audio, sr = librosa.load(audio_path, sr=32000, mono=True)
    segment_samples = int(segment_duration * sr)

    at = AudioTagging(checkpoint_path=ckpt_path, device=device)
    
    # List untuk menyimpan probabilitas setiap potongan
    all_probs = []
    times = []

    for start_sample in range(0, len(audio), segment_samples):
        end_sample = min(start_sample + segment_samples, len(audio))
        audio_segment = audio[start_sample:end_sample]
        audio_segment = audio_segment[None, :]  # (batch_size, segment_samples)

        clipwise_output, _ = at.inference(audio_segment)
        all_probs.append(clipwise_output[0])
        times.append(start_sample / sr)

    # Menggabungkan semua probabilitas
    all_probs = np.vstack(all_probs)
    avg_probs = np.mean(all_probs, axis=0)

    # Menampilkan hasil
    plt.figure(figsize=(10, 6))

    # Plot probabilitas untuk setiap kelas
    sorted_indexes = np.argsort(avg_probs)[::-1]

    # Ambil 5 kelas teratas
    top_k = 5
    top_probs = [avg_probs[i] for i in sorted_indexes[:top_k]]
    top_labels = [labels[i] for i in sorted_indexes[:top_k]]

    plt.barh(top_labels, top_probs)
    plt.xlabel('Probability')
    plt.ylabel('Class')
    plt.title('Overall Audio Tagging Probabilities')
    plt.xlim(-10, 1.0)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    audio_path = '1cef6a00-abu-bakr-al-shatri_62.wav'
    process_audio_and_plot(audio_path)
