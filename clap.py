import librosa
import numpy as np

# Function to detect claps in audio
def detect_claps(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Find peaks in the onset envelope
    peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    
    # Check if there are any peaks (claps)
    if len(peaks) > 0:
        print("Claps detected at time(s):", peaks / sr)
    else:
        print("No claps detected in the audio file.")

# Example usage
audio_file = "clap.wav"
detect_claps(audio_file)