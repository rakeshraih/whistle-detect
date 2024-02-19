import librosa
from sklearn.neighbors import KNeighborsClassifier

# Load audio file
audio_file = "cat.wav"
audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)

# Create a KNeighborsClassifier object
clf = KNeighborsClassifier(n_neighbors=5)

# Print the shape of the audio data array
print("Shape of audio data array:", audio_data.shape)

# Print the sample rate
print("Sample rate:", sample_rate)

# Print the duration of the audio
duration_sec = len(audio_data) / sample_rate
print("Duration:", duration_sec, "seconds")