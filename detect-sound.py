import pyaudio
import numpy as np

# Parameters for audio recording
FORMAT = pyaudio.paInt16  # 16-bit integer format
CHANNELS = 1  # Monaural audio
RATE = 44100  # Sample rate (samples per second)
CHUNK = 1024  # Number of frames per buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

# Record audio data
try:
    while True:
        data = stream.read(CHUNK)
        # Process audio data here
        # Example: Calculate root mean square (RMS) amplitude
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        print("RMS amplitude:", rms)

except KeyboardInterrupt:
    print("Recording stopped.")

# Stop stream
stream.stop_stream()
stream.close()
audio.termina
