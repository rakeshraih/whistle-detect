import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load audio files and extract MFCC features
def extract_features(audio_files):
    features = []
    labels = []
    for file in audio_files:
        audio_data, _ = librosa.load(file, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=13)
        features.append(np.mean(mfccs, axis=1))  # Use mean MFCC coefficients
        labels.append(file.split('/')[-1].split('_')[0])  # Assuming filenames are labeled with categories
    return np.array(features), np.array(labels)

# Example usage
audio_files = ['cat.wav']  # Example audio files
features, labels = extract_features(audio_files)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a classifier (Random Forest for example)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
