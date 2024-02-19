import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to extract MFCC features from audio file
def extract_features(audio_files):
    features = []
    labels = []
    for file in audio_files:
        # Load audio file
        audio_data, sample_rate = librosa.load(file, sr=None)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
        # Take the mean of MFCC coefficients
        features.append(np.mean(mfccs, axis=1))
        # Label cat sounds as 1 and non-cat sounds as 0
        labels.append(1 if "cat" in file else 0)
    return np.array(features), np.array(labels)

# Example usage
audio_files = ["cat.wav"]
features, labels = extract_features(audio_files)

# Check if the number of features and labels match
if len(features) != len(labels):
    raise ValueError("Number of features and labels do not match!")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a classifier (Random Forest for example)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
