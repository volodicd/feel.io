from tqdm import tqdm
import os
import librosa
import soundfile as sf
import pandas as pd

root_path = "."
# Define function to normalize audio
def normalize_audio(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio
        y_normalized = librosa.util.normalize(y)  # Normalize amplitude
        sf.write(output_path, y_normalized, sr)  # Save normalized audio
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Load annotations CSV
annotations_csv = "ravdess_annotations.csv"
ravdess_annotations = pd.read_csv(annotations_csv)

# Define paths
normalized_path = "normalized_audio"
os.makedirs(normalized_path, exist_ok=True)

# Normalize all audio files with a progress bar
print("Starting audio normalization...")
for _, row in tqdm(ravdess_annotations.iterrows(), total=len(ravdess_annotations), desc="Normalizing audio"):
    src = row["file_path"]
    dest = os.path.join(normalized_path, os.path.basename(src))
    normalize_audio(src, dest)

print("Audio normalization completed.")

def resample_audio(file_path, output_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, y_resampled, target_sr)

# Create a directory for resampled audio
resampled_path = os.path.join(root_path, "resampled_audio")
os.makedirs(resampled_path, exist_ok=True)

# Resample all audio files
for _, row in ravdess_annotations.iterrows():
    src = row["file_path"]
    dest = os.path.join(resampled_path, os.path.basename(src))
    resample_audio(src, dest)


import numpy as np

def extract_mel_spectrogram(file_path, n_mels=128, max_length=150):
    """
    Extract Mel Spectrogram and ensure consistent time dimension.
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)  # Load audio
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)  # Extract Mel Spectrogram
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel scale

        # Pad or truncate to ensure consistent time dimension
        if mel_spec_db.shape[1] < max_length:
            # Pad with zeros
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, max_length - mel_spec_db.shape[1])), mode='constant')
        else:
            # Truncate
            mel_spec_db = mel_spec_db[:, :max_length]

        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Example: Extract Mel Spectrogram for the first file
example_file = ravdess_annotations["file_path"].iloc[0]
mel_spec = extract_mel_spectrogram(example_file)
print("Mel Spectrogram shape:", mel_spec.shape)


import torch
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence

# Custom collate function to handle variable-length spectrograms
def collate_fn(batch):
    features, labels = zip(*batch)
    features = [torch.tensor(f, dtype=torch.float32) for f in features]
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad features to the same size
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    return features_padded, labels


# Dataset class
class RAVDESSDataset(Dataset):
    def __init__(self, annotations, feature_extraction_fn):
        self.annotations = annotations
        self.feature_extraction_fn = feature_extraction_fn

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        file_path = row["file_path"]
        label = row["emotion"] - 1  # Adjust labels to start from 0
        features = self.feature_extraction_fn(file_path)
        return features, label


# Instantiate the dataset
dataset = RAVDESSDataset(annotations=ravdess_annotations, feature_extraction_fn=extract_mel_spectrogram)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for features, labels in dataloader:
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    break

print(ravdess_annotations["emotion"].value_counts())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# Define the RNN Model
class EmotionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, input_size, time_steps)
        x = x.permute(0, 2, 1)  # Rearrange to (batch_size, time_steps, input_size)
        _, (hidden, _) = self.lstm(x)  # hidden: (1, batch_size, hidden_size)
        out = self.fc(hidden[-1])  # Use the last hidden state
        return out


# Hyperparameters
input_size = 128  # Mel bands
hidden_size = 64  # LSTM hidden layer size
num_classes = len(ravdess_annotations["emotion"].unique())  # Number of emotion classes
num_epochs = 50
batch_size = 32
learning_rate = 0.001

# Instantiate the model
model = EmotionRNN(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data to device (if using GPU)
        features, labels = features, labels

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
