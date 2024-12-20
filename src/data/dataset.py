import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import torchvision.transforms as T
import random
from typing import Dict


class EmotionAugmentation:
    """Data augmentation for emotion recognition"""

    def __init__(self, split='train'):
        self.split = split
        if split == 'train':
            self.image_transform = T.Compose([
                T.Grayscale (num_output_channels=1),  # Emotion works better with grayscale
                T.Resize ((48, 48)),  # Standard size for emotion detection
                T.RandomHorizontalFlip (p=0.5),
                T.RandomAffine (degrees=5, translate=(0.05, 0.05)),  # Reduced intensity
                T.ToTensor (),
                T.Normalize ([0.5], [0.5])  # Simpler normalization for grayscale
            ])
            self.audio_params = {'pitch_shift': (-2, 2), 'speed_change': (0.9, 1.1), 'noise_factor': 0.005}
        else:
            self.image_transform = T.Compose ([
                T.Grayscale (num_output_channels=1),
                T.Resize ((48, 48)),
                T.ToTensor (),
                T.Normalize ([0.5], [0.5])
            ])
            self.audio_params = None

    def augment_audio(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.split != 'train' or self.audio_params is None:
            return waveform
        # Pitch shift
        if random.random() < 0.5:
            n_steps = random.uniform(*self.audio_params['pitch_shift'])
            waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
        # Speed change
        if random.random() < 0.5:
            speed_factor = random.uniform(*self.audio_params['speed_change'])
            waveform = librosa.effects.time_stretch(waveform, rate=speed_factor)
        # Add noise
        if random.random() < 0.5:
            noise = np.random.randn(len(waveform))
            waveform = waveform + self.audio_params['noise_factor'] * noise
        return waveform


class MultiModalEmotionDataset(Dataset):
    def __init__(self, image_data: pd.DataFrame, audio_data: pd.DataFrame,
                 text_data: pd.DataFrame, split='train', max_audio_length=10, max_text_length=50):
        self.split = split
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length

        # Filter data
        self.image_data = image_data[image_data['split'] == split].reset_index(drop=True)
        self.audio_data = audio_data[audio_data['split'] == split].reset_index(drop=True)
        self.text_data = text_data[text_data['split'] == split].reset_index(drop=True)

        # Initialize augmentation
        self.augmentation = EmotionAugmentation(split)
        self.tokenizer = lambda text: [ord(c) for c in text[:max_text_length]]

    def __len__(self):
        return min(len(self.image_data), len(self.audio_data), len(self.text_data))

    def __getitem__ (self, idx: int) -> Dict[str, torch.Tensor]:
        # Print debugging information for alignment check
        # print (f"Index {idx}")
        # print (f"Image Label: {self.image_data.iloc[idx]['emotion']}")
        # print (f"Audio Label: {self.audio_data.iloc[idx]['emotion']}")
        # print (f"Text Label: {self.text_data.iloc[idx]['emotion']}")

        # Load image
        image_path = self.image_data.iloc[idx]['path']
        image = self.augmentation.image_transform(Image.open (image_path).convert ("RGB"))

        # Load audio
        audio_path = self.audio_data.iloc[idx]['path']
        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = self._pad_audio(self.augmentation.augment_audio (waveform))

        # Process text
        text = self.text_data.iloc[idx]['text']
        text_tensor = self._pad_text(self.tokenizer (text))

        # Emotion label
        emotion = self.image_data.iloc[idx]['emotion']

        return {
            'image': image,
            'audio': torch.tensor (waveform, dtype=torch.float32).unsqueeze (0),
            'text': text_tensor,
            'emotion': torch.tensor (emotion, dtype=torch.long)
        }

    def _pad_audio(self, waveform: np.ndarray) -> np.ndarray:
        target_length = 16000 * self.max_audio_length
        if len(waveform) > target_length:
            return waveform[:target_length]
        return np.pad(waveform, (0, target_length - len(waveform)), mode='constant')

    def _pad_text(self, tokens: list) -> torch.Tensor:
        if len(tokens) < self.max_text_length:
            tokens.extend([0] * (self.max_text_length - len(tokens)))
        else:
            tokens = tokens[:self.max_text_length]
        return torch.tensor(tokens, dtype=torch.long)



