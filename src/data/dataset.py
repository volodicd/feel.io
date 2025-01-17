import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import torchvision.transforms as T
import random
from typing import Dict
from transformers import BertTokenizer
import yaml

def load_dataset_config(config_path: str = 'configs/dataset_config.yaml') -> Dict:
    """Load dataset configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class EmotionAugmentation:
    def __init__(self, config: Dict, split='train'):
        """
        Initialize augmentation with configuration.

        Args:
            config: Configuration dictionary
            split: 'train' or 'test'
        """
        self.split = split
        self.config = config['augmentation'][split]

        if split == 'train':
            self.image_transform = T.Compose([
                T.Grayscale(num_output_channels=config['preprocessing']['image']['channels']),
                T.Resize(config['preprocessing']['image']['size']),
                T.RandomHorizontalFlip(p=self.config['image']['horizontal_flip_prob']),
                T.RandomAffine(
                    degrees=self.config['image']['rotation_degrees'],
                    translate=self.config['image']['translate']
                ),
                T.ToTensor(),
                T.Normalize(
                    self.config['image']['normalize_mean'],
                    self.config['image']['normalize_std']
                )
            ])
            self.audio_params = {
                'pitch_shift': self.config['audio']['pitch_shift_range'],
                'speed_change': self.config['audio']['speed_change_range'],
                'noise_factor': self.config['audio']['noise_factor']
            }
        else:
            self.image_transform = T.Compose([
                T.Grayscale(num_output_channels=config['preprocessing']['image']['channels']),
                T.Resize(config['preprocessing']['image']['size']),
                T.ToTensor(),
                T.Normalize(
                    self.config['image']['normalize_mean'],
                    self.config['image']['normalize_std']
                )
            ])
            self.audio_params = None

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            return self.image_transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return torch.zeros(1, *self.config['preprocessing']['image']['size'])

    def augment_audio(self, waveform: np.ndarray, sr: int = None) -> np.ndarray:
        if self.split != 'train' or self.audio_params is None:
            return waveform

        if sr is None:
            sr = self.config['preprocessing']['audio']['sample_rate']

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
    def __init__(self, config_path: str = 'configs/dataset_config.yaml', split='train'):
        """
        Initialize dataset with configuration.

        Args:
            config_path: Path to configuration file
            split: 'train' or 'test'
        """
        self.config = load_dataset_config(config_path)
        self.split = split

        # Load data
        self.image_data = pd.read_csv(self.config['paths']['fer2013'])
        self.audio_data = pd.read_csv(self.config['paths']['ravdess'])
        self.text_data = pd.read_csv(self.config['paths']['goemotions'])

        # Filter by split
        self.image_data = self.image_data[self.image_data['split'] == split].reset_index(drop=True)
        self.audio_data = self.audio_data[self.audio_data['split'] == split].reset_index(drop=True)
        self.text_data = self.text_data[self.text_data['split'] == split].reset_index(drop=True)

        # Initialize augmentation
        self.augmentation = EmotionAugmentation(self.config, split)

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config['preprocessing']['text']['tokenizer'])

        # Set maximum lengths
        self.max_audio_length = self.config['preprocessing']['audio']['max_length']
        self.max_text_length = self.config['preprocessing']['text']['max_length']

    def __len__(self):
        return min(len(self.image_data), len(self.audio_data), len(self.text_data))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image_path = self.image_data.iloc[idx]['path']
        image = self.augmentation.process_image(image_path)

        # Load audio
        audio_path = self.audio_data.iloc[idx]['path']
        waveform, sr = librosa.load(
            audio_path,
            sr=self.config['preprocessing']['audio']['sample_rate'],
            mono=True
        )
        waveform = self._pad_audio(self.augmentation.augment_audio(waveform))
        audio = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # Process text
        text = self.text_data.iloc[idx]['text']
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        text_tensor = encoded['input_ids'].squeeze(0)

        # Get emotion label
        emotion = self.image_data.iloc[idx]['emotion']

        return {
            'image': image,
            'audio': audio,
            'text': text_tensor,
            'emotion': torch.tensor(emotion, dtype=torch.long)
        }

    def _pad_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Ensure consistent audio length"""
        target_length = self.config['preprocessing']['audio']['sample_rate'] * self.max_audio_length

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=-1)

        if len(waveform) > target_length:
            return waveform[:target_length]
        return np.pad(waveform, (0, target_length - len(waveform)), mode='constant')