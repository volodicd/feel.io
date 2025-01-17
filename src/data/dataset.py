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
        self.split = split
        self.config = config
        augmentation_config = config['augmentation'][split]
        preprocessing_config = config['preprocessing']

        self.image_transform = T.Compose([
            T.Grayscale(num_output_channels=preprocessing_config['image']['channels']),
            T.Resize(preprocessing_config['image']['size']),
            T.RandomHorizontalFlip(p=augmentation_config['image']['horizontal_flip_prob']),
            T.RandomAffine(
                degrees=augmentation_config['image']['rotation_degrees'],
                translate=augmentation_config['image']['translate']
            ),
            T.ToTensor(),
            T.Normalize(
                preprocessing_config['image']['normalize_mean'],
                preprocessing_config['image']['normalize_std']
            )
        ])
        if split == 'train':
            self.audio_params = augmentation_config['audio']
        else:
            self.audio_params = None

class MultiModalEmotionDataset(Dataset):
    def __init__(self, image_data: pd.DataFrame, audio_data: pd.DataFrame, text_data: pd.DataFrame,
                 config: Dict, split='train'):
        self.split = split
        self.config = config
        self.max_audio_length = config['preprocessing']['audio']['max_length']
        self.max_text_length = config['preprocessing']['text']['max_length']

        # Filter data
        self.image_data = image_data[image_data['split'] == split].reset_index(drop=True)
        self.audio_data = audio_data[audio_data['split'] == split].reset_index(drop=True)
        self.text_data = text_data[text_data['split'] == split].reset_index(drop=True)

        # Initialize augmentation
        self.augmentation = EmotionAugmentation(config, split)
        self.tokenizer = BertTokenizer.from_pretrained(config['preprocessing']['text']['tokenizer'])
