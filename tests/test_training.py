import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image
import soundfile as sf

from src.training.train import EmotionTrainer
from src.data.dataset import MultiModalEmotionDataset


class TestEmotionTrainer(unittest.TestCase):
    def setUp(self):
        """
        Setup a temporary directory and create minimal CSVs + dummy image/audio/text data files.
        Then create DataLoader with MultiModalEmotionDataset.
        We'll run a short 'train' to test the pipeline.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

        # Create minimal CSV for images, audio, and text
        self.image_df = self._create_dummy_image_data()
        self.audio_df = self._create_dummy_audio_data()
        self.text_df = self._create_dummy_text_data()

        # Minimal config to run quick training
        self.config = {
            'batch_size': 2,
            'num_workers': 0,
            'learning_rate': 1e-4,
            'weight_decay': 0.0,
            'epochs': 2,          # run just 2 epochs for speed
            'patience': 2,
            'scheduler_patience': 1,
            'grad_clip': 1.0
        }

    def _create_dummy_image_data(self):
        image_rows = []
        emotions = [0, 1]
        splits = ['train', 'test']

        for emotion in emotions:
            for split in splits:
                for i in range(2):
                    img_path = self.data_dir / f"img_{emotion}_{split}_{i}.jpg"
                    image_rows.append({
                        'path': str(img_path),
                        'emotion': emotion,
                        'split': split
                    })
                    # Save dummy image
                    img = Image.new('RGB', (32, 32), color='white')
                    img.save(img_path)

        return pd.DataFrame(image_rows)

    def _create_dummy_audio_data(self):
        audio_rows = []
        emotions = [0, 1]
        splits = ['train', 'test']

        for emotion in emotions:
            for split in splits:
                for j in range(2):
                    wav_path = self.data_dir / f"wav_{emotion}_{split}_{j}.wav"
                    audio_rows.append({
                        'path': str(wav_path),
                        'emotion': emotion,
                        'split': split
                    })
                    # Save dummy audio
                    sample_rate = 16000
                    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
                    wave = np.sin(2 * np.pi * 440 * t)
                    sf.write(str(wav_path), wave, sample_rate)

        return pd.DataFrame(audio_rows)

    def _create_dummy_text_data(self):
        text_rows = []
        dummy_sentences = [
            "I am very happy today!",
            "This is so sad and disappointing.",
            "I am extremely angry with this.",
            "Such a surprising outcome!",
            "Neutral emotions are hard to read."
        ]
        emotions = [0, 1]
        splits = ['train', 'test']

        for emotion in emotions:
            for split in splits:
                for i in range(2):
                    text_rows.append({
                        'text': np.random.choice(dummy_sentences),
                        'emotion': emotion,
                        'split': split
                    })

        return pd.DataFrame(text_rows)

    def test_emotion_trainer(self):
        """
        Create small datasets, train briefly, ensure no crash & logs basic metrics.
        """
        # Combine 'train' and 'test' splits
        train_dataset = MultiModalEmotionDataset(
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='train'
        )
        val_dataset = MultiModalEmotionDataset(
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='test'
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )

        trainer = EmotionTrainer(self.config)
        trainer.train(train_loader, val_loader)

        # Check log and checkpoint directories
        self.assertTrue(trainer.log_dir.exists(), "No log_dir created by trainer.")
        self.assertTrue(trainer.checkpoint_dir.exists(), "No checkpoint_dir created by trainer.")

        # Check if training.log was created
        training_log = trainer.log_dir / 'training.log'
        self.assertTrue(training_log.exists(), "training.log was not created.")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
