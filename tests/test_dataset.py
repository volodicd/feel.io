import unittest
import pandas as pd
import numpy as np
import random
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import soundfile as sf
import torch

from src.data.dataset import MultiModalEmotionDataset

class TestMultiModalEmotionDataset(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary directory and synthetic CSV data
        for images, audio, and text, then init the dataset.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

        # Emotion labels
        self.num_emotions = 3  # e.g., 3 emotions: [0, 1, 2]
        self.emotion_labels = [0, 1, 2]

        # Dummy image data
        self.image_df = self._create_dummy_image_data()

        # Dummy audio data
        self.audio_df = self._create_dummy_audio_data()

        # Dummy text data
        self.text_df = self._create_dummy_text_data()

    def _create_dummy_image_data(self):
        image_rows = []
        for emotion in self.emotion_labels:
            for split in ['train', 'test']:
                for i in range(2):
                    img_path = self.data_dir / f"image_{emotion}_{split}_{i}.jpg"
                    image_rows.append({
                        'path': str(img_path),
                        'emotion': emotion,
                        'split': split
                    })
                    # Create dummy image file
                    img = Image.new('RGB', (64, 64), color='white')
                    img.save(img_path)
        return pd.DataFrame(image_rows)

    def _create_dummy_audio_data(self):
        audio_rows = []
        for emotion in self.emotion_labels:
            for split in ['train', 'test']:
                for i in range(2):
                    wav_path = self.data_dir / f"audio_{emotion}_{split}_{i}.wav"
                    audio_rows.append({
                        'path': str(wav_path),
                        'emotion': emotion,
                        'split': split
                    })
                    # Create dummy audio file
                    sample_rate = 16000
                    duration = 1.0
                    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                    wave = np.sin(2 * np.pi * 440 * t)  # 440 Hz
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
        for emotion in self.emotion_labels:
            for split in ['train', 'test']:
                for i in range(2):
                    text_rows.append({
                        'text': random.choice(dummy_sentences),
                        'emotion': emotion,
                        'split': split
                    })
        return pd.DataFrame(text_rows)

    def test_train_dataset(self):
        """
        Test the 'train' split of MultiModalEmotionDataset.
        """
        train_dataset = MultiModalEmotionDataset(
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='train'
        )

        # Check dataset length
        self.assertEqual(len(train_dataset), 2 * self.num_emotions,
                         "Train dataset length mismatch.")

        # Check a random item
        item = train_dataset[0]
        self.assertIn('audio', item)
        self.assertIn('image', item)
        self.assertIn('text', item)
        self.assertIn('emotion', item)

        # Check shapes
        self.assertEqual(item['audio'].shape, (1, 16000 * 10), "Audio shape mismatch.")
        self.assertEqual(item['image'].shape, (3, 224, 224), "Image shape mismatch.")
        self.assertIsInstance(item['text'], torch.Tensor, "Text should be a tensor.")
        self.assertEqual(item['text'].shape[0], 50, "Text tensor length mismatch.")

    def test_test_dataset(self):
        """
        Test the 'test' split of MultiModalEmotionDataset.
        """
        test_dataset = MultiModalEmotionDataset(
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='test'
        )

        # Check dataset length
        self.assertEqual(len(test_dataset), 2 * self.num_emotions,
                         "Test dataset length mismatch.")

        # Check a random item
        item = test_dataset[1]
        self.assertIn('audio', item)
        self.assertIn('image', item)
        self.assertIn('text', item)
        self.assertIn('emotion', item)

        # Check shapes
        self.assertEqual(item['audio'].shape, (1, 16000 * 10), "Audio shape mismatch.")
        self.assertEqual(item['image'].shape, (3, 224, 224), "Image shape mismatch.")
        self.assertIsInstance(item['text'], torch.Tensor, "Text should be a tensor.")
        self.assertEqual(item['text'].shape[0], 50, "Text tensor length mismatch.")

    # def test_data_preprocessing (self):
    #     """Test data preprocessing and validation steps."""
    #     dataset = MultiModalEmotionDataset (
    #         image_data=self.image_df,
    #         audio_data=self.audio_df,
    #         text_data=self.text_df,
    #         split='train'
    #     )
    #
    #     # Test image normalization
    #     item = dataset[0]
    #     image_tensor = item['image']
    #
    #     # Expected values for a white image after normalization
    #     expected_values = [
    #         (1.0 - 0.485) / 0.229,  # For red channel
    #         (1.0 - 0.456) / 0.224,  # For green channel
    #         (1.0 - 0.406) / 0.225  # For blue channel
    #     ]
    #
    #     for channel in range (3):
    #         channel_mean = image_tensor[channel].mean ()
    #         self.assertTrue (
    #             torch.allclose (
    #                 channel_mean,
    #                 torch.tensor (expected_values[channel]),
    #                 atol=0.1  # Allowing some tolerance for numerical precision
    #             ),
    #             f"Channel {channel} normalization is incorrect. "
    #             f"Expected ~{expected_values[channel]:.2f}, got {channel_mean:.2f}"
    #         )
    #
    #     # Test audio padding
    #     self.assertEqual (
    #         item['audio'].shape[1],
    #         16000 * 10,  # 10 seconds at 16kHz
    #         "Audio should be padded to exactly 10 seconds"
    #     )
    #
    #     # Test text padding
    #     self.assertEqual (
    #         item['text'].shape[0],
    #         50,
    #         "Text should be padded to exactly 50 tokens"
    #     )

    def test_augmentation_consistency (self):
        """Test that augmentation behaves differently for train/test."""
        train_dataset = MultiModalEmotionDataset (
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='train'
        )

        test_dataset = MultiModalEmotionDataset (
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='test'
        )

        # Get same item from both datasets multiple times
        train_items = [train_dataset[0]['image'] for _ in range (5)]
        test_items = [test_dataset[0]['image'] for _ in range (5)]

        # Train items should be different (augmented)
        train_differences = [
            (train_items[i] - train_items[i + 1]).abs ().mean ()
            for i in range (len (train_items) - 1)
        ]
        self.assertTrue (
            all (diff > 0 for diff in train_differences),
            "Train augmentation should produce different results"
        )

        # Test items should be identical (no augmentation)
        test_differences = [
            (test_items[i] - test_items[i + 1]).abs ().mean ()
            for i in range (len (test_items) - 1)
        ]
        self.assertTrue (
            all (diff == 0 for diff in test_differences),
            "Test items should be identical with no augmentation"
        )

    def test_loading_performance (self):
        """Test data loading performance."""
        import time

        dataset = MultiModalEmotionDataset (
            image_data=self.image_df,
            audio_data=self.audio_df,
            text_data=self.text_df,
            split='train'
        )

        start_time = time.time ()
        _ = dataset[0]
        load_time = time.time () - start_time

        self.assertLess (
            load_time,
            0.1,  # 100ms threshold
            f"Data loading too slow: {load_time:.3f}s"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
