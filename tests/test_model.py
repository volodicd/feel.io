import unittest
import torch
import numpy as np
from src.models.model import ImprovedEmotionModel, MultiModalLoss

class TestImprovedEmotionModel(unittest.TestCase):
    def setUp(self):
        """
        Prepare dummy batches of audio, image, and text data for testing the model forward pass.
        """
        self.batch_size = 4
        self.num_emotions = 7  # default in ImprovedEmotionModel
        self.audio_length = 16000  # 1 second if sample_rate=16000
        self.image_height = 224
        self.image_width = 224
        self.text_length = 50  # Maximum length for text sequences

        # Create dummy audio: shape [batch_size, 1, audio_length]
        self.dummy_audio = torch.randn(self.batch_size, 1, self.audio_length)

        # Create dummy images: shape [batch_size, 3, H, W]
        self.dummy_image = torch.randn(self.batch_size, 3, self.image_height, self.image_width)

        # Create dummy text: shape [batch_size, text_length]
        self.dummy_text = torch.randint(0, 100, (self.batch_size, self.text_length))

        # Initialize the model and loss
        self.model = ImprovedEmotionModel(num_emotions=self.num_emotions, dropout=0.5)
        self.criterion = MultiModalLoss()

    def test_forward_pass(self):
        """
        Test that the model forward pass returns the correct keys and output shapes.
        """
        outputs = self.model(self.dummy_audio, self.dummy_image, self.dummy_text)

        # Verify dictionary keys
        expected_keys = ['image_pred', 'audio_pred', 'text_pred', 'fusion_pred']
        self.assertTrue(all(k in outputs for k in expected_keys),
                        f"Model output missing one of the expected keys {expected_keys}")

        # Verify output shapes
        for key in expected_keys:
            preds = outputs[key]
            self.assertEqual(preds.shape, (self.batch_size, self.num_emotions),
                             f"{key} has incorrect shape. Expected {[self.batch_size, self.num_emotions]}")

    def test_loss_computation(self):
        """
        Test the multi-modal loss with random targets.
        Ensure the loss is computed without error and is a scalar.
        """
        outputs = self.model(self.dummy_audio, self.dummy_image, self.dummy_text)
        # Create random targets
        targets = torch.randint(0, self.num_emotions, (self.batch_size,))

        loss = self.criterion(outputs, targets)
        self.assertIsInstance(loss.item(), float, "Loss is not a scalar float value")

        # Test backward pass
        loss.backward()
        # If we got here without error, backward pass succeeded

    def tearDown(self):
        """
        Cleanup if needed. For now, nothing special.
        """
        pass


if __name__ == '__main__':
    unittest.main()
