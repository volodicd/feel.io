# src/training/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
from torch.utils.tensorboard import SummaryWriter

# Import from your new project structure
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel, MultiModalLoss

# If these utilities exist in src/utils/visualization.py
from src.utils.visualization import ModelVisualizer, LRFinder


class EmotionTrainer:
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.

        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_device()
        self.setup_tensorboard()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def setup_device(self):
        """Setup computing device for training"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logging.info("Using MPS device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU device")

    def setup_logging(self):
        """Initialize logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path('logs') / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_directories(self):
        """Create necessary directories for saving results"""
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.plot_dir = self.log_dir / 'plots'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def setup_tensorboard(self):
        """Initialize tensorboard writer"""
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')

    def setup_model(self):
        """Initialize model, optimizer, criterion and scheduler"""
        self.model = ImprovedEmotionModel().to(self.device)
        self.criterion = MultiModalLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['scheduler_patience'],
            factor=0.5,
            verbose=True
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        self.setup_model()
        best_acc = 0.0
        early_stop_counter = 0

        for epoch in range(self.config['epochs']):
            logging.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")

            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            logging.info(f"Training Loss: {train_loss:.4f}")

            # Validation phase
            val_loss, val_metrics, predictions = self.validate(val_loader, epoch)
            logging.info(f"Validation Loss: {val_loss:.4f}")

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Log metrics to tensorboard
            self.log_metrics(train_metrics, 'train', epoch)
            self.log_metrics(val_metrics, 'val', epoch)

            # Optionally visualize
            self.plot_predictions(predictions, val_metrics['true_labels'], epoch)
            self.plot_roc_curves(predictions, val_metrics['true_labels'], epoch)

            # Check for improvement in fusion accuracy
            if val_metrics['fusion_accuracy'] > best_acc:
                best_acc = val_metrics['fusion_accuracy']
                self.save_checkpoint(epoch, val_metrics)
                early_stop_counter = 0
                logging.info(f"New best accuracy: {best_acc:.4f}")
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config['patience']:
                logging.info("Early stopping triggered")
                break

        logging.info(f"Best validation accuracy: {best_acc:.4f}")
        self.writer.close()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        predictions = {'image': [], 'audio': [], 'fusion': []}
        all_targets = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')
        for batch in progress_bar:
            # Move data to device
            audio = batch['audio'].to(self.device)
            image = batch['image'].to(self.device)
            targets = batch['emotion'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(audio, image)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            self.optimizer.step()

            total_loss += loss.item()
            for key in outputs:
                pred = outputs[key].argmax(1)
                predictions[key.replace('_pred', '')].extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        metrics = {
            'loss': total_loss / len(train_loader),
            'true_labels': np.array(all_targets)
        }
        for key, preds in predictions.items():
            metrics[f'{key}_accuracy'] = np.mean(
                np.array(preds) == metrics['true_labels']
            )

        return metrics['loss'], metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        predictions = {'image': [], 'audio': [], 'fusion': []}
        all_targets = []

        for batch in tqdm(val_loader, desc='Validation'):
            audio = batch['audio'].to(self.device)
            image = batch['image'].to(self.device)
            targets = batch['emotion'].to(self.device)

            outputs = self.model(audio, image)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            for key in outputs:
                pred = outputs[key].argmax(1)
                predictions[key.replace('_pred', '')].extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        metrics = {
            'loss': total_loss / len(val_loader),
            'true_labels': np.array(all_targets)
        }
        for key, preds in predictions.items():
            metrics[f'{key}_accuracy'] = np.mean(
                np.array(preds) == metrics['true_labels']
            )

        return metrics['loss'], metrics, predictions

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_acc_{metrics["fusion_accuracy"]:.3f}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

    def log_metrics(self, metrics: Dict, phase: str, epoch: int):
        """Log metrics to tensorboard"""
        for key, value in metrics.items():
            if key == 'true_labels':
                continue
            self.writer.add_scalar(f"{phase}/{key}", value, epoch)

    def plot_predictions(self, predictions: Dict[str, List[int]],
                         true_labels: np.ndarray, epoch: int):
        """(Optional) Example: plot confusion matrices or store them."""
        pass  # Implementation omitted for brevity

    def plot_roc_curves(self, predictions: Dict[str, List[int]],
                        true_labels: np.ndarray, epoch: int):
        """(Optional) Example: plot ROC curves for each emotion."""
        pass  # Implementation omitted for brevity


def main():
    # Example training config
    config = {
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'epochs': 50,
        'patience': 20,
        'scheduler_patience': 2,
        'grad_clip': 1.0
    }

    try:
        # Load CSVs from your new processed data folder
        image_data = pd.read_csv('data/processed/fer2013.csv')
        expw_data = pd.read_csv('data/processed/expw.csv')
        audio_data = pd.read_csv('data/processed/ravdess.csv')

        # Combine image datasets
        image_data = pd.concat([
            image_data,
            expw_data,
        ]).reset_index(drop=True)

        # Load CSV
        goemotions_data = pd.read_csv ('data/processed/goemotions.csv')

        # Then combine image_data, audio_data, and text_data in your MultiModalEmotionDataset
        train_dataset = MultiModalEmotionDataset (
            image_data=image_data,
            audio_data=audio_data,
            text_data=goemotions_data,
            split='train'
        )

        val_dataset = MultiModalEmotionDataset (
            image_data=image_data,
            audio_data=audio_data,
            text_data=goemotions_data,
            split='test'
        )



        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # Initialize and train
        trainer = EmotionTrainer(config)
        trainer.train(train_loader, val_loader)

    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
