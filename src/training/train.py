import torch
import pandas as pd
from pathlib import Path
import yaml
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel, MultiModalLoss
from src.utils.visualization import ModelVisualizer, LRFinder
from src.utils.data_aligment import align_datasets, label_level_align

def load_config(config_path: str):
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class EmotionTrainer:
    def __init__ (self, config: Dict):
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
        self.scaler = torch.amp.GradScaler() if self.config['mixed_precision'] else None
        self.visualizer = ModelVisualizer(self.plot_dir)

    def setup_device (self):
        """Setup CUDA device for training."""
        if not torch.cuda.is_available():
            raise RuntimeError("This script requires CUDA GPU. No GPU found!")

        self.device = torch.device("cuda")
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        logging.info(f"Using CUDA Device {gpu_id}: {gpu_name}")

        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def setup_logging (self):
        """Initialize logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(self.config['logging']['log_dir']) / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / 'training.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        logging.info (f"{'=' * 50}")
        logging.info ("Training Session Started")
        logging.info (f"Timestamp: {timestamp}")
        logging.info (f"Log file: {log_file}")
        logging.info ("Configuration:")
        for key, value in self.config.items():
            logging.info(f"  {key}: {value}")
        logging.info (f"{'=' * 50}")

    def setup_directories (self):
        """Create necessary directories for saving results."""
        self.checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        self.plot_dir = self.log_dir / 'plots'
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir.mkdir(exist_ok=True, parents=True)

    def setup_tensorboard (self):
        """Initialize tensorboard writer."""
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')

    def setup_model (self):
        """Initialize model, optimizer, and scheduler."""
        self.model = ImprovedEmotionModel().cuda()
        self.criterion = MultiModalLoss().cuda()

        optimizer_config = self.config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=optimizer_config['scheduler']['patience'],
            factor=optimizer_config['scheduler']['factor'],
            verbose=True
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with logging and early stopping."""
        self.setup_model()
        best_acc = 0.0
        early_stop_counter = 0

        train_losses, val_losses = [], []

        for epoch in range(self.config['training']['epochs']):
            logging.info(f"Starting epoch {epoch + 1}/{self.config['training']['epochs']}")

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics = self.validate(val_loader, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.visualizer.plot_loss(train_losses, val_losses, epoch)
            self.log_metrics({'train_loss': train_loss, 'val_loss': val_loss, 'fusion_accuracy': val_metrics['fusion_accuracy']}, epoch)

            # Early stopping and checkpoint saving
            if val_metrics['fusion_accuracy'] > best_acc:
                best_acc = val_metrics['fusion_accuracy']
                early_stop_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                logging.info(f"New best accuracy: {best_acc:.4f}")
            else:
                early_stop_counter += 1
                logging.info(f"No improvement. Early stopping counter: {early_stop_counter}")

            if early_stop_counter >= self.config['training']['patience']:
                logging.info("Early stopping triggered.")
                break

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for a single epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            inputs, labels = batch
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(enabled=self.config['mixed_precision']):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}: Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, labels = batch
                inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                with torch.amp.autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        logging.info(f"Epoch {epoch + 1}: Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, {'fusion_accuracy': accuracy}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save the model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}_acc_{metrics['fusion_accuracy']:.4f}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")

    def log_metrics(self, metrics: Dict, epoch: int):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

def main():
    config_path = 'configs/training_config.yaml'
    config = load_config(config_path)

    trainer = EmotionTrainer(config)

    train_dataset = MultiModalEmotionDataset(config['dataset']['train'])
    val_dataset = MultiModalEmotionDataset(config['dataset']['val'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=RandomSampler(train_dataset),
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
