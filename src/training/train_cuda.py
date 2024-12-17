import torch
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel, MultiModalLoss
from src.utils.visualization import ModelVisualizer, LRFinder


class EmotionTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_device()
        self.setup_tensorboard()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    def setup_device(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not found!")

        self.device = torch.device("cuda")
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        logging.info(f"Using CUDA Device {gpu_id}: {gpu_name}")
        torch.backends.cudnn.benchmark = True

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path('logs') / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_directories(self):
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.plot_dir = self.log_dir / 'plots'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

    def setup_tensorboard(self):
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')

    def setup_model(self):
        self.model = ImprovedEmotionModel().to(self.device)
        self.criterion = MultiModalLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=self.config['scheduler_patience'], factor=0.5, verbose=True
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        self.setup_model()
        best_acc = 0.0
        early_stop_counter = 0

        for epoch in range(self.config['epochs']):
            logging.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.scheduler.step(val_loss)
            self.log_metrics(train_metrics, 'train', epoch)
            self.log_metrics(val_metrics, 'val', epoch)

            if val_metrics['fusion_accuracy'] > best_acc:
                best_acc = val_metrics['fusion_accuracy']
                self.save_checkpoint(epoch, val_metrics)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config['patience']:
                logging.info("Early stopping triggered")
                break

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc='Training')

        for batch in progress_bar:
            audio, image, targets = batch['audio'].to(self.device), batch['image'].to(self.device), batch['emotion'].to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(image=image, audio=audio)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss, {}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        predictions, all_targets = [], []

        for batch in tqdm(val_loader, desc='Validation'):
            audio, image, targets = batch['audio'].to(self.device), batch['image'].to(self.device), batch['emotion'].to(self.device)
            with torch.cuda.amp.autocast():
                outputs = self.model(image=image, audio=audio)
                loss = self.criterion(outputs, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss, {'fusion_accuracy': 0.8}  # Replace with real metrics calculation

    def save_checkpoint(self, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics, 'config': self.config
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    def log_metrics(self, metrics: Dict, phase: str, epoch: int):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, epoch)


def main():
    config = {
        'batch_size': 16, 'num_workers': 4, 'learning_rate': 1e-4,
        'weight_decay': 0.1, 'epochs': 10, 'patience': 5, 'scheduler_patience': 2
    }

    image_data = pd.concat([
        pd.read_csv('data/processed/fer2013.csv').assign(split='train'),
        pd.read_csv('data/processed/expw.csv').assign(split='test')
    ]).reset_index(drop=True)

    audio_data = pd.read_csv('data/processed/ravdess.csv').assign(split='train').reset_index(drop=True)
    goemotions_data = pd.read_csv('data/processed/goemotions.csv').assign(split='train').reset_index(drop=True)

    train_dataset = MultiModalEmotionDataset(
        image_data=image_data, audio_data=audio_data, text_data=goemotions_data, split='train'
    )
    val_dataset = MultiModalEmotionDataset(
        image_data=image_data, audio_data=audio_data, text_data=goemotions_data, split='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    trainer = EmotionTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
