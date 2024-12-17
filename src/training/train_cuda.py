import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class MultiModalLoss:
    def __call__ (self, outputs, targets):
        """CrossEntropy loss for all outputs."""
        loss_image = F.cross_entropy (outputs['image_pred'], targets)
        loss_audio = F.cross_entropy (outputs['audio_pred'], targets)
        loss_fusion = F.cross_entropy (outputs['fusion_pred'], targets)
        return loss_image + loss_audio + loss_fusion


class EmotionTrainer:
    def __init__ (self, config):
        self.config = config
        self.setup_logging ()
        self.setup_directories ()
        self.setup_device ()
        self.setup_tensorboard ()
        self.loss_fn = MultiModalLoss ()
        self.scaler = torch.cuda.amp.GradScaler ()  # Mixed precision training

    def setup_logging (self):
        timestamp = datetime.now ().strftime ("%Y%m%d_%H%M%S")
        self.log_dir = Path ('logs') / timestamp
        self.log_dir.mkdir (parents=True, exist_ok=True)
        logging.basicConfig (
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_directories (self):
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir (exist_ok=True)

    def setup_device (self):
        if not torch.cuda.is_available ():
            raise RuntimeError ("This script requires CUDA GPU. No GPU found!")
        self.device = torch.device ("cuda")
        torch.backends.cudnn.benchmark = True
        logging.info (f"Using CUDA Device: {torch.cuda.get_device_name (0)}")

    def setup_tensorboard (self):
        self.writer = SummaryWriter (self.log_dir / 'tensorboard')

    def train (self, train_loader: DataLoader, val_loader: DataLoader):
        self.model = ImprovedEmotionModel ().to (self.device)
        optimizer = torch.optim.AdamW (
            self.model.parameters (),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = OneCycleLR (
            optimizer, max_lr=self.config['learning_rate'],
            epochs=self.config['epochs'],
            steps_per_epoch=len (train_loader)
        )

        best_acc = 0.0
        for epoch in range (self.config['epochs']):
            logging.info (f"Epoch {epoch + 1}/{self.config['epochs']}")
            self.train_epoch (train_loader, optimizer, scheduler, epoch)
            val_acc = self.validate (val_loader, epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint (epoch, best_acc)

    def train_epoch (self, loader, optimizer, scheduler, epoch):
        self.model.train ()
        total_loss = 0.0
        for batch in tqdm (loader, desc=f"Epoch {epoch + 1} Training"):
            audio = batch['audio'].to (self.device, non_blocking=True)
            image = batch['image'].to (self.device, non_blocking=True)
            targets = batch['emotion'].to (self.device, non_blocking=True)
            optimizer.zero_grad ()

            with torch.cuda.amp.autocast ():
                outputs = self.model (image=image, audio=audio)
                loss = self.loss_fn (outputs, targets)

            self.scaler.scale (loss).backward ()
            self.scaler.step (optimizer)
            self.scaler.update ()
            scheduler.step ()
            total_loss += loss.item ()

        avg_loss = total_loss / len (loader)
        logging.info (f"Train Loss: {avg_loss:.4f}")

    @torch.no_grad ()
    def validate (self, loader, epoch):
        self.model.eval ()
        total_correct = 0
        total_samples = 0
        for batch in tqdm (loader, desc=f"Validation"):
            audio = batch['audio'].to (self.device, non_blocking=True)
            image = batch['image'].to (self.device, non_blocking=True)
            targets = batch['emotion'].to (self.device, non_blocking=True)

            outputs = self.model (image=image, audio=audio)
            preds = outputs['fusion_pred'].argmax (dim=1)
            total_correct += (preds == targets).sum ().item ()
            total_samples += targets.size (0)

        accuracy = total_correct / total_samples
        logging.info (f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def save_checkpoint (self, epoch, accuracy):
        checkpoint_path = self.checkpoint_dir / f"model_epoch{epoch}_acc{accuracy:.4f}.pt"
        torch.save (self.model.state_dict (), checkpoint_path)
        logging.info (f"Model saved at {checkpoint_path}")


if __name__ == '__main__':
    config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 10,
        'batch_size': 16,
        'num_workers': 4
    }

    train_dataset = MultiModalEmotionDataset (
        image_data='data/processed/fer2013.csv',
        audio_data='data/processed/ravdess.csv',
        text_data='data/processed/goemotions.csv',
        split='train'
    )
    val_dataset = MultiModalEmotionDataset (
        image_data='data/processed/fer2013.csv',
        audio_data='data/processed/ravdess.csv',
        text_data='data/processed/goemotions.csv',
        split='test'
    )

    train_loader = DataLoader (train_dataset, batch_size=config['batch_size'],
                               sampler=RandomSampler (train_dataset), num_workers=config['num_workers'])
    val_loader = DataLoader (val_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=config['num_workers'])

    trainer = EmotionTrainer (config)
    trainer.train (train_loader, val_loader)
