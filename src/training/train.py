import yaml
import torch
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

# Import from your project structure
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel
from src.utils.visualization import ModelVisualizer, LRFinder

def load_config(config_path: str):
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class EmotionTrainer:
    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_device()
        self.setup_tensorboard()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
        self.visualizer = ModelVisualizer(self.log_dir / 'plots')

    def setup_device(self):
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

    def setup_logging(self):
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

        logging.info(f"Configuration: {self.config}")

    def setup_directories(self):
        """Create necessary directories for saving results."""
        self.checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup_tensorboard(self):
        """Initialize TensorBoard writer."""
        if self.config['logging']['tensorboard']:
            self.writer = SummaryWriter(self.log_dir / 'tensorboard')

    def setup_model(self):
        """Initialize model, optimizer, and scheduler."""
        self.model = ImprovedEmotionModel().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.config['loss']['class_weights']).to(self.device))

        optimizer_config = self.config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )

        scheduler_config = optimizer_config['scheduler']
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor'],
            verbose=True
        )

    def train(self, train_loader, val_loader):
        """Main training loop."""
        self.setup_model()
        best_acc = 0.0
        early_stop_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics = self.validate(val_loader, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Plot and log metrics
            self.visualizer.plot_loss(train_losses, val_losses)
            self.log_metrics({'train_loss': train_loss, 'val_loss': val_loss, **val_metrics}, epoch)

            # Check for early stopping
            if val_metrics['fusion_accuracy'] > best_acc:
                best_acc = val_metrics['fusion_accuracy']
                early_stop_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config['training']['patience']:
                logging.info("Early stopping triggered.")
                break

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
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

        return total_loss / len(train_loader)

    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        return total_loss / len(val_loader), {'fusion_accuracy': accuracy}

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved: {checkpoint_path}")

    def log_metrics(self, metrics: Dict, epoch: int):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)


def main():
    config_path = 'configs/training_config.yaml'
    config = load_config(config_path)

    trainer = EmotionTrainer(config)

    # Load datasets
    train_dataset = MultiModalEmotionDataset(config['dataset']['train'])
    val_dataset = MultiModalEmotionDataset(config['dataset']['val'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
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
