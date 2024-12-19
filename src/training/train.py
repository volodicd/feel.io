import os
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
# Import from your project structure
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel, MultiModalLoss
from src.utils.visualization import ModelVisualizer, LRFinder
from src.utils.data_aligment import align_datasets


class EmotionTrainer:
    def __init__ (self, config: Dict):
        """
        Initialize trainer with configuration.
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.setup_logging ()
        self.setup_directories ()
        self.setup_device ()
        self.setup_tensorboard ()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def setup_device (self):
        """Setup device for training with proper initialization for M1"""
        # Enable MPS fallback for unsupported operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        if torch.backends.mps.is_available ():
            self.device = torch.device ("mps")
            logging.info ("Using Apple M1 GPU (MPS) with CPU fallback for unsupported operations")
            logging.info (
                f"PYTORCH_ENABLE_MPS_FALLBACK is set to: {os.getenv ('PYTORCH_ENABLE_MPS_FALLBACK', 'Not Set')}")
        else:
            self.device = torch.device ("cpu")
            logging.info ("MPS not available, using CPU")

        logging.info (f"PyTorch version: {torch.__version__}")
        logging.info (f"Device: {self.device}")

    def setup_logging (self):
        """Initialize logging configuration"""
        timestamp = datetime.now ().strftime ("%Y%m%d_%H%M%S")
        self.log_dir = Path ('logs') / timestamp
        self.log_dir.mkdir (parents=True, exist_ok=True)

        # Reset any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler (handler)

        log_file = self.log_dir / 'training.log'

        logging.basicConfig (
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler (log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler ()
            ]
        )

        logging.info (f"{'=' * 50}")
        logging.info ("Training Session Started")
        logging.info (f"Timestamp: {timestamp}")
        logging.info (f"Log file: {log_file}")
        logging.info (f"Configuration:")
        for key, value in self.config.items ():
            logging.info (f"  {key}: {value}")
        logging.info (f"{'=' * 50}")

    def setup_directories (self):
        """Create necessary directories for saving results"""
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.plot_dir = self.log_dir / 'plots'
        self.checkpoint_dir.mkdir (exist_ok=True)
        self.plot_dir.mkdir (exist_ok=True)

    def setup_tensorboard (self):
        """Initialize tensorboard writer"""
        self.writer = SummaryWriter (self.log_dir / 'tensorboard')

    def setup_model (self):
        """Initialize model and move components to appropriate devices"""
        try:
            # Initialize model
            self.model = ImprovedEmotionModel (
                num_emotions=len (self.emotion_labels),
                dropout=0.5,
                vocab_size=20000,
                embed_dim=128,
                rnn_hidden=256
            )

            # Audio components stay on CPU
            self.model.audio_encoder = self.model.audio_encoder.to ('cpu', dtype=torch.float32)
            self.model.audio_norm = self.model.audio_norm.to ('cpu', dtype=torch.float32)

            # Other components go to MPS/CPU depending on availability
            components_to_device = [
                'image_encoder', 'text_embedding', 'text_rnn', 'text_proj',
                'cross_attention', 'image_norm', 'text_norm', 'fusion_norm'
            ]

            for component in components_to_device:
                if hasattr (self.model, component):
                    setattr (self.model, component,
                             getattr (self.model, component).to (self.device, dtype=torch.float32))

            # Handle classifiers
            for name, classifier in self.model.classifiers.items ():
                if name == 'audio':
                    classifier.to ('cpu', dtype=torch.float32)
                else:
                    classifier.to (self.device, dtype=torch.float32)

            # Initialize loss and optimizers
            self.criterion = MultiModalLoss ().to (self.device, dtype=torch.float32)
            self.optimizer = torch.optim.AdamW (
                self.model.parameters (),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (
                self.optimizer,
                mode='min',
                patience=self.config['scheduler_patience'],
                factor=0.5,
                verbose=True
            )

        except Exception as e:
            logging.error (f"Error in model setup: {str (e)}")
            raise

    def train (self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        self.setup_model ()
        best_acc = 0.0
        early_stop_counter = 0

        for epoch in range (self.config['epochs']):
            # Training phase
            train_loss, train_metrics = self.train_epoch (train_loader, epoch)
            logging.info (f"\n{'-' * 20} Epoch {epoch + 1}/{self.config['epochs']} {'-' * 20}")
            logging.info ("Training Phase:")
            logging.info (f"  Loss: {train_loss:.4f}")
            logging.info (f"  Image Accuracy: {train_metrics['image_accuracy']:.4f}")
            logging.info (f"  Audio Accuracy: {train_metrics['audio_accuracy']:.4f}")
            logging.info (f"  Fusion Accuracy: {train_metrics['fusion_accuracy']:.4f}")

            # Validation phase
            val_loss, val_metrics, predictions = self.validate (val_loader, epoch)
            logging.info ("Validation Phase:")
            logging.info (f"  Loss: {val_loss:.4f}")
            logging.info (f"  Image Accuracy: {val_metrics['image_accuracy']:.4f}")
            logging.info (f"  Audio Accuracy: {val_metrics['audio_accuracy']:.4f}")
            logging.info (f"  Fusion Accuracy: {val_metrics['fusion_accuracy']:.4f}")

            # Update learning rate scheduler
            self.scheduler.step (val_loss)

            # Log metrics
            self.log_metrics (train_metrics, 'train', epoch)
            self.log_metrics (val_metrics, 'val', epoch)

            # Check for improvement
            if val_metrics['fusion_accuracy'] > best_acc:
                best_acc = val_metrics['fusion_accuracy']
                self.save_checkpoint (epoch, val_metrics)
                early_stop_counter = 0
                logging.info (f"New best accuracy: {best_acc:.4f}")
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config['patience']:
                logging.info ("Early stopping triggered")
                break

        logging.info (f"Best validation accuracy: {best_acc:.4f}")
        self.writer.close ()

    def train_epoch (self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train ()
        total_loss = 0.0
        predictions = {'image': [], 'audio': [], 'fusion': []}
        all_targets = []

        progress_bar = tqdm (train_loader, desc=f'Epoch {epoch + 1} Training')
        for idx, batch in enumerate (progress_bar):
            try:
                # Process audio on CPU
                audio = batch['audio'].to ('cpu', dtype=torch.float32).contiguous ()
                audio_features = self.model.audio_encoder (audio)

                # Log shapes for debugging
                if idx == 0:  # Only log first batch
                    logging.info (f"Audio input shape: {audio.shape}")
                    logging.info (f"Audio features shape before squeeze: {audio_features.shape}")

                # Remove the last dimension if it's 1
                if audio_features.size (-1) == 1:
                    audio_features = audio_features.squeeze (-1)
                    if idx == 0:  # Only log first batch
                        logging.info (f"Audio features shape after squeeze: {audio_features.shape}")

                audio_features = self.model.audio_norm (audio_features)
                audio_pred = self.model.classifiers['audio'] (audio_features)

                # Process other inputs on MPS/CPU
                image = batch['image'].to (self.device, dtype=torch.float32).contiguous ()
                targets = batch['emotion'].to (self.device, dtype=torch.long).contiguous ()

                # Move audio predictions to main device for loss calculation
                audio_pred = audio_pred.to (self.device)

                # Clear gradients
                self.optimizer.zero_grad (set_to_none=True)

                # Forward pass
                outputs = self.model (
                    image=image,
                    audio=audio,
                    audio_features=audio_features.to (self.device),
                    text_input=batch.get ('text', None)
                )

                outputs['audio_pred'] = audio_pred
                loss = self.criterion (outputs, targets)

                # Check for NaN/Inf values
                if not torch.isfinite (loss):
                    raise ValueError (f"Loss is {loss}, training cannot continue")

                # Backward pass
                loss.backward ()

                if self.config.get ('grad_clip'):
                    torch.nn.utils.clip_grad_norm_ (
                        self.model.parameters (),
                        self.config['grad_clip']
                    )

                self.optimizer.step ()

                # Update metrics
                total_loss += loss.item ()

                with torch.no_grad ():
                    for key in outputs:
                        pred = outputs[key].argmax (1).cpu ()
                        predictions[key.replace ('_pred', '')].extend (pred.numpy ())
                    all_targets.extend (targets.cpu ().numpy ())

                # Update progress bar
                progress_bar.set_postfix ({'loss': loss.item ()})

            except Exception as e:
                logging.error (f"Error in training batch: {str (e)}")
                continue

        metrics = {
            'loss': total_loss / len (train_loader),
            'true_labels': np.array (all_targets)
        }
        for key, preds in predictions.items ():
            metrics[f'{key}_accuracy'] = np.mean (
                np.array (preds) == metrics['true_labels']
            )

        return metrics['loss'], metrics

    @torch.no_grad ()
    def validate (self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict, Dict]:
        """Validate the model"""
        self.model.eval ()
        total_loss = 0.0
        predictions = {'image': [], 'audio': [], 'fusion': []}
        all_targets = []

        for batch in tqdm (val_loader, desc='Validation'):
            try:
                # Process audio on CPU
                audio = batch['audio'].to ('cpu', dtype=torch.float32).contiguous ()
                audio_features = self.model.audio_encoder (audio)
                # Remove the last dimension if it's 1
                if audio_features.size (-1) == 1:
                    audio_features = audio_features.squeeze (-1)
                audio_features = self.model.audio_norm (audio_features)
                audio_pred = self.model.classifiers['audio'] (audio_features)

                # Process other inputs on MPS/CPU
                image = batch['image'].to (self.device, dtype=torch.float32).contiguous ()
                targets = batch['emotion'].to (self.device, dtype=torch.long).contiguous ()

                # Move audio predictions to main device for loss calculation
                audio_pred = audio_pred.to (self.device)

                # Forward pass
                outputs = self.model (
                    image=image,
                    audio=audio,
                    audio_features=audio_features.to (self.device),
                    text_input=batch.get ('text', None)
                )

                outputs['audio_pred'] = audio_pred
                loss = self.criterion (outputs, targets)

                total_loss += loss.item ()

                for key in outputs:
                    pred = outputs[key].argmax (1).cpu ()
                    predictions[key.replace ('_pred', '')].extend (pred.numpy ())
                all_targets.extend (targets.cpu ().numpy ())

            except Exception as e:
                logging.error (f"Error in validation batch: {str (e)}")
                continue

        metrics = {
            'loss': total_loss / len (val_loader),
            'true_labels': np.array (all_targets)
        }
        for key, preds in predictions.items ():
            metrics[f'{key}_accuracy'] = np.mean (
                np.array (preds) == metrics['true_labels']
            )
        return metrics['loss'], metrics, predictions

    def save_checkpoint (self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict (),
            'optimizer_state_dict': self.optimizer.state_dict (),
            'scheduler_state_dict': self.scheduler.state_dict (),
            'metrics': metrics,
            'config': self.config
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_acc_{metrics["fusion_accuracy"]:.3f}.pt'
        torch.save (checkpoint, checkpoint_path)
        logging.info (f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint (self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load (checkpoint_path, map_location='cpu')
        self.model.load_state_dict (checkpoint['model_state_dict'])
        self.optimizer.load_state_dict (checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict (checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

    def log_metrics (self, metrics: Dict, phase: str, epoch: int):
        """Log metrics to tensorboard"""
        for key, value in metrics.items ():
            if key == 'true_labels':
                continue
            self.writer.add_scalar (f"{phase}/{key}", value, epoch)


def main ():
    # Set multiprocessing start method to 'fork' to match Linux/CUDA behavior
    import multiprocessing
    if multiprocessing.get_start_method (allow_none=True) != 'fork':
        multiprocessing.set_start_method ('fork', force=True)

    # Configuration optimized for M1
    config = {
        'batch_size': 32,  # M1 can handle larger batches
        'num_workers': 4,  # Adjusted for M1
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 15,
        'scheduler_patience': 5,
        'grad_clip': 1.0,
        'pin_memory': True
    }

    try:
        # Load datasets
        logging.info ("Loading datasets...")
        image_data = pd.concat ([
            pd.read_csv ('data/processed/fer2013.csv'),
            pd.read_csv ('data/processed/expw.csv')
        ])
        audio_data = pd.read_csv ('data/processed/ravdess.csv')
        text_data = pd.read_csv ('data/processed/goemotions.csv')

        # Align datasets
        logging.info ("Aligning datasets...")
        aligned_data = align_datasets (image_data, audio_data, text_data)

        if aligned_data is None:
            raise RuntimeError ("Failed to align datasets")

        logging.info (
            f"Aligned dataset sizes - Train: {len (aligned_data['image'][aligned_data['image']['split'] == 'train'])}, "
            f"Test: {len (aligned_data['image'][aligned_data['image']['split'] == 'test'])}")

        # Create datasets with aligned data
        train_dataset = MultiModalEmotionDataset (
            image_data=aligned_data['image'],
            audio_data=aligned_data['audio'],
            text_data=aligned_data['text'],
            split='train'
        )

        val_dataset = MultiModalEmotionDataset (
            image_data=aligned_data['image'],
            audio_data=aligned_data['audio'],
            text_data=aligned_data['text'],
            split='test'
        )

        # Data loaders with specific multiprocessing context
        import torch.multiprocessing as mp
        train_loader = DataLoader (
            train_dataset,
            batch_size=config['batch_size'],
            sampler=RandomSampler (train_dataset),
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            multiprocessing_context=mp.get_context ('fork')
        )

        val_loader = DataLoader (
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        # Initialize and train
        trainer = EmotionTrainer (config)
        trainer.train (train_loader, val_loader)

    except Exception as e:
        logging.error (f"Training error: {str (e)}")
        raise


if __name__ == '__main__':
    main ()