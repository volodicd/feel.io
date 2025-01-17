# src/training/train.py

import torch
import pandas as pd
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
from src.models.model import ImprovedEmotionModel, MultiModalLoss
from src.utils.visualization import ModelVisualizer, LRFinder
from src.utils.data_aligment import align_datasets, label_level_align


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
        self.scaler = torch.amp.GradScaler()  # For mixed precision training

    def setup_device (self):
        """Setup CUDA device for training with proper initialization"""
        if not torch.cuda.is_available ():
            raise RuntimeError ("This script requires CUDA GPU. No GPU found!")

        self.device = torch.device ("cuda")
        gpu_id = torch.cuda.current_device ()
        gpu_name = torch.cuda.get_device_name (gpu_id)
        logging.info (f"Using CUDA Device {gpu_id}: {gpu_name}")

        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Log GPU information
        total_memory = torch.cuda.get_device_properties (gpu_id).total_memory
        logging.info (f"Total GPU Memory: {total_memory / 1e9:.2f} GB")

    def setup_logging (self):
        """Initialize logging configuration"""
        timestamp = datetime.now ().strftime ("%Y%m%d_%H%M%S")
        self.log_dir = Path ('logs') / timestamp
        self.log_dir.mkdir (parents=True, exist_ok=True)

        # Reset any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler (handler)

        # Create log file path
        log_file = self.log_dir / 'training.log'

        # Configure root logger
        logging.basicConfig (
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler (log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler ()
            ]
        )

        # Log initial training information
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
        """Initialize model, optimizer, criterion and scheduler with CUDA support"""
        self.model = ImprovedEmotionModel ().cuda ()
        self.criterion = MultiModalLoss ().cuda ()
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

    def train (self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with CUDA optimizations"""
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
            logging.info (f"  Text Accuracy: {train_metrics['text_accuracy']:.4f}")
            logging.info (f"  Fusion Accuracy: {train_metrics['fusion_accuracy']:.4f}")

            # Validation phase
            val_loss, val_metrics, predictions = self.validate (val_loader, epoch)
            logging.info ("Validation Phase:")
            logging.info (f"  Loss: {val_loss:.4f}")
            logging.info (f"  Image Accuracy: {val_metrics['image_accuracy']:.4f}")
            logging.info (f"  Audio Accuracy: {val_metrics['audio_accuracy']:.4f}")
            logging.info (f"  Text Accuracy: {val_metrics['text_accuracy']:.4f}")
            logging.info (f"  Fusion Accuracy: {val_metrics['fusion_accuracy']:.4f}")


            # Update learning rate scheduler
            self.scheduler.step (val_loss)

            # Log metrics
            self.log_metrics (train_metrics, 'train', epoch)
            self.log_metrics (val_metrics, 'val', epoch)

            # Log GPU memory usage
            memory_allocated = torch.cuda.memory_allocated (0)
            memory_cached = torch.cuda.memory_reserved (0)
            logging.info ("Resources:")
            logging.info (f"  GPU Memory Allocated: {memory_allocated / 1e9:.2f}GB")
            logging.info (f"  GPU Memory Cached: {memory_cached / 1e9:.2f}GB")

            # Check for improvement
            average_val_accuracy = (
                                           val_metrics['image_accuracy'] +
                                           val_metrics['audio_accuracy'] +
                                           val_metrics['text_accuracy'] +
                                           val_metrics['fusion_accuracy']
                                   ) / 4
            if average_val_accuracy > best_acc:
                best_acc = average_val_accuracy
                self.save_checkpoint (epoch, val_metrics)
                early_stop_counter = 0
                logging.info (f"New best accuracy: {best_acc:.4f}")
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config['patience']:
                logging.info ("Early stopping triggered")
                break

            # Clear cache periodically
            torch.cuda.empty_cache ()


        logging.info (f"Best validation accuracy: {best_acc:.4f}")
        self.writer.close ()

    # In train.py, update the train_epoch method
    def train_epoch (self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        self.model.train ()
        total_loss = 0.0
        predictions = {'image': [], 'audio': [], 'text': [], 'fusion': []}
        all_targets = []

        progress_bar = tqdm (train_loader, desc=f'Epoch {epoch + 1} Training')
        for batch in progress_bar:
            # Move data to GPU
            audio = batch['audio'].cuda (non_blocking=True)
            image = batch['image'].cuda (non_blocking=True)
            text_input = batch['text'].cuda (non_blocking=True)
            targets = batch['emotion'].cuda (non_blocking=True)

            # Clear gradients
            self.optimizer.zero_grad (set_to_none=True)

            # Mixed precision forward pass
            with torch.amp.autocast (device_type='cuda'):
                outputs = self.model (image=image, audio=audio, text_input=text_input)
                loss = self.criterion (outputs, targets)

            # Scaled backpropagation
            self.scaler.scale (loss).backward ()

            if self.config.get ('grad_clip'):
                self.scaler.unscale_ (self.optimizer)
                torch.nn.utils.clip_grad_norm_ (
                    self.model.parameters (),
                    self.config['grad_clip']
                )

            self.scaler.step (self.optimizer)
            self.scaler.update ()

            # Update metrics
            total_loss += loss.item ()

            with torch.no_grad ():
                for key in ['image_pred', 'audio_pred', 'text_pred', 'fusion_pred']:
                    if key in outputs:
                        pred = outputs[key].argmax (1).cpu ()
                        predictions[key.replace ('_pred', '')].extend (pred.numpy ())
                all_targets.extend (targets.cpu ().numpy ())

            # Update progress bar
            used_memory = torch.cuda.memory_allocated () / 1e9
            progress_bar.set_postfix ({
                'loss': loss.item (),
                'GPU Memory': f'{used_memory:.2f}GB'
            })

        metrics = {
            'loss': total_loss / len (train_loader),
            'true_labels': np.array (all_targets)
        }

        # Calculate accuracies for all available modalities
        for key in predictions:
            if predictions[key]:  # Only calculate if predictions exist
                metrics[f'{key}_accuracy'] = np.mean (
                    np.array (predictions[key]) == metrics['true_labels']
                )


        return metrics['loss'], metrics

    @torch.no_grad ()
    def validate (self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict, Dict]:
        """Validate the model with CUDA optimizations"""
        self.model.eval ()
        total_loss = 0.0
        predictions = {'image': [], 'audio': [], 'text': [], 'fusion': []}
        all_targets = []

        stream = torch.cuda.Stream ()
        with torch.cuda.stream (stream):
            for batch in tqdm (val_loader, desc='Validation'):
                audio = batch['audio'].cuda (non_blocking=True)
                image = batch['image'].cuda (non_blocking=True)
                text_input = batch['text'].cuda (non_blocking=True)
                targets = batch['emotion'].cuda (non_blocking=True)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model (image=image, audio=audio, text_input=text_input)  # Include text input
                    loss = self.criterion (outputs, targets)

                total_loss += loss.item ()
                for key in outputs:
                    pred = outputs[key].argmax (1).cpu ()
                    predictions[key.replace ('_pred', '')].extend (pred.numpy ())
                all_targets.extend (targets.cpu ().numpy ())

        torch.cuda.synchronize ()

        metrics = {
            'loss': total_loss / len (val_loader),
            'true_labels': np.array (all_targets)
        }
        for key, preds in predictions.items ():
            metrics[f'{key}_accuracy'] = np.mean (
                np.array (preds) == metrics['true_labels']
            )
        all_preds_fusion = np.array (predictions['fusion'])
        fusion_conf_matrix = confusion_matrix (all_targets, all_preds_fusion)
        logging.info (f"Confusion Matrix for Fusion:\n{fusion_conf_matrix}")
        if len (predictions['image']) > 0:
            all_preds_image = np.array (predictions['image'])
            image_conf_matrix = confusion_matrix (all_targets, all_preds_image)
            logging.info (f"Confusion Matrix for Image:\n{image_conf_matrix}")

            # 2) Audio
        if len (predictions['audio']) > 0:
            all_preds_audio = np.array (predictions['audio'])
            audio_conf_matrix = confusion_matrix (all_targets, all_preds_audio)
            logging.info (f"Confusion Matrix for Audio:\n{audio_conf_matrix}")

            # 3) Text
        if len (predictions['text']) > 0:
            all_preds_text = np.array (predictions['text'])
            text_conf_matrix = confusion_matrix (all_targets, all_preds_text)
            logging.info (f"Confusion Matrix for Text:\n{text_conf_matrix}")

        return metrics['loss'], metrics, predictions

    def save_checkpoint (self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict (),
            'optimizer_state_dict': self.optimizer.state_dict (),
            'scheduler_state_dict': self.scheduler.state_dict (),
            'scaler_state_dict': self.scaler.state_dict (),  # Save AMP scaler
            'metrics': metrics,
            'config': self.config
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_acc_{metrics["fusion_accuracy"]:.3f}.pt'
        torch.save (checkpoint, checkpoint_path)
        logging.info (f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint (self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load (checkpoint_path, map_location='cuda')
        self.model.load_state_dict (checkpoint['model_state_dict'])
        self.optimizer.load_state_dict (checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict (checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict (checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

    def log_metrics (self, metrics: Dict, phase: str, epoch: int):
        """Log metrics to tensorboard"""
        for key, value in metrics.items ():
            if key == 'true_labels':
                continue
            self.writer.add_scalar (f"{phase}/{key}", value, epoch)


def main ():
    # Configuration optimized for CUDA
    config = {
        'batch_size': 16,
        'num_workers': 4,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'epochs': 20,
        'patience': 5,
        'scheduler_patience': 3,
        'grad_clip': 1.0,
        'warmup_steps': 1000,
        'pin_memory': True,
        'cuda_non_blocking': True,
        'amp': True,
    }

    def get_lr_schedule (optimizer, warmup_steps):
        def lr_lambda (step):
            if step < warmup_steps:
                return float (step) / float (max (1, warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR (optimizer, lr_lambda)

    try:
        # Load datasets
        logging.info ("Loading datasets...")
        image_data = pd.concat ([
            pd.read_csv ('data/processed/fer2013.csv'),
            pd.read_csv ('data/processed/expw.csv')
        ])
        audio_data = pd.read_csv ('data/processed/ravdess.csv')
        text_data = pd.read_csv ('data/processed/goemotions.csv')
        max_token = text_data['text'].max()
        print(f"Max token: {max_token}")
        # Align datasets
        print ("=== Checking distribution BEFORE alignment ===")
        for emotion in range (7):
            for split in ["train", "test"]:
                count_img = len (image_data[(image_data["emotion"] == emotion) & (image_data["split"] == split)])
                count_aud = len (audio_data[(audio_data["emotion"] == emotion) & (audio_data["split"] == split)])
                count_txt = len (text_data[(text_data["emotion"] == emotion) & (text_data["split"] == split)])
                print (f"emotion={emotion}, split={split}, "
                       f"count_img={count_img}, count_aud={count_aud}, count_txt={count_txt}")

        logging.info ("Aligning datasets...")
        aligned_data = label_level_align(image_data, audio_data, text_data)
        if aligned_data is None:
            print ("aligned_data returned None—no overlap found!")
        else:
            print ("\n=== Checking distribution AFTER alignment ===")
            for dom in ["image", "audio", "text"]:
                df_dom = aligned_data[dom]
                print (f"---- {dom.upper ()} TRAIN ----")
                print (df_dom[df_dom["split"] == "train"]["emotion"].value_counts ())
                print (f"---- {dom.upper ()} TEST ----")
                print (df_dom[df_dom["split"] == "test"]["emotion"].value_counts ())
        if aligned_data is None:
            raise RuntimeError ("Failed to align datasets")
        print ("Image domain total:", len (aligned_data["image"]))
        print ("Audio domain total:", len (aligned_data["audio"]))
        print ("Text domain total:", len (aligned_data["text"]))

        print ("Train images:", len (aligned_data["image"][aligned_data["image"]["split"] == "train"]))
        print ("Test images:", len (aligned_data["image"][aligned_data["image"]["split"] == "test"]))
        print ("Train audio:", len (aligned_data["audio"][aligned_data["audio"]["split"] == "train"]))
        print ("Test audio:", len (aligned_data["audio"][aligned_data["audio"]["split"] == "test"]))
        print ("Train Text:", len (aligned_data["text"][aligned_data["text"]["split"] == "train"]))
        print ("Test Text:", len (aligned_data["text"][aligned_data["text"]["split"] == "test"]))

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

        # Data loaders
        train_loader = DataLoader (
            train_dataset,
            batch_size = config['batch_size'],
            sampler=RandomSampler (train_dataset),
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            persistent_workers=True
        )

        val_loader = DataLoader (
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            persistent_workers=True
        )

        # Initialize and train
        trainer = EmotionTrainer (config)
        trainer.train (train_loader, val_loader)

    except Exception as e:
        logging.error (f"Training error: {str (e)}")
        raise


if __name__ == '__main__':
    main ()
