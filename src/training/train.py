import torch
import pandas as pd
from pathlib import Path
import json
import yaml
import os
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix

import torch.nn as nn
from collections import Counter


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataset import MultiModalEmotionDataset
from src.models.model import ImprovedEmotionModel, MultiModalLoss

from src.utils.visualization import ModelVisualizer, LRFinder
from src.utils.data_aligment import label_level_align


def load_config(config_path: str):
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class EmotionTrainer:
    def __init__ (self, config_path: str = 'configs/training_config.yaml'):
        """
        Initialize trainer with configuration.
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        self.setup_device()
        self.setup_tensorboard()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.visualizer = ModelVisualizer (self.plot_dir)  # Initialize visualizer
        self.scaler = torch.amp.GradScaler()


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
        timestamp = datetime.now ().strftime ("%Y%m%d_%H%M%S")
        self.log_dir = Path (self.config['logging']['log_dir']) / timestamp
        self.checkpoint_dir = self.log_dir / self.config['logging']['checkpoints_dir']
        self.plot_dir = self.log_dir / self.config['logging']['plots_dir']

        self.log_dir.mkdir (parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir (exist_ok=True)
        self.plot_dir.mkdir (exist_ok=True)

    def setup_tensorboard (self):
        """Initialize tensorboard writer"""
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')

    def setup_model (self):
        """Initialize model, optimizer, criterion and scheduler with CUDA support"""
        self.model = ImprovedEmotionModel ().cuda ()
        self.criterion = MultiModalLoss ().cuda ()
        self.optimizer = torch.optim.AdamW (
            self.model.parameters (),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (
            self.optimizer,
            mode='min',
            patience=self.config['training']['scheduler_patience'],
            factor=0.5,
            verbose=True
        )

    def lr_finder(self, train_loader: DataLoader):
        """Use LRFinder to determine the best learning rate."""
        lr_finder = LRFinder (self.model, self.optimizer, self.criterion, self.device)
        lrs, losses = lr_finder.range_test (train_loader, start_lr=1e-7, end_lr=1e-1, num_iter=100)
        fig = lr_finder.plot_lr_find (lrs, losses)
        fig.savefig (self.plot_dir / 'lr_finder.png')
        logging.info ("Saved LR Finder plot.")

    def train (self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with CUDA optimizations"""
        self.setup_model ()
        best_acc = 0.0
        early_stop_counter = 0
        # Plot model architecture before training
        self.visualizer.plot_model_architecture(
            self.model, input_shape=((1, 16000), (224, 224), 50)
        )

        for epoch in range(self.config['training']['epochs']):

            # Training phase
            train_loss, train_metrics = self.train_epoch (train_loader, epoch)
            logging.info (f"\n{'-' * 20} Epoch {epoch + 1}/{self.config['training']['epochs']} {'-' * 20}")
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

            if early_stop_counter >= self.config['training']['patience']:
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

        label_counts = Counter (train_loader.dataset.audio_data['emotion'])
        total_samples = sum (label_counts.values ())
        # Add error handling and ensure ordered classes
        class_weights = torch.FloatTensor ([
            total_samples / (len (label_counts) * label_counts.get (i, 1))  # Added .get() with default
            for i in range (7)  # Explicit range for 7 emotions
        ]).cuda ()
        criterion_audio = nn.CrossEntropyLoss (weight=class_weights)

        progress_bar = tqdm (train_loader, desc=f'Epoch {epoch + 1} Training')
        for batch_idx, batch in enumerate (progress_bar):
            # Move data to GPU
            audio = batch['audio'].cuda (non_blocking=True)
            image = batch['image'].cuda (non_blocking=True)
            text_input = batch['text'].cuda (non_blocking=True)
            targets = batch['emotion'].cuda (non_blocking=True)

            # Mixed precision forward pass
            with torch.amp.autocast (device_type='cuda'):
                outputs = self.model (image=image, audio=audio, text_input=text_input)
                # Apply weighted loss only to audio predictions
                loss_audio = criterion_audio (outputs['audio_pred'], targets)
                loss_others = self.criterion (
                    {k: v for k, v in outputs.items () if k != 'audio_pred'},
                    targets
                )
                loss = 0.4 * loss_audio + 0.6 * loss_others

                # Only scale loss if using gradient accumulation
                if self.config['optimization'].get ('accumulation_steps', 1) > 1:
                    loss = loss / self.config['optimization']['accumulation_steps']

            # Scaled backpropagation
            self.scaler.scale (loss).backward ()

            # Only update optimizer after accumulating enough gradients
            if (batch_idx + 1) % self.config['optimization'].get ('accumulation_steps', 1) == 0:
                # Gradient clipping if configured
                if self.config['training'].get ('grad_clip'):
                    self.scaler.unscale_ (self.optimizer)
                    torch.nn.utils.clip_grad_norm_ (
                        self.model.parameters (),
                        self.config['training']['grad_clip']
                    )

                # Optimizer steps
                self.scaler.step (self.optimizer)
                self.scaler.update ()
                self.optimizer.zero_grad (set_to_none=True)

            # Update metrics
            total_loss += loss.item () * self.config['optimization']['accumulation_steps']

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

    def export_model (self, export_path: str):
        """Export model to ONNX with optimizations for browser deployment"""
        self.model.eval ()
        os.makedirs (export_path, exist_ok=True)

        # Create dummy inputs with adjusted audio length
        dummy_inputs = {
            'image': torch.randn (1, 1, 48, 48).to (self.device),
            'audio': torch.randn (1, 16000).to (self.device),  # Raw audio
            'text_input': torch.randint (0, 30522, (1, 50)).to (self.device)  # BERT tokens
        }

        # Wrapper class for pre-processing and feature extraction
        class ModelWrapper (nn.Module):
            def __init__ (self, model):
                super ().__init__ ()
                self.model = model
                self.spectrogram = model.spectrogram
                self.normalize = model.normalize
                self.text_embedding = model.text_embedding
                self.text_rnn = model.text_rnn
                self.text_proj = model.text_proj
                self.audio_encoder = model.audio_encoder

            def forward (self, image, audio, text_input):
                if image is not None:
                    image_features = self.model.image_encoder (image)
                else:
                    image_features = None

                if audio is not None:
                    # Process audio into spectrogram
                    spec = self.spectrogram (audio)  # [B, 64, T]
                    # Take magnitude of the complex spectrogram (extract real part)
                    spec = torch.abs (spec)  # Convert complex to magnitude
                    spec = torch.log1p (spec)  # Apply log transformation
                    spec = spec.squeeze (1)  # [B, 64, T]
                    spec = spec.transpose (1, 2)  # [B, T, 64]
                    spec = self.normalize (spec)
                    spec = spec.transpose (1, 2)  # [B, 64, T]
                    audio_features = self.audio_encoder (spec)
                else:
                    audio_features = None

                if text_input is not None:
                    embedded = self.text_embedding (text_input)
                    rnn_out, _ = self.text_rnn (embedded)
                    text_features = self.text_proj (rnn_out.mean (dim=1))
                else:
                    text_features = None

                outputs = {}
                if image_features is not None:
                    outputs['image_pred'] = self.model.classifiers['image'] (image_features)
                if audio_features is not None:
                    outputs['audio_pred'] = self.model.classifiers['audio'] (audio_features)
                if text_features is not None:
                    outputs['text_pred'] = self.model.classifiers['text'] (text_features)

                if any (x is not None for x in [image_features, audio_features, text_features]):
                    fusion_features = self.model.fuse_modalities (image_features, audio_features, text_features)
                    outputs['fusion_pred'] = self.model.classifiers['fusion'] (fusion_features)

                return outputs

        wrapped_model = ModelWrapper (self.model)

        # Test the model with dummy inputs
        with torch.no_grad ():
            outputs = wrapped_model (**dummy_inputs)
            for key in ['image_pred', 'audio_pred', 'text_pred', 'fusion_pred']:
                assert key in outputs, f"Missing {key} in model outputs"
                assert outputs[key].shape[1] == len (self.emotion_labels), f"Shape mismatch for {key}"

        # Export to ONNX
        try:
            torch.onnx.export (
                wrapped_model,
                args=(dummy_inputs['image'], dummy_inputs['audio'], dummy_inputs['text_input']),
                f=os.path.join (export_path, 'emotion_model.onnx'),
                input_names=['image', 'audio', 'text_input'],
                output_names=['image_pred', 'audio_pred', 'text_pred', 'fusion_pred'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'audio': {0: 'batch_size'},
                    'text_input': {0: 'batch_size'},
                    'image_pred': {0: 'batch_size'},
                    'audio_pred': {0: 'batch_size'},
                    'text_pred': {0: 'batch_size'},
                    'fusion_pred': {0: 'batch_size'}
                },
                do_constant_folding=True,
                opset_version=17,  # Ensure using opset version 17 or above
                keep_initializers_as_inputs=True,
                verbose=True
            )

            # Save model metadata
            model_metadata = {
                'version': '1.0',
                'input_shapes': {'image': [1, 48, 48], 'audio': [1, 16000], 'text': [50]},
                'preprocessing': {
                    'image': {'size': [48, 48], 'channels': 1, 'normalize_mean': [0.5], 'normalize_std': [0.5]},
                    'audio': {'sample_rate': 16000,
                              'mel_spec_params': {'n_mels': 64, 'n_fft': 400, 'win_length': 400, 'hop_length': 160,
                                                  'power': 2.0}},
                    'text': {'max_length': 50, 'vocab_size': 30522, 'tokenizer': 'bert-base-uncased'}
                },
                'labels': self.emotion_labels
            }

            with open (os.path.join (export_path, 'model_metadata.json'), 'w') as f:
                json.dump (model_metadata, f, indent=2)

        except Exception as e:
            logging.error (f'Error during export: {str (e)}')
            raise

        logging.info (f'Model successfully exported to {export_path}')
        return True

def main ():
    trainer = EmotionTrainer('configs/training_config.yaml')

    try:
        # Load datasets
        logging.info ("Loading datasets...")
        image_data = pd.concat ([
            pd.read_csv ('data/processed/fer2013.csv'),
            pd.read_csv ('data/processed/expw.csv')
        ])
        audio_data = pd.read_csv ('data/processed/ravdess.csv')
        text_data = pd.read_csv ('data/processed/goemotions.csv')

        logging.info ("Aligning datasets...")
        aligned_data = label_level_align(image_data, audio_data, text_data)
        if aligned_data is None:
            raise RuntimeError("Failed to align datasets")

        train_ratio = trainer.config['data']['train_ratio']
        train_size = int (len (aligned_data['image']) * train_ratio)

        for key in aligned_data:
            aligned_data[key] = aligned_data[key].sample (n=train_size, random_state=42)
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
            #config=config,  # Pass the config here
            split='train'
        )

        val_dataset = MultiModalEmotionDataset (
            image_data=aligned_data['image'],
            audio_data=aligned_data['audio'],
            text_data=aligned_data['text'],
            #config=config,  # Pass the config here
            split='test'
        )

        # Data loaders
        train_loader = DataLoader (
            train_dataset,
            batch_size=trainer.config['training']['batch_size'],
            sampler=RandomSampler (train_dataset),
            num_workers=trainer.config['training']['num_workers'],
            pin_memory=trainer.config['data']['pin_memory'],
            persistent_workers=trainer.config['data']['persistent_workers']
        )

        val_loader = DataLoader (
            val_dataset,
            batch_size=trainer.config['training']['batch_size'],
            shuffle=False,
            num_workers=trainer.config['training']['num_workers'],
            pin_memory=trainer.config['data']['pin_memory'],
            persistent_workers=trainer.config['data']['persistent_workers']
        )

        # Initialize and train
        trainer.train(train_loader, val_loader)
        export_dir = 'exported_models'
        os.makedirs (export_dir, exist_ok=True)
        trainer.export_model (export_dir)

    except Exception as e:
        logging.error (f"Training error: {str (e)}")
        raise


if __name__ == '__main__':
    main ()
