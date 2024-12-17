from datetime import datetime
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_contour
)
from optuna.pruners import SuccessiveHalvingPruner, HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
import plotly
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model import ImprovedEmotionModel, MultiModalLoss


class EnhancedHyperparameterOptimizer:
    def __init__ (
            self,
            train_dataset,
            val_dataset,
            n_trials=100,
            n_jobs=-1,
            storage=None
    ):
        # Verify CUDA availability
        if not torch.cuda.is_available ():
            raise RuntimeError ("CUDA is required for this optimization process")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = torch.device ("cuda")
        self.n_trials = n_trials
        self.n_jobs = self.setup_parallel_resources (n_jobs)
        self.storage = storage

        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler ()

        # Set up CUDA optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.setup_logging ()
        self.setup_visualization ()

        # Log GPU information
        gpu_id = torch.cuda.current_device ()
        gpu_name = torch.cuda.get_device_name (gpu_id)
        logging.info (f"Using CUDA Device {gpu_id}: {gpu_name}")
        total_memory = torch.cuda.get_device_properties (gpu_id).total_memory
        logging.info (f"Total GPU Memory: {total_memory / 1e9:.2f} GB")

    def setup_logging (self):
        """Initialize logging configuration"""
        timestamp = datetime.now ().strftime ("%Y%m%d_%H%M%S")
        self.log_dir = Path ("logs/hyperopt") / timestamp
        self.log_dir.mkdir (parents=True, exist_ok=True)
        self.viz_dir = self.log_dir / "visualizations"
        self.viz_dir.mkdir (exist_ok=True)

        logging.basicConfig (
            filename=self.log_dir / "optimization.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def setup_visualization (self):
        """Initialize visualization settings and requirements"""
        try:
            import plotly
        except ImportError:
            raise ImportError ("Plotly is required for visualization. Please install it with 'pip install plotly'")

        # Ensure the visualization directory exists
        if not hasattr (self, 'viz_dir'):
            self.viz_dir = self.log_dir / "visualizations" if hasattr (self, 'log_dir') else Path (
                "logs/hyperopt/visualizations")
            self.viz_dir.mkdir (parents=True, exist_ok=True)

        logging.info (f"Visualization directory set up at: {self.viz_dir}")

    def setup_parallel_resources (self, n_jobs: int) -> int:
        """Setup parallel processing considering available GPUs."""
        n_gpus = torch.cuda.device_count ()
        if n_jobs < 0:
            n_jobs = n_gpus if n_gpus > 0 else mp.cpu_count ()
        return min (n_jobs, n_gpus)

    def suggest_hyperparameters (self, trial: optuna.Trial) -> dict:
        """Define the hyperparameter search space."""
        return {
            'batch_size': trial.suggest_int ('batch_size', 16, 128, step=16),
            'learning_rate': trial.suggest_float ('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float ('weight_decay', 1e-6, 1e-3, log=True),
            'dropout': trial.suggest_float ('dropout', 0.1, 0.5),
            'scheduler_patience': trial.suggest_int ('scheduler_patience', 2, 5),
            'gradient_clip': trial.suggest_float ('gradient_clip', 0.1, 1.0),

            # Loss weights for different modalities (should sum to less than 1)
            'image_loss_weight': trial.suggest_float ('image_loss_weight', 0.1, 0.3),
            'audio_loss_weight': trial.suggest_float ('audio_loss_weight', 0.1, 0.3),
            'text_loss_weight': trial.suggest_float ('text_loss_weight', 0.1, 0.3),
        }

    def objective (self, trial: optuna.Trial) -> float:
        """CUDA-optimized objective function."""
        params = self.suggest_hyperparameters (trial)

        try:
            # Create model and move to GPU
            model = ImprovedEmotionModel (
                num_emotions=7,
                dropout=params['dropout']
            ).cuda ()

            # Create data loaders with CUDA optimizations
            train_loader = DataLoader (
                self.train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            val_loader = DataLoader (
                self.val_dataset,
                batch_size=params['batch_size'] * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

            optimizer = torch.optim.AdamW (
                model.parameters (),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (
                optimizer,
                mode='min',
                patience=params['scheduler_patience'],
                factor=0.5
            )

            criterion = MultiModalLoss (
                weights={
                    'image': params['image_loss_weight'],
                    'audio': params['audio_loss_weight'],
                    'text': params['text_loss_weight'],
                    'fusion': 1 - params['image_loss_weight'] -
                              params['audio_loss_weight'] -
                              params['text_loss_weight']
                }
            ).cuda ()

            best_accuracy = 0.0
            early_stopping_counter = 0

            for epoch in range (20):
                torch.cuda.empty_cache ()
                train_loss = self.train_epoch (
                    model, train_loader, optimizer, criterion, params['gradient_clip']
                )
                val_loss, accuracy = self.validate (model, val_loader, criterion)
                scheduler.step (val_loss)
                trial.report (accuracy, epoch)
                print (f"Trial {trial.number}, Epoch {epoch}, Accuracy: {accuracy:.4f}")

                if trial.should_prune () and epoch > 5:
                    raise optuna.TrialPruned ()

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    early_stopping_counter = 0
                    self.save_trial_checkpoint (trial, model, best_accuracy)
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= 10:
                    break

            return best_accuracy

        except Exception as e:
            logging.error (f"Trial {trial.number} failed: {str (e)}")
            return 0.0  # Return a low score instead of pruning

    def train_epoch (self, model, train_loader, optimizer, criterion, gradient_clip):
        """Train for one epoch."""
        model.train ()
        total_loss = 0.0

        for batch in train_loader:
            audio = batch['audio'].cuda (non_blocking=True)
            image = batch['image'].cuda (non_blocking=True)
            text = batch['text'].cuda (non_blocking=True)
            targets = batch['emotion'].cuda (non_blocking=True)

            optimizer.zero_grad (set_to_none=True)
            with torch.amp.autocast ('cuda'):
                outputs = model (audio, image, text)
                loss = criterion (outputs, targets)

            self.scaler.scale (loss).backward ()
            self.scaler.unscale_ (optimizer)
            torch.nn.utils.clip_grad_norm_ (model.parameters (), gradient_clip)
            self.scaler.step (optimizer)
            self.scaler.update ()

            total_loss += loss.item ()

        return total_loss / len (train_loader)

    @torch.no_grad ()
    def validate (self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval ()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            audio = batch['audio'].cuda (non_blocking=True)
            image = batch['image'].cuda (non_blocking=True)
            text = batch['text'].cuda (non_blocking=True)
            targets = batch['emotion'].cuda (non_blocking=True)

            with torch.amp.autocast ('cuda'):
                outputs = model (audio, image, text)
                loss = criterion (outputs, targets)

            _, predicted = outputs['fusion_pred'].max (1)
            total += targets.size (0)
            correct += predicted.eq (targets).sum ().item ()
            total_loss += loss.item ()

        accuracy = correct / total if total > 0 else 0.0
        return total_loss / len (val_loader), accuracy

    def optimization_callback (self, study: optuna.Study, trial: optuna.Trial):
        """Log trial progress."""
        logging.info (f"Trial {trial.number} finished with value: {trial.value} and params: {trial.params}")

    def optimize (self) -> Dict[str, Any]:
        """Wrapper method to run hyperparameter optimization."""
        return self.optimize_parallel ()

    def optimize_parallel (self) -> Dict[str, Any]:
        """Parallel optimization wrapper to align with updated method calls."""
        return self.aggregate_results (optuna.create_study (
            direction="maximize",
            sampler=TPESampler (n_startup_trials=5, multivariate=True),
            pruner=MedianPruner (n_startup_trials=10, n_warmup_steps=5)
        ))

    def aggregate_results (self, study: optuna.Study) -> Dict[str, Any]:
        """Aggregate results of the study."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            logging.error ("No completed trials found.")
            return {'error': 'No completed trials'}

        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len (study.trials),
            'completed_trials': len (completed_trials),
            'pruned_trials': len ([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len ([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        }

        # Save results
        results_path = self.viz_dir / 'optimization_results.json'
        with open (results_path, 'w') as f:
            json.dump (results, f, indent=4)
        logging.info (f"Optimization results saved to: {results_path}")

        return results
