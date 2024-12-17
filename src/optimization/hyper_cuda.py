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
                # Clear cache before each epoch
                torch.cuda.empty_cache ()

                train_loss = self.train_epoch (
                    model, train_loader, optimizer, criterion, params['gradient_clip']
                )

                val_loss, accuracy = self.validate (model, val_loader, criterion)

                scheduler.step (val_loss)
                trial.report (accuracy, epoch)

                if trial.should_prune ():
                    raise optuna.TrialPruned ()

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    early_stopping_counter = 0
                    self.save_trial_checkpoint (trial, model, best_accuracy)
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= 5:
                    break

            return best_accuracy

        except Exception as e:
            logging.error (f"Trial {trial.number} failed: {str (e)}")
            raise optuna.TrialPruned ()

    def train_epoch (self, model, train_loader, optimizer, criterion, gradient_clip):
        """CUDA-optimized training loop with mixed precision."""
        model.train ()
        total_loss = 0.0

        for batch in train_loader:
            # Move data to GPU efficiently
            audio = batch['audio'].cuda (non_blocking=True)
            image = batch['image'].cuda (non_blocking=True)
            text = batch['text'].cuda (non_blocking=True)
            targets = batch['emotion'].cuda (non_blocking=True)

            # Clear gradients
            optimizer.zero_grad (set_to_none=True)

            # Mixed precision forward pass
            with torch.amp.autocast ('cuda'):
                outputs = model (audio, image, text)
                loss = criterion (outputs, targets)

            # Scaled backpropagation
            self.scaler.scale (loss).backward ()

            if gradient_clip > 0:
                self.scaler.unscale_ (optimizer)
                torch.nn.utils.clip_grad_norm_ (model.parameters (), gradient_clip)

            self.scaler.step (optimizer)
            self.scaler.update ()

            total_loss += loss.item ()

        return total_loss / len (train_loader)

    @torch.no_grad ()
    def validate (self, model, val_loader, criterion):
        """CUDA-optimized validation loop."""
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

        return total_loss / len (val_loader), (correct / total if total > 0 else 0.0)

    def optimization_callback (self, study: optuna.Trial, trial: optuna.Trial):
        """Callback function for optimization progress logging."""
        if trial.number % 10 == 0:
            self.visualize_optimization_results (study)
            logging.info (f"\nTrial {trial.number} finished with state: {trial.state}")

            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                logging.info (f"Current best value: {study.best_value}")
                logging.info ("Current best params:")
                for key, value in study.best_params.items ():
                    logging.info (f"    {key}: {value}")
            else:
                logging.info ("No completed trials yet")

    def visualize_optimization_results (self, study: optuna.Study):
        """Create and save Optuna optimization plots."""
        try:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logging.info ("No completed trials yet, skipping visualization")
                return

            fig1 = plot_optimization_history (study)
            fig2 = plot_parallel_coordinate (study)
            fig3 = plot_param_importances (study)
            fig4 = plot_contour (study)

            fig1.write_image (str (self.viz_dir / 'optimization_history.png'))
            fig2.write_image (str (self.viz_dir / 'parallel_coordinate.png'))
            fig3.write_image (str (self.viz_dir / 'param_importances.png'))
            fig4.write_image (str (self.viz_dir / 'param_contour.png'))
        except Exception as e:
            logging.error (f"Error in visualization: {str (e)}")

    def save_trial_checkpoint (self, trial: optuna.Trial, model: nn.Module, accuracy: float):
        """Save trial checkpoint for a given trial/model snapshot."""
        checkpoint_path = self.viz_dir / f'trial_{trial.number}_acc_{accuracy:.3f}.pt'
        torch.save ({
            'trial_number': trial.number,
            'model_state': model.state_dict (),
            'trial_params': trial.params,
            'accuracy': accuracy
        }, checkpoint_path)
        logging.info (f"Saved trial checkpoint: {checkpoint_path}")

    def optimize_parallel (self) -> Dict[str, Any]:
        """
        Run parallel hyperparameter optimization.
        Returns a dictionary of final optimization results.
        """
        storage = (optuna.storages.RDBStorage (
            self.storage,
            heartbeat_interval=60,
            grace_period=120
        ) if self.storage else None)

        study = optuna.create_study (
            direction="maximize",
            pruner=HyperbandPruner (
                min_resource=1,
                max_resource=20,
                reduction_factor=3,
                bootstrap_count=0
            ),
            sampler=TPESampler (n_startup_trials=5, multivariate=True),
            storage=storage,
            load_if_exists=True
        )

        try:
            # Run optimization
            study.optimize (
                self.objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                timeout=None,
                callbacks=[self.optimization_callback],
                gc_after_trial=True,
                catch=(Exception,)
            )

            # Check for completed trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                self.visualize_optimization_results (study)
                return self.aggregate_results (study)
            else:
                logging.error ("No trials completed successfully. No optimal parameters were found.")
                return {'error': 'No trials completed successfully'}

        except Exception as e:
            logging.error (f"Optimization failed: {str (e)}")
            raise

    def aggregate_results (self, study: optuna.Study) -> Dict[str, Any]:
        """Aggregate final hyperparameter optimization results."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            logging.error ("No completed trials found to aggregate results.")
            return {'error': 'No completed trials found'}

        durations = [
            t.duration.total_seconds () for t in completed_trials if t.duration is not None
        ]

        results = {
            'best_params': study.best_params if study.best_trial else {},
            'best_value': study.best_value if study.best_trial else None,
            'n_trials': len (study.trials),
            'parameter_importance': optuna.importance.get_param_importances (study),
            'trial_statistics': {
                'mean_duration': np.mean (durations) if durations else 0,
                'success_rate': len (completed_trials) / len (study.trials) if study.trials else 0,
                'pruned_trials': len ([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len ([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            }
        }

        # Save results
        with open (self.viz_dir / 'optimization_results.json', 'w') as f:
            json.dump (results, f, indent=4, default=str)

        logging.info ("Optimization results successfully aggregated and saved.")
        return results
