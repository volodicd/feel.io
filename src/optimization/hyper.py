# src/optimization/hyper.py

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

# Import from your new structure
from src.models.model import ImprovedEmotionModel, MultiModalLoss


class EnhancedHyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna for ImprovedEmotionModel.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        device,
        n_trials=100,
        n_jobs=-1,
        storage=None
    ):
        """
        Args:
            train_dataset: A PyTorch Dataset for training
            val_dataset: A PyTorch Dataset for validation
            device: PyTorch device (cpu, cuda, mps, etc.)
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 to use all cores)
            storage: Optuna storage path (e.g., sqlite:///example.db) or None
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.n_trials = n_trials
        self.n_jobs = self.setup_parallel_resources(n_jobs)
        self.storage = storage

        self.setup_logging()
        self.setup_visualization()

    def setup_parallel_resources(self, n_jobs: int) -> int:
        """Setup and return number of parallel jobs."""
        n_gpus = torch.cuda.device_count()
        if n_jobs < 0:
            n_jobs = mp.cpu_count()
        return min(n_jobs, n_gpus if n_gpus > 0 else mp.cpu_count())

    def setup_logging(self):
        """Initialize logging configuration"""
        self.log_dir = Path('logs/hyperopt')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=self.log_dir / 'optimization.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_visualization(self):
        """Setup directory for hyperopt visualizations"""
        self.viz_dir = self.log_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)

    def suggest_hyperparameters (self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space for the model and training process."""
        return {
            # Model architecture
            'hidden_dim': trial.suggest_int ('hidden_dim', 64, 512, step=64),
            'num_attention_heads': trial.suggest_int ('num_attention_heads', 4, 12),
            'num_encoder_layers': trial.suggest_int ('num_encoder_layers', 1, 4),
            'dropout': trial.suggest_float ('dropout', 0.1, 0.5),

            # Training parameters
            'batch_size': trial.suggest_categorical ('batch_size', [16, 32, 64]),
            'learning_rate': trial.suggest_float ('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float ('weight_decay', 1e-5, 1e-3, log=True),

            # Loss weights
            'image_loss_weight': trial.suggest_float ('image_loss_weight', 0.2, 0.4),
            'audio_loss_weight': trial.suggest_float ('audio_loss_weight', 0.2, 0.4),
            'text_loss_weight': trial.suggest_float ('text_loss_weight', 0.2, 0.4),

            # Optimization parameters
            'scheduler_patience': trial.suggest_int ('scheduler_patience', 2, 5),
            'gradient_clip': trial.suggest_float ('gradient_clip', 0.1, 1.0)
        }

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.

        Returns:
            The best validation accuracy (fusion accuracy) achieved.
        """
        # Get hyperparameters
        params = self.suggest_hyperparameters(trial)

        try:
            # Create model with dynamic hyperparams
            model = ImprovedEmotionModel(
                num_emotions=7,  # or pass if flexible
                dropout=params['dropout']
            ).to(self.device)

            # If your model has hyperparams like hidden_dim, num_heads, etc.
            # you'd inject them into the model's __init__ or a similar approach.

            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=params['batch_size'] * 2,
                shuffle=False,
                num_workers=2
            )

            # Setup training components
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=params['scheduler_patience'],
                factor=0.5
            )

            criterion = MultiModalLoss (weights={
                'image': params['image_loss_weight'],
                'audio': params['audio_loss_weight'],
                'text': params['text_loss_weight'],
                'fusion': 1 - params['image_loss_weight'] - params['audio_loss_weight'] - params['text_loss_weight']
            })

            best_accuracy = 0.0
            early_stopping_counter = 0

            for epoch in range(20):
                # Training
                train_loss = self.train_epoch(
                    model, train_loader, optimizer, criterion, params['gradient_clip']
                )

                # Validation
                val_loss, accuracy = self.validate(model, val_loader, criterion)

                # Update scheduler
                scheduler.step(val_loss)

                # Report intermediate value to Optuna
                trial.report(accuracy, epoch)

                # Pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    early_stopping_counter = 0
                    self.save_trial_checkpoint(trial, model, best_accuracy)
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= 5:
                    break

            return best_accuracy

        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        gradient_clip: float
    ) -> float:
        """Train for one epoch, returning average loss."""
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            audio = batch['audio'].to (self.device)
            image = batch['image'].to (self.device)
            text = batch['text'].to (self.device)
            targets = batch['emotion'].to (self.device)

            outputs = model (audio, image, text)
            loss = criterion (outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate model performance, returning (val_loss, fusion_accuracy)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            audio = batch['audio'].to (self.device)
            image = batch['image'].to (self.device)
            text = batch['text'].to (self.device)
            targets = batch['emotion'].to (self.device)

            outputs = model (audio, image, text)
            loss = criterion (outputs, targets)

            # Evaluate fusion accuracy
            _, predicted = outputs['fusion_pred'].max (1)
            total += targets.size (0)
            correct += predicted.eq (targets).sum ().item ()

            total_loss += loss.item()

        fusion_acc = correct / total if total > 0 else 0.0
        return total_loss / len(val_loader), fusion_acc

    def save_trial_checkpoint(self, trial: optuna.Trial, model: nn.Module, accuracy: float):
        """Save trial checkpoint for a given trial/model snapshot."""
        checkpoint_path = self.viz_dir / f'trial_{trial.number}_acc_{accuracy:.3f}.pt'
        torch.save({
            'trial_number': trial.number,
            'model_state': model.state_dict(),
            'trial_params': trial.params,
            'accuracy': accuracy
        }, checkpoint_path)
        logging.info(f"Saved trial checkpoint: {checkpoint_path}")

    def optimize_parallel(self) -> Dict[str, Any]:
        """
        Run parallel hyperparameter optimization.
        Returns a dictionary of final optimization results.
        """
        storage = (optuna.storages.RDBStorage(
            self.storage,
            heartbeat_interval=60,
            grace_period=120
        ) if self.storage else None)

        study = optuna.create_study(
            direction="maximize",
            pruner=HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3),
            sampler=TPESampler(n_startup_trials=10, multivariate=True),
            storage=storage,
            load_if_exists=True
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=None,
            callbacks=[self.optimization_callback],
            gc_after_trial=True
        )

        # Visualize results
        self.visualize_optimization_results(study)
        return self.aggregate_results(study)

    def optimization_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function for optimization progress logging."""
        if trial.number % 10 == 0:
            self.visualize_optimization_results(study)
            logging.info(f"\nTrial {trial.number} finished:")
            logging.info(f"Current best value: {study.best_value}")
            logging.info("Current best params:")
            for key, value in study.best_params.items():
                logging.info(f"    {key}: {value}")

    def visualize_optimization_results(self, study: optuna.Study):
        """Create and save Optuna optimization plots."""
        # e.g. plot optimization history, parallel coordinate, etc.
        # This is optional. Example usage:
        fig1 = plot_optimization_history(study)
        fig2 = plot_parallel_coordinate(study)
        fig3 = plot_param_importances(study)
        fig4 = plot_contour(study)

        fig1.write_image(str(self.viz_dir / 'optimization_history.png'))
        fig2.write_image(str(self.viz_dir / 'parallel_coordinate.png'))
        fig3.write_image(str(self.viz_dir / 'param_importances.png'))
        fig4.write_image(str(self.viz_dir / 'param_contour.png'))

    def aggregate_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Aggregate final hyperparameter optimization results."""
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'parameter_importance': optuna.importance.get_param_importances(study),
            'trial_statistics': {
                'mean_duration': np.mean([
                    t.duration.total_seconds() for t in study.trials if t.duration
                ]) if study.trials else 0,
                'success_rate': (
                    len([t for t in study.trials if t.value is not None]) / len(study.trials)
                ) if study.trials else 0,
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        }

        # Save results
        with open(self.viz_dir / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)

        return results
