# src/utils/visualization.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchviz import make_dot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ModelVisualizer:
    """
    Visualize model architecture and training dynamics.
    Provides tools for visualizing model structure, attention weights,
    and feature maps during training.
    """

    def __init__(self, save_dir: Path):
        """
        Initialize visualizer with save directory.

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.viz_dir = self.save_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            filename=self.viz_dir / 'visualization.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def plot_model_architecture (self, model: nn.Module, input_shape: Tuple) -> None:
        """
        Create and save model architecture visualization using torchviz.

        Args:
            model: PyTorch model to visualize
            input_shape: Tuple of input shapes (audio_shape, image_shape, text_length).
                         Example: ((1, 16000), (1, 224, 224), 50)
        """
        try:
            device = next (model.parameters ()).device  # Get the model's device
            # Ensure dummy inputs have correct shapes
            dummy_audio = torch.randn (1, *input_shape[0]).to (device)  # Audio: [1, samples]
            dummy_image = torch.randn(1, 1, input_shape[1][0], input_shape[1][1]).to(device)
            dummy_text = torch.randint (0, 1000, (1, input_shape[2])).to (device)  # Text: [1, seq_len]
            logging.info (
                f"Audio shape: {dummy_audio.shape}, Image shape: {dummy_image.shape}, Text shape: {dummy_text.shape}")

            # Generate computational graph
            with torch.no_grad ():
                outputs = model (dummy_audio, dummy_image, dummy_text)
                dot = make_dot (outputs['fusion_pred'], params=dict (model.named_parameters ()))

            # Save visualization
            output_path = self.viz_dir / "model_architecture"
            dot.render (str (output_path), format="png", cleanup=True)
            logging.info (f"Saved model architecture visualization to {output_path}.png")

        except Exception as e:
            logging.error (f"Error generating model architecture plot: {str (e)}")
            raise

    def plot_text_embeddings (self, embeddings: torch.Tensor, epoch: int) -> None:
        """
        Visualize text embeddings as a heatmap.

        Args:
            embeddings: Tensor of shape [num_samples, embedding_dim]
            epoch: Current epoch number
        """
        try:
            plt.figure (figsize=(12, 6))
            sns.heatmap (embeddings.cpu ().numpy (),
                         cmap='viridis',
                         annot=False)

            plt.title (f'Text Embeddings - Epoch {epoch}')
            output_path = self.viz_dir / f'text_embeddings_epoch_{epoch}.png'
            plt.savefig (output_path)
            plt.close ()

            logging.info (f"Saved text embeddings visualization to {output_path}")
        except Exception as e:
            logging.error (f"Error generating text embeddings plot: {str (e)}")
            raise

    def plot_attention_weights(self, attention_weights: torch.Tensor,
                               epoch: int, layer_name: str = "") -> None:
        """
        Visualize attention weights from model.

        Args:
            attention_weights: Attention weight tensor, shape [num_heads, seq_len_q, seq_len_k]
            epoch: Current epoch number
            layer_name: Name of attention layer
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_weights.cpu().numpy(),
                        cmap='viridis',
                        annot=True,
                        fmt='.2f')

            plt.title(f'Attention Weights - {layer_name} - Epoch {epoch}')
            output_path = self.viz_dir / f'attention_weights_{layer_name}_epoch_{epoch}.png'
            plt.savefig(output_path)
            plt.close()

            logging.info(f"Saved attention weights visualization to {output_path}")

        except Exception as e:
            logging.error(f"Error generating attention weights plot: {str(e)}")
            raise

    def plot_feature_maps(self, feature_maps: Dict[str, torch.Tensor],
                          epoch: int) -> None:
        """
        Visualize feature maps from different layers.

        Args:
            feature_maps: Dictionary of feature tensors {layer_name: tensor}
            epoch: Current epoch number
        """
        try:
            for name, features in feature_maps.items():
                # Expecting features shape: [B, C, H, W] or [B, channels, seq_len]
                features = features.cpu().numpy()
                n_features = min(16, features.shape[1])

                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for idx in range(n_features):
                    ax = axes[idx // 4, idx % 4]
                    im = ax.imshow(features[0, idx], cmap='viridis')
                    ax.axis('off')

                plt.colorbar(im, ax=axes.ravel().tolist())
                plt.suptitle(f'{name} Feature Maps - Epoch {epoch}')

                output_path = self.viz_dir / f'feature_maps_{name}_epoch_{epoch}.png'
                plt.savefig(output_path)
                plt.close()

                logging.info(f"Saved feature maps visualization to {output_path}")

        except Exception as e:
            logging.error(f"Error generating feature maps plot: {str(e)}")
            raise


class LRFinder:
    """Learning rate finder implementation (Smith's approach)."""

    def __init__(self, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device):
        """
        Initialize LR finder.

        Args:
            model: Model to train
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Store initial state
        self.init_params = [p.clone().detach() for p in model.parameters()]
        self.init_opt = optimizer.state_dict()

    def reset(self) -> None:
        """Reset model and optimizer to initial state."""
        for p, init_p in zip(self.model.parameters(), self.init_params):
            p.data = init_p.clone()
        self.optimizer.load_state_dict(self.init_opt)

    def range_test(self, train_loader: DataLoader,
                   start_lr: float = 1e-7,
                   end_lr: float = 10,
                   num_iter: int = 100,
                   step_mode: str = "exp") -> Tuple[List[float], List[float]]:
        """
        Perform learning rate range test.

        Args:
            train_loader: DataLoader for training data
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            step_mode: Learning rate increase mode ("exp" or "linear")

        Returns:
            (lrs, losses): Lists of recorded learning rates and corresponding losses
        """
        self.reset()
        self.optimizer.param_groups[0]['lr'] = start_lr

        if step_mode == "exp":
            gamma = (end_lr / start_lr) ** (1 / num_iter)
            update_fn = lambda lr: lr * gamma
        else:
            step = (end_lr - start_lr) / num_iter
            update_fn = lambda lr: lr + step

        lrs, losses = [], []
        best_loss = float('inf')

        try:
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= num_iter:
                    break

                loss = self._training_step(batch)
                lrs.append(self.optimizer.param_groups[0]['lr'])
                losses.append(loss)

                self.optimizer.param_groups[0]['lr'] = update_fn(self.optimizer.param_groups[0]['lr'])

                # Stop if loss diverges
                if loss > 4 * best_loss:
                    logging.info("Stopping early due to diverging loss")
                    break
                if loss < best_loss:
                    best_loss = loss

        except Exception as e:
            logging.error(f"Error during LR range test: {str(e)}")
            raise

        return lrs, losses

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step for LR Finder."""
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for current batch."""
        audio = batch['audio'].to(self.device)
        image = batch['image'].to(self.device)
        targets = batch['emotion'].to(self.device)

        outputs = self.model(audio, image)
        return self.criterion(outputs, targets)

    def plot_lr_find(self, lrs: List[float], losses: List[float],
                     skip_start: int = 10, skip_end: int = 5,
                     log_scale: bool = True) -> plt.Figure:
        """
        Plot results from LR Finder.

        Args:
            lrs: Learning rates recorded
            losses: Corresponding losses
            skip_start: Number of initial batches to skip
            skip_end: Number of final batches to skip
            log_scale: Use log scale on X-axis

        Returns:
            Matplotlib Figure object
        """
        fig = plt.figure(figsize=(10, 6))
        plt.plot(lrs[skip_start:-skip_end], losses[skip_start:-skip_end])

        if log_scale:
            plt.xscale('log')

        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        return fig


class EnsembleModel(nn.Module):
    """Ensemble of emotion recognition models with weighted averaging."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.

        Args:
            models: List of models to ensemble
            weights: Optional weights for each model
        """
        super().__init__()

        if not models:
            raise ValueError("Must provide at least one model")

        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")

        self.weights = weights

    def forward(self, audio: torch.Tensor, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ensemble.

        Args:
            audio: Audio input tensor
            image: Image input tensor

        Returns:
            Dictionary with weighted ensemble predictions: 'image_pred', 'audio_pred', 'fusion_pred'
        """
        # Collect outputs from each model
        outputs_list = []
        for model in self.models:
            with torch.no_grad():
                outputs_list.append(model(audio, image))

        # Weighted average of predictions
        ensemble_output = {
            key: sum(w * out[key] for w, out in zip(self.weights, outputs_list))
            for key in ['image_pred', 'audio_pred', 'fusion_pred']
        }

        return ensemble_output
