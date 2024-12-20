# src/models/model.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

# Import building blocks from your new components subfolder
from src.models.components.blocks import ResidualBlock
from src.models.components.attention import MultiHeadAttention


class ImprovedEmotionModel(nn.Module):
    """
    Multimodal emotion recognition model combining audio, image, and text inputs.
    Uses CNN-based encoders for image/audio, an RNN-based encoder for text,
    and an attention mechanism for final fusion.
    """

    def __init__(self, num_emotions: int = 7, dropout: float = 0.5,
                 vocab_size: int = 20000, embed_dim: int = 128, rnn_hidden: int = 256):
        """
        Args:
            num_emotions: Number of emotion classes
            dropout: Dropout probability
            vocab_size: Vocabulary size for text embedding
            embed_dim: Dimension of the text embedding
            rnn_hidden: Hidden size for the bidirectional RNN (output will be projected to 256)
        """
        super().__init__()

        # -------------------------------------------------------------
        # 1. Image Encoder
        # -------------------------------------------------------------
        self.image_encoder = nn.Sequential (
            # Block 1
            nn.Conv2d (1, 64, 3, padding=1),
            nn.BatchNorm2d (64),
            nn.ReLU (inplace=True),
            nn.Conv2d (64, 64, 3, padding=1),
            nn.BatchNorm2d (64),
            nn.ReLU (inplace=True),
            nn.MaxPool2d (2),
            nn.Dropout (0.3),

            # Block 2
            nn.Conv2d (64, 128, 3, padding=1),
            nn.BatchNorm2d (128),
            nn.ReLU (inplace=True),
            nn.Conv2d (128, 128, 3, padding=1),
            nn.BatchNorm2d (128),
            nn.ReLU (inplace=True),
            nn.MaxPool2d (2),
            nn.Dropout (0.4),

            # Dense layers
            nn.Flatten (),
            nn.Linear (128 * 12 * 12, 512),
            nn.ReLU (inplace=True),
            nn.Dropout (0.5),
            nn.Linear (512, 256)
        )

        # -------------------------------------------------------------
        # 2. Audio Encoder (1D conv blocks)
        # -------------------------------------------------------------
        self.audio_encoder = nn.Sequential(
            self._make_audio_block(1, 64),
            self._make_audio_block(64, 128),
            self._make_audio_block(128, 256),
            nn.AdaptiveAvgPool1d(1)
        )

        # -------------------------------------------------------------
        # 3. Text Encoder
        # -------------------------------------------------------------
        # We define an Embedding + Bidirectional LSTM (or GRU).
        self.embed_dim = embed_dim
        self.rnn_hidden = rnn_hidden

        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # Bidirectional LSTM (hidden size rnn_hidden//2 each direction => final hidden is rnn_hidden)
        self.text_rnn = nn.LSTM(input_size=embed_dim,
                                hidden_size=rnn_hidden // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)
        # Project RNN output from rnn_hidden to 256 for fusion
        self.text_proj = nn.Linear(rnn_hidden, 256)

        # -------------------------------------------------------------
        # 4. Cross-modal attention or fusion
        # -------------------------------------------------------------
        self.cross_attention = MultiHeadAttention(dim=256, num_heads=8, dropout=dropout)

        # -------------------------------------------------------------
        # 5. Layer normalization & Classification heads
        # -------------------------------------------------------------
        self.image_norm = nn.LayerNorm(256)
        self.audio_norm = nn.LayerNorm(256)
        self.text_norm = nn.LayerNorm(256)
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        # Create a classification head for each modality plus the fusion
        classifier_config = [
            ('image', self.image_norm),
            ('audio', self.audio_norm),
            ('text', self.text_norm),
            ('fusion', self.fusion_norm)
        ]

        self.classifiers = nn.ModuleDict({
            name: self._make_classifier(norm_layer, num_emotions, dropout)
            for name, norm_layer in classifier_config
        })
        self.init_weights()

    # -------------------------------------------------------------
    # Utility function for audio blocks
    # -------------------------------------------------------------
    def _make_audio_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a simple 1D conv block for audio, with two conv layers and ReLU."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    # -------------------------------------------------------------
    # Utility function to create classifier heads
    # -------------------------------------------------------------
    def _make_classifier(self, norm_layer: nn.Module, num_emotions: int, dropout: float) -> nn.Sequential:
        """
        Create classification head for a single modality or the fusion output.
        Applies layer normalization, then two linear layers with ReLU + dropout.
        """
        return nn.Sequential(
            norm_layer,
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )

    # -------------------------------------------------------------
    # Forward pass (audio, image, text)
    # -------------------------------------------------------------

    def init_weights (self):
        for m in self.modules ():
            if isinstance (m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_ (m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_ (m.bias, 0)
            elif isinstance (m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_ (m.weight, 1)
                nn.init.constant_ (m.bias, 0)

    def forward (self,
                 image: Optional[torch.Tensor] = None,  # shape: [B, 3, H, W]
                 audio: Optional[torch.Tensor] = None,  # shape: [B, 1, T_audio]
                 text_input: Optional[torch.Tensor] = None  # shape: [B, seq_len] of token IDs
                 ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: Tensor of shape [batch_size, 3, H, W] or None
            audio: Tensor of shape [batch_size, 1, audio_length] or None
            text_input: Tensor of shape [batch_size, seq_len] or None

        Returns:
            Dictionary with keys: 'image_pred', 'audio_pred', 'text_pred', 'fusion_pred'
        """
        # Initialize predictions dictionary
        predictions = {}

        # Placeholder features for missing modalities
        batch_size = 1  # Default batch size
        device = torch.device("cpu")

        if image is not None:
            batch_size = image.size (0)
            device = image.device
        elif audio is not None:
            batch_size = audio.size (0)
            device = audio.device
        elif text_input is not None:
            batch_size = text_input.size (0)
            device = text_input.device

        zero_features = torch.zeros((batch_size, 256), device=device)

        # ---------------------------------------------------------
        # 1. Encode Image (if provided)
        # ---------------------------------------------------------
        if image is not None:
            image_features = self.image_encoder(image)  # [B, 256, 1, 1]
            image_features = image_features.squeeze(-1).squeeze (-1)  # [B, 256]
            image_pred = self.classifiers['image'](image_features)
            predictions['image_pred'] = image_pred
        else:
            image_features = zero_features  # Replace with zeros if image is missing

        # ---------------------------------------------------------
        # 2. Encode Audio (if provided)
        # ---------------------------------------------------------
        if audio is not None:
            audio_features = self.audio_encoder(audio).squeeze (-1)  # [B, 256]
            audio_pred = self.classifiers['audio'](audio_features)
            predictions['audio_pred'] = audio_pred
        else:
            audio_features = zero_features  # Replace with zeros if audio is missing

        # ---------------------------------------------------------
        # 3. Encode Text (if provided)
        # ---------------------------------------------------------
        if text_input is not None:
            embedded = self.text_embedding (text_input)  # [B, seq_len, embed_dim]
            rnn_out, _ = self.text_rnn (embedded)  # [B, seq_len, rnn_hidden]
            text_feat = rnn_out.mean (dim=1)  # Mean pooling [B, rnn_hidden]
            text_features = self.text_proj (text_feat)  # [B, 256]
            text_pred = self.classifiers['text'] (text_features)
            predictions['text_pred'] = text_pred
        else:
            text_features = zero_features  # Replace with zeros if text is missing

        # ---------------------------------------------------------
        # 4. Fusion using only available features
        # ---------------------------------------------------------
        features = []
        if image is not None:
            features.append(image_features)
        if audio is not None:
            features.append(audio_features)
        if text_input is not None:
            features.append(text_features)

        # Stack only available features
        if len(features) > 0:
            # Concatenate all available features
            fused_features = torch.cat(features, dim=1)  # [B, 256 * num_modalities]
            # Pad with zeros for missing modalities
            if len(features) < 3:
                padding = torch.zeros((fused_features.size(0), 256 * (3 - len(features))), device=device)
                fused_features = torch.cat([fused_features, padding], dim=1)
            # Process through MLP
            fusion_features = self.fusion_mlp(fused_features)
        else:
            fusion_features = zero_features

        # Final fusion prediction
        fusion_pred = self.classifiers['fusion'](fusion_features)
        predictions['fusion_pred'] = fusion_pred

        return predictions


class MultiModalLoss(nn.Module):
    """
    Combined loss function for multimodal emotion recognition.
    Weighs losses from different modalities (image, audio, text, fusion).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: e.g. {'image':0.25, 'audio':0.25, 'text':0.25, 'fusion':0.25}
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        # Default if user doesn't provide weights for text
        self.weights = weights or {'image': 0.35, 'audio': 0.20, 'text': 0.20, 'fusion': 0.25}

        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Loss weights must sum to 1")

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted sum of cross-entropy losses from each modality.

        Args:
            outputs: Dict with keys 'image_pred', 'audio_pred', 'text_pred' (optional), 'fusion_pred'
            targets: Ground truth labels, shape [B]

        Returns:
            Weighted sum of losses.
        """
        losses = {}
        total_loss = 0.0
        for key in self.weights.keys():
            pred_key = f'{key}_pred'
            if pred_key in outputs:  # text might be missing
                loss_val = self.criterion(outputs[pred_key], targets)
                total_loss += self.weights[key] * loss_val

        return total_loss


