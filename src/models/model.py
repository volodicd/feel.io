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
    Uses CNN-based encoders for image/audio, an RNN-based encoder for text (from scratch),
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
        self.image_encoder = nn.Sequential(
            ResidualBlock(3, 64, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256, stride=1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
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
        # 3. Text Encoder (from scratch, no pretrained)
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
    def forward(self,
                audio: torch.Tensor,      # shape: [B, 1, T_audio]
                image: torch.Tensor,      # shape: [B, 3, H, W]
                text_input: Optional[torch.Tensor] = None  # shape: [B, seq_len] of token IDs
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio: Tensor of shape [batch_size, 1, audio_length]
            image: Tensor of shape [batch_size, 3, H, W]
            text_input: Optional tensor of shape [batch_size, seq_len] representing tokenized text IDs

        Returns:
            Dictionary with keys: 'image_pred', 'audio_pred', 'text_pred', 'fusion_pred'
        """

        # ---------------------------------------------------------
        # 1. Encode Image
        # ---------------------------------------------------------
        image_features = self.image_encoder(image)  # [B, 256, 1, 1]
        image_features = image_features.squeeze(-1).squeeze(-1)  # [B, 256]
        image_pred = self.classifiers['image'](image_features)

        # ---------------------------------------------------------
        # 2. Encode Audio
        # ---------------------------------------------------------
        audio_features = self.audio_encoder(audio).squeeze(-1)  # [B, 256]
        audio_pred = self.classifiers['audio'](audio_features)

        # ---------------------------------------------------------
        # 3. Encode Text (if provided)
        # ---------------------------------------------------------
        if text_input is not None:
            # text_input: [B, seq_len]
            embedded = self.text_embedding(text_input)        # [B, seq_len, embed_dim]
            # Pass through LSTM
            rnn_out, (h, c) = self.text_rnn(embedded)         # [B, seq_len, rnn_hidden], h shape: [2, B, rnn_hidden//2]

            # We can take the final hidden state from both directions (concatenate)
            # or we can do mean pooling. Let's do mean pooling for simplicity:
            text_feat = rnn_out.mean(dim=1)  # shape: [B, rnn_hidden]

            # Project to 256
            text_features = self.text_proj(text_feat)  # [B, 256]
            text_pred = self.classifiers['text'](text_features)
        else:
            # If text is missing, we can set text_features = 0 or skip it
            # For now, let's do a zero tensor if text isn't provided.
            text_features = torch.zeros_like(audio_features)  # shape [B, 256]
            text_pred = None

        # ---------------------------------------------------------
        # 4. Fusion with Cross-Attention
        # ---------------------------------------------------------
        # We have 3 embeddings: image_features, audio_features, text_features
        # Each is [B, 256]. Let's treat them as "tokens" in cross-attention.
        # Minimal approach: stack them: shape [B, 3, 256]
        stack = torch.stack([image_features, audio_features, text_features], dim=1)  # [B, 3, 256]

        # Let's do a self-attention among these 3 tokens for fusion:
        fused, _ = self.cross_attention(stack, stack, stack)  # shape [B, 3, 256]

        # For final fusion, maybe we just mean-pool or take first token. Let's do mean-pooling across tokens:
        fused_features = fused.mean(dim=1)  # [B, 256]

        # Classification on fused features
        fusion_pred = self.classifiers['fusion'](fused_features)

        # Return dictionary of predictions
        predictions = {
            'image_pred': image_pred,
            'audio_pred': audio_pred,
            'fusion_pred': fusion_pred
        }
        if text_pred is not None:
            predictions['text_pred'] = text_pred

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
        self.weights = weights or {'image': 0.25, 'audio': 0.25, 'text': 0.25, 'fusion': 0.25}

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


