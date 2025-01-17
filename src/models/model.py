# src/models/model.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from src.models.components.blocks import ResidualBlock, ResidualBlock1D
from src.models.components.attention import MultiHeadAttention


class ImprovedEmotionModel (nn.Module):
    """
    Multimodal emotion recognition model combining audio, image, and text inputs.
    Uses CNN-based encoders for image/audio, an RNN-based encoder for text,
    and an advanced attention mechanism for fusion.
    """

    def __init__ (self, num_emotions: int = 7, dropout: float = 0.5,
                  vocab_size: int = 30522, embed_dim: int = 128, rnn_hidden: int = 256):
        """
        Args:
            num_emotions: Number of emotion classes
            dropout: Dropout probability
            vocab_size: Vocabulary size for text embedding
            embed_dim: Dimension of the text embedding
            rnn_hidden: Hidden size for the bidirectional RNN
        """
        super ().__init__ ()

        # -------------------------------------------------------------
        # 1. Image Encoder with ResidualBlocks
        # -------------------------------------------------------------
        self.image_encoder = nn.Sequential (
            # Initial convolution
            nn.Conv2d (1, 64, 3, padding=1),
            nn.BatchNorm2d (64),
            nn.ReLU (inplace=True),

            # Residual blocks with increasing channels
            ResidualBlock (64, 64),
            ResidualBlock (64, 128, stride=2),  # Spatial reduction
            ResidualBlock (128, 128),
            ResidualBlock (128, 256, stride=2),  # Further reduction
            ResidualBlock (256, 256),

            # Global pooling and final projection
            nn.AdaptiveAvgPool2d ((1, 1)),
            nn.Flatten (),
            nn.Dropout (0.3),
            nn.Linear (256, 256)
        )


        # -------------------------------------------------------------
        # 2. Enhanced Audio Encoder
        # -------------------------------------------------------------
        self.audio_encoder = nn.Sequential(
            # Initial 1D convolution
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            ResidualBlock1D(128, 128),
            nn.LSTM(
                input_size=128,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        # -------------------------------------------------------------
        # 3. Text Encoder
        # -------------------------------------------------------------
        self.embed_dim = embed_dim
        self.rnn_hidden = rnn_hidden

        self.text_embedding = nn.Embedding (num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=0)
        self.text_rnn = nn.LSTM (input_size=embed_dim,
                                 hidden_size=rnn_hidden // 2,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.text_proj = nn.Linear (rnn_hidden, 256)

        # -------------------------------------------------------------
        # 4. Advanced Fusion Mechanism
        # -------------------------------------------------------------
        # Modality dimension
        self.modality_dim = 256

        # Learnable modality tokens
        self.image_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.audio_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.text_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))

        # Modality presence embedding
        self.presence_embedding = nn.Parameter (torch.randn (2, self.modality_dim))

        # Modality-specific projections
        self.image_proj = nn.Linear (256, self.modality_dim)
        self.audio_proj = nn.Linear (256, self.modality_dim)
        self.text_proj = nn.Linear (256, self.modality_dim)

        # Cross-modal attention
        self.fusion_attention = MultiHeadAttention (
            dim=self.modality_dim,
            num_heads=8,
            dropout=dropout
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential (
            nn.Linear (self.modality_dim * 3, 512),
            nn.LayerNorm (512),
            nn.ReLU (),
            nn.Dropout (dropout),
            nn.Linear (512, 256)
        )

        # -------------------------------------------------------------
        # 5. Layer Normalization & Classification Heads
        # -------------------------------------------------------------
        self.image_norm = nn.LayerNorm (256)
        self.audio_norm = nn.LayerNorm (256)
        self.text_norm = nn.LayerNorm (256)
        self.fusion_norm = nn.LayerNorm (256)

        # Create classification heads
        classifier_config = [
            ('image', self.image_norm),
            ('audio', self.audio_norm),
            ('text', self.text_norm),
            ('fusion', self.fusion_norm)
        ]

        self.classifiers = nn.ModuleDict ({
            name: self._make_classifier (norm_layer, num_emotions, dropout)
            for name, norm_layer in classifier_config
        })

        self.init_weights ()


    def _make_classifier (self, norm_layer: nn.Module, num_emotions: int, dropout: float) -> nn.Sequential:
        """Create a classification head"""
        return nn.Sequential (
            norm_layer,
            nn.Linear (256, 128),
            nn.ReLU (),
            nn.Dropout (dropout),
            nn.Linear (128, num_emotions)
        )

    def init_weights (self):
        """Improved weight initialization"""
        for m in self.modules ():
            if isinstance (m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_ (m.weight)
                if m.bias is not None:
                    nn.init.zeros_ (m.bias)
            elif isinstance (m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_ (m.weight)
                nn.init.zeros_ (m.bias)
        nn.init.normal_ (self.image_token, mean=0.0, std=0.02)
        nn.init.normal_ (self.audio_token, mean=0.0, std=0.02)
        nn.init.normal_ (self.text_token, mean=0.0, std=0.02)
        nn.init.normal_ (self.presence_embedding, mean=0.0, std=0.02)

    def fuse_modalities (self, image_features: Optional[torch.Tensor] = None,
                         audio_features: Optional[torch.Tensor] = None,
                         text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advanced fusion mechanism using attention"""
        batch_size = next (x.size (0) for x in [image_features, audio_features, text_features]
                           if x is not None)
        device = next (x.device for x in [image_features, audio_features, text_features]
                       if x is not None)

        # Initialize feature list and presence mask
        features = []
        presence_mask = torch.zeros (batch_size, 3, device=device)

        # Process each modality
        modalities = [
            (image_features, self.image_token, self.image_proj, 0),
            (audio_features, self.audio_token, self.audio_proj, 1),
            (text_features, self.text_token, self.text_proj, 2)
        ]

        for features_tensor, token, proj, idx in modalities:
            if features_tensor is not None:
                # Project features and add modality token
                proj_features = proj (features_tensor)
                mod_token = token.expand (batch_size, -1, -1)
                features.append (proj_features.unsqueeze (1) + mod_token)
                presence_mask[:, idx] = 1
            else:
                # Use learned token with presence embedding
                mod_token = token.expand (batch_size, -1, -1)
                absent_emb = self.presence_embedding[1:2].expand (batch_size, 1, -1)
                features.append (mod_token + absent_emb)

        # Combine features
        combined_features = torch.cat (features, dim=1)  # [batch_size, 3, modality_dim]

        # Create proper attention mask
        # First expand for number of attention heads
        presence_mask = presence_mask.unsqueeze (1)  # [batch_size, 1, 3]
        # Then expand to proper shape [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_mask = presence_mask.unsqueeze (2).expand (batch_size, self.fusion_attention.num_heads, 3, 3)

        # Apply attention
        attended_features, _ = self.fusion_attention (
            combined_features, combined_features, combined_features,
            mask=attention_mask
        )

        # Final fusion through MLP
        fused = attended_features.reshape (batch_size, -1)
        return self.fusion_mlp (fused)

    def forward (self,
                 image: Optional[torch.Tensor] = None,
                 audio: Optional[torch.Tensor] = None,
                 text_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass handling all modalities"""
        predictions = {}

        # Process image if available
        if image is not None:
            image_features = self.image_encoder (image)
            predictions['image_pred'] = self.classifiers['image'] (image_features)
        else:
            image_features = None

        # Process audio if available
        if audio is not None:
            audio_features = self.audio_encoder (audio).squeeze (-1)
            predictions['audio_pred'] = self.classifiers['audio'] (audio_features)
        else:
            audio_features = None

        # Process text if available
        if text_input is not None:
            embedded = self.text_embedding (text_input)
            rnn_out, _ = self.text_rnn (embedded)
            text_features = self.text_proj (rnn_out.mean (dim=1))
            predictions['text_pred'] = self.classifiers['text'] (text_features)
        else:
            text_features = None

        # Perform fusion if any modality is present
        if any (x is not None for x in [image_features, audio_features, text_features]):
            fusion_features = self.fuse_modalities (image_features, audio_features, text_features)
            predictions['fusion_pred'] = self.classifiers['fusion'] (fusion_features)

        return predictions


class MultiModalLoss (nn.Module):
    """Combined loss function for multimodal emotion recognition"""

    def __init__ (self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: e.g. {'image':0.25, 'audio':0.25, 'text':0.25, 'fusion':0.25}
        """
        super ().__init__ ()
        self.criterion = nn.CrossEntropyLoss ()
        self.weights = weights or {'image': 0.3, 'audio': 0.3, 'text': 0.3, 'fusion': 0.1}

        if not np.isclose (sum (self.weights.values ()), 1.0):
            raise ValueError ("Loss weights must sum to 1")

    def forward (self, outputs, targets):
        total_loss = 0.0
        batch_size = targets.size (0)

        for key in self.weights.keys ():
            pred_key = f'{key}_pred'
            if pred_key in outputs:
                # Calculate cross-entropy loss
                logits = outputs[pred_key]
                ce_loss = self.criterion (logits, targets)

                # Add L2 regularization (apply only to model weights, not outputs)
                l2_reg = sum (torch.sum (p ** 2) for p in self.parameters ()) * 0.0001

                # Combine losses
                loss_val = ce_loss + l2_reg
                total_loss += self.weights[key] * loss_val

        return total_loss