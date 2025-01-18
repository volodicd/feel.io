import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from src.models.components.blocks import ResidualBlock
from src.models.components.attention import MultiHeadAttention
import torchaudio.transforms as T

def load_model_conf(config_path: str = 'configs/model_config.yaml') -> Dict:
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ImprovedEmotionModel(nn.Module):
    """
    Multimodal emotion recognition model combining audio, image, and text inputs.
    Uses CNN-based encoders for image/audio, an RNN-based encoder for text,
    and an advanced attention mechanism for fusion.
    """
    def __init__(self, config_path: str = 'configs/model_config.yaml'):
        """
        Args:
            config_path: Path to model configuration file
        """
        super().__init__()
        self.config = load_model_conf(config_path)

        model_config = self.config['model']
        attention_config = self.config['attention']
        dropouts = self.config['dropouts']
        fusion_config = self.config['fusion']
        image_config = self.config['image_encoder']
        audio_config = self.config['audio_encoder']
        text_config = self.config['text_encoder']

        # -------------------------------------------------------------
        # 1. Image Encoder with ResidualBlocks
        # -------------------------------------------------------------
        self.image_encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(1, image_config['initial_channels'], 3, padding=1),
            nn.BatchNorm2d(image_config['initial_channels']),
            nn.ReLU(inplace=True),

            # Residual blocks with increasing channels
            ResidualBlock(image_config['channels'][0], image_config['channels'][0]),
            ResidualBlock(image_config['channels'][0], image_config['channels'][1], stride=2),  # Spatial reduction
            ResidualBlock(image_config['channels'][1], image_config['channels'][1]),
            ResidualBlock(image_config['channels'][1], image_config['channels'][2], stride=2),  # Further reduction
            ResidualBlock(image_config['channels'][2], image_config['channels'][2]),

            # Global pooling and final projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropouts['image_encoder']),
            nn.Linear(image_config['channels'][2], image_config['channels'][2])
        )

        # -------------------------------------------------------------
        # 2. Audio Encoder (1D conv blocks)
        # -------------------------------------------------------------
        self.spectrogram = T.MelSpectrogram (
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
        )
        self.audio_encoder = nn.Sequential (
            ResidualBlock (in_channels=1, out_channels=32, stride=2),  # Initial channels
            ResidualBlock (32, 64, stride=2),
            ResidualBlock (64, 128, stride=2),
            nn.AdaptiveAvgPool2d ((1, 1)),  # Global pooling for flattened output
            nn.Flatten (),
            nn.Linear (128, 256),  # Project to a fixed embedding size
            nn.ReLU (),
            nn.Dropout (0.3),
        )

        # -------------------------------------------------------------
        # 3. Text Encoder
        # -------------------------------------------------------------
        self.embed_dim = text_config['embedding_dim']
        self.rnn_hidden = model_config['rnn_hidden']

        self.text_embedding = nn.Embedding(num_embeddings=text_config['vocab_size'],
                                           embedding_dim=self.embed_dim, padding_idx=0)
        self.text_rnn = nn.LSTM(input_size=self.embed_dim,
                                hidden_size=self.rnn_hidden // 2,
                                num_layers=text_config['rnn_layers'],
                                batch_first=True,
                                bidirectional=text_config['bidirectional'],)
        # -------------------------------------------------------------
        # 4. Advanced Fusion Mechanism
        # -------------------------------------------------------------
        # Modality dimension
        self.modality_dim = fusion_config['modality_dim']

        # Learnable modality tokens
        self.image_token = nn.Parameter(torch.randn(1, 1, self.modality_dim))
        self.audio_token = nn.Parameter(torch.randn(1, 1, self.modality_dim))
        self.text_token = nn.Parameter(torch.randn(1, 1, self.modality_dim))

        # Modality presence embedding
        self.presence_embedding = nn.Parameter(torch.randn(3, self.modality_dim))

        # Modality-specific projections
        self.image_proj = nn.Linear(image_config['channels'][2], self.modality_dim)
        self.audio_proj = nn.Linear(audio_config['channels'][2], self.modality_dim)
        self.text_proj = nn.Linear(self.rnn_hidden, self.modality_dim)

        # Cross-modal attention
        self.fusion_attention = MultiHeadAttention(
            dim=self.modality_dim,
            num_heads=attention_config['num_heads'],
            dropout=dropouts['attention']
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.modality_dim * 3, fusion_config['mlp_hidden']),
            nn.LayerNorm(fusion_config['mlp_hidden']),
            nn.ReLU(),
            nn.Dropout(dropouts['fusion']),
            nn.Linear(fusion_config['mlp_hidden'], fusion_config['modality_dim'])
        )

        # -------------------------------------------------------------
        # 5. Layer Normalization & Classification Heads
        # -------------------------------------------------------------
        self.image_norm = nn.LayerNorm(fusion_config['modality_dim'])
        self.audio_norm = nn.LayerNorm(fusion_config['modality_dim'])
        self.text_norm = nn.LayerNorm(fusion_config['modality_dim'])
        self.fusion_norm = nn.LayerNorm(fusion_config['modality_dim'])

        # Create classification heads
        classifier_config = [
            ('image', self.image_norm),
            ('audio', self.audio_norm),
            ('text', self.text_norm),
            ('fusion', self.fusion_norm)
        ]

        self.classifiers = nn.ModuleDict({
            name: self._make_classifier(norm_layer, model_config['num_emotions'], dropouts['classifier'])
            for name, norm_layer in classifier_config
        })

        self.init_weights()

    def _make_classifier(self, norm_layer: nn.Module, num_emotions: int, dropout: float) -> nn.Sequential:
        """Create a classification head"""
        return nn.Sequential(
            norm_layer,
            nn.Linear(self.modality_dim, self.modality_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.modality_dim // 2, num_emotions)
        )

    def init_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.image_token, mean=0.0, std=0.02)
        nn.init.normal_(self.audio_token, mean=0.0, std=0.02)
        nn.init.normal_(self.text_token, mean=0.0, std=0.02)
        nn.init.normal_(self.presence_embedding, mean=0.0, std=0.02)

    def fuse_modalities (self, image_features=None, audio_features=None, text_features=None):
        """
        Improved fusion mechanism with normalized modality weighting and attention dropout.
        """
        batch_size = next (x.size (0) for x in [image_features, audio_features, text_features] if x is not None)
        device = next (x.device for x in [image_features, audio_features, text_features] if x is not None)

        # Initialize feature list and modality presence mask
        features = []
        modality_weights = []
        presence_mask = torch.zeros (batch_size, 3, device=device)

        # Process each modality with learnable weights
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
                weighted_features = proj_features.unsqueeze (1) + mod_token
                features.append (weighted_features)

                # Learnable modality-specific weight
                modality_weight = torch.sigmoid (self.presence_embedding[idx:idx + 1]).expand (batch_size, -1)
                modality_weights.append (modality_weight)

                presence_mask[:, idx] = 1
            else:
                # Use learned token for missing modality
                mod_token = token.expand (batch_size, -1, -1)
                absent_emb = self.presence_embedding[1:2].expand (batch_size, 1, -1)
                features.append (mod_token + absent_emb)
                modality_weights.append (torch.zeros (batch_size, self.modality_dim, device=device))

        # Normalize modality weights
        modality_weights = torch.stack (modality_weights, dim=1)  # [batch_size, 3, modality_dim]
        modality_weights = torch.softmax (modality_weights, dim=1)  # Normalize across modalities

        # Combine features and weights
        combined_features = torch.cat (features, dim=1)  # [batch_size, 3, modality_dim]
        combined_features = combined_features * modality_weights.unsqueeze (-1)

        # Cross-modal attention with dropout
        attention_mask = presence_mask.unsqueeze (1).repeat (1, 3, 1)  # [batch_size, 3, 3]
        attended_features, _ = self.fusion_attention (
            combined_features, combined_features, combined_features, mask=attention_mask
        )
        attended_features = self.attention_dropout (attended_features)

        # Fusion through MLP
        fused = attended_features.reshape (batch_size, -1)  # Flatten for MLP input
        return self.fusion_mlp (fused)


    def forward(self,
                image: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                text_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass handling all modalities"""
        predictions = {}

        # Process image if available
        if image is not None:
            image_features = self.image_encoder(image)
            predictions['image_pred'] = self.classifiers['image'](image_features)
        else:
            image_features = None

        # Process audio if available
        if audio is not None:
            spectrogram = self.spectrogram (audio)  # Convert raw audio to spectrogram
            spectrogram = spectrogram.unsqueeze (1)  # Add channel dimension, shape: [batch_size, 1, n_mels, time_steps]
            assert spectrogram.dim () == 4, f"Unexpected shape: {spectrogram.shape}"  # Debugging step
            audio_features = self.audio_encoder (spectrogram)  # Pass to the encoder
            predictions['audio_pred'] = self.classifiers['audio'] (audio_features)
        else:
            audio_features = None

        # Process text if available
        if text_input is not None:
            embedded = self.text_embedding(text_input)
            rnn_out, _ = self.text_rnn(embedded)
            text_features = self.text_proj(rnn_out.mean(dim=1))
            predictions['text_pred'] = self.classifiers['text'](text_features)
        else:
            text_features = None

        # Perform fusion if any modality is present
        if any(x is not None for x in [image_features, audio_features, text_features]):
            fusion_features = self.fuse_modalities(image_features, audio_features, text_features)
            predictions['fusion_pred'] = self.classifiers['fusion'](fusion_features)

        return predictions

class MultiModalLoss(nn.Module):
    """Combined loss function for multimodal emotion recognition"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: e.g. {'image':0.25, 'audio':0.25, 'text':0.25, 'fusion':0.25}
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weights = weights or {'image': 0.3, 'audio': 0.3, 'text': 0.3, 'fusion': 0.1}

        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Loss weights must sum to 1")

    def forward(self, outputs, targets):
        total_loss = 0.0

        for key in self.weights.keys():
            pred_key = f'{key}_pred'
            if pred_key in outputs:
                # Just use cross entropy loss - L2 regularization should be handled by optimizer
                loss_val = self.criterion(outputs[pred_key], targets)
                total_loss += self.weights[key] * loss_val

        return total_loss