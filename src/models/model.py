import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from src.models.components.blocks import ResidualBlock, ResidualBlock1D
from src.models.components.attention import MultiHeadAttention
import torchaudio.transforms as T
import torch.nn.functional as F


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
        # Add normalization layer
        self.normalize = nn.Sequential (
            nn.LayerNorm (64),  # Normalize mel bins
            nn.ReLU ()
        )

        self.audio_encoder = nn.Sequential (
            # Increase kernel size and reduce stride for better temporal features
            nn.Conv1d (64, audio_config['initial_channels'], kernel_size=31, stride=1, padding=15),
            nn.BatchNorm1d (audio_config['initial_channels']),
            nn.ReLU (inplace=True),
            nn.Dropout (0.2),

            # Keep existing ResidualBlock1D layers but with modified channels
            ResidualBlock1D(audio_config['channels'][0], audio_config['channels'][0]),
            ResidualBlock1D(audio_config['channels'][0], audio_config['channels'][1], 2),
            ResidualBlock1D(audio_config['channels'][1], audio_config['channels'][1]),
            ResidualBlock1D(audio_config['channels'][1], audio_config['channels'][2], 2),

            nn.AdaptiveAvgPool1d (1),
            nn.Flatten (),
            nn.Dropout (dropouts['audio_encoder']),
            nn.Linear (audio_config['channels'][2], audio_config['channels'][2])
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
        # -------------------------------------------------------------
        # 4. Advanced Fusion Mechanism
        # -------------------------------------------------------------
        # Modality dimension
        self.modality_dim = fusion_config['modality_dim']

        # Add uncertainty and importance weights
        self.modality_uncertainty = nn.Parameter (torch.zeros (3))
        self.modality_importance = nn.Parameter (torch.ones (3))

        # Modality-specific projections (MOVE THESE HERE)
        self.image_proj = nn.Linear (image_config['channels'][2], self.modality_dim)
        self.audio_proj = nn.Linear (audio_config['channels'][2], self.modality_dim)
        self.text_proj = nn.Linear (self.rnn_hidden, self.modality_dim)

        # Positional encodings and absence tokens
        self.image_pos_encoding = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.audio_pos_encoding = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.text_pos_encoding = nn.Parameter (torch.randn (1, 1, self.modality_dim))

        self.image_absence_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.audio_absence_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.text_absence_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))

        # Learnable modality tokens
        self.image_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.audio_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))
        self.text_token = nn.Parameter (torch.randn (1, 1, self.modality_dim))

        # Modality presence embedding
        self.presence_embedding = nn.Parameter (torch.randn (3, self.modality_dim))

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

    def fuse_modalities (self, image_features: Optional[torch.Tensor] = None,
                         audio_features: Optional[torch.Tensor] = None,
                         text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Robust fusion mechanism with proper missing modality handling"""
        # Get batch size and device from available features
        batch_size = next (x.size (0) for x in [image_features, audio_features, text_features]
                           if x is not None)
        device = next (x.device for x in [image_features, audio_features, text_features]
                       if x is not None)

        # Initialize storage for features and presence mask
        features = []
        presence_mask = torch.zeros (batch_size, 3, device=device)

        # Define modality configurations
        modalities = [
            (image_features, self.image_proj, self.image_pos_encoding,
             self.image_absence_token, 0),
            (audio_features, self.audio_proj, self.audio_pos_encoding,
             self.audio_absence_token, 1),
            (text_features, self.text_proj, self.text_pos_encoding,
             self.text_absence_token, 2)
        ]

        # Process each modality
        for feat, proj, pos_enc, absence_token, idx in modalities:
            if feat is not None:
                # Project and add positional encoding
                proj_feat = proj (feat)
                if pos_enc is not None:
                    proj_feat = proj_feat + pos_enc.expand (batch_size, -1, self.modality_dim)

                # Add to features list
                features.append (proj_feat.unsqueeze (1))
                presence_mask[:, idx] = 1.0
            else:
                # Handle missing modality with learned absence token and uncertainty
                uncertainty = torch.sigmoid (self.modality_uncertainty[idx])
                missing_feat = absence_token.expand (batch_size, 1, -1) * uncertainty
                features.append (missing_feat)

        # Combine all features
        combined_features = torch.cat (features, dim=1)  # [batch_size, 3, modality_dim]

        # Create attention mask based on presence
        presence_mask = presence_mask.unsqueeze (1).expand (batch_size, self.fusion_attention.num_heads, 3)
        attention_mask = presence_mask.unsqueeze (-1) * presence_mask.unsqueeze (-2)

        # Apply learned modality importance
        importance_weights = F.softmax (self.modality_importance, dim=0).view (1, 3, 1, 1)
        print (f"importance_weights before expand: {importance_weights.shape}")
        importance_weights = importance_weights.expand (batch_size, self.fusion_attention.num_heads, 3, 3)
        print (f"importance_weights after expand: {importance_weights.shape}")
        print (f"importance_weights shape: {importance_weights.shape}")
        print (f"attention_mask shape before multiplication: {attention_mask.shape}")
        attention_mask = attention_mask * importance_weights

        # Apply fusion attention
        attended_features, _ = self.fusion_attention (
            combined_features, combined_features, combined_features,
            mask=attention_mask
        )

        # Weighted pooling based on presence and importance
        presence_weights = presence_mask.mean (1).unsqueeze (-1)  # [batch_size, 3, 1]
        weighted_features = attended_features * presence_weights * importance_weights.view (1, 3, 1)

        # Pool features and apply final MLP
        pooled_features = weighted_features.sum (dim=1) / (presence_weights.sum (dim=1) + 1e-8)
        return self.fusion_mlp (pooled_features)


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
            # Generate spectrogram
            spectrogram = torch.log1p (self.spectrogram (audio))
            spectrogram = spectrogram.squeeze (1)  # Remove the extra channel dim
            # Now shape is [batch_size, n_mels, time_steps]
            audio_features = self.audio_encoder (spectrogram)
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