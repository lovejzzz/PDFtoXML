"""CNN/ResNet encoder + Transformer decoder for OMR.

Architecture options:
- CNNEncoder: Simple 4-block CNN (from scratch)
- ResNetEncoder: Pretrained ResNet-18 backbone (ImageNet features)
Both produce a grid of feature vectors from the input image.
- Decoder: Transformer decoder with cross-attention to encoder features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Simple CNN that converts (B, 1, H, W) → (B, S, D) feature sequence."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        B, D, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, D)


class ResNetEncoder(nn.Module):
    """Pretrained ResNet-18 encoder: (B, 1, H, W) → (B, S, D).

    Uses ImageNet-pretrained features. Adapts 1-channel grayscale input
    by repeating to 3 channels. Projects ResNet features to d_model.
    """

    def __init__(self, d_model: int = 256, freeze_layers: int = 2):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Take layers up to layer3 (output stride 16)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # /4, 64ch
        self.layer2 = resnet.layer2  # /8, 128ch
        self.layer3 = resnet.layer3  # /16, 256ch

        # Freeze early layers for transfer learning
        layers_to_freeze = [self.layer0, self.layer1, self.layer2, self.layer3][:freeze_layers]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # Project to d_model if needed (layer3 outputs 256 channels)
        resnet_out_ch = 256
        self.proj = nn.Conv2d(resnet_out_ch, d_model, 1) if resnet_out_ch != d_model else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat grayscale to 3 channels for pretrained conv1
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)

        B, D, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class OMRModel(nn.Module):
    """Image-to-sequence model for OMR.

    Encoder: CNN → feature grid → flattened sequence
    Decoder: Transformer decoder with cross-attention to encoder features
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1400,
        pad_idx: int = 0,
        encoder_type: str = "cnn",
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Encoder
        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(d_model, freeze_layers=2)
        else:
            self.encoder = CNNEncoder(d_model)
        self.enc_pos = PositionalEncoding(d_model, max_len=2000, dropout=dropout)

        # Decoder
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dec_pos = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask for autoregressive decoding."""
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        images: (B, 1, H, W)
        tgt_tokens: (B, T) — teacher-forced target tokens (shifted right)
        Returns: logits (B, T, vocab_size)
        """
        # Encode image
        enc_out = self.encoder(images)  # (B, S, D)
        enc_out = self.enc_pos(enc_out)

        # Decode tokens
        tgt_emb = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.dec_pos(tgt_emb)

        T = tgt_tokens.size(1)
        causal_mask = self._make_causal_mask(T, tgt_tokens.device)

        # Padding mask for target
        tgt_pad_mask = tgt_tokens == self.pad_idx

        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=enc_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        logits = self.output_proj(dec_out)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 1400,
    ) -> list[list[int]]:
        """Greedy autoregressive generation."""
        self.eval()
        B = images.size(0)
        device = images.device

        # Encode
        enc_out = self.encoder(images)
        enc_out = self.enc_pos(enc_out)

        # Start with SOS
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.token_embed(generated) * math.sqrt(self.d_model)
            tgt_emb = self.dec_pos(tgt_emb)

            T = generated.size(1)
            causal_mask = self._make_causal_mask(T, device)

            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=enc_out,
                tgt_mask=causal_mask,
            )

            logits = self.output_proj(dec_out[:, -1, :])  # (B, vocab)
            next_token = logits.argmax(dim=-1)  # (B,)

            # Set finished sequences to pad
            next_token[finished] = self.pad_idx
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Check for EOS
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        # Convert to lists, strip SOS/EOS/PAD
        results = []
        for b in range(B):
            seq = generated[b].tolist()
            # Remove SOS
            if seq[0] == sos_idx:
                seq = seq[1:]
            # Truncate at EOS
            if eos_idx in seq:
                seq = seq[: seq.index(eos_idx)]
            # Remove PAD
            seq = [t for t in seq if t != self.pad_idx]
            results.append(seq)

        return results
