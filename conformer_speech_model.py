"""
conformer_speech_model.py
==========================
Conformer Attention + Hybrid HMM-DNN Speech Recognition Model for Nepali.

Architecture:
  Input Normalized MFCC (39-dim)
       ↓
  Input Projection (Linear 39 → d_model)
       ↓
  Positional Encoding
       ↓
  N × ConformerBlock
     ├── Feed-Forward (Macaron style, ×0.5)
     ├── Multi-Head Self-Attention
     ├── Depthwise Convolution (Gated Linear Unit + SiLU + BatchNorm)
     └── Feed-Forward (Macaron style, ×0.5)
       ↓
  CTC Classifier Head (Linear → num_classes)
       ↓
  CTC Loss → Backpropagation → AdamW Weight Update
"""

import os
import sys
import argparse
import math
import io
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_pytorch_nepali import (
    TextTokenizer, MultilingualSpeechDataset, pad_collate_fn,
    extract_features_from_array, normalize_nepali_text, TSV_PATH, DATA_DIR
)
from audio_augmentation import AudioAugmentor

MODEL_SAVE_PATH = "conformer_speech_model.pt"

# ─── Sub-Modules ────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(ctx)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model=128, kernel_size=15, dropout=0.1):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                            padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, d)
        x = x.transpose(1, 2)  # → (B, d, T)
        x = self.glu(self.pw1(x))
        x = self.act(self.bn(self.dw(x)))
        x = self.drop(self.pw2(x))
        return x.transpose(1, 2)  # → (B, T, d)


class FeedForward(nn.Module):
    def __init__(self, d_model=128, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv(self.conv_norm(x))
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_pe(max_len, d_model))

    def _build_pe(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        if T > self.pe.size(1):
            self.pe = self._build_pe(T + 1000, self.d_model).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, :T]


# ─── Full Conformer Model ────────────────────────────────────────────────────

class ConformerSpeechModel(nn.Module):
    """
    Conformer Attention model with 4x Convolutional Subsampling for high-precision
    End-to-End Nepali Speech Recognition and Hybrid HMM-DNN interface.
    """
    def __init__(self, num_classes, num_features=39, d_model=128, num_blocks=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.subsampling = nn.Sequential(
            nn.Conv1d(num_features, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model=d_model)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def compute_output_lengths(self, input_lengths):
        """Computes time length after 4x convolutional subsampling."""
        l1 = (input_lengths + 1) // 2
        l2 = (l1 + 1) // 2
        return l2

    def forward(self, x):
        # x: (B, T, F) -> transpose to (B, F, T) for 1D conv subsampling
        x = x.transpose(1, 2)
        x = self.subsampling(x)
        x = x.transpose(1, 2)  # (B, T//4, d_model)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)  # (B, T//4, num_classes)

    def hmm_emission_log_likes(self, feat_tensor):
        """
        Hybrid HMM-DNN interface: converts DNN posteriors P(s|x) to emission log-likelihoods log P(x|s).
        """
        self.eval()
        with torch.no_grad():
            log_post = torch.log_softmax(self.forward(feat_tensor), dim=-1)
            num_classes = max(1, log_post.shape[-1])
            log_prior = math.log(1.0 / num_classes)
            return log_post - log_prior  # (B, T, K)


# ─── Training Engine ─────────────────────────────────────────────────────────

def train_conformer(epochs=10, batch_size=8, lr=5e-4, max_samples=500, nepali_source="huggingface", english_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = TextTokenizer()
    dataset = MultilingualSpeechDataset(
        nepali_source=nepali_source,
        nepali_tsv_path=TSV_PATH if nepali_source == "local" else None,
        nepali_data_dir=DATA_DIR if nepali_source == "local" else None,
        english_dir=english_dir,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )

    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Please check your network connection or dataset path.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    num_classes = len(tokenizer.char_map)
    print(f"Total training samples: {len(dataset)}")
    print(f"Vocabulary size (including blank/pad): {num_classes}")

    model = ConformerSpeechModel(num_classes=num_classes, d_model=128).to(device)
    criterion = nn.CTCLoss(blank=1, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n=== Conformer Attention Nepali Speech Recognition Training Pipeline ===")
    print("Dataset (pujanpaudel/nepali_speech_to_text) -> 39-MFCC CMVN -> SpecAugment -> Conformer Attention -> CTC Loss -> Weight Update\n")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (features, targets, in_lens, tgt_lens) in enumerate(loader):
            # SpecAugment Data Augmentation applied during training
            aug = torch.stack([AudioAugmentor.spec_augment(f) for f in features]).to(device)
            targets = targets.to(device)

            logits = model(aug)
            log_probs = logits.log_softmax(2).transpose(0, 1)  # (T, B, C)
            out_lens = model.compute_output_lengths(in_lens)
            loss = criterion(log_probs, targets, out_lens, tgt_lens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                print(f"Epoch [{epoch:02d}/{epochs:02d}] | Batch [{batch_idx+1:03d}/{len(loader):03d}] | Conformer CTC Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"--> Epoch [{epoch:02d}/{epochs:02d}] Completed | Avg Train Loss: {avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving best model checkpoint to '{MODEL_SAVE_PATH}'...")
            torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.char_map, "d_model": 128}, MODEL_SAVE_PATH)

    print(f"\nConformer training completed! Model saved to '{MODEL_SAVE_PATH}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer Speech Model Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=500, help="Max Nepali dataset samples")
    parser.add_argument("--dataset", type=str, default="huggingface", choices=["huggingface", "local"],
                        help="Dataset source: 'huggingface' (pujanpaudel/nepali_speech_to_text) or 'local'")
    parser.add_argument("--english_dir", type=str, default=None,
                        help="English audio source: 'peoples_speech' (HuggingFace) or local folder path")
    args = parser.parse_args()

    train_conformer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        nepali_source=args.dataset,
        english_dir=args.english_dir
    )
