"""
train_hybrid_conformer.py
=========================
Train the Custom Conformer-HMM Hybrid Speech Recognition Model on Any HuggingFace Dataset.

Pipeline:
  1. Load Dataset (from HuggingFace repository e.g., 'username/my_dataset' or 'pujanpaudel/nepali_speech_to_text')
  2. Build / Update Devanagari Character Tokenizer & Vocabulary
  3. Extract 39-MFCC Acoustic Features with Cepstral Mean & Variance Normalization (CMVN)
  4. Train Conformer Acoustic DNN (Multi-Head Self-Attention + Depthwise Conv + Macaron FeedForward) via CTC Loss & SpecAugment
  5. Fit / Initialize HMM Transition Matrix A and Prior Distribution π from state alignments
  6. Save Checkpoints:
     • 'conformer_speech_model.pt' (DNN Acoustic Model weights + Tokenizer)
     • 'persistent_hmm_decoder.pkl' (HMM Transition Matrix + Prior)
"""

import os
import sys
import io
import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train_pytorch_nepali import (
    TextTokenizer, MultilingualSpeechDataset, pad_collate_fn,
    extract_features_from_array, normalize_nepali_text
)
from conformer_speech_model import ConformerSpeechModel
from audio_augmentation import AudioAugmentor

MODEL_SAVE_PATH = "conformer_speech_model.pt"
HMM_SAVE_PATH = "persistent_hmm_decoder.pkl"


def fit_hmm_matrices(model, dataset, num_classes, device, sample_limit=200):
    """
    Fits the HMM transition matrix A and prior distribution π
    using emission state sequences predicted by the trained Conformer model.
    """
    print(f"\n--- Initializing HMM Transition & Prior Matrices for {num_classes} phonetic states ---")
    K = num_classes
    A_counts = np.ones((K, K), dtype=np.float64) * 1e-3  # Laplace smoothing
    pi_counts = np.ones(K, dtype=np.float64) * 1e-3

    model.eval()
    processed = 0

    with torch.no_grad():
        for i in range(min(len(dataset), sample_limit)):
            feat_tensor, _ = dataset[i]
            if feat_tensor.shape[0] == 0:
                continue
            x = feat_tensor.unsqueeze(0).to(device)  # (1, T, 39)
            logits = model(x)  # (1, T, K)
            states = torch.argmax(logits, dim=2)[0].cpu().numpy().tolist()

            if len(states) == 0:
                continue

            pi_counts[states[0]] += 1.0
            for t in range(len(states) - 1):
                A_counts[states[t], states[t + 1]] += 1.0

            processed += 1

    # Normalize prior and transition matrices
    pi_prior = pi_counts / np.sum(pi_counts)
    A_matrix = A_counts / np.sum(A_counts, axis=1, keepdims=True)

    # Boost self-loop diagonal for stability
    for k in range(K):
        A_matrix[k, k] += 0.5
    A_matrix = A_matrix / np.sum(A_matrix, axis=1, keepdims=True)

    # Save to persistent HMM decoder file
    print(f"Saving persistent HMM transition decoder matrix to '{HMM_SAVE_PATH}'...")
    with open(HMM_SAVE_PATH, "wb") as f:
        pickle.dump({
            "A": A_matrix,
            "pi": pi_prior,
            "count": processed
        }, f)
    print(f"HMM state matrices successfully fitted and saved (K={K} states).")
    return A_matrix, pi_prior


def train_hybrid_conformer(
    dataset_name="pujanpaudel/nepali_speech_to_text",
    train_split="train",
    val_split=None,
    epochs=10,
    batch_size=8,
    lr=5e-4,
    max_samples=500,
    d_model=128,
    num_blocks=4,
    n_heads=4,
    resume_ckpt=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name} (split: {train_split})")

    # Handle checkpoint resuming with strict vocabulary & dimension locking
    ck = None
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Loading pre-trained checkpoint for fine-tuning from: '{resume_ckpt}'...")
        # Create a safety backup copy of the existing best checkpoint
        backup_path = f"{os.path.splitext(resume_ckpt)[0]}_backup.pt"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copyfile(resume_ckpt, backup_path)
            print(f"Created safety backup: '{backup_path}'")

        try:
            ck = torch.load(resume_ckpt, map_location=device)
            if "tokenizer" in ck:
                tokenizer = TextTokenizer(char_map=ck["tokenizer"], freeze_vocab=True)
                print(f"Locked tokenizer vocabulary from checkpoint ({len(tokenizer.char_map)} classes).")
            else:
                tokenizer = TextTokenizer()
        except Exception as e:
            print(f"Warning: Could not read checkpoint tokenizer ({e}). Using fresh tokenizer.")
            tokenizer = TextTokenizer()
    else:
        tokenizer = TextTokenizer()

    print("Loading training dataset...")
    train_dataset = MultilingualSpeechDataset(
        nepali_source=dataset_name,
        split=train_split,
        tokenizer=tokenizer,
        max_samples=max_samples
    )

    if len(train_dataset) == 0:
        print(f"ERROR: No samples loaded from dataset '{dataset_name}'. Check dataset name and network connection.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    num_classes = len(tokenizer.char_map)
    print(f"Training samples: {len(train_dataset)} | Vocabulary size: {num_classes}")

    # Optional validation loader
    val_loader = None
    if val_split:
        print(f"Loading validation dataset (split: '{val_split}')...")
        val_dataset = MultilingualSpeechDataset(
            nepali_source=dataset_name,
            split=val_split,
            tokenizer=tokenizer,
            max_samples=min(max_samples // 5, 100)
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # Initialize Conformer Neural Acoustic Model
    model = ConformerSpeechModel(
        num_classes=num_classes,
        num_features=39,
        d_model=d_model,
        num_blocks=num_blocks,
        n_heads=n_heads
    ).to(device)

    # Load model weights strictly
    if ck is not None and "model_state" in ck:
        try:
            model.load_state_dict(ck["model_state"], strict=True)
            print("Successfully loaded pre-trained model weights with 100% layer match!")
        except Exception as e:
            print(f"Strict load warning: {e}. Attempting compatible load...")
            model.load_state_dict(ck["model_state"], strict=False)

    criterion = nn.CTCLoss(blank=1, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 65)
    print("      CONFORMER-HMM HYBRID MODEL TRAINING PIPELINE")
    print("=" * 65)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (features, targets, in_lens, tgt_lens) in enumerate(train_loader):
            # Apply SpecAugment during training
            aug_features = torch.stack([AudioAugmentor.spec_augment(f) for f in features]).to(device)
            targets = targets.to(device)

            # 1. Conformer Forward Pass
            logits = model(aug_features)
            log_probs = logits.log_softmax(2).transpose(0, 1)  # (T_sub, B, num_classes)
            out_lens = model.compute_output_lengths(in_lens)

            # 2. CTC Acoustic Loss
            loss = criterion(log_probs, targets, out_lens, tgt_lens)

            # 3. Backpropagation & Gradient Clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 4. Weight Update
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch:02d}/{epochs:02d}] | Batch [{batch_idx+1:03d}/{len(train_loader):03d}] | Conformer CTC Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / max(len(train_loader), 1)

        val_msg = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vf, vt, vi, vtg in val_loader:
                    vf = vf.to(device)
                    vt = vt.to(device)
                    v_logits = model(vf)
                    v_log_p = v_logits.log_softmax(2).transpose(0, 1)
                    v_out_lens = model.compute_output_lengths(vi)
                    val_loss += criterion(v_log_p, vt, v_out_lens, vtg).item()
            val_msg = f" | Val Loss: {val_loss / max(len(val_loader), 1):.4f}"

        print(f"--> Epoch [{epoch:02d}/{epochs:02d}] Completed | Avg Train Loss: {avg_train_loss:.4f}{val_msg}\n")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f"Saving best Conformer model checkpoint to '{MODEL_SAVE_PATH}'...")
            torch.save({
                "model_state": model.state_dict(),
                "tokenizer": tokenizer.char_map,
                "d_model": d_model,
                "num_blocks": num_blocks
            }, MODEL_SAVE_PATH)

    print("\n--- Conformer Acoustic Model Training Completed ---")

    # Fit HMM Transition and Prior Matrices
    fit_hmm_matrices(model, train_dataset, num_classes, device, sample_limit=min(len(train_dataset), 300))

    print("\n" + "=" * 65)
    print("CONFORMER-HMM HYBRID MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"• Acoustic Model saved: '{MODEL_SAVE_PATH}'")
    print(f"• HMM Decoder saved:    '{HMM_SAVE_PATH}'")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conformer-HMM Hybrid Speech Recognition Model")
    parser.add_argument("--dataset", type=str, default="pujanpaudel/nepali_speech_to_text",
                        help="HuggingFace dataset repository name (e.g. 'pujanpaudel/nepali_speech_to_text' or 'username/my_dataset')")
    parser.add_argument("--train_split", type=str, default="train", help="Dataset split for training (e.g. 'train')")
    parser.add_argument("--val_split", type=str, default=None, help="Dataset split for validation (e.g. 'validation')")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples to stream from dataset")
    parser.add_argument("--d_model", type=int, default=128, help="Conformer hidden dimension")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of Conformer blocks")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path to pre-trained checkpoint to continue fine-tuning (e.g. 'conformer_speech_model.pt')")
    args = parser.parse_args()

    train_hybrid_conformer(
        dataset_name=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        resume_ckpt=args.resume_ckpt
    )
