import os
import sys
import argparse
import io
import re
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

from audio_augmentation import AudioAugmentor

# Resolve stdout encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATASET_DIR = r"C:\Users\user\Downloads\asr_nepali_0\asr_nepali"
TSV_PATH = os.path.join(DATASET_DIR, "utt_spk_text.tsv")
DATA_DIR = os.path.join(DATASET_DIR, "data")
MODEL_SAVE_PATH = "nepali_speech_crnn.pt"

# --- 1. Devanagari Vocabulary & Tokenizer ---

BASE_NEPALI_CHARS = (
    # Vowels
    "अआइईउऊऋएऐओऔ"
    # Consonants
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
    # Matras (Dependent Vowel Signs)
    "ािीुूृेैोौ"
    # Diacritics / Modifiers
    "्ंँः़ऽ"
    # Nepali & Latin Digits
    "०१२३४५६७८९0123456789"
    # Punctuation & Symbols
    "।?!,-. "
    # Basic Latin alphabet (for code-mixed loan words)
    "abcdefghijklmnopqrstuvwxyz"
)

def normalize_nepali_text(text: str) -> str:
    """Normalizes Nepali Devanagari text: NFC normalization, cleaning punctuation, lowercase english."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip())
    # Convert english characters to lowercase
    text = text.lower()
    # Replace non-breaking spaces and redundant whitespaces
    text = re.sub(r"\s+", " ", text)
    # Remove unwanted special characters/emojis while retaining valid Devanagari & basic punctuation
    text = re.sub(r"[^\w\s" + re.escape("।?!,-.ािीुूृेैोौ्ंँः़ऽ") + "]", "", text)
    return text.strip()


class TextTokenizer:
    """
    Devanagari & Multilingual Character Tokenizer.
    Index 0: <pad>
    Index 1: <blank> (CTC blank)
    Index 2: ' ' (space)
    Index 3: <unk>
    """
    def __init__(self, char_map=None, freeze_vocab=False):
        self.freeze_vocab = freeze_vocab
        if char_map:
            self.char_map = dict(char_map)
            self.rev_map = {v: k for k, v in self.char_map.items()}
        else:
            self.char_map = {"<pad>": 0, "<blank>": 1, " ": 2, "<unk>": 3}
            self.rev_map = {0: "<pad>", 1: "<blank>", 2: " ", 3: "<unk>"}
            # Pre-populate base Nepali alphabets and matras
            self._init_base_vocab()

    def _init_base_vocab(self):
        idx = max(self.char_map.values()) + 1
        for ch in BASE_NEPALI_CHARS:
            if ch not in self.char_map:
                self.char_map[ch] = idx
                self.rev_map[idx] = ch
                idx += 1

    def build_vocab(self, texts):
        if self.freeze_vocab:
            return
        idx = max(self.char_map.values()) + 1
        for text in texts:
            norm_text = normalize_nepali_text(text)
            for ch in norm_text:
                if ch not in self.char_map:
                    self.char_map[ch] = idx
                    self.rev_map[idx] = ch
                    idx += 1

    def encode(self, text: str):
        norm_text = normalize_nepali_text(text)
        return [self.char_map.get(c, self.char_map.get("<unk>", 3)) for c in norm_text]

    def decode(self, indices):
        return "".join([self.rev_map.get(i, "") for i in indices if i not in (0, 1, 3)])


# --- 2. Acoustic Feature Extraction with CMVN ---

def apply_energy_vad(audio_arr: np.ndarray, sr: int = 16000, top_db: float = 38.0, pad_sec: float = 0.25) -> np.ndarray:
    """
    Applies gentle energy-based Voice Activity Detection (VAD) to trim dead silence
    while preserving all soft initial/final consonants with 250ms acoustic padding.
    """
    if audio_arr is None or len(audio_arr) < int(sr * 0.5):
        return audio_arr
    try:
        trimmed, index = librosa.effects.trim(audio_arr, top_db=top_db, frame_length=512, hop_length=128)
        start_sample = max(0, int(index[0] - pad_sec * sr))
        end_sample = min(len(audio_arr), int(index[1] + pad_sec * sr))
        if end_sample > start_sample + int(sr * 0.2):
            return audio_arr[start_sample:end_sample]
    except Exception:
        pass
    return audio_arr


def extract_features_from_array(audio_arr: np.ndarray, sr: int = 16000, n_mfcc: int = 13, max_duration_sec: float = 60.0, apply_vad: bool = True) -> np.ndarray:
    """
    Extracts 39-dimensional normalized MFCCs (13 MFCC + Delta + Delta-Delta)
    with Cepstral Mean & Variance Normalization (CMVN) and conservative VAD.
    """
    if sr != 16000:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio_arr = audio_arr.astype(np.float32)

    # Apply conservative VAD to trim dead silence on long recordings
    if apply_vad and len(audio_arr) > int(sr * 3.5):
        audio_arr = apply_energy_vad(audio_arr, sr=sr, top_db=38.0, pad_sec=0.25)

    # Support up to 60s speech duration
    if max_duration_sec is not None:
        max_samples = int(sr * max_duration_sec)
        if len(audio_arr) > max_samples:
            audio_arr = audio_arr[:max_samples]

    # Ensure non-empty
    if len(audio_arr) < 400:
        audio_arr = np.pad(audio_arr, (0, 400 - len(audio_arr)))

    mfcc = librosa.feature.mfcc(y=audio_arr, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2])  # (39, T)

    # Apply per-utterance CMVN
    mean = np.mean(feat, axis=1, keepdims=True)
    std = np.std(feat, axis=1, keepdims=True) + 1e-6
    feat_norm = (feat - mean) / std

    return feat_norm.astype(np.float32)


# --- 3. PyTorch Dataset ---

class MultilingualSpeechDataset(Dataset):
    def __init__(self, nepali_source="huggingface", split="train", hf_dataset_name=None,
                 nepali_tsv_path=None, nepali_data_dir=None,
                 english_dir=None, tokenizer=None, max_samples=500):
        self.data = []
        self.tokenizer = tokenizer or TextTokenizer()
        all_texts = []

        # Determine HuggingFace dataset name
        target_hf_dataset = None
        if hf_dataset_name:
            target_hf_dataset = hf_dataset_name
        elif nepali_source not in ("local", None):
            if nepali_source == "huggingface":
                target_hf_dataset = "pujanpaudel/nepali_speech_to_text"
            else:
                target_hf_dataset = nepali_source

        # 1. Load Dataset from HuggingFace
        if target_hf_dataset:
            print(f"Loading speech dataset from HuggingFace: '{target_hf_dataset}' (split='{split}')...")
            try:
                from datasets import load_dataset, Audio
                hf_token = os.environ.get("HF_TOKEN", None)
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

                # Download & cache all parquet shards to guarantee 100% (all 7,481 samples) load
                try:
                    print("Downloading full dataset shards to local cache for reliable 100% sample loading...")
                    hf_dataset = load_dataset(
                        target_hf_dataset,
                        split=split,
                        token=hf_token
                    )
                    if "audio" in hf_dataset.features:
                        hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
                    total_available = len(hf_dataset)
                    print(f"Full dataset downloaded and verified ({total_available} total records available).")
                except Exception as dl_err:
                    print(f"Notice: Direct download fallback to streaming mode ({dl_err}).")
                    hf_dataset = load_dataset(
                        target_hf_dataset,
                        split=split,
                        streaming=True,
                        token=hf_token
                    )
                    if "audio" in hf_dataset.features:
                        hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))

                nep_count = 0
                dataset_iter = iter(hf_dataset)
                retries = 0
                max_retries = 5
                while True:
                    try:
                        item = next(dataset_iter)
                        retries = 0  # Reset retry counter on successful item fetch
                    except StopIteration:
                        break
                    except Exception as stream_err:
                        retries += 1
                        if retries <= max_retries:
                            print(f"\n[Network Notice] Streaming hiccup ({stream_err}). Retrying in 2s (Attempt {retries}/{max_retries})...")
                            import time
                            time.sleep(2)
                            continue
                        elif nep_count > 0:
                            print(f"\n[Network Notice] Proceeding to training with {nep_count} loaded samples.")
                            break
                        else:
                            raise stream_err

                    # Auto-detect text field
                    text = ""
                    for col in ("transcription", "text", "sentence", "normalized_text", "transcript", "target_text", "label"):
                        if col in item and item[col]:
                            text = str(item[col])
                            break

                    text = normalize_nepali_text(text)
                    if not text:
                        continue

                    # Auto-detect audio field
                    audio_info = item.get("audio", {})
                    if isinstance(audio_info, dict):
                        audio_bytes = audio_info.get("bytes", None)
                        audio_arr = audio_info.get("array", None)
                        audio_path = audio_info.get("path", None)
                    else:
                        audio_bytes = None
                        audio_arr = None
                        audio_path = str(audio_info) if audio_info else None

                    if audio_bytes:
                        self.data.append((audio_bytes, text, "bytes"))
                    elif audio_arr is not None:
                        self.data.append((audio_arr, text, "array"))
                    elif audio_path and os.path.exists(audio_path):
                        self.data.append((audio_path, text, "file"))
                    else:
                        continue

                    all_texts.append(text)
                    nep_count += 1

                    if nep_count % 100 == 0:
                        print(f"  Loaded {nep_count} speech samples...")

                    if max_samples and nep_count >= max_samples:
                        break

                print(f"Successfully loaded {nep_count} speech samples from '{target_hf_dataset}' [{split}].")
            except Exception as e:
                print(f"WARNING: Failed to load HuggingFace dataset '{target_hf_dataset}': {e}")

        # 1b. Fallback: Load Nepali Dataset from local TSV folder
        elif nepali_tsv_path and os.path.exists(nepali_tsv_path):
            print(f"Indexing Nepali dataset from local TSV: {nepali_tsv_path}...")
            with open(nepali_tsv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            nep_count = 0
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                utt_id, spk_id, text = parts[0], parts[1], parts[2]
                audio_path = os.path.join(nepali_data_dir, utt_id[:2], f"{utt_id}.flac")
                if os.path.exists(audio_path):
                    self.data.append((audio_path, normalize_nepali_text(text), "file"))
                    all_texts.append(text)
                    nep_count += 1
                    if max_samples and nep_count >= max_samples:
                        break
            print(f"Loaded {nep_count} Nepali speech samples from local files.")

        # 2. Optional: Load English Dataset
        if english_dir in ("huggingface", "peoples_speech"):
            print("Loading English dataset from HuggingFace: MLCommons/peoples_speech (streaming)...")
            try:
                from datasets import load_dataset, Audio
                eng_ds = load_dataset(
                    "MLCommons/peoples_speech", "clean",
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                ).cast_column("audio", Audio(decode=False))

                eng_count = 0
                for item in eng_ds:
                    text = item.get("text", "").strip()
                    audio_bytes = item.get("audio", {}).get("bytes", None)
                    if not text or not audio_bytes:
                        continue
                    self.data.append((audio_bytes, text, "bytes"))
                    all_texts.append(text)
                    eng_count += 1
                    if max_samples and eng_count >= max_samples:
                        break
                print(f"Loaded {eng_count} English samples.")
            except Exception as e:
                print(f"WARNING: Failed to load English dataset: {e}")

        # Update vocabulary from all collected texts
        self.tokenizer.build_vocab(all_texts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        kind = entry[2]

        if kind == "bytes":
            audio_bytes, text, _ = entry
            try:
                audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
                feat = extract_features_from_array(audio_arr, sr=sr, n_mfcc=13)
            except Exception:
                feat = np.zeros((39, 100), dtype=np.float32)
        elif kind == "array":
            audio_arr, text, _ = entry
            feat = extract_features_from_array(audio_arr, sr=16000, n_mfcc=13)
        else:
            audio_path, text, _ = entry
            try:
                audio_arr, sr = sf.read(audio_path)
                feat = extract_features_from_array(audio_arr, sr=sr, n_mfcc=13)
            except Exception:
                feat = np.zeros((39, 100), dtype=np.float32)

        if feat is None or feat.shape[1] == 0:
            feat = np.zeros((39, 100), dtype=np.float32)

        target_seq = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        feature_tensor = torch.tensor(feat.T, dtype=torch.float32)  # (T, 39)
        return feature_tensor, target_seq


# Dynamic Collate Function for Padding
def pad_collate_fn(batch):
    features, targets = zip(*batch)
    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    # Pad features (time_steps, n_features) -> (batch, max_time, n_features)
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    # Concatenate targets for CTCLoss
    concatenated_targets = torch.cat(targets)

    return padded_features, concatenated_targets, input_lengths, target_lengths


# --- 4. Deep Learning Architecture (CRNN) ---

class NepaliSpeechCRNN(nn.Module):
    def __init__(self, num_classes, num_features=39, hidden_size=256, num_layers=3, dropout=0.2):
        super(NepaliSpeechCRNN, self).__init__()
        
        # 1D Convolutional feature extractor with residual/normalization
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        
        # Bidirectional Recurrent Network (Bi-GRU)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, time_steps, n_features)
        x = x.transpose(1, 2)  # (batch, n_features, time_steps)
        x = self.conv(x)       # (batch, 256, time_steps)
        x = x.transpose(1, 2)  # (batch, time_steps, 256)
        
        out, _ = self.rnn(x)   # (batch, time_steps, hidden_size * 2)
        logits = self.fc(out)  # (batch, time_steps, num_classes)
        return logits


# --- 5. Training Engine ---

def train_model(epochs=10, batch_size=8, lr=5e-4, max_samples=500, nepali_source="huggingface", english_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = TextTokenizer()
    dataset = MultilingualSpeechDataset(
        nepali_source=nepali_source,
        nepali_tsv_path=TSV_PATH if nepali_source == "local" else None,
        nepali_data_dir=DATA_DIR if nepali_source == "local" else None,
        english_dir=english_dir,
        tokenizer=tokenizer,
        max_samples=max_samples
    )

    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Please check your network connection or dataset path.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    num_classes = len(tokenizer.char_map)
    print(f"Total training samples: {len(dataset)}")
    print(f"Vocabulary size (including blank/pad): {num_classes}")

    # Model, CTC Loss, Optimizer
    model = NepaliSpeechCRNN(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=1, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n--- Starting Nepali CRNN Speech Model Training Pipeline ---")
    print("Dataset (pujanpaudel/nepali_speech_to_text) -> 39-MFCC CMVN -> SpecAugment -> CRNN -> CTC Loss -> Weight Update\n")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (features, targets, input_lengths, target_lengths) in enumerate(dataloader):
            # Apply SpecAugment Data Augmentation (Frequency + Time Masking)
            features = torch.stack([AudioAugmentor.spec_augment(f) for f in features]).to(device)
            targets = targets.to(device)

            # 1. Forward Pass (Prediction)
            logits = model(features)  # (batch, time_steps, num_classes)
            
            # Log probabilities for CTC Loss: (time_steps, batch, num_classes)
            log_probs = logits.log_softmax(2).transpose(0, 1)

            # 2. Compute CTC Loss Function
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # 3. Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 4. Update Weights
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch [{epoch:02d}/{epochs:02d}] | Batch [{batch_idx+1:03d}/{len(dataloader):03d}] | CTC Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"--> Epoch [{epoch:02d}/{epochs:02d}] Completed | Avg Train CTC Loss: {avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving best model checkpoint to '{MODEL_SAVE_PATH}'...")
            torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.char_map}, MODEL_SAVE_PATH)

    print(f"\nTraining completed! Final model saved to '{MODEL_SAVE_PATH}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Nepali Speech Recognition Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=500, help="Max Nepali dataset samples (e.g. 500, 1000, 5000)")
    parser.add_argument("--dataset", type=str, default="huggingface", choices=["huggingface", "local"],
                        help="Dataset source: 'huggingface' (pujanpaudel/nepali_speech_to_text) or 'local'")
    parser.add_argument("--english_dir", type=str, default=None,
                        help="Optional English audio dataset source")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        nepali_source=args.dataset,
        english_dir=args.english_dir
    )
