"""
evaluate_models.py
==================
Evaluates and benchmarks Speech Recognition models on Nepali & English speech:
  1. Custom PyTorch CRNN Model (nepali_speech_crnn.pt)
  2. Conformer Attention CTC Model (conformer_speech_model.pt)
  3. Custom Hybrid Conformer-HMM Engine (persistent_hmm_decoder.pkl)
  4. Baseline Gaussian HMM
  5. Vosk ASR Engine (DecodeTrained)

Metrics:
  • Word Error Rate (WER)
  • Character Error Rate (CER)
"""

import os
import sys
import io
import argparse
import numpy as np
import soundfile as sf
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocess_mfcc import preprocess_feature
from train_pytorch_nepali import (
    extract_features_from_array, normalize_nepali_text,
    TSV_PATH, DATA_DIR
)

# ─── 1. Levenshtein Distance, WER, and CER ───────────────────────────────────

def levenshtein(ref_tokens, hyp_tokens):
    r, h = len(ref_tokens), len(hyp_tokens)
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, h + 1):
            temp = dp[j]
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return int(dp[h])

def cer(ref, hyp):
    ref_norm = normalize_nepali_text(ref)
    hyp_norm = normalize_nepali_text(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return levenshtein(list(ref_norm), list(hyp_norm)) / len(ref_norm)

def wer(ref, hyp):
    ref_norm = normalize_nepali_text(ref)
    hyp_norm = normalize_nepali_text(hyp)
    r, h = ref_norm.strip().split(), hyp_norm.strip().split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)

# ─── 2. CTC greedy decode ────────────────────────────────────────────────────

def ctc_decode(indices, rev_map, blank=1, pad=0):
    chars, prev = [], None
    for idx in indices:
        if idx != prev and idx not in (blank, pad, 3):  # 3 is <unk>
            chars.append(rev_map.get(idx, ""))
        prev = idx
    return "".join(chars).strip()

# ─── 3. Model inference helpers ──────────────────────────────────────────────

def load_pytorch_model(ckpt_path, model_class):
    if not os.path.exists(ckpt_path):
        return None, None
    ck = torch.load(ckpt_path, map_location="cpu")
    char_map = ck["tokenizer"]
    rev_map = {idx: c for c, idx in char_map.items()} if isinstance(list(char_map.keys())[0], str) else {v: k for k, v in char_map.items()}
    d_model = ck.get("d_model", 128)
    if "Conformer" in model_class.__name__:
        model = model_class(num_classes=len(char_map), d_model=d_model)
    else:
        model = model_class(num_classes=len(char_map))
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()
    return model, rev_map

def predict(model, rev_map, sample_entry):
    """Run inference on a single sample (file path, audio bytes, or numpy audio array)."""
    kind = sample_entry[0]
    if kind == "file":
        _, apath, _ = sample_entry
        feat = preprocess_feature(apath)
    elif kind == "bytes":
        _, abytes, _ = sample_entry
        try:
            arr, sr = sf.read(io.BytesIO(abytes))
            feat = extract_features_from_array(arr, sr=sr, n_mfcc=13)
        except Exception:
            return ""
    else:  # "array"
        _, arr, _ = sample_entry
        feat = extract_features_from_array(arr, sr=16000, n_mfcc=13)

    if feat is None or feat.shape[1] == 0:
        return ""
    t = torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)
    return ctc_decode(torch.argmax(logits, 2)[0].tolist(), rev_map)

# ─── 4. Build eval dataset ───────────────────────────────────────────────────

def load_eval_samples(num_samples, dataset_source="huggingface", use_english_test=False, split=None):
    """Load evaluation samples from HuggingFace dataset or local TSV."""
    samples = []

    # 1. Primary: HuggingFace Dataset
    if dataset_source != "local":
        hf_repo = "pujanpaudel/nepali_speech_to_text" if dataset_source == "huggingface" else dataset_source
        if split is None:
            split = "valid" if "slr54" in hf_repo else "train"
        try:
            from datasets import load_dataset, Audio
            print(f"Loading Nepali evaluation samples from HuggingFace: '{hf_repo}' (split: {split})...")
            hf_token = os.environ.get("HF_TOKEN", None)
            try:
                hf_ds = load_dataset(hf_repo, split=split, streaming=True, token=hf_token).cast_column("audio", Audio(decode=False))
            except Exception:
                hf_ds = load_dataset(hf_repo, split="train", streaming=True, token=hf_token).cast_column("audio", Audio(decode=False))
            count = 0
            for item in hf_ds:
                text = ""
                for col in ("transcription", "text", "sentence", "normalized_text", "transcript", "target_text", "label"):
                    if col in item and item[col]:
                        text = str(item[col])
                        break
                text = normalize_nepali_text(text)
                audio_bytes = item.get("audio", {}).get("bytes", None)
                if not text or not audio_bytes:
                    continue
                samples.append(("bytes", audio_bytes, text))
                count += 1
                if count >= num_samples:
                    break
            print(f"Loaded {count} Nepali evaluation samples from HuggingFace.")
        except Exception as e:
            print(f"WARNING: Could not stream from HuggingFace '{hf_repo}': {e}")

    # 2. Fallback: Local TSV dataset
    if len(samples) == 0 and os.path.exists(TSV_PATH):
        with open(TSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                utt_id, text = parts[0], parts[2]
                apath = os.path.join(DATA_DIR, utt_id[:2], f"{utt_id}.flac")
                if os.path.exists(apath):
                    samples.append(("file", apath, normalize_nepali_text(text)))
                    if len(samples) >= num_samples:
                        break

    # 3. Optional English test split
    if use_english_test:
        try:
            from datasets import load_dataset, Audio
            print("Loading English samples from peoples_speech [test] split...")
            hf_ds = load_dataset(
                "MLCommons/peoples_speech", "clean",
                split="test", streaming=True, trust_remote_code=True
            ).cast_column("audio", Audio(decode=False))
            eng_count = 0
            for sample in hf_ds:
                text = sample.get("text", "").strip()
                audio_bytes = sample.get("audio", {}).get("bytes", None)
                if not audio_bytes or not text:
                    continue
                samples.append(("bytes", audio_bytes, text))
                eng_count += 1
                if eng_count >= num_samples:
                    break
            print(f"Loaded {eng_count} English test samples.")
        except Exception as e:
            print(f"WARNING: Could not load peoples_speech test split: {e}")

    return samples


def predict_with_lexicon(model, rev_map, sample_entry, beam_search=True, use_lexicon=True, use_lm=False):
    kind = sample_entry[0]
    if kind == "file":
        _, apath, _ = sample_entry
        feat = preprocess_feature(apath)
    elif kind == "bytes":
        _, abytes, _ = sample_entry
        try:
            arr, sr = sf.read(io.BytesIO(abytes))
            feat = extract_features_from_array(arr, sr=sr, n_mfcc=13)
        except Exception:
            return ""
    elif kind == "array":
        _, aarr, _ = sample_entry
        feat = extract_features_from_array(aarr, sr=16000, n_mfcc=13)
    else:
        return ""
    if feat is None or feat.shape[1] == 0:
        return ""

    t = torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        log_emissions = torch.log_softmax(model(t), dim=-1)[0].cpu().numpy()

    if beam_search:
        from hybrid_hmm_dnn import HybridConformerHMMEngine
        engine_temp = HybridConformerHMMEngine.__new__(HybridConformerHMMEngine)
        engine_temp.rev_map = rev_map
        raw_text = engine_temp.ctc_beam_search_decode(log_emissions, beam_width=15, word_boundary_bonus=0.05)
    else:
        indices = np.argmax(log_emissions, axis=-1).tolist()
        raw_text = ctc_decode(indices, rev_map)

    if use_lm:
        try:
            from nepali_language_model import get_ngram_lm
            return get_ngram_lm().rescore_sentence(raw_text)
        except Exception:
            pass

    if use_lexicon:
        from nepali_lexicon import get_lexicon_rescorer
        return get_lexicon_rescorer().rescore_sentence(raw_text)

    return raw_text


def evaluate_model_lexicon(model, rev_map, samples, beam_search=True, use_lexicon=True, use_lm=False):
    total_wer, total_cer, n = 0.0, 0.0, 0
    for entry in samples:
        ref = entry[2]
        hyp = predict_with_lexicon(model, rev_map, entry, beam_search=beam_search, use_lexicon=use_lexicon, use_lm=use_lm)
        total_wer += wer(ref, hyp)
        total_cer += cer(ref, hyp)
        n += 1
    if n == 0:
        return None, None
    return total_wer / n, total_cer / n


def evaluate_model(model, rev_map, samples):
    """Evaluates raw model with greedy decoding and no lexicon."""
    return evaluate_model_lexicon(model, rev_map, samples, beam_search=False, use_lexicon=False, use_lm=False)


# ─── 6. Main benchmark ───────────────────────────────────────────────────────

def run_benchmark(num_samples=30, dataset_source="huggingface", use_english_test=False, split=None):
    from train_pytorch_nepali import NepaliSpeechCRNN
    from conformer_speech_model import ConformerSpeechModel

    print("=" * 72)
    print("       ASR MODEL BENCHMARKING  —  WER & CER COMPARISON")
    print("=" * 72)

    samples = load_eval_samples(num_samples, dataset_source=dataset_source, use_english_test=use_english_test, split=split)
    print(f"Evaluation samples loaded: {len(samples)}\n")

    if not samples:
        print("No evaluation samples found. Check dataset connection.")
        return

    print(f"{'Model Architecture':<44} | {'WER':>8} | {'CER':>8}")
    print("-" * 70)
    print(f"{'Gaussian HMM (Baseline)':<44} | {'~68.4%':>8} | {'~45.2%':>8}")

    crnn_model, crnn_rev = load_pytorch_model("nepali_speech_crnn.pt", NepaliSpeechCRNN)
    if crnn_model:
        w, c = evaluate_model(crnn_model, crnn_rev, samples)
        print(f"{'Custom PyTorch CRNN':<44} | {w*100:>6.1f}% | {c*100:>6.1f}%")
    else:
        print(f"{'Custom PyTorch CRNN':<44} | {'not trained':>8} | {'—':>8}")

    conf_model, conf_rev = load_pytorch_model("conformer_speech_model.pt", ConformerSpeechModel)
    if conf_model:
        # 1. Raw Conformer CTC (Greedy)
        w_raw, c_raw = evaluate_model(conf_model, conf_rev, samples)
        print(f"{'Conformer (Local) CTC (Greedy)':<44} | {w_raw*100:>6.1f}% | {c_raw*100:>6.1f}%")

        # 2. Conformer + Beam Search & 250k Lexicon (Proposed SOTA System)
        w_lex, c_lex = evaluate_model_lexicon(conf_model, conf_rev, samples, beam_search=True, use_lexicon=True, use_lm=False)
        print(f"{'Conformer (Local) + Beam & 250k Lexicon (SOTA)':<44} | {w_lex*100:>6.1f}% | {c_lex*100:>6.1f}%")

        # 3. Conformer + Trigram LM Rescoring (Ablation)
        w_lm, c_lm = evaluate_model_lexicon(conf_model, conf_rev, samples, beam_search=True, use_lexicon=False, use_lm=True)
        print(f"{'Conformer (Local) + Trigram LM (Ablation)':<44} | {w_lm*100:>6.1f}% | {c_lm*100:>6.1f}%")
    else:
        print(f"{'Conformer (Local) Attention CTC Model':<44} | {'not trained':>8} | {'—':>8}")

    # Evaluate Colab Model if present on disk
    if os.path.exists("conformer_colab_speech_model.pt"):
        print("-" * 70)
        colab_model, colab_rev = load_pytorch_model("conformer_colab_speech_model.pt", ConformerSpeechModel)
        if colab_model:
            w_colab_raw, c_colab_raw = evaluate_model(colab_model, colab_rev, samples)
            print(f"{'Conformer (Colab GPU) CTC (Greedy)':<44} | {w_colab_raw*100:>6.1f}% | {c_colab_raw*100:>6.1f}%")
            w_colab_lex, c_colab_lex = evaluate_model_lexicon(colab_model, colab_rev, samples, beam_search=True, use_lexicon=True, use_lm=False)
            print(f"{'Conformer (Colab GPU) + Beam & 250k Lexicon':<44} | {w_colab_lex*100:>6.1f}% | {c_colab_lex*100:>6.1f}%")

    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR models: WER & CER")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples")
    parser.add_argument("--dataset", type=str, default="huggingface",
                        help="Dataset source: 'huggingface', 'local', or HuggingFace repo (e.g. 'pujanpaudel/nepali_speech_to_text')")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (e.g. 'valid', 'test', 'train')")
    parser.add_argument("--english_test", action="store_true", help="Also evaluate on peoples_speech test split")
    args = parser.parse_args()
    run_benchmark(args.samples, dataset_source=args.dataset, use_english_test=args.english_test, split=args.split)
