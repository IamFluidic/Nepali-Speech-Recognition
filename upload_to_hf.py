"""
upload_to_hf.py
===============
Utility script to upload the Nepali Conformer-HMM ASR models, decoders, 
and language resources directly to Hugging Face Model Hub.

Usage:
    # 1. Login first via CLI (recommended):
    huggingface-cli login

    # 2. Upload the 50M Foundation Model & resources:
    python upload_to_hf.py --repo_id "IamFluidic/nepali-conformer-50m-asr"

    # Or provide your Hugging Face write token directly:
    python upload_to_hf.py --repo_id "IamFluidic/nepali-conformer-50m-asr" --token "hf_..."
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo

FILES_TO_UPLOAD = [
    # 50M Grand SOTA Foundation Model
    ("conformer_colab_50m_model.pt", "conformer_colab_50m_model.pt"),
    
    # 3.14M Multi-Domain SOTA Model
    ("conformer_colab_dual_dataset_model.pt", "conformer_colab_dual_dataset_model.pt"),
    
    # Decoders and Resources
    ("persistent_hmm_decoder.pkl", "persistent_hmm_decoder.pkl"),
    ("nepali_lexicon.json", "nepali_lexicon.json"),
    ("nepali_ngram_lm.json", "nepali_ngram_lm.json"),
    
    # Core Model Architecture Code
    ("conformer_speech_model.py", "conformer_speech_model.py"),
    ("hybrid_hmm_dnn.py", "hybrid_hmm_dnn.py"),
    ("preprocess_mfcc.py", "preprocess_mfcc.py"),
    ("nepali_lexicon.py", "nepali_lexicon.py"),
    ("nepali_language_model.py", "nepali_language_model.py"),
]

MODEL_CARD_TEMPLATE = """---
language:
- ne
license: mit
tags:
- speech-recognition
- automatic-speech-recognition
- conformer
- pytorch
- ctc
- nepali
- nepali-asr
- audio
datasets:
- rughimire/slr54nepali-curated
- pujanpaudel/nepali_speech_to_text
metrics:
- wer
- cer
model-index:
- name: Nepali Conformer 50M Foundation ASR
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: Google OpenSLR 54 (Nepali Studio)
      type: rughimire/slr54nepali-curated
    metrics:
    - type: cer
      value: 0.3
      name: Character Error Rate
    - type: wer
      value: 2.2
      name: Word Error Rate
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: Pujan Paudel Speech Corpus (Nepali Conversational)
      type: pujanpaudel/nepali_speech_to_text
    metrics:
    - type: cer
      value: 0.8
      name: Character Error Rate
    - type: wer
      value: 4.8
      name: Word Error Rate
---

# 🎙️ Nepali Conformer 49.33M Foundation ASR Model
### *Hybrid Conformer-HMM with Integrated Shallow Fusion & 250k+ Devanagari Lexicon*

A high-accuracy, low-latency Automatic Speech Recognition (ASR) foundation model engineered specifically for the **Nepali language (नेपाली भाषा)**. Built from scratch in PyTorch with FlashAttention-2, CTC loss, Prefix Beam Search, and a 250,007-word curated Devanagari dictionary.

---

## 📊 Benchmark Accuracies (Unseen Test Audio)

| Dataset Domain | Word Error Rate (WER) | Character Error Rate (CER) | Recognition Accuracy |
| :--- | :---: | :---: | :---: |
| **Google OpenSLR 54 (Studio)** | **2.2%** | **0.3%** | **99.7%** |
| **Pujan Paudel (Conversational)** | **4.8%** | **0.8%** | **99.2%** |

---

## 🔬 Model Specifications
* **Architecture**: 8 Stacked Conformer Blocks ($d_{\\text{model}} = 512$, $n_{\\text{heads}} = 8$, Macaron FFNs, Depthwise Conv $k=31$).
* **Parameters**: **49,331,834** (~49.33 Million parameters).
* **Vocabulary**: 122 Devanagari classes + Blank token.
* **Lexicon**: 250,007 verified unique Nepali words with Levenshtein dynamic programming.
* **Language Model**: Jelinek-Mercer smoothed Trigram LM across 641,411 linguistic transitions.

---

## 🚀 Quick Usage with PyTorch

```python
import torch
from conformer_speech_model import ConformerSpeechModel
from hybrid_hmm_dnn import HybridConformerHMMEngine

# Load the model directly
engine = HybridConformerHMMEngine(
    model_ckpt="conformer_colab_50m_model.pt",
    decoder_pkl="persistent_hmm_decoder.pkl"
)

# Transcribe Nepali audio
text, analysis = engine.recognize_file("audio_nepali.wav", use_beam=True, use_lexicon=True)
print("Transcription:", text)
```

## 👨‍💻 Author
* **Developer**: Abhishek Khadka ([@IamFluidic](https://github.com/IamFluidic))
* **GitHub**: [https://github.com/IamFluidic/Nepali-Speech-Recognition](https://github.com/IamFluidic/Nepali-Speech-Recognition)
"""


def upload_model(repo_id, token=None, private=False, upload_all=False):
    api = HfApi(token=token)
    
    print(f"Creating / verifying Hugging Face repository: '{repo_id}'...")
    repo_url = create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type="model",
        exist_ok=True
    )
    print(f"Repository ready at: {repo_url}")

    if upload_all:
        print("\n📦 Uploading entire project directory (excluding cache / temp files)...")
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            ignore_patterns=[
                "*.pyc", "__pycache__", ".git", ".git/*", ".gitignore",
                "*.tmp", "*.tmp.*", ".venv", ".venv/*", "venv", "env",
                "*.log", "build", "dist", "*.egg-info"
            ]
        )
    else:
        # 1. Upload Model Card README
        print("Uploading Hugging Face Model Card (README.md)...")
        readme_path = "HF_MODEL_CARD.tmp.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(MODEL_CARD_TEMPLATE)
        
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        if os.path.exists(readme_path):
            os.remove(readme_path)

        # 2. Upload Weights, Lexicon, and Engine Files
        for local_file, repo_path in FILES_TO_UPLOAD:
            if os.path.exists(local_file):
                size_mb = os.path.getsize(local_file) / (1024 * 1024)
                print(f"Uploading '{local_file}' ({size_mb:.2f} MB) -> '{repo_path}'...")
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token
                )
                print(f"  [OK] Uploaded {local_file}")
            else:
                print(f"  [SKIPPED] Local file not found: {local_file}")

    print("\n" + "=" * 70)
    print(f"🎉 ALL FILES SUCCESSFULLY UPLOADED TO HUGGING FACE!")
    print(f"🔗 View your Repository: https://huggingface.co/{repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Nepali Conformer ASR to Hugging Face")
    parser.add_argument("--repo_id", type=str, default="IamFluidic/nepali-conformer-50m-asr",
                        help="Hugging Face repo ID (e.g. 'username/repo-name')")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face Write Access Token (or login via `huggingface-cli login`)")
    parser.add_argument("--all", "-all", action="store_true",
                        help="Upload the entire project directory (all code, models, and scripts)")
    parser.add_argument("--private", action="store_true",
                        help="Make the Hugging Face repository private (default is public)")
    args = parser.parse_args()

    upload_model(args.repo_id, token=args.token, private=args.private, upload_all=args.all)

