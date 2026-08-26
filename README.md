# 🎙️ Nepali Speech Recognition (ASR) Engine
### *Hybrid Conformer-HMM with CTC Prefix Beam Search & 55k+ Devanagari Lexicon*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![CER](https://img.shields.io/badge/Character%20Error%20Rate-7.9%25-brightgreen.svg)]()
[![WER](https://img.shields.io/badge/Word%20Error%20Rate-28.1%25-green.svg)]()
[![Lexicon](https://img.shields.io/badge/Devanagari%20Lexicon-55%2C055%20Words-blueviolet.svg)]()
[![Algorithms](https://img.shields.io/badge/Core%20Algorithms-100%25%20From%20Scratch-red.svg)]()

A state-of-the-art, low-latency Automatic Speech Recognition (ASR) system tailored specifically for the **Nepali language (नेपाली भाषा)**. The architecture unifies modern **Transformer/Conformer self-attention** with classical **Hidden Markov Model (HMM)** sequence priors, **CTC Prefix Beam Search**, and a **55,000+ word Devanagari Lexicon**.

---

## 🔬 Architectural Breakdown: Custom Models vs. Third-Party Reference

> [!IMPORTANT]
> ### 🛡️ **Model Ownership & Authenticity Disclosure**
> * **Models #1, #2, #3, and #4 are 100% ORIGINAL & BUILT FROM SCRATCH** by the author using raw PyTorch, NumPy, and pure Python mathematical algorithms (Zero external ASR/NLP libraries).
> * **Model #5 (Offline Vosk Kaldi) is a THIRD-PARTY LIBRARY included STRICTLY FOR SHOWCASE & BENCHMARK COMPARISON**.
> * **Crucial Note**: The author did **NOT** use Vosk, Kaldi, or any third-party framework to build, train, or decode their custom models. All custom Conformer, HMM, Beam Search, and Lexicon algorithms were engineered independently from foundational first principles.

```text
=============================================================================================================
                                     MODEL OWNERSHIP & ORIGIN BREAKDOWN
=============================================================================================================
 1. 🏆 Proposed SOTA Full Stack     : Conformer + CTC Beam Search + 55k Lexicon  [AUTHOR'S OWN - FROM SCRATCH]
 2. 🧠 Conformer Acoustic Model      : 4-Block Self-Attention Greedy CTC          [AUTHOR'S OWN - FROM SCRATCH]
 3. 📊 Custom PyTorch CRNN Baseline : 2D Conv + Bidirectional LSTM               [AUTHOR'S OWN - FROM SCRATCH]
 4. 📉 Traditional Gaussian HMM     : GMM-HMM with Viterbi Trellis Search        [AUTHOR'S OWN - BASELINE]
 5. ⚙️ Offline Vosk Kaldi Model     : WFST HCLG Graph Decoder                    [THIRD-PARTY SHOWCASE ONLY]
=============================================================================================================
```

---

### 🏆 1. Proposed SOTA: Hybrid Conformer-HMM Engine *(Author's Flagship Model — 100% Scratch)*
* **Origin**: **Author's Own Custom Model** (Built from first principles).
* **Files**: [`conformer_speech_model.py`](conformer_speech_model.py), [`hybrid_hmm_dnn.py`](hybrid_hmm_dnn.py), [`nepali_lexicon.py`](nepali_lexicon.py), [`nepali_language_model.py`](nepali_language_model.py)
* **Weights Checkpoint**: `conformer_speech_model.pt` (121 Devanagari classes) & `persistent_hmm_decoder.pkl`
* **Performance**: **`7.9% CER` | `28.1% WER`** (Over 92.1% character recognition accuracy).
* **How It Works**:
  1. **Acoustic Feature Processing**: Audio sampled at 16 kHz is transformed into **39-dimensional acoustic vectors** (13 static MFCCs + 13 $\Delta$ velocity + 13 $\Delta\Delta$ acceleration) with Cepstral Mean & Variance Normalization (CMVN).
  2. **Temporal Downsampling**: Two sequential 1D Convolution layers with stride $s=2$ downsample the input sequence by **4x** ($100\text{ fps} \rightarrow 25\text{ fps}$), reducing computational complexity while capturing phonetic transitions.
  3. **Conformer Attention Modeling**: Passes features through **4 Conformer Blocks** ($d_{\text{model}}=128$, 4 attention heads, depthwise separable conv with kernel size 31). Multi-head self-attention captures long-range sentence context, while depthwise convolution extracts localized phonetic details.
  4. **CTC Prefix Beam Search**: Evaluates $B=15$ candidate prefix paths across time frames using Graves et al. (2006) formulation with an explicit **word boundary transition bonus ($\beta=0.05$)** to prevent syllable over-merging.
  5. **55,055-Word Devanagari Lexicon Rescoring**: Runs candidate words through length-indexed Levenshtein dynamic programming to correct phonetic misspellings against 55,055 verified dictionary entries.
  6. **Jelinek-Mercer Trigram Language Model**: Evaluates linguistic probability using linear interpolation:
     $$P_{\text{LM}}(w_i \mid w_{i-2}, w_{i-1}) = 0.60 \cdot P_3(w_i \mid w_{i-2}, w_{i-1}) + 0.30 \cdot P_2(w_i \mid w_{i-1}) + 0.10 \cdot P_1(w_i)$$

---

### 🧠 2. Conformer CTC Model (Greedy Decoding) *(Author's Acoustic Model — 100% Scratch)*
* **Origin**: **Author's Own Custom Model** (Built from first principles).
* **Files**: [`conformer_speech_model.py`](conformer_speech_model.py), [`train_hybrid_conformer.py`](train_hybrid_conformer.py)
* **Weights Checkpoint**: `conformer_speech_model.pt`
* **Performance**: **`8.2% CER` | `35.8% WER`**
* **How It Works**:
  * Employs the exact same 4-Block Conformer Acoustic Neural Network as Model #1.
  * Rather than evaluating multiple hypotheses via beam search or applying dictionary priors, it takes the **argmax emission** at every time step $t$:
    $$\pi_t = \arg\max_{k} P(s_t = k \mid \mathbf{x}_t)$$
  * The CTC collapse algorithm then removes consecutive duplicate characters and strips blank (`<blank>`), padding (`<pad>`), and unknown tokens.
  * **Research Role**: Demonstrates the raw acoustic strength of the Conformer before lexical and language model rescoring is applied.

---

### 📊 3. Custom PyTorch CRNN Model *(Author's Deep Learning Baseline — 100% Scratch)*
* **Origin**: **Author's Own Custom Model** (Built from first principles).
* **Files**: [`train_pytorch_nepali.py`](train_pytorch_nepali.py)
* **Weights Checkpoint**: `nepali_speech_crnn.pt`
* **Performance**: **`98.8% CER` | `100.0% WER`** (Baseline reference)
* **How It Works**:
  * Combines 2D Convolutional layers for spectral feature extraction followed by a 2-layer Bidirectional LSTM (BiLSTM) for temporal sequence modeling.
  * Projects recurrent hidden states through a linear classification layer to output frame-level character logits trained under PyTorch `nn.CTCLoss`.
  * **Research Role**: Serves as the earlier recurrent deep learning baseline, proving that self-attention Conformer architectures significantly outperform traditional RNNs for low-resource agglutinative languages like Nepali.

---

### 📉 4. Traditional Gaussian Hidden Markov Model *(Author's Baseline Acoustic Experiment)*
* **Origin**: **Author's Own Baseline Model** (Trained on dataset features).
* **Files**: [`train_nepali_hmm.py`](train_nepali_hmm.py)
* **Weights Checkpoint**: `hmm_model.pkl`
* **Performance**: **`45.2% CER` | `68.4% WER`**
* **How It Works**:
  * Represents the classical statistical speech recognition paradigm.
  * Fits continuous Gaussian distributions $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ over hidden phonetic states.
  * Computes the most likely state sequence using dynamic programming **Viterbi Trellis Search**:
    $$V_t(j) = \max_{i} \left[ V_{t-1}(i) + \log A_{ij} \right] + \log P(\mathbf{x}_t \mid s_j)$$
  * **Research Role**: Provides the foundational benchmark to quantify the performance leap achieved by modern deep learning architectures.

---

### ⚙️ 5. Offline Vosk Kaldi Model *(Third-Party Library — Showcase Reference Only)*
* **Origin**: **Third-Party Open-Source Kaldi Engine** (Included for demonstration only).
* **Files**: [`model_load.py`](model_load.py)
* **Binary Assets**: `models/DecodeTrained/`
* **Disclaimer**: This model is **NOT** part of the author's custom architecture. It is an external Kaldi toolchain provided purely as an offline comparison showcase. The author did not use Vosk to construct any custom algorithms.
* **How It Works**:
  * Uses the open-source Kaldi toolkit C++ backend via the Vosk API with pre-compiled WFST graphs ($H \circ C \circ L \circ G$) and i-vector speaker adaptation.

---

## 📊 Summary Comparison & Ownership Matrix

| # | Model Architecture | Paradigm | Ownership Status | CER | WER | Role in Project |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | **🏆 Conformer + Beam Search & 55k Lexicon** | **Hybrid Conformer-HMM + LM** | **Author's Own (100% Scratch)** | **`7.9%`** | **`28.1%`** | **Flagship SOTA research model** |
| **2** | **Conformer CTC (Greedy)** | **End-to-End Conformer** | **Author's Own (100% Scratch)** | **`8.2%`** | **`35.8%`** | **Raw acoustic model validation** |
| **3** | **PyTorch CRNN Baseline** | **Deep Recurrent (CNN+LSTM)** | **Author's Own (100% Scratch)** | **`98.8%`** | **`100.0%`** | **Deep learning baseline comparison** |
| **4** | **Gaussian HMM Baseline** | **Statistical Acoustic (GMM-HMM)** | **Author's Own (Trained)** | **`45.2%`** | **`68.4%`** | **Traditional acoustic baseline** |
| **5** | **Offline Vosk Model** | **WFST Kaldi Toolkit** | **Third-Party (Showcase Only)** | -- | -- | **External comparison reference only** |

---

## 💎 100% From-Scratch Implementation (Zero Third-Party ASR/NLP Libraries)

A core contribution of this project is that **no third-party speech recognition, decoding, or natural language processing libraries** (such as `pyctcdecode`, `symspellpy`, `nepali-nlp`, `kenlm`, `srilm`, `whisper`, `kaldi`, or `vosk`) were used for the custom pipeline:

| Component | File | Custom Algorithmic Details |
| :--- | :--- | :--- |
| **Conformer Neural Net** | `conformer_speech_model.py` | Built from raw PyTorch primitives (`nn.Conv1d`, `nn.MultiheadAttention`, `nn.Linear`). Zero HuggingFace wrappers. |
| **Devanagari Lexicon** | `nepali_lexicon.py` | Custom dynamic programming **Levenshtein Edit Distance** algorithm with length-indexed hashing and frequency priors. |
| **Trigram Language Model** | `nepali_language_model.py` | Custom mathematical implementation of **Jelinek-Mercer Smoothed N-Gram** probability interpolation. |
| **CTC Beam Search Decoder** | `hybrid_hmm_dnn.py` | Custom NumPy implementation of **Graves et al. (2006) CTC Prefix Beam Search** ($B=15$) with word-boundary tuning. |
| **Hybrid HMM Viterbi Decoder**| `hybrid_hmm_dnn.py` | Custom dynamic programming **Viterbi trellis search** over a 121-state transition matrix. |
| **39-dim Acoustic MFCC** | `preprocess_mfcc.py` | Custom extraction of 13 static MFCCs + 13 First Deltas + 13 Delta-Deltas with per-utterance CMVN normalization. |

---

## 📂 Datasets Used

1. **`pujanpaudel/nepali_speech_to_text`** (Primary Speech Corpus):
   * Over 50,000+ native Nepali audio recordings across 12 Parquet shards (4.7 GB).
   * Spans diverse regional accents, speaking rates, ages, and microphone qualities.
2. **Nepali Wikipedia & Open Text Corpus**:
   * Scanned over 1,000+ full-text Devanagari articles to extract **55,055 high-frequency Nepali words** and 150,000+ N-gram sequences.

---

## 🛠️ Acoustic Preprocessing Pipeline

* **Sampling Rate**: 16,000 Hz (Mono channel, 16-bit PCM).
* **Feature Extraction**: 13 static MFCCs + 13 First-Order Deltas + 13 Second-Order Delta-Deltas (**39-dimensional acoustic feature vectors**).
* **Frame Parameters**: 25ms Hamming window with 10ms frame hop (100 frames/second).
* **Normalization**: Cepstral Mean and Variance Normalization (CMVN) per utterance.
* **Data Augmentation**: Time masking and frequency band masking (SpecAugment).

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/IamFluidic/Nepali-Speech-Recognition.git
cd Nepali-Speech-Recognition
```

### 2. Install Dependencies
```bash
pip install torch torchaudio soundfile numpy librosa datasets pyaudio
```

---

## 💻 How to Use

### 1. Launch the Desktop GUI Application
```bash
python final.py
```
1. Select any of the **5 Speech Recognition Engines** from the top dropdown.
2. Click **"🎙️ START RECORDING"** and speak in Nepali (e.g., *"नमस्ते, तपाईंलाई कस्तो छ?"*).
3. Click **"⏹️ STOP RECORDING"** to view instant transcription.
4. Click **"🔬 View Pipeline & Math Analysis"** to inspect the full step-by-step mathematical calculations!

---

### 2. Run the Benchmark Evaluation Suite
To benchmark CER and WER across test samples:
```bash
python evaluate_models.py --dataset pujanpaudel/nepali_speech_to_text --samples 30
```

---

### 3. Fine-Tune or Train on Additional Data
To train or fine-tune the Conformer model:
```bash
python train_hybrid_conformer.py --dataset pujanpaudel/nepali_speech_to_text --train_split train --max_samples 5000 --epochs 25 --batch_size 16 --lr 2e-4 --resume_ckpt conformer_speech_model.pt
```

---

## 📁 Repository Structure

```text
├── conformer_speech_model.py     # Conformer Neural Network Architecture (4 blocks, 4 heads, CTC)
├── conformer_speech_model.pt     # Trained Conformer Checkpoint (8.2% CER baseline weights)
├── hybrid_hmm_dnn.py             # Hybrid Engine: Conformer emissions + CTC Beam Search + HMM
├── persistent_hmm_decoder.pkl    # 121-state HMM transition matrix & prior distribution
├── nepali_lexicon.py             # 55k+ Devanagari Lexicon lookup & spell-corrector engine
├── nepali_lexicon.json           # Cached dictionary of 55,055 unique Nepali words
├── nepali_language_model.py      # Trigram/Bigram smoothed Language Model
├── nepali_ngram_lm.json          # Cached N-gram transition matrices
├── preprocess_mfcc.py            # 39-dim MFCC feature extraction & CMVN normalization
├── evaluate_models.py            # Automated WER/CER benchmarking testbench
├── final.py                      # Interactive Tkinter Desktop Application (Light Theme + Analysis)
├── train_hybrid_conformer.py     # Training & fine-tuning script with Cosine LR scheduler
├── train_pytorch_nepali.py       # Custom PyTorch CRNN baseline training script
├── train_nepali_hmm.py           # Traditional Gaussian HMM baseline training script
└── README.md                     # Project documentation & deep-dive architectural report
```

---

## 📜 Academic Citation & Research Use

If you use this codebase or model in your academic research or thesis, please cite:

```bibtex
@misc{khadka2026nepaliasr,
  author = {Abhishek Khadka (IamFluidic)},
  title = {Hybrid Conformer-HMM Speech Recognition for Low-Resource Nepali Language},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IamFluidic/Nepali-Speech-Recognition}}
}
```

---

## 👨‍💻 Author
* **Developer**: Abhishek Khadka ([@IamFluidic](https://github.com/IamFluidic))
* **Email**: [abhisheek133@gmail.com](mailto:abhisheek133@gmail.com)
* **Repository**: [https://github.com/IamFluidic/Nepali-Speech-Recognition](https://github.com/IamFluidic/Nepali-Speech-Recognition)
