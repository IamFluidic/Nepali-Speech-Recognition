"""
hybrid_hmm_dnn.py
=================
Custom Hybrid Conformer-HMM Speech Recognition Engine.

Combines:
  1. Conformer Attention DNN → Acoustic State Posteriors P(s_t | x_t)
  2. HMM Transition Matrix A_{ij} & Prior Distribution π_i
  3. Dynamic Programming Viterbi Decoder → Optimal State Path argmax P(Q, X | λ)
  4. Online Baum-Welch Parameter Accumulation → Saves to persistent_hmm_decoder.pkl
"""

import os
import sys
import pickle
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conformer_speech_model import ConformerSpeechModel
from train_pytorch_nepali import NepaliSpeechCRNN
from preprocess_mfcc import preprocess_feature

HMM_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persistent_hmm_decoder.pkl")

class HybridConformerHMMEngine:
    """
    Custom Hybrid Conformer-HMM Speech Recognizer.
    Maintains a persistent HMM transition matrix A and prior distribution π
    that adapts incrementally over time with every audio recording.
    """
    def __init__(self, model_ckpt="conformer_speech_model.pt", fallback_ckpt="nepali_speech_crnn.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dnn_model = None
        self.tokenizer = None
        self.rev_map = {}
        self.num_classes = 0

        # Load DNN Acoustic Model — Hybrid engine prefers Conformer, then CRNN fallback
        ckpt_dir = os.path.dirname(os.path.abspath(__file__))
        path_conformer = os.path.join(ckpt_dir, model_ckpt)
        path_crnn = os.path.join(ckpt_dir, fallback_ckpt)

        ckpt_to_load = None
        if os.path.exists(path_conformer):
            ckpt_to_load = path_conformer
        elif os.path.exists(path_crnn):
            ckpt_to_load = path_crnn

        if ckpt_to_load and os.path.exists(ckpt_to_load):
            print(f"Loading acoustic DNN model from checkpoint: {ckpt_to_load}")
            ck = torch.load(ckpt_to_load, map_location=self.device)
            self.tokenizer = ck["tokenizer"]
            self.rev_map = {v: k for k, v in self.tokenizer.items()}
            self.num_classes = len(self.tokenizer)

            d_model = ck.get("d_model", 128)
            model_cls = ConformerSpeechModel if "conformer" in ckpt_to_load else NepaliSpeechCRNN
            if "conformer" in ckpt_to_load:
                self.dnn_model = model_cls(num_classes=self.num_classes, d_model=d_model).to(self.device)
            else:
                self.dnn_model = model_cls(num_classes=self.num_classes).to(self.device)
            self.dnn_model.load_state_dict(ck["model_state"], strict=False)
            self.dnn_model.eval()
        else:
            print(f"Warning: No trained acoustic model checkpoint found at {path_conformer} or {path_crnn}")

        # Initialize or Load Persistent HMM Parameters (A_matrix, pi_prior, adaptation_count)
        self.init_or_load_hmm_parameters()

    def reset_hmm_parameters(self):
        """Resets persistent HMM transition & prior parameters to balanced state."""
        K = self.num_classes if self.num_classes > 0 else 100
        print(f"Resetting HMM transition & prior parameters for K={K} states...")
        self.A_matrix = np.full((K, K), 0.1 / (K - 1) if K > 1 else 1.0)
        np.fill_diagonal(self.A_matrix, 0.8)
        self.pi_prior = np.full(K, 0.2 / (K - 1) if K > 1 else 1.0)
        if K > 1:
            self.pi_prior[1] = 0.8  # blank token state default
        self.pi_prior /= self.pi_prior.sum()
        self.A_matrix /= self.A_matrix.sum(axis=1, keepdims=True)
        self.adaptation_count = 1
        self.save_hmm_parameters()

    def init_or_load_hmm_parameters(self):
        """Loads persistent HMM transition matrix A and prior π from disk or initializes them."""
        if os.path.exists(HMM_SAVE_PATH):
            try:
                print(f"Loading persistent HMM decoder parameters from: {HMM_SAVE_PATH}")
                with open(HMM_SAVE_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.A_matrix = data["A"]
                    self.pi_prior = data["pi"]
                    self.adaptation_count = data.get("count", 1)

                # Check if loaded parameters are corrupted (e.g., non-blank prior > 0.3 or dimension mismatch)
                K = self.num_classes if self.num_classes > 0 else 100
                if self.A_matrix.shape != (K, K) or len(self.pi_prior) != K:
                    print("Dimension mismatch in HMM state parameters. Auto-resetting HMM parameters.")
                    self.reset_hmm_parameters()
                else:
                    # Check for runaway prior bias (non-blank index != 1 has prior > 0.30)
                    corrupted = False
                    for idx in range(K):
                        if idx != 1 and self.pi_prior[idx] > 0.30:
                            corrupted = True
                            break
                    if corrupted:
                        print("Detected runaway single-state prior bias in HMM state file. Auto-resetting HMM parameters.")
                        self.reset_hmm_parameters()
            except Exception as e:
                print(f"Error loading HMM state file ({e}). Resetting HMM parameters.")
                self.reset_hmm_parameters()
        else:
            self.reset_hmm_parameters()

    def save_hmm_parameters(self):
        """Persists updated HMM parameters back to disk."""
        with open(HMM_SAVE_PATH, "wb") as f:
            pickle.dump({
                "A": self.A_matrix,
                "pi": self.pi_prior,
                "count": self.adaptation_count
            }, f)

    def viterbi_decode(self, log_emissions, acoustic_scale=6.0):
        """
        Viterbi Decoding Algorithm for Hybrid HMM-DNN.
        log_emissions: (time_steps, num_classes) numpy array of acoustic log-likelihoods.
        acoustic_scale: Multiplier for acoustic emission likelihoods vs HMM transitions (default 6.0).
        Computes Viterbi lattice with numerical stabilization.
        """
        scaled_emissions = log_emissions * acoustic_scale
        T, K = scaled_emissions.shape
        log_A = np.log(np.maximum(self.A_matrix, 1e-12))
        log_pi = np.log(np.maximum(self.pi_prior, 1e-12))

        viterbi = np.zeros((T, K))
        backpointer = np.zeros((T, K), dtype=int)

        # 1. Initialization (t = 0)
        viterbi[0] = log_pi + scaled_emissions[0]
        backpointer[0] = 0

        # 2. Recursion (t = 1 ... T-1)
        for t in range(1, T):
            for j in range(K):
                prob = viterbi[t - 1] + log_A[:, j]
                best_prev = np.argmax(prob)
                viterbi[t, j] = prob[best_prev] + scaled_emissions[t, j]
                backpointer[t, j] = best_prev

        # 3. Termination & Backtracking optimal state path
        best_last_state = np.argmax(viterbi[T - 1])
        best_path = [best_last_state]

        for t in range(T - 1, 0, -1):
            best_last_state = backpointer[t, best_last_state]
            best_path.insert(0, best_last_state)

        return best_path

    def update_hmm_online(self, state_path, learning_rate=0.005):
        """
        Online Baum-Welch Parameter Accumulation with Laplace smoothing & prior clipping.
        Prevents single-state runaway bias.
        """
        # Sanity check: do not update HMM if state_path is degenerate (all frames same state or <= 2 distinct non-blank states in long sequence)
        non_blank_states = [s for s in state_path if s not in (0, 1)]
        if len(set(non_blank_states)) <= 1:
            return

        K = self.A_matrix.shape[0]
        obs_A = np.zeros((K, K))

        for t in range(len(state_path) - 1):
            i, j = state_path[t], state_path[t + 1]
            obs_A[i, j] += 1.0

        row_sums = obs_A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        obs_A_norm = obs_A / row_sums

        # Exponential moving average with Laplace floor smoothing
        self.A_matrix = (1 - learning_rate) * self.A_matrix + learning_rate * obs_A_norm + 1e-4
        self.A_matrix = self.A_matrix / self.A_matrix.sum(axis=1, keepdims=True)

        first_state = state_path[0]
        self.pi_prior[first_state] += learning_rate
        # Cap non-blank priors (index != 1) at 0.15 max to avoid initial state lock-in
        for idx in range(K):
            if idx != 1 and self.pi_prior[idx] > 0.15:
                self.pi_prior[idx] = 0.15

        self.pi_prior /= self.pi_prior.sum()

        self.adaptation_count += 1
        self.save_hmm_parameters()

    def check_audio_energy(self, audio_path, energy_threshold=0.002):
        """Calculates normalized RMS energy and peak amplitude to filter out silence while supporting long speech."""
        try:
            import scipy.io.wavfile as wavfile
            sr, samples = wavfile.read(audio_path)
            if len(samples) == 0:
                return False
            # Normalize int16 audio to float range [-1, 1]
            if samples.dtype == np.int16:
                float_samples = samples.astype(np.float32) / 32768.0
            else:
                float_samples = samples.astype(np.float32)
            rms = np.sqrt(np.mean(float_samples ** 2))
            peak = np.max(np.abs(float_samples))
            return rms >= energy_threshold or peak >= 0.015
        except Exception as e:
            print(f"VAD energy check error: {e}")
            return True  # Fallback: proceed with inference if file reading fails

    def transcribe(self, audio_path):
        """Transcribes raw audio using the Hybrid Conformer-HMM Viterbi Decoder."""
        if self.dnn_model is None:
            return "Error: No DNN model loaded"

        # Energy-based Voice Activity Detection (VAD)
        if not self.check_audio_energy(audio_path, energy_threshold=0.002):
            return "No Speech Detected"

        feat = preprocess_feature(audio_path)
        if feat is None or feat.shape[1] == 0:
            return "No Speech Detected"

        feat_tensor = torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 1. Compute acoustic emission log-likelihoods / posteriors from DNN
        with torch.no_grad():
            if hasattr(self.dnn_model, "hmm_emission_log_likes"):
                log_emissions = self.dnn_model.hmm_emission_log_likes(feat_tensor)[0].cpu().numpy()
            else:
                log_post = torch.log_softmax(self.dnn_model(feat_tensor), dim=-1)[0].cpu().numpy()
                log_emissions = log_post - math.log(1.0 / max(1, self.num_classes))

    def ctc_beam_search_decode(self, log_emissions: np.ndarray, beam_width: int = 20, blank: int = 1, pad: int = 0, word_boundary_bonus: float = 0.05) -> str:
        """
        Tuned CTC Prefix Beam Search Decoder with Word Boundary Prior.
        Evaluates top candidate prefix paths across time frames with space boundary tuning.
        """
        T, K = log_emissions.shape
        space_indices = {k for k, v in self.rev_map.items() if v == " "}

        # Initialize beams: {prefix_tuple: (p_blank, p_non_blank)}
        beams = {(): (0.0, float("-inf"))}

        for t in range(T):
            next_beams = {}
            top_classes = np.argsort(log_emissions[t])[-min(K, 30):]

            for prefix, (p_b, p_nb) in beams.items():
                p_tot = np.logaddexp(p_b, p_nb)

                for c in top_classes:
                    log_p = log_emissions[t, c]

                    # Apply word boundary reward on valid space insertions
                    if c in space_indices and len(prefix) > 0 and prefix[-1] not in space_indices:
                        log_p += word_boundary_bonus

                    if c in (blank, pad):
                        curr_b, curr_nb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        next_beams[prefix] = (np.logaddexp(curr_b, p_tot + log_p), curr_nb)
                    else:
                        last_char = prefix[-1] if len(prefix) > 0 else None
                        new_prefix = prefix + (c,)

                        if c == last_char:
                            curr_b, curr_nb = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                            next_beams[new_prefix] = (curr_b, np.logaddexp(curr_nb, p_b + log_p))

                            curr_b, curr_nb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                            next_beams[prefix] = (curr_b, np.logaddexp(curr_nb, p_nb + log_p))
                        else:
                            curr_b, curr_nb = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                            next_beams[new_prefix] = (curr_b, np.logaddexp(curr_nb, p_tot + log_p))

            # Prune next_beams to top beam_width
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda item: np.logaddexp(item[1][0], item[1][1]),
                reverse=True
            )
            beams = dict(sorted_beams[:beam_width])

        # Pick highest scoring prefix
        best_prefix = max(beams.keys(), key=lambda p: np.logaddexp(beams[p][0], beams[p][1]))
        decoded_chars = [self.rev_map.get(idx, "") for idx in best_prefix if idx not in (blank, pad, 3)]
        return "".join(decoded_chars).strip()

    def transcribe(self, audio_path: str, use_beam_search: bool = True, use_lexicon: bool = True, adapt_online: bool = False) -> str:
        """
        Transcribes raw audio using Conformer Acoustic Posteriors, Beam Search,
        Word Boundary Tuning, and N-gram Language Model Rescoring.
        """
        if self.dnn_model is None:
            return "Error: No DNN model loaded"

        # Energy-based Voice Activity Detection (VAD)
        if not self.check_audio_energy(audio_path, energy_threshold=0.002):
            return "No Speech Detected"

        feat = preprocess_feature(audio_path)
        if feat is None or feat.shape[1] == 0:
            return "No Speech Detected"

        feat_tensor = torch.tensor(feat.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 1. Compute acoustic emission log-likelihoods / posteriors from DNN
        with torch.no_grad():
            if hasattr(self.dnn_model, "hmm_emission_log_likes"):
                log_emissions = self.dnn_model.hmm_emission_log_likes(feat_tensor)[0].cpu().numpy()
            else:
                log_post = torch.log_softmax(self.dnn_model(feat_tensor), dim=-1)[0].cpu().numpy()
                log_emissions = log_post - math.log(1.0 / max(1, self.num_classes))

        # 2. Sequence Decoding (Beam Search with Word Boundary Tuning or Viterbi)
        if use_beam_search:
            raw_text = self.ctc_beam_search_decode(log_emissions, beam_width=15, word_boundary_bonus=0.45)
        else:
            state_path = self.viterbi_decode(log_emissions, acoustic_scale=6.0)
            non_blank_viterbi = [s for s in state_path if s not in (0, 1)]
            if len(set(non_blank_viterbi)) <= 1:
                state_path = np.argmax(log_emissions, axis=-1).tolist()

            decoded_chars = []
            prev = None
            for s in state_path:
                if s != prev and s not in (0, 1):
                    decoded_chars.append(self.rev_map.get(s, ""))
                prev = s
            raw_text = "".join(decoded_chars).strip()

        if not raw_text:
            return "No Speech Detected"

        # 3. Apply Nepali N-Gram Language Model & Lexicon Context Rescoring
        if use_lexicon:
            try:
                from nepali_language_model import get_ngram_lm
                lm = get_ngram_lm()
                final_text = lm.rescore_sentence(raw_text)
            except Exception as e:
                try:
                    from nepali_lexicon import get_lexicon_rescorer
                    rescorer = get_lexicon_rescorer()
                    final_text = rescorer.rescore_sentence(raw_text)
                except Exception:
                    final_text = raw_text
        else:
            final_text = raw_text

        # 4. Accumulate online learning into HMM transition matrix (only if explicitly enabled)
        if adapt_online:
            greedy_path = np.argmax(log_emissions, axis=-1).tolist()
            self.update_hmm_online(greedy_path)

        return final_text if final_text else "No Speech Detected"


if __name__ == "__main__":
    engine = HybridConformerHMMEngine()
    print("Hybrid Conformer-HMM Engine initialized successfully with Beam Search, Word Penalty & Trigram LM!")
    print(f"HMM state dimension: {engine.A_matrix.shape}")
    print(f"Total online adaptation steps accumulated: {engine.adaptation_count}")

