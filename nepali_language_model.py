"""
nepali_language_model.py
========================
Nepali N-Gram (Unigram, Bigram, Trigram) Language Model with Jelinek-Mercer Smoothing.
Provides grammatical sequence scoring for Speech Recognition decoding.
"""

import os
import sys
import json
import math
import unicodedata
import re
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LM_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nepali_ngram_lm.json")

def normalize_nepali_word(w: str) -> str:
    if not w:
        return ""
    w = unicodedata.normalize("NFC", str(w).strip())
    w = re.sub(r"[^\u0900-\u097F\u0966-\u096F]", "", w)
    return w.strip()

class NepaliNGramLM:
    """
    Nepali Trigram + Bigram + Unigram Language Model.
    """
    def __init__(self, lm_path: str = LM_SAVE_PATH):
        self.lm_path = lm_path
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.total_unigrams = 0
        self.vocab_size = 0
        self.load_or_build()

    def load_or_build(self):
        if os.path.exists(self.lm_path):
            try:
                with open(self.lm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.unigrams = Counter(data.get("unigrams", {}))
                    self.bigrams = Counter(data.get("bigrams", {}))
                    self.trigrams = Counter(data.get("trigrams", {}))
                    self.total_unigrams = sum(self.unigrams.values()) or 1
                    self.vocab_size = len(self.unigrams) or 1
                print(f"Loaded Nepali N-gram LM ({len(self.unigrams):,} unigrams, {len(self.bigrams):,} bigrams, {len(self.trigrams):,} trigrams).")
                return
            except Exception as e:
                print(f"Notice loading LM cache: {e}. Rebuilding...")

        self.build_lm()

    def build_lm(self):
        print("Building Nepali N-gram Language Model from Wikipedia and speech transcriptions...")
        counts_1 = Counter()
        counts_2 = Counter()
        counts_3 = Counter()

        # Extract text from datasets
        try:
            from datasets import load_dataset, Audio
            hf_token = os.environ.get("HF_TOKEN", None)

            # 1. Dataset transcriptions
            print("Indexing speech dataset sentences...")
            ds = load_dataset("pujanpaudel/nepali_speech_to_text", split="train", streaming=True, token=hf_token)
            if "audio" in ds.features:
                ds = ds.cast_column("audio", Audio(decode=False))

            n = 0
            for item in ds:
                text = ""
                for col in ("transcription", "text", "sentence", "normalized_text", "transcript"):
                    if col in item and item[col]:
                        text = str(item[col])
                        break
                if text:
                    words = [normalize_nepali_word(tok) for tok in text.split()]
                    words = ["<s>"] + [w for w in words if len(w) >= 1] + ["</s>"]
                    for i in range(len(words)):
                        counts_1[words[i]] += 1
                        if i >= 1:
                            counts_2[f"{words[i-1]} {words[i]}"] += 1
                        if i >= 2:
                            counts_3[f"{words[i-2]} {words[i-1]} {words[i]}"] += 1
                n += 1
                if n >= 15000:
                    break

            # 2. Wikipedia Nepali Articles
            print("Indexing Nepali Wikipedia articles...")
            try:
                wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.ne", split="train", streaming=True)
                w_n = 0
                for item in wiki_ds:
                    text = item.get("text", "")
                    for sentence in re.split(r"[।\n?!.]", text):
                        words = [normalize_nepali_word(tok) for tok in sentence.split()]
                        words = ["<s>"] + [w for w in words if len(w) >= 1] + ["</s>"]
                        if len(words) > 2:
                            for i in range(len(words)):
                                counts_1[words[i]] += 1
                                if i >= 1:
                                    counts_2[f"{words[i-1]} {words[i]}"] += 1
                                if i >= 2:
                                    counts_3[f"{words[i-2]} {words[i-1]} {words[i]}"] += 1
                    w_n += 1
                    if w_n >= 1000:
                        break
            except Exception as e:
                print(f"Wikipedia LM notice: {e}")

        except Exception as e:
            print(f"Notice building LM: {e}")

        # Keep most frequent N-grams for compact storage & speed
        self.unigrams = Counter(dict(counts_1.most_common(50000)))
        self.bigrams = Counter(dict(counts_2.most_common(120000)))
        self.trigrams = Counter(dict(counts_3.most_common(150000)))
        self.total_unigrams = sum(self.unigrams.values()) or 1
        self.vocab_size = len(self.unigrams) or 1

        try:
            with open(self.lm_path, "w", encoding="utf-8") as f:
                json.dump({
                    "unigrams": dict(self.unigrams),
                    "bigrams": dict(self.bigrams),
                    "trigrams": dict(self.trigrams),
                }, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved Nepali N-gram LM ({len(self.unigrams):,} unigrams, {len(self.bigrams):,} bigrams, {len(self.trigrams):,} trigrams) to '{self.lm_path}'.")
        except Exception as e:
            print(f"Warning saving LM: {e}")

    def score_word(self, word: str, prev_word: str = "<s>", prev_prev_word: str = "") -> float:
        """
        Computes log P(word | prev_prev_word, prev_word) with smoothed Jelinek-Mercer interpolation.
        """
        w = normalize_nepali_word(word)
        w1 = normalize_nepali_word(prev_word) if prev_word else "<s>"
        w2 = normalize_nepali_word(prev_prev_word) if prev_prev_word else ""

        # 1. Unigram probability P1(w)
        p1 = (self.unigrams.get(w, 0) + 1.0) / (self.total_unigrams + self.vocab_size)

        # 2. Bigram probability P2(w | w1)
        c_w1 = self.unigrams.get(w1, 0)
        c_w1_w = self.bigrams.get(f"{w1} {w}", 0)
        p2 = (c_w1_w + 1.0) / (c_w1 + self.vocab_size) if c_w1 > 0 else p1

        # 3. Trigram probability P3(w | w2, w1)
        if w2:
            c_w2_w1 = self.bigrams.get(f"{w2} {w1}", 0)
            c_w2_w1_w = self.trigrams.get(f"{w2} {w1} {w}", 0)
            p3 = (c_w2_w1_w + 1.0) / (c_w2_w1 + self.vocab_size) if c_w2_w1 > 0 else p2
        else:
            p3 = p2

        # Interpolated probability: 0.6 * trigram + 0.3 * bigram + 0.1 * unigram
        p_interp = 0.60 * p3 + 0.30 * p2 + 0.10 * p1
        return math.log(max(1e-9, p_interp))

    def rescore_sentence(self, sentence: str, lm_weight: float = 0.15) -> str:
        """
        Grammatical context-aware spell rescoring using the Trigram/Bigram model.
        Keeps acoustic prediction dominant while using LM as a gentle tie-breaker.
        """
        if not sentence or sentence.strip() in ("", "No Speech Detected", "Error"):
            return sentence

        from nepali_lexicon import get_lexicon_rescorer, levenshtein_distance
        rescorer = get_lexicon_rescorer()

        raw_words = sentence.strip().split()
        if not raw_words:
            return sentence

        rescored_words = []
        for i, word in enumerate(raw_words):
            w = normalize_nepali_word(word)
            if not w or len(w) <= 2:
                rescored_words.append(word)
                continue

            # If word is already a valid dictionary word, never alter it
            if w in rescorer.word_counts:
                rescored_words.append(w)
                continue

            prev_w = rescored_words[i - 1] if i >= 1 else "<s>"
            prev_prev_w = rescored_words[i - 2] if i >= 2 else ""

            # Candidate words within edit distance == 1
            candidates = [w]
            w_len = len(w)
            for l in range(max(1, w_len - 1), w_len + 2):
                for cand in rescorer.words_by_len.get(l, []):
                    if levenshtein_distance(w, cand) <= 1 and rescorer.word_counts.get(cand, 0) >= 3:
                        candidates.append(cand)

            candidates = list(set(candidates))
            best_cand = w
            best_score = float("-inf")

            for cand in candidates:
                dist = levenshtein_distance(w, cand)
                lm_score = self.score_word(cand, prev_w, prev_prev_w)
                freq_score = math.log10(rescorer.word_counts.get(cand, 1) + 1)

                # Acoustic Edit Distance strongly dominates (dist * 6.0), LM guides tie-breaks
                score = - (dist * 6.0) + (lm_weight * lm_score) + (0.3 * freq_score)
                if score > best_score:
                    best_score = score
                    best_cand = cand

            rescored_words.append(best_cand)

        return " ".join(rescored_words)


_ngram_lm = None

def get_ngram_lm() -> NepaliNGramLM:
    global _ngram_lm
    if _ngram_lm is None:
        _ngram_lm = NepaliNGramLM()
    return _ngram_lm

if __name__ == "__main__":
    lm = get_ngram_lm()
    print("Testing sentence rescoring with Trigram LM:")
    test_1 = "नेपाल को राजधानी काठमाडौ हो"
    print("Raw:     ", test_1)
    print("Rescored:", lm.rescore_sentence(test_1))
