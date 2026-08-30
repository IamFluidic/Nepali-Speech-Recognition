"""
expand_nepali_lexicon.py
========================
Streams IRIIS-RESEARCH/Nepali-Text-Corpus, Nepali Wikipedia, and Speech datasets
to expand nepali_lexicon.json and nepali_ngram_lm.json to 250,000+ unique Devanagari words.
"""

import os
import sys
import json
import re
import unicodedata
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LEXICON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nepali_lexicon.json")
LM_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nepali_ngram_lm.json")

def normalize_nepali_word(w: str) -> str:
    if not w:
        return ""
    w = unicodedata.normalize("NFC", str(w).strip())
    # Keep only Devanagari characters, matras, halant, and digits
    w = re.sub(r"[^\u0900-\u097F\u0966-\u096F]", "", w)
    return w.strip()

def build_expanded_lexicon(target_vocab_size=250000):
    print("=" * 70, flush=True)
    print("   EXPANDING WITH IRIIS-RESEARCH/Nepali-Text-Corpus & WIKIPEDIA", flush=True)
    print("=" * 70, flush=True)

    word_counts = Counter()
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    # 1. Load existing lexicon counts if present
    if os.path.exists(LEXICON_FILE):
        try:
            with open(LEXICON_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                word_counts.update(existing)
            print(f"Loaded existing baseline lexicon: {len(existing):,} words.", flush=True)
        except Exception as e:
            print(f"Notice loading existing lexicon: {e}", flush=True)

    # 2. Ingest IRIIS-RESEARCH/Nepali-Text-Corpus (Column: 'Article')
    try:
        from datasets import load_dataset
        hf_token = os.environ.get("HF_TOKEN", None)
        print("\n[1/2] Streaming 'IRIIS-RESEARCH/Nepali-Text-Corpus' (News Articles)...", flush=True)
        iriis_ds = load_dataset("IRIIS-RESEARCH/Nepali-Text-Corpus", split="train", streaming=True, token=hf_token)
        count = 0
        for item in iriis_ds:
            article_text = item.get("Article", "") or item.get("text", "")
            if article_text:
                tokens = [normalize_nepali_word(t) for t in article_text.split()]
                tokens = [t for t in tokens if len(t) >= 1]
                for i, w in enumerate(tokens):
                    word_counts[w] += 1
                    unigrams[w] += 1
                    if i >= 1:
                        bigrams[f"{tokens[i-1]} {w}"] += 1
                    if i >= 2:
                        trigrams[f"{tokens[i-2]} {tokens[i-1]} {w}"] += 1
            count += 1
            if count % 1000 == 0:
                print(f"  Processed {count:,} news articles... (Total unique words: {len(word_counts):,})", flush=True)
            if len(word_counts) >= target_vocab_size or count >= 20000:
                break
        print(f"Finished streaming IRIIS corpus ({count:,} articles). Total unique words: {len(word_counts):,}", flush=True)
    except Exception as e:
        print(f"IRIIS corpus notice: {e}", flush=True)

    # 3. Stream Nepali Wikipedia text corpus ('wikimedia/wikipedia', '20231101.ne')
    try:
        from datasets import load_dataset
        print("\n[2/2] Streaming Nepali Wikipedia text corpus ('wikimedia/wikipedia')...", flush=True)
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.ne", split="train", streaming=True)
        wiki_count = 0
        for item in wiki_ds:
            text = item.get("text", "")
            if text:
                tokens = [normalize_nepali_word(t) for t in text.split()]
                tokens = [t for t in tokens if len(t) >= 1]
                for i, w in enumerate(tokens):
                    word_counts[w] += 1
                    unigrams[w] += 1
                    if i >= 1:
                        bigrams[f"{tokens[i-1]} {w}"] += 1
                    if i >= 2:
                        trigrams[f"{tokens[i-2]} {tokens[i-1]} {w}"] += 1
            wiki_count += 1
            if wiki_count % 1000 == 0:
                print(f"  Processed {wiki_count:,} Wikipedia articles... (Total unique words: {len(word_counts):,})", flush=True)
            if len(word_counts) >= target_vocab_size or wiki_count >= 15000:
                break
        print(f"Finished streaming Wikipedia corpus ({wiki_count:,} articles). Total unique words: {len(word_counts):,}", flush=True)
    except Exception as e:
        print(f"Wikipedia notice: {e}", flush=True)

    # 4. Filter and Save Lexicon
    print(f"\nFinalizing Lexicon & Language Model...", flush=True)
    print(f"Total unique Devanagari words extracted: {len(word_counts):,}", flush=True)
    filtered_lexicon = {w: cnt for w, cnt in word_counts.items() if len(w) >= 1}
    with open(LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_lexicon, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(filtered_lexicon):,} unique words to '{LEXICON_FILE}'.", flush=True)

    # 5. Filter and Save N-gram Language Model
    print("\nCompiling high-frequency N-gram Language Model...", flush=True)
    top_unigrams = dict(unigrams.most_common(100000))
    top_bigrams = dict(bigrams.most_common(250000))
    top_trigrams = dict(trigrams.most_common(300000))

    lm_data = {
        "unigrams": top_unigrams,
        "bigrams": top_bigrams,
        "trigrams": top_trigrams
    }
    with open(LM_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(lm_data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved N-gram LM ({len(top_unigrams):,} unigrams, {len(top_bigrams):,} bigrams, {len(top_trigrams):,} trigrams) to '{LM_SAVE_PATH}'.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"   EXPANSION COMPLETE: {len(filtered_lexicon):,} WORDS IN DEVANAGARI LEXICON!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    build_expanded_lexicon(target_vocab_size=250000)
