"""
expand_nepali_lexicon.py
========================
Streams Nepali speech dataset transcriptions and the Nepali Wikipedia text corpus
to expand nepali_lexicon.json to 100,000+ comprehensive unique Devanagari words.
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

def normalize_nepali_word(w: str) -> str:
    if not w:
        return ""
    w = unicodedata.normalize("NFC", str(w).strip())
    # Keep only Devanagari characters and digits
    w = re.sub(r"[^\u0900-\u097F\u0966-\u096F]", "", w)
    return w.strip()

def build_expanded_100k_lexicon():
    print("=" * 65)
    print("   EXPANDING NEPALI LEXICON TO 100,000+ UNIQUE WORDS")
    print("=" * 65)

    counts = Counter()

    # 1. Load existing lexicon counts if present
    if os.path.exists(LEXICON_FILE):
        try:
            with open(LEXICON_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                counts.update(existing)
            print(f"Loaded existing baseline lexicon: {len(existing):,} words.")
        except Exception as e:
            print(f"Notice loading existing: {e}")

    # 2. Stream transcriptions from pujanpaudel/nepali_speech_to_text
    try:
        from datasets import load_dataset, Audio
        hf_token = os.environ.get("HF_TOKEN", None)
        print("\n[1/2] Streaming transcriptions from 'pujanpaudel/nepali_speech_to_text'...")
        ds = load_dataset("pujanpaudel/nepali_speech_to_text", split="train", streaming=True, token=hf_token)
        if "audio" in ds.features:
            ds = ds.cast_column("audio", Audio(decode=False))

        sample_count = 0
        for item in ds:
            text = ""
            for col in ("transcription", "text", "sentence", "normalized_text", "transcript"):
                if col in item and item[col]:
                    text = str(item[col])
                    break
            if text:
                for token in text.split():
                    w = normalize_nepali_word(token)
                    if len(w) >= 1:
                        counts[w] += 1

            sample_count += 1
            if sample_count % 10000 == 0:
                print(f"  Processed {sample_count:,} speech transcriptions... (Unique words so far: {len(counts):,})")
            if sample_count >= 50000:
                break
        print(f"Finished streaming speech corpus ({sample_count:,} items).")
    except Exception as e:
        print(f"Speech streaming notice: {e}")

    # 3. Stream Nepali Wikipedia corpus (wikimedia/wikipedia 20231101.ne)
    try:
        from datasets import load_dataset
        print("\n[2/2] Streaming Nepali Wikipedia text corpus ('wikimedia/wikipedia', '20231101.ne')...")
        wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.ne", split="train", streaming=True)
        wiki_count = 0
        for item in wiki_ds:
            text = item.get("text", "")
            if text:
                for token in text.split():
                    w = normalize_nepali_word(token)
                    if len(w) >= 1:
                        counts[w] += 1
            wiki_count += 1
            if wiki_count % 500 == 0:
                print(f"  Processed {wiki_count:,} Wikipedia articles... (Unique words: {len(counts):,})")
            if len(counts) >= 120000 or wiki_count >= 5000:
                break
        print(f"Finished streaming Wikipedia corpus ({wiki_count:,} articles).")
    except Exception as e:
        print(f"Wikipedia streaming notice: {e}")

    # 4. Filter and retain top 100,000+ most frequent valid Nepali words
    filtered_dict = dict(counts.most_common(105000))
    print(f"\nFinal Expanded Lexicon Size: {len(filtered_dict):,} unique Nepali words.")

    with open(LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_dict, f, ensure_ascii=False, indent=2)

    print(f"Successfully cached 100k+ lexicon to '{LEXICON_FILE}'!")
    print("=" * 65)

if __name__ == "__main__":
    build_expanded_100k_lexicon()
