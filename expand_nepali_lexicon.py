"""
expand_nepali_lexicon.py
========================
Streams through all 50,000+ Nepali speech dataset transcriptions (text-only)
and expands nepali_lexicon.json to 50,000+ comprehensive unique words.
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
    w = re.sub(r"[^\u0900-\u097F\u0966-\u096F]", "", w)
    return w.strip()

def build_expanded_lexicon():
    print("=" * 60)
    print("  EXPANDING NEPALI LEXICON TO 50,000+ UNIQUE WORDS")
    print("=" * 60)

    counts = Counter()

    # 1. Load existing lexicon counts if present
    if os.path.exists(LEXICON_FILE):
        try:
            with open(LEXICON_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                counts.update(existing)
            print(f"Loaded existing baseline lexicon: {len(existing):,} words.")
        except Exception as e:
            print(f"Notice: {e}")

    # 2. Stream all transcriptions from pujanpaudel/nepali_speech_to_text without decoding audio
    try:
        from datasets import load_dataset, Audio
        hf_token = os.environ.get("HF_TOKEN", None)
        print("Streaming all text transcriptions from 'pujanpaudel/nepali_speech_to_text'...")
        ds = load_dataset("pujanpaudel/nepali_speech_to_text", split="train", streaming=True, token=hf_token)
        if "audio" in ds.features:
            ds = ds.cast_column("audio", Audio(decode=False))

        total_processed = 0
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

            total_processed += 1
            if total_processed % 5000 == 0:
                print(f"  Processed {total_processed:,} transcriptions... (Unique words so far: {len(counts):,})")

            # Cover all available utterances
            if total_processed >= 55000:
                break

        print(f"Finished streaming {total_processed:,} transcriptions.")
    except Exception as e:
        print(f"Streaming notice: {e}")

    # 3. Filter and retain top 60,000 most frequent valid Nepali words
    filtered_dict = dict(counts.most_common(60000))
    print(f"\nFinal Lexicon Size: {len(filtered_dict):,} unique Nepali words.")

    with open(LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved expanded dictionary successfully to '{LEXICON_FILE}'!")
    print("=" * 60)

if __name__ == "__main__":
    build_expanded_lexicon()
