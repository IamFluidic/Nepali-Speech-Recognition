import os
import argparse
import pickle
import numpy as np
from hmmlearn import hmm
from preprocess_mfcc import preprocess_feature

DATASET_DIR = r"C:\Users\user\Downloads\asr_nepali_0\asr_nepali"
TSV_PATH = os.path.join(DATASET_DIR, "utt_spk_text.tsv")
DATA_DIR = os.path.join(DATASET_DIR, "data")
MODEL_OUTPUT = "hmm_model.pkl"

def find_audio_path(utt_id):
    # Utterance FLAC files are stored in data/XX/<utt_id>.flac where XX is the first 2 chars of utt_id
    prefix = utt_id[:2]
    audio_path = os.path.join(DATA_DIR, prefix, f"{utt_id}.flac")
    if os.path.exists(audio_path):
        return audio_path
    return None

def train_nepali_dataset(max_samples=500, n_components=5, n_iter=100):
    print(f"Loading dataset mapping from: {TSV_PATH}")
    if not os.path.exists(TSV_PATH):
        print(f"Error: {TSV_PATH} not found.")
        return

    features_list = []
    labels = []
    processed_count = 0

    with open(TSV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            utt_id, spk_id, text = parts[0], parts[1], parts[2]
            audio_path = find_audio_path(utt_id)

            if audio_path:
                try:
                    feat = preprocess_feature(audio_path)
                    if feat is not None and feat.shape[1] > 0:
                        # Transpose to shape (num_frames, num_features)
                        features_list.append(feat.T)
                        labels.append(text)
                        processed_count += 1
                        if processed_count % 50 == 0:
                            print(f"Processed {processed_count}/{max_samples} audio files...")
                        if max_samples and processed_count >= max_samples:
                            break
                except Exception as e:
                    print(f"Skipping {utt_id}: {e}")

    if not features_list:
        print("No valid audio features extracted!")
        return

    print(f"Successfully processed {len(features_list)} audio files.")
    print("Combining features for HMM training...")
    
    X = np.vstack(features_list)
    lengths = [f.shape[0] for f in features_list]

    print(f"Training GaussianHMM (components={n_components}, iter={n_iter})...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=n_iter)
    model.fit(X, lengths)

    print(f"Saving trained model to {MODEL_OUTPUT}...")
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(model, f)

    print("HMM Model training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMM model on Nepali ASR dataset")
    parser.add_argument("--max_samples", type=int, default=300, help="Maximum number of audio files to process for training")
    parser.add_argument("--n_components", type=int, default=5, help="Number of HMM states/components")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of HMM fitting iterations")
    args = parser.parse_args()

    train_nepali_dataset(max_samples=args.max_samples, n_components=args.n_components, n_iter=args.n_iter)
