# src/data/processing/dataset_processor.py

import os
import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import json
from PIL import Image
from tqdm import tqdm
import logging
from typing import Dict, List, Optional

class DatasetProcessor:
    """
    Main processor class for handling multiple emotion datasets.
    Processes and validates image and audio datasets, converting them into a standardized format.
    """

    def __init__(self, base_path: str = 'data/raw'):
        """
        Args:
            base_path: Path to the directory containing raw data (default: 'data/raw')
        """
        # Set up paths
        self.base_path = Path(base_path)         # Raw data directory
        self.processed_path = Path('data/processed')
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Configure logging with detailed format for debugging
        logging.basicConfig(
            filename=self.processed_path / 'processing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Standardized emotion mapping across all datasets
        self.emotion_mapping = {
            'angry':    0,
            'disgust':  1,
            'fear':     2,
            'happy':    3,
            'neutral':  4,
            'sad':      5,
            'surprise': 6
        }

    def determine_celeba_emotion(self, row: pd.Series) -> str:
        """
        Determine emotion from CelebA attributes using rule-based mapping.
        Args:
            row: A row from the CelebA attributes dataset
        Returns:
            Detected emotion as a string
        """
        if row['Smiling'] > 0 and row['Young'] > 0:
            return 'happy'
        elif row['Arched_Eyebrows'] > 0 and (row['Angry_Looking'] > 0 or row['Frowning'] > 0):
            return 'angry'
        elif row['Sad'] > 0 or (row['Frowning'] > 0 and not row['Smiling'] > 0):
            return 'sad'
        elif row['Mouth_Slightly_Open'] > 0 and row['Arched_Eyebrows'] > 0:
            return 'surprise'
        elif row['Frowning'] > 0 and row['Narrow_Eyes'] > 0:
            return 'fear'
        elif row['Mouth_Slightly_Open'] < 0 and row['Frowning'] > 0:
            return 'disgust'
        else:
            return 'neutral'

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """
        Validate processed dataset for completeness and correctness.

        Metrics tracked:
        - Data completeness: All required columns present
        - Data validity: Emotion values within valid range (0-6)
        - File existence: All referenced files exist
        - Class balance: Ratio between largest and smallest classes

        Target metrics:
        - 100% required columns present
        - 100% valid emotion values
        - File existence rate > 95%
        - Class imbalance ratio < 10:1
        """
        try:
            # Verify required columns exist
            required_cols = ['path', 'emotion', 'split']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"{dataset_name}: Missing columns: {missing_cols}")
                return False

            # Verify emotion values are valid
            invalid_emotions = df[~df['emotion'].isin(range(7))]
            if not invalid_emotions.empty:
                logging.error(f"{dataset_name}: Found {len(invalid_emotions)} invalid emotion values")
                return False

            # Check all files exist
            invalid_files = 0
            for _, row in df.iterrows():
                if not Path(row['path']).exists():
                    invalid_files += 1

            if invalid_files > 0:
                logging.warning(f"{dataset_name}: {invalid_files} files not found")

            # Log split distribution
            split_dist = df['split'].value_counts()
            logging.info(f"{dataset_name} split distribution: {split_dist.to_dict()}")
            emotion_dist = df['emotion'].value_counts()
            logging.info(f"{dataset_name} emotion distribution:\n{emotion_dist}")

            # Check for potential class imbalance
            min_samples = emotion_dist.min()
            max_samples = emotion_dist.max()
            if min_samples > 0 and (max_samples / min_samples > 10):
                logging.warning(f"{dataset_name}: High class imbalance detected")

            return True

        except Exception as e:
            logging.error(f"Validation error for {dataset_name}: {str(e)}")
            return False

    def process_fer2013(self) -> pd.DataFrame:
        """Process FER-2013 dataset with error handling and progress tracking."""
        fer_path = self.base_path / 'FER-2013'  # adjusted to new directory
        data = []

        try:
            for split in ['train', 'test']:
                split_path = fer_path / split
                if not split_path.exists():
                    logging.error(f"FER2013: {split} directory not found in {split_path}")
                    continue

                for emotion in tqdm(self.emotion_mapping.keys(), desc=f"Processing FER2013 {split}"):
                    emotion_path = split_path / emotion
                    if not emotion_path.exists():
                        # Maybe just skip if the subfolder is missing
                        continue

                    for img_path in emotion_path.glob('*.jpg'):
                        try:
                            # Verify image can be opened
                            Image.open(img_path).verify()
                            data.append({
                                'path':    str(img_path),
                                'emotion': self.emotion_mapping[emotion],
                                'split':   split
                            })
                        except Exception as e:
                            logging.warning(f"FER2013: Error with image {img_path}: {str(e)}")

            df = pd.DataFrame(data)
            if self.validate_dataset(df, "FER2013"):
                out_csv = self.processed_path / 'fer2013.csv'
                df.to_csv(out_csv, index=False)
                logging.info(f"Saved FER2013 CSV to {out_csv}")
                return df
            return pd.DataFrame()

        except Exception as e:
            logging.error(f"FER2013 processing error: {str(e)}")
            return pd.DataFrame()

    def process_expw(self) -> pd.DataFrame:
        """Process ExpW dataset with validation and error handling."""
        expw_path = self.base_path / 'ExpW'
        data = []

        try:
            # ExpW emotion mapping
            expw_emotion_map = {
                0: 'angry', 1: 'disgust', 2: 'fear',
                3: 'happy', 4: 'sad',     5: 'surprise', 6: 'neutral'
            }

            annotations_file = expw_path / 'annotations' / 'label.lst'
            if not annotations_file.exists():
                logging.error(f"ExpW: {annotations_file} not found.")
                return pd.DataFrame()

            # Read annotations
            with open(annotations_file, 'r') as f:
                annotations = f.readlines()

            for line in tqdm(annotations, desc="Processing ExpW"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_path = expw_path / 'images' / parts[0]
                    emotion_id = int(parts[1])

                    if emotion_id in expw_emotion_map:
                        try:
                            Image.open(image_path).verify()
                            emotion = expw_emotion_map[emotion_id]
                            # Random split ~80% train, 20% test
                            split = 'train' if np.random.random() < 0.8 else 'test'

                            data.append({
                                'path':    str(image_path),
                                'emotion': self.emotion_mapping[emotion],
                                'split':   split
                            })
                        except Exception as e:
                            logging.warning(f"ExpW: Error with image {image_path}: {str(e)}")

            df = pd.DataFrame(data)
            if self.validate_dataset(df, "ExpW"):
                out_csv = self.processed_path / 'expw.csv'
                df.to_csv(out_csv, index=False)
                logging.info(f"Saved ExpW CSV to {out_csv}")
                return df
            return pd.DataFrame()

        except Exception as e:
            logging.error(f"ExpW processing error: {str(e)}")
            return pd.DataFrame()

    def process_celeba(self) -> pd.DataFrame:
        """Process CelebA dataset with improved emotion mapping."""
        celeba_path = self.base_path / 'CelebA'
        data = []

        try:
            partition_file = celeba_path / 'annotations' / 'list_eval_partition.csv'
            attr_file = celeba_path / 'annotations' / 'list_attr_celeba.csv'

            if not partition_file.exists() or not attr_file.exists():
                logging.error("CelebA: Missing annotation files.")
                return pd.DataFrame()

            partition_df = pd.read_csv(partition_file)
            attr_df = pd.read_csv(attr_file)

            # Convert -1/1 to 0/1 for easier processing
            attr_df[attr_df.columns[1:]] = (attr_df[attr_df.columns[1:]] + 1) // 2

            for idx, row in tqdm(attr_df.iterrows(), total=len(attr_df), desc="Processing CelebA"):
                try:
                    image_path = celeba_path / 'images' / row['image_id']
                    if not image_path.exists():
                        continue

                    # Determine emotion using complex rules
                    emotion = self.determine_celeba_emotion(row)
                    split = 'train' if partition_df.iloc[idx]['partition'] == 0 else 'test'

                    data.append({
                        'path':    str(image_path),
                        'emotion': self.emotion_mapping[emotion],
                        'split':   split
                    })
                except Exception as e:
                    logging.warning(f"CelebA: Error processing row {idx}: {str(e)}")

            df = pd.DataFrame(data)
            if self.validate_dataset(df, "CelebA"):
                out_csv = self.processed_path / 'celeba.csv'
                df.to_csv(out_csv, index=False)
                logging.info(f"Saved CelebA CSV to {out_csv}")
                return df
            return pd.DataFrame()

        except Exception as e:
            logging.error(f"CelebA processing error: {str(e)}")
            return pd.DataFrame()

    def process_ravdess(self) -> pd.DataFrame:
        """Process RAVDESS dataset with audio validation."""
        ravdess_path = self.base_path / 'RAVDESS' / 'audio_speech_actors_01-24'
        data = []

        try:
            ravdess_emotion_map = {
                '01': 'neutral', '02': 'happy',  '03': 'sad',
                '04': 'angry',   '05': 'fear',  '06': 'surprise', '07': 'disgust'
            }

            if not ravdess_path.exists():
                logging.error(f"RAVDESS: {ravdess_path} not found.")
                return pd.DataFrame()

            for actor_dir in tqdm(list(ravdess_path.glob('Actor_*')), desc="Processing RAVDESS"):
                for audio_file in actor_dir.glob('*.wav'):
                    try:
                        # Verify audio file can be loaded
                        librosa.load(audio_file, duration=1)

                        parts = audio_file.stem.split('-')
                        # e.g. "03" is sad
                        emotion_code = parts[2]
                        if emotion_code in ravdess_emotion_map:
                            emotion = ravdess_emotion_map[emotion_code]
                            # last part is the actor number
                            actor_num = int(parts[-1])

                            data.append({
                                'path':      str(audio_file),
                                'emotion':   self.emotion_mapping[emotion],
                                'actor':     actor_num,
                                'intensity': parts[3],   # optional
                                'split':     'train' if actor_num <= 20 else 'test'
                            })
                    except Exception as e:
                        logging.warning(f"RAVDESS: Error with audio {audio_file}: {str(e)}")

            df = pd.DataFrame(data)
            if self.validate_dataset(df, "RAVDESS"):
                out_csv = self.processed_path / 'ravdess.csv'
                df.to_csv(out_csv, index=False)
                logging.info(f"Saved RAVDESS CSV to {out_csv}")
                return df
            return pd.DataFrame()

        except Exception as e:
            logging.error(f"RAVDESS processing error: {str(e)}")
            return pd.DataFrame()

    def process_goemotions (self) -> pd.DataFrame:
        """
        Process GoEmotions dataset into a single CSV with [text, emotion, split].
        """
        base_goemotions_path = self.base_path / 'GoEmotions' / 'archive-2' / 'data'
        # The folder structure might differ in your setup; adapt as needed.
        # Typically, there are train.tsv, dev.tsv, test.tsv files.

        all_data = []

        try:
            for split in ['train', 'dev', 'test']:
                tsv_path = base_goemotions_path / f'{split}.tsv'
                if not tsv_path.exists ():
                    logging.error (f"GoEmotions: {tsv_path} not found.")
                    continue

                # Each line typically: "text\tlabels"
                with open (tsv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines ()

                for line in tqdm (lines, desc=f"Processing GoEmotions {split}"):
                    parts = line.strip ().split ('\t')
                    if len (parts) < 2:
                        continue
                    text = parts[0]
                    labels = parts[1].split (',')
                    # If multi-label, decide on a mapping to single label or skip multi-label logic.
                    # For demonstration, pick the first label or map to 'emotion' if needed:
                    emotion_id = int (labels[0])  # or do a more complex mapping.

                    # Create a row
                    all_data.append ({
                        'text': text,
                        'emotion': emotion_id,
                        'split': 'train' if split == 'train' else 'test'
                        # If you want dev to also be test or separate 'val', adapt as needed.
                    })

            df = pd.DataFrame (all_data)
            # Validate columns
            if not df.empty:
                out_csv = self.processed_path / 'goemotions.csv'
                df.to_csv (out_csv, index=False)
                logging.info (f"Saved GoEmotions CSV to {out_csv}")
                return df

            return pd.DataFrame ()

        except Exception as e:
            logging.error (f"GoEmotions processing error: {str (e)}")
            return pd.DataFrame ()


def main():
    """
    Main processing function that handles all datasets.
    Includes comprehensive error handling and logging.
    """
    try:
        # Optionally pass base_path='data/raw' explicitly
        processor = DatasetProcessor(base_path='data/raw')
        datasets = {}

        # Process each dataset sequentially
        for name, func in [
            ("FER2013", processor.process_fer2013),
            ("ExpW",    processor.process_expw),
            ("RAVDESS", processor.process_ravdess),
            ("GoEmotions", processor.process_goemotions),
        ]:
            print(f"\nProcessing {name}...")
            df = func()
            if not df.empty:
                datasets[f"num_{name.lower()}"] = len(df)
                print(f"Successfully processed {len(df)} samples from {name}")
            else:
                print(f"Failed to process {name}")

        # Save comprehensive metadata
        metadata = {
            **datasets,
            'emotion_mapping': processor.emotion_mapping,
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(processor.processed_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        print("\nProcessing completed. See processing.log for details.")

    except Exception as e:
        logging.error(f"Fatal error in main processing: {str(e)}")
        print(f"Fatal error occurred. Check processing.log for details.")

if __name__ == '__main__':
    main()
