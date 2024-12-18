# src/data/processing/dataset_processor.py
"""
Dataset processing script for multiple datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import json
from PIL import Image
from tqdm import tqdm
import logging

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

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """
        Validate processed dataset for completeness and correctness.
        """
        try:
            # Define required columns based on dataset type
            if dataset_name == "GoEmotions":
                required_cols = ['text', 'emotion', 'split']
            else:
                required_cols = ['path', 'emotion', 'split']

            # Verify required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"{dataset_name}: Missing columns: {missing_cols}")
                return False

            # Verify emotion values are valid
            invalid_emotions = df[~df['emotion'].isin(range(7))]
            if not invalid_emotions.empty:
                logging.error(f"{dataset_name}: Found {len (invalid_emotions)} invalid emotion values")
                return False

            # Check file existence only for datasets with paths
            if 'path' in df.columns:
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
            logging.info (f"{dataset_name} emotion distribution:\n{emotion_dist}")

            # Check for potential class imbalance
            min_samples = emotion_dist.min()
            max_samples = emotion_dist.max()
            if min_samples > 0 and (max_samples / min_samples > 10):
                logging.warning(f"{dataset_name}: High class imbalance detected")

            return True

        except Exception as e:
            logging.error(f"Validation error for {dataset_name}: {str (e)}")
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
                        # just skip if the subfolder is missing
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
                df = self.hybrid_balance(df)
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
                df = self.hybrid_balance(df)
                out_csv = self.processed_path / 'expw.csv'
                df.to_csv(out_csv, index=False)
                logging.info(f"Saved ExpW CSV to {out_csv}")
                return df
            return pd.DataFrame()

        except Exception as e:
            logging.error(f"ExpW processing error: {str(e)}")
            return pd.DataFrame()


# Unused func (dataset was delete due to the lack of time)
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
                    # emotion = self.determine_celeba_emotion(row) deleted dataset
                    split = 'train' if partition_df.iloc[idx]['partition'] == 0 else 'test'

                    data.append({
                        'path':    str(image_path),
                        # 'emotion': self.emotion_mapping[emotion],
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

    def process_goemotions(self) -> pd.DataFrame:
        """Process GoEmotions dataset with proper emotion mapping."""
        base_goemotions_path = self.base_path / 'GoEmotions' / 'archive-2' / 'data'
        all_data = []

        try:
            # Create GoEmotions to standard emotion mapping
            goemotions_mapping = {
                # Anger group
                'anger': 'angry', 'annoyance': 'angry', 'disapproval': 'angry',
                # Disgust
                'disgust': 'disgust',
                # Fear group
                'fear': 'fear', 'nervousness': 'fear',
                # Joy/Happy group
                'joy': 'happy', 'amusement': 'happy', 'approval': 'happy',
                'excitement': 'happy', 'gratitude': 'happy', 'love': 'happy',
                'optimism': 'happy', 'relief': 'happy', 'pride': 'happy',
                'admiration': 'happy', 'desire': 'happy', 'caring': 'happy',
                # Sadness group
                'sadness': 'sad', 'disappointment': 'sad', 'embarrassment': 'sad',
                'grief': 'sad', 'remorse': 'sad',
                # Surprise group
                'surprise': 'surprise', 'realization': 'surprise',
                'confusion': 'surprise', 'curiosity': 'surprise',
                # Neutral
                'neutral': 'neutral'
            }

            # Load emotions list
            emotions_file = base_goemotions_path / 'emotions.txt'
            with open(emotions_file, 'r', encoding='utf-8') as f:
                emotions_list = [line.strip() for line in f.readlines()]

            idx_to_emotion = {i: emotion for i, emotion in enumerate(emotions_list)}
            idx_to_emotion = {i: emotion for i, emotion in enumerate (emotions_list)}

            for split in ['train', 'dev', 'test']:
                tsv_path = base_goemotions_path / f'{split}.tsv'
                if not tsv_path.exists():
                    logging.error(f"GoEmotions: {tsv_path} not found.")
                    continue

                with open(tsv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in tqdm(lines, desc=f"Processing GoEmotions {split}"):
                    parts = line.strip().split('\t')
                    if len (parts) < 2:
                        continue

                    text = parts[0]
                    labels = parts[1].split (',')

                    # Get the first emotion label
                    emotion_idx = int(labels[0])
                    emotion_name = idx_to_emotion[emotion_idx]

                    # Map to standard emotion
                    if emotion_name in goemotions_mapping:
                        standard_emotion = goemotions_mapping[emotion_name]

                        all_data.append({
                            'text': text,
                            'emotion': self.emotion_mapping[standard_emotion],
                            'split': 'train' if split == 'train' else 'test'
                        })

            df = pd.DataFrame(all_data)
            if not df.empty:
                # Add validation
                if self.validate_dataset (df, "GoEmotions"):
                    out_csv = self.processed_path / 'goemotions.csv'
                    df.to_csv (out_csv, index=False)
                    logging.info (f"Saved GoEmotions CSV to {out_csv}")
                    return df

            return pd.DataFrame ()

        except Exception as e:
            logging.error (f"GoEmotions processing error: {str (e)}")
            return pd.DataFrame ()

    def hybrid_balance(self, df: pd.DataFrame, target_column: str = 'emotion') -> pd.DataFrame:
        """
        Perform hybrid balancing: oversample minority classes and undersample majority classes.
        Args:
            df: Input DataFrame with the emotion column
            target_column: Name of the column containing class labels

        Returns:
            A balanced DataFrame.
        """
        from sklearn.utils import resample

        # Group by class
        grouped = df.groupby(target_column)

        # Determine max and min sample sizes
        max_size = grouped.size().max()
        min_size = grouped.size().min()

        # Target number for hybrid balance (midpoint between min and max)
        target_size = (max_size + min_size) // 2

        balanced_data = []

        for emotion, group in grouped:
            if len(group) < target_size:
                # Oversample minority class
                oversampled = resample(group, replace=True, n_samples=target_size, random_state=42)
                balanced_data.append(oversampled)
            elif len(group) > target_size:
                # Undersample majority class
                undersampled = resample(group, replace=False, n_samples=target_size, random_state=42)
                balanced_data.append(undersampled)
            else:
                balanced_data.append(group)

        # Combine all balanced classes
        return pd.concat(balanced_data).reset_index(drop=True)


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
