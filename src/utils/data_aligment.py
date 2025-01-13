import pandas as pd
import logging
from typing import Dict, Optional


def align_datasets (image_data: pd.DataFrame, audio_data: pd.DataFrame, text_data: pd.DataFrame) -> Optional[
    Dict[str, pd.DataFrame]]:
    """Align datasets by emotion label and split"""
    aligned_data = []

    try:
        # Group data by emotion and split
        image_groups = image_data.groupby (['emotion', 'split'])
        audio_groups = audio_data.groupby (['emotion', 'split'])
        text_groups = text_data.groupby (['emotion', 'split'])

        # For each emotion and split combination
        for (emotion, split), img_group in image_groups:
            if (emotion, split) in audio_groups.groups and (emotion, split) in text_groups.groups:
                # Get corresponding groups
                aud_group = audio_groups.get_group ((emotion, split))
                txt_group = text_groups.get_group ((emotion, split))

                # Get minimum size
                max_len = max (len (img_group), len (aud_group), len (txt_group))
                min_len = min (len (img_group), len (aud_group), len (txt_group))
                target_size = (max_len + min_len) // 2  # e.g. midpoint
                img_sample_size = max(min_len, min(len(img_group), target_size))   # or oversample if group < target_size
                aud_sample_size = max(min_len, min(len(aud_group), target_size))
                txt_sample_size = max(min_len, min(len(txt_group), target_size))
                img_samples = img_group.sample (n=img_sample_size, random_state=42)
                aud_samples = aud_group.sample (n=aud_sample_size, random_state=42)
                txt_samples = txt_group.sample (n=txt_sample_size, random_state=42)

                # Combine
                aligned_data.append ({
                    'image': img_samples,
                    'audio': aud_samples,
                    'text': txt_samples
                })

        # Combine all aligned data
        if aligned_data:
            return {
                'image': pd.concat ([d['image'] for d in aligned_data]).reset_index (drop=True),
                'audio': pd.concat ([d['audio'] for d in aligned_data]).reset_index (drop=True),
                'text': pd.concat ([d['text'] for d in aligned_data]).reset_index (drop=True)
            }

        return None

    except Exception as e:
        logging.error (f"Error in dataset alignment: {str (e)}")
        return None