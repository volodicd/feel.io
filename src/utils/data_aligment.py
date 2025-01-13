import pandas as pd
import logging
from typing import Dict, Optional


def align_datasets (image_data: pd.DataFrame, audio_data: pd.DataFrame, text_data: pd.DataFrame):
    """Align datasets by emotion label and split, sampling up to min_size across the 3 domains."""
    aligned_data = []

    # Group data by emotion and split
    image_groups = image_data.groupby(['emotion', 'split'])
    audio_groups = audio_data.groupby(['emotion', 'split'])
    text_groups = text_data.groupby(['emotion', 'split'])

    for (emotion, split), img_group in image_groups:
        if (emotion, split) in audio_groups.groups and (emotion, split) in text_groups.groups:
            aud_group = audio_groups.get_group((emotion, split))
            txt_group = text_groups.get_group((emotion, split))

            # min_size approach
            min_size = min(len(img_group), len(aud_group), len(txt_group))
            if min_size == 0:
                continue  # skip if any domain has zero
            # sample = min_size from each domain
            img_samples = img_group.sample(n=min_size, random_state=42)
            aud_samples = aud_group.sample(n=min_size, random_state=42)
            txt_samples = txt_group.sample(n=min_size, random_state=42)

            aligned_data.append({
                'image': img_samples,
                'audio': aud_samples,
                'text': txt_samples
            })

    if aligned_data:
        return {
            'image': pd.concat(d['image'] for d in aligned_data).reset_index(drop=True),
            'audio': pd.concat(d['audio'] for d in aligned_data).reset_index(drop=True),
            'text': pd.concat(d['text'] for d in aligned_data).reset_index(drop=True)
        }
    return None