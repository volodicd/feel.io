import pandas as pd
import logging
from typing import Dict, Optional
from sklearn.utils import resample

def align_datasets(
    image_data: pd.DataFrame,
    audio_data: pd.DataFrame,
    text_data: pd.DataFrame
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Align datasets by emotion label and split, but use a 'mid-range' approach:
    oversample smaller domain and undersample bigger domain so we keep more data
    than a strict min(...) approach.
    """
    aligned_data = []

    # Utility function
    def sample_or_oversample(df: pd.DataFrame, size: int, random_state=42) -> pd.DataFrame:
        """
        If df has fewer rows than `size`, oversample (replace=True).
        If df has more rows, undersample (replace=False).
        If exactly the same, just return as is.
        """
        n = len(df)
        if n == 0:
            # Return empty, or skip.
            return df  # We'll handle zero outside
        if n < size:
            # oversample
            return resample(df, replace=True, n_samples=size, random_state=random_state)
        elif n > size:
            # undersample
            return resample(df, replace=False, n_samples=size, random_state=random_state)
        else:
            return df  # exactly size

    try:
        # Group data by emotion and split
        image_groups = image_data.groupby(['emotion', 'split'])
        audio_groups = audio_data.groupby(['emotion', 'split'])
        text_groups = text_data.groupby(['emotion', 'split'])

        for (emotion, split), img_group in image_groups:
            # Only proceed if we also have audio + text for (emotion, split)
            if (emotion, split) in audio_groups.groups and (emotion, split) in text_groups.groups:
                aud_group = audio_groups.get_group((emotion, split))
                txt_group = text_groups.get_group((emotion, split))

                if len(img_group) == 0 or len(aud_group) == 0 or len(txt_group) == 0:
                    continue  # skip if any domain has zero

                # compute min and max
                max_len = max(len(img_group), len(aud_group), len(txt_group))
                min_len = min(len(img_group), len(aud_group), len(txt_group))
                if min_len == 0:
                    continue

                # pick a midpoint or whichever target you want:
                target_size = (max_len + min_len) // 2
                # If you prefer to oversample all smaller sets to match max_len:
                # target_size = max_len
                # Or if you want a smaller target for memory reasons, pick something else:
                # target_size = 2000  # arbitrary

                # oversample/undersample each domain
                img_samples = sample_or_oversample(img_group, target_size)
                aud_samples = sample_or_oversample(aud_group, target_size)
                txt_samples = sample_or_oversample(txt_group, target_size)

                # If any ended up 0 after resampling (shouldn’t happen, but just in case):
                if len(img_samples) == 0 or len(aud_samples) == 0 or len(txt_samples) == 0:
                    continue

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

    except Exception as e:
        logging.error(f"Error in dataset alignment: {str(e)}")
        return None
