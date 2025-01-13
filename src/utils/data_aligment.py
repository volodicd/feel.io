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


import pandas as pd
import logging
from typing import Dict, Optional
from sklearn.utils import resample

def label_level_align(
    image_data: pd.DataFrame,
    audio_data: pd.DataFrame,
    text_data: pd.DataFrame
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Pure label-level alignment:
      - We group only by (emotion, split).
      - We do *not* try to match the same instance or user ID.
      - For each domain (image, audio, text), we pick a 'target_size' (like midpoint or max).
      - We oversample/undersample each domain to the same target_size for that emotion+split.
      - Then we cross-join them randomly so we get tri-modal samples for that label.
    """

    # If you want to store results for each (emotion, split)
    aligned_parts = []

    # For each emotion, for each split, do label-level pairing
    for emotion in range(7):
        for split in ["train", "test"]:
            # All images with that emotion+split
            img_group = image_data[
                (image_data["emotion"] == emotion) & (image_data["split"] == split)
            ]
            # All audio with that emotion+split
            aud_group = audio_data[
                (audio_data["emotion"] == emotion) & (audio_data["split"] == split)
            ]
            # All text with that emotion+split
            txt_group = text_data[
                (text_data["emotion"] == emotion) & (text_data["split"] == split)
            ]

            # If any domain is empty, skip
            if len(img_group) == 0 or len(aud_group) == 0 or len(txt_group) == 0:
                continue

            # Decide how many samples you want to keep for each domain
            # For instance, pick a target size = midpoint among them
            max_len = max(len(img_group), len(aud_group), len(txt_group))
            min_len = min(len(img_group), len(aud_group), len(txt_group))
            if min_len == 0:
                continue

            target_size = (max_len + min_len) // 2  # or use max_len, or something else

            # Oversample/undersample each domain to target_size
            # We'll define a quick helper:
            def sample_or_oversample(df, size: int) -> pd.DataFrame:
                if len(df) == 0:
                    return df
                if len(df) < size:
                    return resample(df, replace=True, n_samples=size, random_state=42)
                elif len(df) > size:
                    return resample(df, replace=False, n_samples=size, random_state=42)
                else:
                    return df

            img_resampled = sample_or_oversample(img_group, target_size)
            aud_resampled = sample_or_oversample(aud_group, target_size)
            txt_resampled = sample_or_oversample(txt_group, target_size)

            # Now cross-join them randomly. The simplest way:
            #   - shuffle each
            #   - reindex them with a common index
            #   - then "merge" them row by row
            img_resampled = img_resampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
            aud_resampled = aud_resampled.sample(frac=1.0, random_state=99).reset_index(drop=True)
            txt_resampled = txt_resampled.sample(frac=1.0, random_state=123).reset_index(drop=True)
            # At this point, all 3 have the same length = target_size.
            # We'll combine them in a "pretend tri-modal sample" row by row.

            img_resampled = img_resampled.reset_index(drop=True)
            aud_resampled = aud_resampled.reset_index(drop=True)
            txt_resampled = txt_resampled.reset_index(drop=True)

            # Tag them so we can separate later:
            # We'll store them in a bigger list, then concat at the end.
            # We'll just rename columns for clarity (or keep separate dict):
            # For the sake of example, let's store them in a dict:
            # (Alternatively, you could store them in one DataFrame with separate columns.)
            aligned_parts.append({
                "image": img_resampled,
                "audio": aud_resampled,
                "text": txt_resampled
            })

    if not aligned_parts:
        return None

    # Combine all
    # We'll just do a big pd.concat for each modality
    all_img = []
    all_aud = []
    all_txt = []
    for chunk in aligned_parts:
        all_img.append(chunk["image"])
        all_aud.append(chunk["audio"])
        all_txt.append(chunk["text"])

    return {
        "image": pd.concat(all_img).reset_index(drop=True),
        "audio": pd.concat(all_aud).reset_index(drop=True),
        "text": pd.concat(all_txt).reset_index(drop=True),
    }

