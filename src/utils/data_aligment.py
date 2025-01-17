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
            assert len (img_resampled) == len (aud_resampled) == len (txt_resampled) == target_size
            print (f"Emotion={emotion}, split={split},  ",
                   f"img_group={len (img_group)}, aud_group={len (aud_group)}, txt_group={len (txt_group)}, ",
                   f"target_size={target_size}, ",
                   f"img_resampled={len (img_resampled)}, aud_resampled={len (aud_resampled)}, txt_resampled={len (txt_resampled)}")

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
