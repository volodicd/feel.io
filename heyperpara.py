# Example usage script (optimize_hyperparameters.py)
import pandas as pd
import torch
from src.data.dataset import MultiModalEmotionDataset
from src.optimization.hyper_cuda import EnhancedHyperparameterOptimizer


def main ():
    # 1. First, load your datasets
    image_data = pd.read_csv ('data/processed/fer2013.csv')
    expw_data = pd.read_csv ('data/processed/expw.csv')
    audio_data = pd.read_csv ('data/processed/ravdess.csv')
    goemotions_data = pd.read_csv ('data/processed/goemotions.csv')

    # Combine image datasets if needed
    image_data = pd.concat ([
        image_data,
        expw_data,
    ]).reset_index (drop=True)

    # 2. Create your datasets
    train_dataset = MultiModalEmotionDataset (
        image_data=image_data,
        audio_data=audio_data,
        text_data=goemotions_data,
        split='train'
    )

    val_dataset = MultiModalEmotionDataset (
        image_data=image_data,
        audio_data=audio_data,
        text_data=goemotions_data,
        split='test'
    )

    # 3. Initialize the optimizer
    optimizer = EnhancedHyperparameterOptimizer (
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=50,  # Number of different hyperparameter combinations to try
        n_jobs=-1,  # Use all available GPUs
        storage="sqlite:///hyper_opt.db"  # Optional: save results to database
    )

    # 4. Run the optimization
    results = optimizer.optimize_parallel ()

    # 5. Print the best parameters
    print ("\nBest hyperparameters found:")
    for param, value in results['best_params'].items ():
        print (f"{param}: {value}")

    print (f"\nBest accuracy achieved: {results['best_value']:.4f}")


if __name__ == "__main__":
    main ()