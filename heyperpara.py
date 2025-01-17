import pandas as pd
from src.data.dataset import MultiModalEmotionDataset
from src.optimization.hyper import EnhancedHyperparameterOptimizer

def main():
    # Load datasets
    image_data = pd.read_csv('data/processed/fer2013.csv')
    expw_data = pd.read_csv('data/processed/expw.csv')
    audio_data = pd.read_csv('data/processed/ravdess.csv')
    goemotions_data = pd.read_csv('data/processed/goemotions.csv')

    image_data = pd.concat([image_data, expw_data]).reset_index(drop=True)

    # Create datasets
    train_dataset = MultiModalEmotionDataset(
        image_data=image_data,
        audio_data=audio_data,
        text_data=goemotions_data,
        split='train'
    )

    val_dataset = MultiModalEmotionDataset(
        image_data=image_data,
        audio_data=audio_data,
        text_data=goemotions_data,
        split='test'
    )

    # Initialize optimizer
    optimizer = EnhancedHyperparameterOptimizer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=50,
        n_jobs=-1,
        storage="sqlite:///hyper_opt.db"
    )

    # Run optimization
    results = optimizer.optimize_parallel()

    # Print results
    print("\nBest hyperparameters found:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    print(f"\nBest accuracy achieved: {results['best_value']:.4f}")

if __name__ == "__main__":
    main()
