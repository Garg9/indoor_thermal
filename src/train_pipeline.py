from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.model_training import train_models


def main():
    print("Loading dataset...")
    df = load_raw_data()

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Training models...")
    train_models(X, y)

    print("\nModel training pipeline completed successfully.")


if __name__ == "__main__":
    main()
