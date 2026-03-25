from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd 

MODEL_PATH = "models/thermal_comfort_model.pkl"


def train_models(X, y):
    """
    Train and compare ML models for thermal comfort prediction.
    """
    # Check if we have at least 2 classes
    unique_classes = pd.Series(y).unique()
    if len(unique_classes) < 2:
        class_counts = pd.Series(y).value_counts().to_dict()
        raise ValueError(
            f"Training failed: The dataset contains only one class: {unique_classes[0]}. "
            f"Machine Learning classifiers require at least 2 distinct classes to train. "
            f"Current distribution: {class_counts}. "
            "Please check your input data (sample_ashrae.csv) for variety."
        )
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved at: {MODEL_PATH}")

    return best_model




if __name__ == "__main__":
    import pandas as pd
    from src.data_loader import load_raw_data
    from src.preprocessing import preprocess_data

    df = load_raw_data()
    X, y = preprocess_data(df)

    train_models(X, y)



# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import os
# from xgboost import XGBClassifier
# from sklearn.metrics import confusion_matrix

# MODEL_PATH = "models/thermal_comfort_model.pkl"


# def train_models(X, y):
#     """
#     Train and compare ML models for thermal comfort prediction.
#     """

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     models = {
#         "Logistic Regression": LogisticRegression(
#             max_iter=2000,
#             class_weight="balanced"
#         ),

#         "Random Forest": RandomForestClassifier(
#             n_estimators=600,
#             max_depth=None,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42,
#             n_jobs=-1,
#             class_weight="balanced_subsample"
#         ),

#         "Gradient Boosting": GradientBoostingClassifier(
#             n_estimators=400,
#             learning_rate=0.05,
#             max_depth=3,
#             random_state=42
#         ),
#         "XGBoost": XGBClassifier(
#             n_estimators=600,
#             max_depth=6,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#             eval_metric="mlogloss"
#         )
#     }

#     best_model = None
#     best_accuracy = 0.0

#     for name, model in models.items():
#         print(f"\nTraining {name}...")
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)

#         print(f"{name} Accuracy: {acc:.4f}")
#         print(classification_report(y_test, y_pred))
#         print("Confusion Matrix:")
#         print(confusion_matrix(y_test, y_pred))

#         if acc > best_accuracy:
#             best_accuracy = acc
#             best_model = model

#     # Ensure models directory exists
#     os.makedirs("models", exist_ok=True)

#     # Save best model
#     joblib.dump(best_model, MODEL_PATH)
#     print(f"\nBest model saved at: {MODEL_PATH}")

#     return best_model




# if __name__ == "__main__":
#     from src.data_loader import load_raw_data
#     from src.preprocessing import preprocess_data

#     df = load_raw_data()
#     X, y = preprocess_data(df)

#     train_models(X, y)
