import pandas as pd


FEATURE_COLUMNS = [
    "Air temperature (C)",
    "Relative humidity (%)",
    "Air velocity (m/s)",
    "Radiant temperature (C)",
    "Clo",
    "Met"
]

TARGET_COLUMN = "Thermal sensation"


def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare data for ML model training.
    """

    # Select required columns
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Remove missing values
    df.dropna(inplace=True)

    # Ensure target is numeric
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df.dropna(inplace=True)

    # Convert to comfort classes
    df["comfort_class"] = df[TARGET_COLUMN].apply(map_comfort_class)

    X = df[FEATURE_COLUMNS]
    y = df["comfort_class"]

    return X, y


def map_comfort_class(value):
    if value <= -1:
        return "Cold"
    elif value >= 1:
        return "Warm"
    else:
        return "Neutral"


# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# FEATURE_COLUMNS = [
#     "Air temperature (C)",
#     "Relative humidity (%)",
#     "Air velocity (m/s)",
#     "Radiant temperature (C)",
#     "Clo",
#     "Met",
#     "Outdoor monthly air temperature (C)",
#     "PMV",
#     "SET"
# ]

# TARGET_COLUMN = "Thermal sensation"


# def preprocess_data(df: pd.DataFrame):
#     """
#     Clean and prepare data for ML model training.
#     """

#     # Select required columns
#     df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

#     # Remove missing values
#     df.fillna(df.median(numeric_only=True), inplace=True)

#     # Ensure target is numeric
#     df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
#     df.dropna(inplace=True)

#     df["temp_diff"] = df["Air temperature (C)"] - df["Outdoor monthly air temperature (C)"]

#     # Convert to comfort classes
#     df["comfort_class"] = df[TARGET_COLUMN].apply(map_comfort_class)

#     print("\nClass Distribution:")
#     print(df["comfort_class"].value_counts())
    
#     X = df[FEATURE_COLUMNS]
#     from sklearn.preprocessing import LabelEncoder

#     le = LabelEncoder()
#     y = le.fit_transform(df["comfort_class"])

#     # Scaling features
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     return X, y


# def map_comfort_class(value):
#     if value <= -1:
#         return "Cold"
#     elif value >= 1:
#         return "Warm"
#     else:
#         return "Neutral"
