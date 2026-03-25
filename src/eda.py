import matplotlib.pyplot as plt
import seaborn as sns


def plot_comfort_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Thermal Comfort Class Distribution")
    plt.xlabel("Comfort Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_temperature_vs_comfort(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x="Thermal sensation",
        y="Air temperature (C)",
        data=df
    )
    plt.title("Air Temperature vs Thermal Sensation")
    plt.tight_layout()
    plt.show()
