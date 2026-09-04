"""
Carseat Sales Classification using KNN
========================================
Predicts whether carseat sales will be high (Sales > 8) from store
location and demographic data, using a K-Nearest Neighbors classifier.

Also runs a KMeans clustering pass (k=2) on the same feature set to
segment stores, and produces a scatter plot of the resulting clusters.

Usage:
    python src/main.py
    python src/main.py --data data/Carseats.csv --neighbors 7 --clusters 2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def load_data(path: Path) -> pd.DataFrame:
    """Load the Carseats dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Download Carseats.csv (e.g. from the ISLR R package datasets) "
            "and place it in the data/ folder."
        )
    return pd.read_csv(path)


def add_target(df: pd.DataFrame, sales_threshold: float = 8.0) -> pd.DataFrame:
    """Create the binary target 'High': 1 if Sales > threshold else 0."""
    df = df.copy()
    df["High"] = (df["Sales"] > sales_threshold).astype(int)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features (one-hot encoded) and target."""
    X = df.drop(columns=["Sales", "High"])
    y = df["High"]
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_knn(
    X_train, y_train, n_neighbors: int = 7
) -> tuple[KNeighborsClassifier, StandardScaler]:
    """Scale features and fit a KNN classifier."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate(model, scaler, X_test, y_test) -> None:
    """Print accuracy, a classification report, and a confusion matrix."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def run_clustering(df: pd.DataFrame, n_clusters: int = 2, output_path: Path = None):
    """Run KMeans on the store features and plot Price vs Income clusters."""
    X_cluster = df.drop(columns=["Sales", "High"])
    X_cluster = pd.get_dummies(X_cluster, drop_first=True)

    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    df = df.copy()
    df["Cluster"] = clusters

    plt.figure()
    plt.scatter(df["Price"], df["Income"], c=df["Cluster"])
    plt.xlabel("Price")
    plt.ylabel("Regional Income")
    plt.title(f"Store Clusters (k={n_clusters})")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Cluster plot saved to '{output_path}'")
    else:
        plt.show()

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carseat sales KNN classification")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Carseats.csv"),
        help="Path to the Carseats CSV file",
    )
    parser.add_argument(
        "--neighbors", type=int, default=7, help="Number of neighbors for KNN"
    )
    parser.add_argument(
        "--clusters", type=int, default=2, help="Number of clusters for KMeans"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/clusters.png"),
        help="Where to save the cluster plot (omit --plot-output to show it instead)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data(args.data)
    df = add_target(df)

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model, scaler = train_knn(X_train, y_train, n_neighbors=args.neighbors)
    evaluate(model, scaler, X_test, y_test)

    run_clustering(df, n_clusters=args.clusters, output_path=args.plot_output)


if __name__ == "__main__":
    main()
