# Carseat Sales Classification using KNN

This project analyzes store location and demographic data to predict whether carseat sales will be high (`Sales > 8`) using a **K-Nearest Neighbors (KNN)** classification model. It also includes a **KMeans** clustering pass to segment stores by profile.

## 📌 Features
- Data preprocessing & feature encoding using `pandas.get_dummies`
- Feature scaling with `StandardScaler`
- Supervised classification using `KNeighborsClassifier` (K=7)
- Unsupervised segmentation using `KMeans` (k=2) with a Price vs. Income scatter plot
- Command-line arguments to tweak neighbors, clusters, split size, and seed

## 📁 Dataset
The dataset contains 400 store observations across 11 variables, including unit sales, competitor prices, income levels, advertising budgets, population, price, and store shelving locations.

Place `Carseats.csv` in the `data/` folder before running the script (the file is not included in this repo — see `.gitignore`).

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/carseat-sales-classification.git
   cd carseat-sales-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the classification pipeline:**
   ```bash
   python src/main.py
   ```

   Optional flags:
   ```bash
   python src/main.py --data data/Carseats.csv --neighbors 7 --clusters 2
   ```

## 📂 Project Structure
```
carseat-sales-classification/
│
├── data/
│   └── Carseats.csv
├── src/
│   └── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📈 Output
The script prints accuracy, a classification report, and a confusion matrix for the KNN model, then saves a cluster scatter plot to `outputs/clusters.png`.
