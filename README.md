# Clustering Analysis & Outlier Detection Using KMeans, Hierarchical Clustering and DBSCAN

This project performs clustering analysis and outlier detection on a 2D dataset of data points using a combination of KMeans, hierarchical clustering, and DBSCAN. The implementation is done in Python with data visualization, metric evaluation, and multiple methods for identifying outliers.

## 📊 Key Features

- **Data Visualization**: Initial plot of raw data.
- **Optimal Cluster Selection**: Silhouette score evaluation across multiple K values.
- **Hierarchical Clustering**: Single-link and complete-link evaluation using CPCC metric.
- **Dendrogram Analysis**: Visual representation of hierarchical relationships.
- **Combined KMeans & Hierarchical**: Initialization of KMeans with hierarchical centroids.
- **Outlier Detection** using:
  - Minimum distance to other points
  - Distance from cluster centroids
  - DBSCAN density-based approach
- **Final Visualization**: Marking outliers and cluster centers.

## 🧪 Technologies Used

- Python **3.12.3**
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## 📁 Dataset

The file `data.csv` contains the dataset used for analysis. 

## ⚙️ Installation & Execution

To run the project, follow these steps:

1. Create and activate a **virtual environment**.
2. Install the required libraries using:
    - `pip install -r requirements.txt`
3. Ensure the following files are in the **same directory**:
    - `data.py`
    - `data.csv`
    - `requirements.txt`
