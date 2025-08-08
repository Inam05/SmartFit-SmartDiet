import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import argparse
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_features(df):
    non_feature_cols = ['subject_id', 'session_type', 'window_start', 'window_end', 'activity']
    X = df.drop(columns=non_feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, X.columns

def find_best_k(X_scaled, k_range=range(2,11)):
    best_k = None
    best_score = -1
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores

def fit_kmeans(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def plot_elbow_silhouette(scores, output_dir):
    k_values, sil_scores = zip(*scores)
    fig, ax1 = plt.subplots(figsize=(10,5))

    color = 'tab:green'
    ax1.plot(k_values, sil_scores, marker='o', color=color)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("KMeans Clustering: Silhouette Scores")

    plt.tight_layout()
    plt_path = os.path.join(output_dir, "kmeans_silhouette_scores.png")
    plt.savefig(plt_path)
    plt.close()
    print(f"Saved silhouette scores plot to {plt_path}")

def plot_clusters_pca(X_scaled, labels, output_dir):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='tab10', legend='full', s=15)
    plt.title(f"KMeans Clusters (k={len(np.unique(labels))}) on PCA projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt_path = os.path.join(output_dir, "kmeans_clusters_pca.png")
    plt.savefig(plt_path)
    plt.close()
    print(f"Saved PCA cluster plot to {plt_path}")

def save_clustered_data(df, labels, output_path):
    df['kmeans_cluster'] = labels
    df.to_csv(output_path, index=False)
    print(f"Saved clustered data to {output_path}")

def save_models(kmeans_model, scaler, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    kmeans_path = os.path.join(model_dir, "kmeans_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(kmeans_model, kmeans_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved KMeans model to {kmeans_path}")
    print(f"Saved scaler to {scaler_path}")

def main(args):
    print("Loading data...")
    df = load_data(args.input_path)

    print("Preprocessing features...")
    X_scaled, scaler, feature_cols = preprocess_features(df)

    print("Finding best k via silhouette score...")
    best_k, scores = find_best_k(X_scaled, k_range=range(2, args.max_k+1))
    print(f"Best k found: {best_k}")

    print("Fitting KMeans with best k...")
    kmeans, labels = fit_kmeans(X_scaled, best_k)

    print("Saving clustered data...")
    save_clustered_data(df, labels, args.output_csv)

    print("Saving models...")
    save_models(kmeans, scaler, args.model_dir)

    print("Plotting results...")
    plot_elbow_silhouette(scores, args.output_dir)
    plot_clusters_pca(X_scaled, labels, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User Profiling Clustering with KMeans")
    parser.add_argument("--input_path", type=str, default="data/processed/pamap2_features.csv",
                        help="Path to features CSV file")
    parser.add_argument("--output_csv", type=str, default="data/processed/pamap2_features_clustered.csv",
                        help="Path to save clustered data CSV")
    parser.add_argument("--output_dir", type=str, default="reports/figures",
                        help="Directory to save plots")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--max_k", type=int, default=10,
                        help="Maximum number of clusters to try")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    main(args)
