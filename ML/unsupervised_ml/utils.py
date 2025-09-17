import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm

def plot_pca3d_clusters(df_features, kmeans, *, scale=True, random_state=42, title=None):
    """
    3D PCA scatter colored by K-Means cluster labels (unsupervised).
    - df_features: DataFrame with only feature columns. If 'performance_metric' sneaks in, it's dropped.
    - kmeans: optional fitted KMeans. If None, a new one is fit on all rows.
    - n_clusters: used only if kmeans is None.
    - scale: Standardize features before PCA/KMeans (recommended).
    Returns: (pca, X_pca, labels, centers_pca_or_None)
    """
    # Be robust if performance_metric is still present

    X = df_features.values

    # Scale
    scaler = StandardScaler() if scale else None
    Xs = scaler.fit_transform(X) if scaler else X

    km = kmeans
    labels = km.predict(Xs)
    centers_s = getattr(km, "cluster_centers_", None)

    # PCA (fit on the same X used for clustering visualization)
    pca = PCA(n_components=3, random_state=random_state)
    X_pca = pca.fit_transform(Xs)

    # Project centers if available
    centers_pca = pca.transform(centers_s) if centers_s is not None else None

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    c=labels, s=18, alpha=0.9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ttl = title or f"PCA(3D) of features colored by K-Means (k={km.n_clusters})"
    ax.set_title(ttl + f"\nExplained var ≈ {pca.explained_variance_ratio_.sum():.2f}")

    # (Centers will be plotted later per your plan)
    plt.show()

    return pca, X_pca, labels, centers_pca

def plot_pca3d_clusters_colored(X_pca, labels, centers_pca=None, title="PCA 3D Clusters"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = np.unique(labels)
    colors = cm.tab10.colors  # up to 10 distinct colors (can switch to 'tab20' for more)

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   color=colors[i % len(colors)], s=18, alpha=0.9, label=f"cluster {lab}")

    # Optionally plot cluster centers
    if centers_pca is not None:
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2],
                   c="black", s=200, marker="X", edgecolor="k", linewidth=1.5, label="centers")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()
    plt.show()
