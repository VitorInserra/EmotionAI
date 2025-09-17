
from ML.data_proc import df_relevant, relevant_bands, relevant_sensors
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from ML.unsupervised_ml.utils import plot_pca3d_clusters, plot_pca3d_clusters_colored


target_col = "performance_metric"

k_means = KMeans(n_clusters=2, max_iter=50, tol=0.000001, verbose=1)
features = df_relevant.drop(columns=[target_col, 'start_time']).values
# X_train, X_val = train_test_split(features, test_size=0.15, random_state=42, shuffle=True)
k_means = k_means.fit(features)
all_labels = k_means.predict(features)

df_viz = df_relevant.copy()
df_viz['cluster'] = all_labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ts_col = "start_time"
if df_viz[ts_col].dtype == object:
    df_viz[ts_col] = pd.to_datetime(df_viz[ts_col])

df_viz = df_viz.sort_values(by=ts_col).reset_index(drop=True)



import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 4))

x_vals = np.arange(len(df_viz))
y_vals = df_viz[target_col].values
labels = df_viz['cluster'].values
unique_labels = np.unique(labels)

cmap = cm.get_cmap("tab20", len(unique_labels)) 
cluster_colors = {
    lbl: cmap(i) for i, lbl in enumerate(unique_labels)
}

# Scatter plot
colors = [cluster_colors[lbl] for lbl in labels]
ax.scatter(x_vals, y_vals, c=colors, s=40, alpha=0.8, edgecolors="k")

ax.set_title("Performance over Time with KMeans Cluster Coloring")
ax.set_xlabel("Trial index (equal spacing)")
ax.set_ylabel(target_col)

# Legend
legend_patches = [
    mpatches.Patch(color=cluster_colors[lbl], label=f"Cluster {lbl}")
    for lbl in unique_labels
]
ax.legend(handles=legend_patches, loc="best")

plt.tight_layout()
plt.show()
