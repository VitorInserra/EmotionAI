
from ML.data_proc import df_relevant, relevant_bands, relevant_sensors
from sklearn.cluster import KMeans


target_col = "performance_metric"

X = df_relevant.drop(columns=[target_col]).values


k_means = KMeans(n_clusters=3, max_iter=300, tol=0.0001, verbose=1)

k_means = k_means.fit(X)

print(k_means)