import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


file_path = ".././data/reddit_embeddings_all.csv"
sample_df = pd.read_csv(file_path, nrows=10000)



X = sample_df.select_dtypes(include=['float64', 'int64']).values


k = 10
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
sample_df["cluster"] = kmeans.fit_predict(X)


pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
sample_df["x"] = reduced[:, 0]
sample_df["y"] = reduced[:, 1]


plt.figure(figsize=(10, 8))
plt.scatter(sample_df["x"], sample_df["y"], c=sample_df["cluster"], cmap="tab10", s=8, alpha=0.6)

plt.title("K-Means Clustering of Word Embeddings (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")


centroids_2d = pca.transform(kmeans.cluster_centers_)

for i in range(k):
    cluster_points = sample_df[sample_df["cluster"] == i]
    center_x, center_y = centroids_2d[i]

    distances = ((cluster_points["x"] - center_x)**2 + (cluster_points["y"] - center_y)**2)
    nearest_word = cluster_points.loc[distances.idxmin()]
    label_col = "word" if "word" in sample_df.columns else sample_df.columns[0]
    plt.text(center_x, center_y, nearest_word[label_col],
             fontsize=10, fontweight="bold",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.show()