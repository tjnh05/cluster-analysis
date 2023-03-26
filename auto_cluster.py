from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import display

colors = ['red', 'blue', 'green', 'orange']

# Load data
data = pd.read_csv('data.csv')

# Preprocessing
X = data.iloc[:, [0, 1]].values

# Determine optimal number of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

index_value = silhouette_scores.index(max(silhouette_scores))
optimal_n_clusters = index_value + 2 if index_value + 2 < len(colors) else len(colors)

# Fit K-Means to the dataset
kmeans = KMeans(n_clusters=optimal_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

c1 = X[y_kmeans == 0]
c2 = X[y_kmeans == 1]

# Visualize the clusters
# ...
for i in range(optimal_n_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
