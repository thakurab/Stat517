import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


# Defining the dataset
hdf_file1 = '/zdata/abhi/elastic/torch_exercises/c110_742-c120_336-c440_096-c111_000-c121_000-c441_000_000-data.hdf'
hdf_file2 = '/zdata/abhi/elastic/torch_exercises/c110_742-c120_339-c440_128-c111_000-c121_000-c441_000_000-data.hdf'

# Reading the input file and plotting it
f1 = h5py.File(hdf_file1, 'r')
print (f1['composition'].shape)
dataset1 = f1['composition']
dataset1 = np.array(dataset1)

f2 = h5py.File(hdf_file2, 'r')
print (f2['composition'].shape)
dataset2 = f2['composition']
dataset2 = np.array(dataset2)
x = np.array(list(zip(dataset1, dataset2)))

# Agglomerative clustering
cluster = AgglomerativeClustering(n_clusters = 72, affinity = 'euclidean').fit(dataset1)
print (cluster.labels_)

# PCA
pca = PCA().fit(dataset1)
plt.figure(figsize = (12, 12))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative variance')
plt.grid(True)
plt.show()
pca = PCA(n_components = 100) # 100 components will cover more than 90% variance.
pca.fit(dataset1)
x_pca = pca.transform(dataset1)
print (x_pca)

# K-Means clustering
range_n_clusters = np.arange(2, 100, 1)
x = x_pca
s = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters = n_clusters, random_state = 7)
    cluster_labels = clusterer.fit_predict(x)
    sil_avg = silhouette_score(x, cluster_labels)
    s.append(sil_avg)

plt.figure().set_size_inches(10, 6)
plt.plot(range_n_clusters, s)
plt.title('Silhouette average')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette average')
plt.grid(True)
plt.show()

for i in range(len(s)):
    if s[i] == max(s):
        n_clust_km = i+2
        print ('Optimal number of clusters: ', n_clust_km)

range_n_clusters = [n_clust_km]
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 5)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters = n_clusters, random_state = 10)
    cluster_labels = clusterer.fit_predict(x)
    silhouette_avg = silhouette_score(x, cluster_labels)
    sample_silhouette_values = silhouette_samples(x, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor = color, edgecolor = color, alpha = 0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title('Silhouette plot for different clusters')
    ax1.set_xlabel('Silhouette coefficient')
    ax1.set_ylabel('Clusters')
    ax1.axvline(x = silhouette_avg, color = 'red', linestyle = '--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(x[:, 0], x[:, 1], marker = '.', s = 100, lw = 0, alpha = 0.7, c = colors, edgecolor = 'k')
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker = 'o', c = 'white', alpha = 1, s = 200, edgecolor = 'k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker = '$%d$' % i, alpha = 1, s = 50, edgecolor = 'k')

    ax2.set_title('Clustering analysis')
    ax2.set_xlabel('First principle component')
    ax2.set_ylabel('Second principle component')
    plt.suptitle(('Silhouette analysis for KMeans clustering on the microstructural dataset with n_clusters = %d' % n_clusters), fontsize = 16, fontweight = 'bold')
plt.show()



# Hierarchical clustering
H = linkage(dataset1, 'ward')
plt.figure(figsize = (10, 10))
dendro = dendrogram(H, leaf_font_size = 30)
plt.title('Dendrogram on microstructural dataset using ward linkage')
plt.show()

H = linkage(dataset1, 'complete')
plt.figure(figsize = (10, 10))
dendro = dendrogram(H, leaf_font_size = 30)
plt.title('Dendrogram on microstructural dataset using complete linkage')
plt.show()

H = linkage(dataset1, 'single', metric = 'correlation')
plt.figure(figsize = (10, 10))
dendro = dendrogram(H, leaf_font_size = 30)
plt.title('Dendrogram on microstructural dataset using single linkage')
plt.show()

# Distance matrix
dm = squareform(pdist(dataset1))
# For euclidean
h = sns.clustermap(dm, metric = 'euclidean')
plt.show()
# For jaccard
h = sns.clustermap(dm, metric = 'jaccard')
plt.show()
# For correlation
h = sns.clustermap(dm, metric = 'correlation')
plt.show()
# For single
h = sns.clustermap(dm, method = 'single')
plt.show()


# Gaussian mixture 
g_m = GaussianMixture(n_components = 72).fit(x)
labels = g_m.predict(x)
plt.scatter(x[:, 0], x[:, 1], c = labels, s = 20, cmap = 'viridis');
plt.title('Clustering analysis using Gaussian mixture')
plt.show()


# DBSCAN
db = DBSCAN(eps = 0.3, min_samples = 10, metric = 'cosine').fit(x)
core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
plt.scatter(x[:, 0], x[:, 1], c = labels, s = 20, cmap = 'viridis')
plt.title('Clustering analysis using DBSCAN')
plt.show()
