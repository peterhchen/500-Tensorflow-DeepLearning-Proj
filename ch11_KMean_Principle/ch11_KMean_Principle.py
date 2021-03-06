from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
np.random.seed(42)
digits = load_digits ()
data = scale (digits.data)

k = 10
n_samples, n_features = data.shape
n_digits = len (np.unique (digits.target))
labels = digits.target
sample_size = 300
print("n_figits: %d, \t n_samples %d, \t n_features %d"
  % (n_digits, n_samples, n_features))
print (82 * '_')

print ('init\t\ttime\tinerrtia\thome\tcompl\tv-means\tAMI\tsilhouette')

def bench_k_means (estimator, name, data):
    t0 = time ()
    estimator.fit (data)
    print ('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
    % (name, (time() -t0), estimator.inertia_,
    metrics.homogeneity_score(labels, estimator.labels_),
    metrics.completeness_score (labels, estimator.labels_),
    metrics.v_measure_score (labels, estimator.labels_),
    metrics.adjusted_rand_score (labels, estimator.labels_),
    metrics.adjusted_mutual_info_score (labels, estimator.labels_),
    metrics.silhouette_score(data, estimator.labels_,  
                            metric='euclidean',
                            sample_size=sample_size)))

bench_k_means (KMeans (init='k-means++', n_clusters=n_digits, n_init=10),
            name="k-means++", data=data)
bench_k_means (KMeans (init='random', n_clusters=n_digits, n_init=10),
            name="random", data=data)
print (82 * '_')

# Visualize the result on PCA-reduced data
reduced_data = PCA(n_components=2).fit_transform (data)
kmeans= KMeans (init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# step size of the mesg, Decrease to increase the quality of the VQ.
h = 0.02 # point in the mesh [x_min, x_max]x[y_min, y_max].
# plot the decision boundary. For that, we weill asigna color for each,
x_min, x_max = reduced_data[:, 0].min() -1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() -1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid (np.arange(x_min, x_max, h), np.arange (y_min, y_max, h))

# Obyaon lables for each point in messg. Use last trained model.
Z = kmeans.predict (np.c_[xx.ravel(), yy.ravel()])
# put hte result into a color plot.
Z = Z.reshape (xx.shape)
plt.figure (1)
plt.clf()
plt.imshow(Z, interpolation = 'nearest', 
    extent = (xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:,1], 'k.', markersize=2)
clf = KMeans (n_clusters=k, init="random", n_init=10)
# plot the centroid as a white X
centroids = kmeans.cluster_centers_
plt.scatter (centroids[:, 0], centroids[:, 1],
                marker= 'x', s=169, linewidths=3,
                color='w', zorder=10)
plt.title ('k-mans clustering on the digits dataset (PCA-reduced data)\n'
            'Centroids are marked with white cross')
plt.xlim (x_min, x_max)
plt.ylim (y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()