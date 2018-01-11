import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex7_utils import *
import scipy.io
import matplotlib.pyplot as plt



def findClosestCentroids(X, centroids):
	K = np.size(centroids, 1)
	idx = []

	for i in range(len(X)):
		norm = np.sum(((X[i] - centroids)**2), axis=1)
		idx.append(norm.argmin())
		
	return idx

def computeCentroids(X, idx, K):
	centroid = np.zeros((K,np.size(X,1)))
	aug_X = np.hstack((np.array(idx)[:,None],X))
	for i in range(K):
		centroid[i] = np.mean(X[aug_X[:,0] == i], axis=0)
	
	return centroid

def runKMeans(X, initial_centroids, max_iters, plot_progress=False):
	K = np.size(initial_centroids, 0)
	centroids = initial_centroids 
	previous_centroids = centroids

	for i in range(max_iters):
		# Centroid assignment
		idx = findClosestCentroids(X, centroids)

		if plot_progress:
			plt.plot(X[:,0],X[:,1], 'bo')
			plt.plot(centroids[:,0], centroids[:,1], 'rx')
			plt.plot(previous_centroids[:,0], previous_centroids[:,1], 'gx')
			plt.show()

			previous_centroids = centroids
			centroids = computeCentroids(X, idx, K)

	return (centroids, idx)
	
def displayData(X):
    """
    Displays 2D data stored in design matrix in a nice grid.
    """
    num_images = len(X)
    rows = int(num_images**.5)
    cols = int(num_images**.5)
    fig, ax = plt.subplots(rows,cols,sharex=True,sharey=True)
    img_num = 0

    for i in range(rows):
        for j in range(cols):
            # Convert column vector into 32x232 pixel matrix
            # You have to transpose to have them display correctly
            img = X[img_num,:].reshape(32,32).T
            ax[i][j].imshow(img,cmap='gray')
            img_num += 1

    return (fig, ax)

def kMeansInitCentroids(X, K):
	return X[np.random.choice(X.shape[0], K)]

# Part 1 -- Find Closest Centroids
raw_mat = scipy.io.loadmat("ex7data2.mat")
X = raw_mat.get("X")

# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, initial_centroids)

# Part 2 -- Compute Means
centroids = computeCentroids(X, idx, K)

# Part 3 -- K-means Clustering
max_iters = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
centroids, idx = runKMeans(X, initial_centroids, max_iters, plot_progress=True)

# Part 4 -- K-means Clustering on Pixels
A = plt.imread("bird_small.png")
plt.imshow(A)
plt.show()

original_shape = np.shape(A)

# Reshape A to get R, G, B values for each pixel
X = A.reshape((np.size(A, 0)*np.size(A, 1), 3))
K = 16
max_iters = 10

# Initialize centroids
initial_centroids = kMeansInitCentroids(X, K)

# Run K-means
centroids, idx = runKMeans(X, initial_centroids, max_iters, plot_progress=False)

# Part 5 -- Image Compression
idx = findClosestCentroids(X, centroids)
X_recovered = centroids[idx,:]
X_recovered = X_recovered.reshape(original_shape)

# Display Images 
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.imshow(A)
ax2.imshow(X_recovered)
