import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ex8_utils import *
import scipy.io
import matplotlib.pyplot as plt



def estimateGaussian(X):
	mu = np.mean(X, axis=0, keepdims=True)
	sigma2 = np.var(X, axis=0, keepdims=True)

	return (mu, sigma2)

def multivariateGaussian(X, mu, sigma2):
	k = np.size(mu,1)
	if ((np.size(sigma2,0) == 1) | (np.size(sigma2,1) == 1)):
		sigma2 = np.diagflat(sigma2)

	# De-mean data 
	X = X - mu

	# Calculate p-values of data
	p = ((1 / (2* (np.pi)**(-k / 2) * np.linalg.det(sigma2)**(-.5))) *
		np.exp(-.5 * np.sum(np.dot(X, np.linalg.inv(sigma2)) * X, 1)))

	return p

def visualizeFit(X, mu, sigma2):
	meshvals = np.arange(0, 35, .5)
	X1, X2 = np.meshgrid(meshvals, meshvals)
	Z = np.hstack((X1.reshape((-1,1)), X2.reshape((-1,1))))
	Z = multivariateGaussian(Z, mu, sigma2).reshape(np.shape(X1))

	mylevels = np.array([10**i for i in np.arange(-20,0,3)])
	fig, ax = plt.subplots(1)
	ax.plot(X[:, 0], X[:, 1], 'bx')
	ax.contour(X1, X2, Z, mylevels)

	return fig, ax

def selectThreshold(yval, pval):
	bestEpsilon = 0
	bestF1 = 0
	F1 = 0

	stepsize = (np.max(pval) - np.min(pval)) / 1000
	evals = np.arange(np.min(pval), np.max(pval), stepsize)
	for epsilon in evals:
		predictions = (pval < epsilon).reshape((-1,1))
		X = np.hstack((predictions, yval))
		fp = np.sum((X[:,0] == 1) & (X[:,1] == 0))
		tp = np.sum((X[:,0] == 1) & (X[:,1] == 1))
		fn = np.sum((X[:,0] == 0) & (X[:,1] == 1))
		prec = tp / (tp + fp)
		rec = tp / (tp + fn)
		F1 = (2 * prec * rec) / (prec + rec)

		if F1 > bestF1:
			bestF1 = F1
			bestEpsilon = epsilon

	return (bestEpsilon, bestF1)
	
	
# Part 1 -- Load Example Data
raw_mat = scipy.io.loadmat("ex8data1.mat")
X = raw_mat.get("X")
Xval = raw_mat.get("Xval")
yval = raw_mat.get("yval")

plt.plot(X[:, 0], X[:, 1], 'bx')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)');
plt.show()

# Part 2 -- Estimate the dataset statistics
mu, sigma2 = estimateGaussian(X) # returns flattened arrays

# Density of data based on multivariate normal distribution
p = multivariateGaussian(X, mu, sigma2)

# Visualize the fit
fig, ax = visualizeFit(X,  mu, sigma2)
fig.show()

# Part 3 -- Find Outliers
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)

outliers = np.where(p < epsilon)
fig, ax = visualizeFit(X,  mu, sigma2)
ax.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
fig.show()

# Part 4 -- Multi-Dimensional Outliers
raw_mat2 = scipy.io.loadmat("ex8data2.mat")
X = raw_mat2.get("X")
Xval = raw_mat2.get("Xval")
yval = raw_mat2.get("yval")

mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)