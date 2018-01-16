import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from functions import (
    gaussian_kernel, dataset3_params
)

#use  cross validation set Xval, yval to determine best C and σ 

def dataset3_params(X, y, Xval, yval):
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))

    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            svm = SVC(kernel='rbf', C=C_vec[i], gamma=sigma_vec[j])
            svm.fit(X, y.ravel())
            scores[i, j] = accuracy_score(yval, svm.predict(Xval))

    max_c_index, max_s_index = np.unravel_index(scores.argmax(), scores.shape)
    return (C_vec[max_c_index], sigma_vec[max_s_index])
	
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex6data1.mat')
X = data['X']  # 51 x 2 matrix
y = data['y']  # 51 x 1 matrix

pos = (y == 1).ravel()         #flattens i.e. makes 1d array
neg = (y == 0).ravel()         #alternative code neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()
print('Training Linear SVM ...\n')
C = 1  #default
svm = SVC(kernel='linear', C=C)
svm.fit(X, y.ravel())
weights = svm.coef_[0]
intercept = svm.intercept_[0]
#draw svm boundary
xp = np.linspace(X.min(), X.max(), 100)
yp = - (weights[0] * xp + intercept) / weights[1]

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.plot(xp, yp)
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

#Gaussian Kernel
print('Evaluating the Gaussian Kernel ...\n')

#linalg.norm returns one of seven different matrix norms
def gaussian_kernel(x1, x2, sigma):
    return np.exp(- (np.linalg.norm(x1 - x2) ** 2).sum() / (2 * (sigma ** 2)))


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)



print(
    'Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {0} :\n'
    .format(sigma),
    '\t{0:.6f}\n(for sigma = 2, this value should be about 0.324652)'
    .format(sim))

input('Program paused. Press enter to continue.\n')
plt.close()

#Visualizing Dataset 2 
data = sio.loadmat('ex6data2.mat')
X = data['X']  # 863 x 2 matrix
y = data['y']  # 863 x 1 matrix

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

#RBF Kernel (Dataset 2)
print('Training SVM with RBF Kernel ...\n')

C = 30
sigma = 30

svm = SVC(kernel='rbf', C=C, gamma=sigma)
svm.fit(X, y.ravel())

x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1, x2)
yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.contour(x1, x2, yp)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# Visualizing Dataset 3 
data = sio.loadmat('ex6data3.mat')
X = data['X']  # 211 x 2 matrix
y = data['y']  # 211 x 1 matrix
Xval = data['Xval']  # 200 x 2 matrix
yval = data['yval']  # 200 x 1 matrix

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(-0.6, 0.3)
plt.ylim(-0.8, 0.6)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()

# RBF Kernel (Dataset 3)
C, sigma = dataset3_params(X, y, Xval, yval)

svm = SVC(kernel='rbf', C=C, gamma=sigma)
svm.fit(X, y.ravel())

x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1, x2)
yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

pos = (y == 1).ravel()
neg = (y == 0).ravel()
plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
plt.xlim(-0.6, 0.3)
plt.ylim(-0.8, 0.6)
plt.contour(x1, x2, yp)
plt.show()

input('Program paused. Press enter to continue.\n')
plt.close()
