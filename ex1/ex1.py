import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def warmUpExercise():
	return np.eye(5)


def plotData(x, y):
    
    fig, ax = plt.subplots() # create empty figure
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")

    return fig


def normalEqn(X,y):
    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i] = computeCost(X, y, theta)
        print('Cost function: ', J_history[i])
    
    return (theta,J_History)


def gradientDescent(X, y, theta, alpha, num_iters):
    
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i] = computeCost(X, y, theta)
        print('Cost function: ',J_history[i])
    
    return (theta, J_history)


def featureNormalize(X):
    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))


def computeCost(X, y, theta):
    m = len(y)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)
    return J

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

print(warmUpExercise()) 
input('Program paused. Press enter to continue.\n')

print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None] # population in 10,0000
y = np.array(data.y) # profit for a food truck
m = len(y) 
fig = plotData(x,y)
fig.show()
input('Program paused. Press enter to continue.\n')
print('Running Gradient Descent ...\n')
ones = np.ones_like(x) #an array of ones of same dimension as x
X = np.hstack((ones,x)) # Add a column of ones to x. hstack means stacking horizontally i.e. columnwise
theta = np.zeros(2) # initialize
iterations = 1500
alpha = 0.01
computeCost(X, y, theta)
theta, hist = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ')
print(theta[0],"\n", theta[1])

# Plot the linear fit
plt.plot(x,y,'rx',x,np.dot(X,theta),'b-')
plt.legend(['Training Data','Linear Regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5],theta) # takes inner product to get y_bar
print('For population = 35,000, we predict a profit of ', predict1*10000)

predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)
input('Program paused. Press enter to continue.\n');
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J 
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i][j] = computeCost(X,y,t)
"""
# Surface plot using J_Vals
fig = plt.figure()
ax = plt.subplot(111,projection='3d')
Axes3D.plot_surface(ax,theta0_vals,theta1_vals,J_vals,cmap=cm.coolwarm)
plt.show()

fig = plt.figure()
ax = plt.subplot(111)
plt.contour(theta0_vals,theta1_vals,J_vals) 
"""
print('Loading data ...','\n')
print('Plotting Data ...','\n')
data = pd.read_csv("ex1data2.txt",names=["size","bedrooms","price"])
s = np.array(data.size)
r = np.array(data.bedrooms)
p = np.array(data.price)
m = len(r) 
s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))
print('First 10 examples from the dataset: \n')
print(" size = ", s[:10],"\n"," bedrooms = ", r[:10], "\n")
input('Program paused. Press enter to continue.\n')
print('Normalizing Features ...\n')
X = featureNormalize(X)
X = np.hstack((np.ones_like(s),X))

print('Running gradient descent ...\n')
alpha = 0.05
num_iters = 400
theta = np.zeros(3)

# Multiple Dimension Gradient Descent
theta, hist = gradientDescent(X, p, theta, alpha, num_iters)

# Plot convergence graph
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(np.arange(len(hist)),hist ,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


print('Theta computed from gradient descent: \n')
print(theta,'\n')

# Estimate the price of a 1650 sq-ft, 3 br house
#the first column of X is all-ones.it doesnot need to be normalized.
normalized_specs = np.array([1,((1650-s.mean())/s.std()),((3-r.mean())/r.std())])
price = np.dot(normalized_specs,theta) 
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ',
      price)
input('Program paused. Press enter to continue.\n')

print('Solving with normal equations...\n')

data = pd.read_csv("ex1data2.txt",names=["sz","bed","price"])
s = np.array(data.sz)
r = np.array(data.bed)
p = np.array(data.price)
m = len(r) 
s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))
X = np.hstack((np.ones_like(s),X))

theta = normalEqn(X, p)

print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1,1650,3],theta) 


print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): \n',
       price)
