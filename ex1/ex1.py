import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def warmUpExercise():
    """
    Example Function in Python
    Instructions: Return the 5x5 identity matrix. In Python,
                  define the object to be returned within the
		  function, and then return it at the bottom
		  with the "return" statement.
    """
    return np.eye(5)


def plotData(x, y):
    """
    plotData -- Plots the data points x and y into a new figure and gives
                the figure axes labels of population and profit. It returns
		at matplotlib figure.
     Instructions: Plot the training data into a figure by manipulating the
		   axes object created for you below. Set the axes labels using
                   the "xlabel" and "ylabel" commands. Assume the 
                   population and revenue data have been passed in
                   as the x and y arguments of this function.
    
     Hint: You can use the 'rx' option with plot to have the markers
           appear as red crosses. Furthermore, you can make the
           markers larger by using plot(..., 'rx', markersize=10)
    """
    
    fig, ax = plt.subplots() # create empty figure and set of axes
    ax.plot(x,y,'rx',markersize=10)
    ax.set_xlabel("Population of City in 10,000s")
    ax.set_ylabel("Profit in $10,000s")

    return fig


def normalEqn(X,y):
    """
    Computes the closed form least squares solution using normal
    equations
    theta = (X^T*X)^{-1}X^T*y
    Returns: Array of least-squares parameters
    """
    
    return np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    
    Performs gradient descent to learn theta by taking n_iters steps and
    updating theta with each step at a learning rate alpha
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)

	# Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)
        print('Cost function has a value of: ', J_history[i])
    
    return (theta,J_History)


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    
    Performs gradient descent to learn theta by taking n_iters steps and
    updating theta with each step at a learning rate alpha
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)

	# Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)
        print('Cost function as a value of: ',J_history[i])
    
    return (theta, J_history)


def featureNormalize(X):
    """
    Normalizes (mean=0, std=1) the features in design matrix X
    returns -- Normalized version of X where the mean of each
               value of each feature is 0 and the standard deviation
	       is 1. This will often help gradient descent learning
	       algorithms to converge more quickly.
    Instructions: First, for each feature dimension, compute the mean
                  of the feature and subtract it from the dataset,
		  storing the mean value in mu. Next, compute the 
		  standard deviation of each feature and divide
		  each feature by it's standard deviation, storing
		  the standard deviation in sigma. 
		  
		  Note that X is a matrix where each column is a 
		  feature and each row is an example. You need 
		  to perform the normalization separately for 
		  each feature. 
		  
		  Hint: You might find the 'mean' and 'std' functions useful.
    """
    
    return np.divide((X - np.mean(X,axis=0)),np.std(X,axis=0))


def computeCost(X, y, theta):
    """
    
    Compute cost using sum of square errors for linear 
    regression using theta as the parameter vector for 
    linear regression to fit the data points in X and y.
    Note: Requires numpy in order to run, but is not imported as part of this
	  script since it is imported in ex1.py and therefore numpy is part of
	  the namespace when the function is actually run.
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J = (np.sum((np.dot(X,theta) - y)**2))/(2*m)

    return J
	
	
# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

print(warmUpExercise()) # Prints object returned by WarmUpExercise 
input('Program paused. Press enter to continue.\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None] # population in 10,0000
y = np.array(data.y) # profit for a food truck
m = len(y) # number of training examples

# Plot Data
fig = plotData(x,y)
fig.show()

input('Program paused. Press enter to continue.\n')

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

ones = np.ones_like(x)
X = np.hstack((ones,x)) # Add a column of ones to x
theta = np.zeros(2) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta, hist = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
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


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J 
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

# Fill out J_Vals 
# Note: There is probably a more efficient way to do this that uses
#	broadcasting instead of the nested for loops
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i][j] = computeCost(X,y,t)


# Surface plot using J_Vals
fig = plt.figure()
ax = plt.subplot(111,projection='3d')
Axes3D.plot_surface(ax,theta0_vals,theta1_vals,J_vals,cmap=cm.coolwarm)
plt.show()

# Contour plot
# TO DO: Currently does not work as expected. Need to find a way to mimic
#	 the logspace option in matlab
fig = plt.figure()
ax = plt.subplot(111)
plt.contour(theta0_vals,theta1_vals,J_vals) 



#----------------------------------------------------------
#Linear regression with multiple variables

## ================ Part 1: Feature Normalization ================

print('Loading data ...','\n')

## Load Data
print('Plotting Data ...','\n')

data = pd.read_csv("ex1data2.txt",names=["size","bedrooms","price"])
s = np.array(data.size)
r = np.array(data.bedrooms)
p = np.array(data.price)
m = len(r) # number of training examples

# Design Matrix
s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))

# Print out some data points
print('First 10 examples from the dataset: \n')
print(" size = ", s[:10],"\n"," bedrooms = ", r[:10], "\n")

input('Program paused. Press enter to continue.\n')

# Scale features to zero mean and standard deviation of 1
print('Normalizing Features ...\n')

X = featureNormalize(X)

# Add intercept term to X
X = np.hstack((np.ones_like(s),X))

## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.05
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)

# Multiple Dimension Gradient Descent
theta, hist = gradientDescent(X, p, theta, alpha, num_iters)

# Plot the convergence graph
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(np.arange(len(hist)),hist ,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(theta,'\n')

# Estimate the price of a 1650 sq-ft, 3 br house

# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
normalized_specs = np.array([1,((1650-s.mean())/s.std()),((3-r.mean())/r.std())])
price = np.dot(normalized_specs,theta) 


print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ',
      price)

input('Program paused. Press enter to continue.\n')

## ================ Part 3: Normal Equations ================
print('Solving with normal equations...\n')

data = pd.read_csv("ex1data2.txt",names=["sz","bed","price"])
s = np.array(data.sz)
r = np.array(data.bed)
p = np.array(data.price)
m = len(r) # number of training examples

# Design Matrix
s = np.vstack(s)
r = np.vstack(r)
X = np.hstack((s,r))

# Add intercept term to X
X = np.hstack((np.ones_like(s),X))

# Calculate the parameters from the normal equation
theta = normalEqn(X, p)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1,1650,3],theta) # You should change this


print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): \n',
       price)