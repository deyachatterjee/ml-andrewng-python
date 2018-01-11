import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


def plotData(X, y):

    """
    Plots the data points X and y into a new figure with + for the positive
    examples and 0 for the negative examples. X is assumed to be an Mx2 array.
    """

    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]

    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")

    return (fig, ax)

def costFunction(theta,X,y):
    """
    Computes loss using sum of square errors for logistic regression
    using theta as the parameter vector for linear regression to fit 
    the data points in X and y.
    """

    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)

    # Gradient
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
  
    return (J, grad)

def sigmoid(z):
    """ Returns the value from using x in the sigmoid function """
    
    return 1.0/(1 +  np.e**(-z))


def predict(theta,X):
    """
    Given a vector of parameter results and training set X,
    returns the model prediction for admission. If predicted
    probability of admission is greater than .5, predict will
    return a value of 1.
    """
    return np.where(np.dot(X,theta) > 5.,1,0)

def mapFeatureVector(X1,X2):
    """
    Feature mapping function to polynomial features. Maps the two features
    X1,X2 to quadratic features used in the regularization exercise. X1, X2
    must be the same size.
    Returns -- New feature array with interactions and quadratic terms
    """
    
    degree = 6
    output_feature_vec = np.ones(len(X1))[:,None]

    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
   
    return output_feature_vec


def costFunctionReg(theta,X,y,reg_param):
    """
    Computes loss using sum of square errors for logistic regression
    using theta as the parameter vector for linear regression to fit 
    the data points in X and y with penalty reg_param (lambda).
    """

    # Initialize some useful values
    m = len(y) # number of training examples
    
    # Cost function J(theta)
    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m +
       (reg_param/m)*np.sum(theta**2))

    # Gradient
    
    # Non-regularized 
    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    
    # Regularized
    grad_reg = grad_0 + (reg_param/m)*theta


    # Replace gradient for theta_0 with non-regularized gradient
    grad_reg[0] = grad_0[0] 
    
    return J


def plotDecisionBoundary(theta,X,y):
    """
    Plots the data points X and y into a new figure with the decision boundary
    defined by theta with + for the positive examples and o for the negative
    examples. X is asssumed to be either:
        1) Mx3 matrix where the first column is all ones for the intercept
        2) MxN with N>3, where the first column is all ones
    """

    # Start with the same plot as before
    fig, ax = plotData(X[:,1:],y)
    
    """
    if len(X[0]<=3):
        # Choose two endpoints and plot the line between them
        plot_x = np.array([min(X[:,1])-2,max(X[:,2])+2])
        # Calculate the decision boundary line
        # Add boundary and adjust axes
        ax.plot(plot_x,plot_y)
        ax.legend(['Admitted','Fail','Pass'])
        ax.set_xbound(30,100)
        ax.set_ybound(30,100)
    else:
    """

    # Create grid space
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    
    # Evaluate z = theta*x over values in the gridspace
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(mapFeatureVector(np.array([u[i]]),
		      np.array([v[j]])),theta)
    
    # Plot contour
    ax.contour(u,v,z,levels=[0])

    return (fig,ax)
	
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('ex2data1.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print("Plotting data with + indicating (y = 1) examples and o indicating",
" (y =0) examples.")

fig, ax = plotData(X, y)

# Specified in plot order
ax.legend(['Admitted', 'Not admitted'])


fig.show()

input('\nProgram paused. Press enter to continue.\n')

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term

# Add intercept term to x and X_test
X = np.hstack((np.ones_like(y)[:,None],X))

# Initialize fitting parameters
initial_theta = np.zeros(3)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): \n', cost)
print('Gradient at initial theta (zeros): \n',grad)

input('\nProgram paused. Press enter to continue.')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

res = minimize(costFunction,
	       initial_theta,
	       method='Newton-CG',
	       args=(X,y),
	       jac=True, 
	       options={'maxiter':400,
			'disp':True})

theta = res.x

# Print theta to screen
print('Cost at theta found by minimize: \n', res.fun)
print('theta: \n', theta)


# Plot Boundary
plotDecisionBoundary(theta, X, y)


input('\nProgram paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.dot([1,45,85],theta))
print('For a student with scores 45 and 85, we predict an ',
      'admission probability of ', prob)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: \n', np.mean(p==y)*100)

input('Program paused. Press enter to continue.\n')