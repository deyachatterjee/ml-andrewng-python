import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


def plotData(X, y):
    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]
    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig, ax)

def costFunction(theta,X,y):
    
    m = len(y) 
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    return (J, grad)

def sigmoid(z):
     return 1.0/(1 +  np.e**(-z))


def predict(theta,X):
    return np.where(np.dot(X,theta) > 5.,1,0)

def mapFeatureVector(X1,X2):
    degree = 6
    output_feature_vec = np.ones(len(X1))[:,None]
    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
   
    return output_feature_vec


def costFunctionReg(theta,X,y,reg_param):
    m = len(y)
    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m +
       (reg_param/m)*np.sum(theta**2)
    # Non-regularized 
    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    # Regularized
    grad_reg = grad_0 + (reg_param/m)*theta
    grad_reg[0] = grad_0[0] 
    
    return J


def plotDecisionBoundary(theta,X,y):
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

    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(mapFeatureVector(np.array([u[i]]),
		      np.array([v[j]])),theta)

    ax.contour(u,v,z,levels=[0])

    return (fig,ax)
	
## Load Data
data = pd.read_csv('ex2data2.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])
fig, ax = plotData(X, y)

ax.legend(['Pass', 'Fail'])

# Labels
ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')
fig.show()

input('\nProgram paused. Press enter to continue.\n')

## Part 1 -- Regularized Logistic Regression
X = mapFeatureVector(X[:,0],X[:,1])
initial_theta = np.zeros(len(X[0,:]))

# Set regularization parameter to 1
reg_param = 1.0

# Optimize for theta letting python choose method
res = minimize(costFunctionReg,
	       initial_theta,
	       args=(X,y,reg_param),
	       tol=1e-6,
	       options={'maxiter':400,
			'disp':True})


theta = res.x
fig.clear()
fig, ax = plotDecisionBoundary(theta,X,y)

ax.legend(['Pass', 'Fail','Decision Boundary'])

# Labels
ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')
ax.set_title('Lambda = 1')

fig.show()

input('\nProgram paused. Press enter to continue.\n')
