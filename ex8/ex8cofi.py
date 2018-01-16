import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt



def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, reg):
	# Unfold the U and W matrices from params
	X = params[:num_movies * num_features].reshape((num_movies, num_features))
	Theta = params[num_movies * num_features:].reshape((num_users, num_features))
	
	# Cost
	J = (.5 * np.sum(((np.dot(Theta,X.T).T - Y) * R)**2) + 
	    ((reg / 2) * np.sum(Theta**2)) +
	    ((reg / 2) * np.sum(X**2)))
	
	# Gradients
	X_grad = np.zeros_like(X)
	for i in range(num_movies):
		idx = np.where(R[i,:]==1)[0] # users who have rated movie i
		temp_theta = Theta[idx,:] # parameter vector for those users 
		temp_Y = Y[idx, :] # ratings given to movie i
		X_grad[i,:] = np.sum(np.dot(np.dot(temp_theta, X[i, :]) - temp_Y.T,
		    temp_theta) + reg*X[i,:], axis=0)

	Theta_grad = np.zeros_like(Theta)
	for j in range(num_users):
		idx = np.where(R[:,j]==1)[0]
		temp_X = X[idx,:]
		temp_Y = Y[idx,j]
		Theta_grad[j,:] = np.sum(np.dot(np.dot(Theta[j], temp_X.T) -
		    temp_Y, temp_X) + reg*Theta[j], axis=0) 
	grad = np.append(X_grad.flatten(), Theta_grad.flatten())
	
	return (J, grad)

def computeNumericalGradient(J,theta):
    
    
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4
    
    for p in range(len(theta)):
        
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1)/(2 * tol)
        perturb[p] = 0

    return numgrad

def checkCostFunction(reg):
    # Create small problem
    X_t = np.random.random((4,3))
    Theta_t = np.random.random((5,3))

    # Zap out most entries
    Y = np.dot(Theta_t, X_t.T)
    Y[(np.random.random(np.shape(Y)) > .5)] = 0
    R = np.zeros_like(Y)
    R[Y != 0] = 1

    # gradient checking
    X = np.random.random(np.shape(X_t))
    Theta = np.random.random(np.shape(Theta_t))
    num_users = np.size(Y, 1)
    num_movies = np.size(Y,0)
    num_features = np.size(Theta_t,1)

    params = np.append(X.flatten(), Theta.flatten())
    
    def reducedCofiCostFunc(p):
        
        return cofiCostFunc(p,Y, R, num_users, num_movies, num_features,0)[0]

    numgrad = computeNumericalGradient(reducedCofiCostFunc,params)
    J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)  
    # Check two gradients
    np.testing.assert_almost_equal(grad, numgrad)

    return

def normalizeRatings(Y, R):
    m, n = np.shape(Y)
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros_like(Y)
    for i in range(m):
        idx = (R[i] == 1)
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]

    return (Ynorm, Ymean)

raw_mat = scipy.io.loadmat("ex8_movies.mat")
R = raw_mat.get("R") # num movies x num users indicator matrix
Y = raw_mat.get("Y") # num movies x num users ratings matrix

# Visualize 
plt.matshow[.]
plt.xlabel("Users")
plt.ylabel("Movies")
plt.show()

# Collaborative Filtering Cost Function
raw_mat2 = scipy.io.loadmat("ex8_movieParams.mat")
X = raw_mat2.get("X") # rows correspond to feature vector of the ith movie 
Theta = raw_mat2.get("Theta") # rows are the parameter vector for jth user

# Reduce data size to have it run faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# Evaluate Cost 
params = np.append(X.flatten(), Theta.flatten())
J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
np.testing.assert_almost_equal(22.22, J,decimal=2, err_msg="Incorrect unregularized error")

# Gradient
checkCostFunction(0)

#Regularization
J, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
np.testing.assert_almost_equal(31.34, J,decimal=2, 
    err_msg="Incorrect regularized cost")

checkCostFunction(1.5)

# Entering ratings for a new users
movieList = pd.read_table("movie_ids.txt",encoding='latin-1',names=["Movie"])
movies = movieList.Movie.tolist()
my_ratings = [0]*len(movies)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991),set
my_ratings[97] = 2

# selected a few movies liked / did not like 
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53]= 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68]= 5
my_ratings[182]= 4
my_ratings[225]= 5
my_ratings[354]= 5

for i in range(len(movies)):
    if my_ratings[i] > 0:
        print("User rated " + str(movies[i]) + ": " + str(my_ratings[i]))

# Learning
raw_mat = scipy.io.loadmat("ex8_movies.mat")
R = raw_mat.get("R") # num movies x num users indicator matrix
Y = raw_mat.get("Y") # num movies x num users ratings matrix

# Add own ratings to Y
ratings_col = np.array(my_ratings).reshape((-1,1))
Y = np.hstack((ratings_col, Y))

# Add indicators to R
R = np.hstack((ratings_col !=0, R))

# Normalize 
Ynorm, Ymean = normalizeRatings(Y,R)

# Useful values
num_users = np.size(Y,1)
num_movies = np.size(Y,0)
num_features = 10

# Set initial parameters
X = np.random.normal(size=(num_movies, num_features))
Theta = np.random.normal(size=(num_users, num_features))

initial_parameters = np.append(X.flatten(), Theta.flatten())
reg = 10

def reducedCofiCostFunc(p):
    
    return cofiCostFunc(p,Y, R, num_users, num_movies, num_features,reg)

results = minimize(reducedCofiCostFunc,
                   initial_parameters,
		   method="CG",
                   jac=True,
                   options={'maxiter':100, "disp":True})

out_params = results.x

# Unfold the returned parameters back into X and Theta
X = np.reshape(out_params[:num_moves*num_features], (num_movies, num_features))
Theta = np.reshape(out_params[num_movies*num_features:],
    (num_users,num_features))

# Recommendation
p = np.dot(X, Theta.T)
my_predictions = p[:,0] + Ymean.T.flatten()
sorted_predictions = np.sort(my_predictions)
sorted_ix = my_predictions.ravel().argsort()

print("\nTop recommendations for you:\n")
for i in range(10):
    j = sorted_ix[-i]
    print("Predicting rating " + str(my_predictions[j]) + 
	" for movie " + str(movies[j]))

print("\n Original ratings provided: \n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated " + str(my_ratings[i]) + " for " + str(movies[i]))
