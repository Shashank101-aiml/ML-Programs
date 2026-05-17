#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
#Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(scale=0.2, size=X.shape)
X = X[:, np.newaxis]
#Locally weighted Regression algorithm function
def lwr(X_train, y_train, query_point, tau):
  #Compute weights using Gaussian Kernel sampling
  weights = np.exp(-np.sum((X_train - query_point) ** 2, axis=1) / (2 * tau ** 2))
  #Create a diagonal weight matrix
  W = np.diag(weights)
  #Add bias term to X_train
  X_bias = np.hstack([np.ones_like(X_train), X_train])
  #Compute the weighted normal equation to find the paarmeters (theta)
  theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y_train
  #Predict for query points
  query_point_bias = np.array([1, query_point])
  pred = query_point_bias @ theta
  return pred
#Function to make predictions using the trained LWR model
def predict_lwr(X_train, y_train, X_test, tau):
  prediction = np.array([lwr(X_train, y_train, x[0], tau) for x in X_test])
  return prediction
tau = 0.5
y_pred = predict_lwr(X, y, X, tau)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Data Points", color="blue", s=10)
plt.plot(X, y_pred, label=f"LWR Prediction (tau={tau})", color="red",linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Locally Weighted Regression (LWR)")
plt.legend()
plt.grid(True)
plt.show()
