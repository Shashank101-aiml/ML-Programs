#A program to load the Iris dataset. Implement the k-Nearest Neighbors
#(k-NN) algorithm for classifying flowers based on their features. Split the dataset
#into training and testing sets and evaluate the model using metrics like accuracy
#and F1-score. Test it for different values of k (e.g., k=1,3,5) and evaluate the
#accuracy. Extend the k-NN algorithm to assign weights based on the distance of
#neighbors (e.g., weight=1/d2). Compare the performance of weighted k-NN and regular k-NN on a synthetic or real-world dataset.

#Code:
#import the required libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt

#Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42 )
def evaluate_knn(k_values, X_train, X_test, y_train, y_test, weighted = False):
  results = []
  for k in k_values:
    if weighted:
      knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    else:
      knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append((k, accuracy, f1))
  return results
k_values = [1, 3, 5]
print("Unweighted k-NN:")
unweighted_results = evaluate_knn(k_values, X_train, X_test, y_train, y_test, weighted=False)
for k, accuracy, f1 in unweighted_results:
  print(f"k={k}, Accuracy={accuracy:.4f}, F1-score={f1:.4f}")
print("Weighted k-NN:")
weighted_results = evaluate_knn(k_values, X_train, X_test, y_train, y_test, weighted=True)
for k, accuracy, f1 in weighted_results:
  print(f"k={k}, Accuracy={accuracy:.4f}, F1-score={f1:.4f}")

weighted_accuracies=[]
unweighted_accuracies = []
for k, accuracy, f1 in unweighted_results:
  unweighted_accuracies.append(accuracy) 
for k, accuracy, f1 in weighted_results:
  weighted_accuracies.append(accuracy)   

#Plotting the results
plt.figure(figsize=(10,6))
plt.plot(k_values, unweighted_accuracies, label='Unweighted K-NN', marker = 'o')
plt.plot(k_values, weighted_accuracies, label='Weighted k-NN', marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Accuracy of k-NN with Different k Values')
plt.show()
