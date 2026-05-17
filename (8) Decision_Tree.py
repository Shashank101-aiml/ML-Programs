#Develop a program to load the titanioc dataset.Split tha data into training and test sets. Train a decision tree classifier . Visualize the tree structure. Evaluate accuracy, precision , recall, and F1-score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("titanic.csv")
df.head()

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode() [0] , inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Cabin'].fillna('U', inplace=True)
df.head()

label_encoder=LabelEncoder()
df['Sex']=label_encoder.fit_transform(df['Sex'])
df['Embarked']=label_encoder.fit_transform(df['Embarked'])
df.drop(columns =['Name','Ticket','Cabin'], inplace=True)
X=df.drop(columns=['Survived'])
y=df['Survived']
df.head()+

#train the model
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

#Evaluation metrices
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test,y_pred)
print(f"Accuracy:{accuracy:4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")
print(f"F1-score:{f1:.4f}")

plt.figure(figsize=(12,8))
plot_tree(clf,filled=True,feature_names=X.columns, class_names=['Not Survivded','Survived'],rounded=True)
plt.title("Decision Tree Classifier for Titanic Dataset")
plt.show()
