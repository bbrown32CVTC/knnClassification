# This is a Python script for K-Nearest Neighbors Classification Model.
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Create the path to the dataset and set dataframe to the data
path_to_file = "../CSC 419 HW 2 Transaction Data.csv"
df = pd.read_csv(path_to_file)

# Visualize the data using Matplotlib
sns.scatterplot(x=df['distance_from_home'], y=df['price_ratio'], hue=df['fraud'])
plt.show()  # Display scatterplot

# Split the data into features (X) and target (y)
X = df.drop('fraud', axis=1)
y = df['fraud']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model by its accuracy against predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Cross Validation
k_values = [i for i in range(1, 31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))

# Plot Cross Validation Results
sns.lineplot(x=k_values, y=scores, marker='o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()

# Train the model using the best k value
best_index = np.argmax(scores)
best_k = k_values[best_index]

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Evaluate accuracy, precision, and recall
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Best k Value:", best_k)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
