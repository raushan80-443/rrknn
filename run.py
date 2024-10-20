import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load the iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# Load the breast cancer

data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0: malignant, 1: benign)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset (scaling features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the generalized social distance function
def generalized_social_distance(x, y, data_set=None, k=2):
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    # Calculate distances between x and all other points in the data_set using the distance_metric
    distances_x = [euclidean_distance(x, point) for point in data_set]
    distances_y = [euclidean_distance(y, point) for point in data_set]

    # Calculate ranks
    mx_y = sum(1 for dx in distances_x if 0 < dx < euclidean_distance(x, y))
    mx_eq = sum(1 for dx in distances_x if dx == euclidean_distance(x, y))
    my_x = sum(1 for dy in distances_y if 0 < dy < euclidean_distance(x, y))
    my_eq = sum(1 for dy in distances_y if dy == euclidean_distance(x, y))

    # Handle the case where mx_y + mx_eq or my_x + my_eq is zero
    if mx_y + mx_eq == 0 or my_x + my_eq == 0:
        return float('inf')

    # Calculate Lk distance
    lk_distance = (mx_y**k + mx_eq**k) / (mx_y + mx_eq) + (my_x**k + my_eq**k) / (my_x + my_eq)

    # Calculate Generalized Social Distance
    generalized_social_distance = lk_distance / (1 + lk_distance)

    return generalized_social_distance

# Wrap the distance function to pass to KNeighborsClassifier
def generalized_social_distance_wrapper(x, y):
    return generalized_social_distance(x, y, data_set=X_train)

# Initialize the KNeighborsClassifier with the custom metric


knn = KNeighborsClassifier(n_neighbors=3, metric=generalized_social_distance_wrapper)

# knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Evaluate KNN
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_classification_report = classification_report(y_test, y_pred_knn)

# Print the results
print("KNN Accuracy:", knn_accuracy)
print("KNN Confusion Matrix:")
print(knn_confusion_matrix)
print("KNN Classification Report:")
print(knn_classification_report)