import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the breast cancer dataset
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

# Precompute the generalized social distance matrix
def compute_distance_matrix(X_train):
    n_samples = X_train.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = generalized_social_distance(X_train[i], X_train[j], data_set=X_train)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix

    return distance_matrix

# Compute the distance matrix for X_train
distance_matrix_train = compute_distance_matrix(X_train)

# Now, we will wrap this distance matrix to use it in the k-NN algorithm.
class PrecomputedDistanceKNN(KNeighborsClassifier):
    def __init__(self, *args, distance_matrix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_matrix = distance_matrix

    def fit(self, X, y):
        self.y_train_ = y
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        n_samples_test = X.shape[0]
        n_samples_train = self.distance_matrix.shape[0]
        distances = np.zeros((n_samples_test, n_samples_train))

        for i, test_sample in enumerate(X):
            for j, train_sample in enumerate(X_train):
                distances[i, j] = generalized_social_distance(test_sample, train_sample, data_set=X_train)

        if return_distance:
            return distances, np.argsort(distances, axis=1)[:, :n_neighbors]
        else:
            return np.argsort(distances, axis=1)[:, :n_neighbors]

# Use the precomputed distance matrix for training
knn = PrecomputedDistanceKNN(n_neighbors=3, distance_matrix=distance_matrix_train)

# Fit the model using the precomputed distance matrix
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)

# Evaluate the model
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_classification_report = classification_report(y_test, y_pred_knn)

# Print the results
print("KNN Accuracy:", knn_accuracy)
print("KNN Confusion Matrix:")
print(knn_confusion_matrix)
print("KNN Classification Report:")
print(knn_classification_report)
