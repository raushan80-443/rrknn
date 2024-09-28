import numpy as np
from collections import Counter
from .social_distance import generalized_social_distance

class SocialDistanceKNN:
    def __init__(self, k=3, distance_metric=None, generalized=False, power=2):
        """
        k : int, the number of neighbors to consider
        distance_metric : function, the distance function (Euclidean, Manhattan, etc.)
        generalized : bool, whether to use the generalized social distance metric
        power : int, power for the generalized social distance metric
        """
        self.k = k
        self.distance_metric = distance_metric
        self.generalized = generalized
        self.power = power

    def fit(self, X_train, y_train):
        """ Store the training data """ 
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """ Predict labels for the test set X_test """
        predictions = []
        for test_point in X_test:
            # Get the neighbors using the updated predict method with weighted voting
            prediction = self.predict_single(test_point)
            predictions.append(prediction)
        return predictions

    def predict_single(self, sample):
        """Predicts the class of a single sample using social distance and weighted voting."""
        distances = []

        # Calculate distances between the sample and all points in the training set
        for i, x in enumerate(self.X_train):
            if self.generalized:
                # Use generalized social distance
                dist = generalized_social_distance(sample, x, self.X_train, self.distance_metric, self.power)
            else:
                # Use the specified regular distance metric (e.g., Euclidean, Manhattan)
                dist = self.distance_metric(sample, x)
            distances.append((dist, self.y_train[i]))

        # Sort distances in ascending order and get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        closest_classes = [label for _, label in distances[:self.k]]

        # Perform weighted voting (weights are inversely proportional to distance)
        weights = [1 / d[0] if d[0] > 0 else 1 for d in distances[:self.k]]
        weighted_votes = Counter()

        for label, weight in zip(closest_classes, weights):
            weighted_votes[label] += weight

        # Return the class with the highest weighted vote
        prediction = weighted_votes.most_common(1)[0][0]
        return prediction

    def get_neighbors(self, test_point):
        """ Find the k nearest neighbors for the test_point """
        distances = []
        for i, train_point in enumerate(self.X_train):
            if self.generalized:
                # Use generalized social distance
                dist = generalized_social_distance(test_point, train_point, self.X_train, self.distance_metric, self.power)
            else:
                # Use regular distance metric
                dist = self.distance_metric(test_point, train_point)
            distances.append((i, dist))

        # Sort by distance and return the indices of the k smallest distances
        distances.sort(key=lambda x: x[1])
        return [idx for idx, dist in distances[:self.k]]
