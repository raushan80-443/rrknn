import numpy as np

def generalized_social_distance(x, y, data_set, distance_metric, k=2):
    # Convert data_set to a numpy array if it's not already
    data_set = np.array(data_set)

    # Calculate distances between x and all other points in the data_set using the distance_metric
    distances_x = [distance_metric(x, point) for point in data_set]
    distances_y = [distance_metric(y, point) for point in data_set]

    # Get the index of x and y in the dataset
    x_idx = np.where(np.all(data_set == x, axis=1))[0][0]  # Find the index of x in data_set
    y_idx = np.where(np.all(data_set == y, axis=1))[0][0]  # Find the index of y in data_set

    # Calculate ranks
    mx_y = sum(1 for dx in distances_x if 0 < dx < distances_y[y_idx])
    mx_eq = sum(1 for dx in distances_x if dx == distances_y[y_idx])
    my_x = sum(1 for dy in distances_y if 0 < dy < distances_x[x_idx])
    my_eq = sum(1 for dy in distances_y if dy == distances_x[x_idx])

    # Handle the case where mx_y + mx_eq or my_x + my_eq is zero
    if mx_y + mx_eq == 0 or my_x + my_eq == 0:
        return float('inf')

    # Calculate Lk distance
    lk_distance = (mx_y**k + mx_eq**k) / (mx_y + mx_eq) + (my_x**k + my_eq**k) / (my_x + my_eq)

    # Calculate Generalized Social Distance
    generalized_social_distance = lk_distance / (1 + lk_distance)

    return generalized_social_distance
