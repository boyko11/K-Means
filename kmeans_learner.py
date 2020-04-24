import numpy as np


class KMeansLearner:

    def __init__(self):
        self.trained_centroids = None
        self.k = None

    def train(self, data=None, k=2):

        self.k = k
        centroids = self.generate_k_random_centroids(data, k)
        iter_count = 1
        current_centroids = centroids.copy()

        converged = False
        while not converged:

            # for each record calculate euclidean distance to all the centroids
            # for each record assign membership to the the closest centroid
            centroid_membership = self.classify_for_centroid(data, current_centroids)

            # update centroids - the mean of all data records which are members of the centroid cluster
            centroid_indices, counts = np.unique(centroid_membership, return_counts=True)
            centroids_updated = np.empty(shape=current_centroids.shape)
            for centroid_index in centroid_indices:
                record_indices_for_this_cluster = np.where(centroid_membership == centroid_index)
                centroid_updated = np.mean(data[record_indices_for_this_cluster], axis=0)
                centroids_updated[centroid_index, :] = centroid_updated

            # converged if the centroids stop changing
            diff = np.sum(np.abs(current_centroids - centroids_updated))
            current_centroids = centroids_updated.copy()
            if diff == 0:
                print("Converged in {} iterations.".format(iter_count))
                converged = True

            iter_count += 1

        self.trained_centroids = current_centroids

    def classify(self, data):
        return self.classify_for_centroid(data, self.trained_centroids)

    def classify_for_centroid(self, data, centroids):

        record_distances_to_centroids = np.zeros(shape=(data.shape[0], self.k))
        for index, centroid in enumerate(centroids):
            euclidean_distances_to_this_centroid = \
                np.linalg.norm(data - centroid, axis=1)
            record_distances_to_centroids[:, index] = euclidean_distances_to_this_centroid

        # for each record assign membership to the the closest centroid
        return np.argmin(record_distances_to_centroids, axis=1)

    @staticmethod
    def generate_k_random_centroids(data, k):

        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        k_random_centroids = np.empty(shape=(k, data.shape[1]))

        for k_index in range(k):
            k_th_random_centroid = []
            for feature_index in range(means.size):
                random_number_for_this_dimensions = np.random.normal(means[feature_index], stds[feature_index], 1)[0]
                k_th_random_centroid.append(random_number_for_this_dimensions)

            k_random_centroids[k_index, :] = k_th_random_centroid

        return k_random_centroids



