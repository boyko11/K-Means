from service.data_service import DataService
import numpy as np
from kmeans_learner import KMeansLearner
from service.plot_service import PlotService


class Runner:

    def __init__(self, normalization_method='z'):
        self.kmeans_learner = None
        self.normalization_method = normalization_method

    def run(self, k=2):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        labels_data = data[:, 1]

        self.kmeans_learner = KMeansLearner()

        normalized_data = DataService.normalize(data, method=self.normalization_method)
        normalized_feature_data = normalized_data[:, 2:]

        self.kmeans_learner.train(normalized_feature_data, k=k)
        cluster_assignments = self.kmeans_learner.classify(normalized_feature_data)

        plot_data = {}

        for cluster_index in range(k):
            cluster_0_assignments_indices = np.where(cluster_assignments == cluster_index)
            data_records_assigned_to_cluster = labels_data[cluster_0_assignments_indices]
            unique_labels, counts_per_label = np.unique(data_records_assigned_to_cluster, return_counts=True)
            plot_data[cluster_index] = counts_per_label

        PlotService().plot_clusters_per_label_barchart(('Benign', 'Malignant'), k, plot_data)


if __name__ == "__main__":

    Runner(normalization_method='z').run(k=2)
