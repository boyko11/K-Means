import numpy as np
from service.plot_service import PlotService


class ReportService:

    def __init__(self):
        pass

    @staticmethod
    def report(cluster_assignments, labels_data, k):

        plot_data = {}

        for cluster_index in range(k):
            cluster_0_assignments_indices = np.where(cluster_assignments == cluster_index)
            data_records_assigned_to_cluster = labels_data[cluster_0_assignments_indices]
            unique_labels, counts_per_label = np.unique(data_records_assigned_to_cluster, return_counts=True)
            plot_data[cluster_index] = counts_per_label

        PlotService().plot_clusters_per_label_barchart(('Benign', 'Malignant'), k, plot_data)
