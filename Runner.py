from service.data_service import DataService
from kmeans_learner import KMeansLearner
from service.report_service import ReportService


class Runner:

    def __init__(self, normalization_method='z'):
        self.kmeans_learner = None
        self.normalization_method = normalization_method
        self.report_service = ReportService()

    def run(self, k=2):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        labels_data = data[:, 1]

        self.kmeans_learner = KMeansLearner()

        normalized_data = DataService.normalize(data, method=self.normalization_method)
        normalized_feature_data = normalized_data[:, 2:]

        self.kmeans_learner.train(normalized_feature_data, k=k)
        cluster_assignments = self.kmeans_learner.classify(normalized_feature_data)

        self.report_service.report(cluster_assignments, labels_data, k)


if __name__ == "__main__":

    Runner(normalization_method='z').run(k=2)
