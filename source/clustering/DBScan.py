from sklearn.cluster import DBSCAN
import numpy as np
from clustering import clusterer


class DBScanClusterer(clusterer.Clusterer):
    miles = .75
    kilometers = miles / 0.621371
    eps = kilometers / 10

    def __init__(self, eps):
        clusterer.Clusterer.__init__(self, "DBScan")
        if eps != 0:
            self.eps = eps
        print("EPS:" + str(self.eps))


    def perform_clustering(self, data, parameters):
        clusterer.Clusterer.perform_clustering(self, data, parameters)
        dbscan = DBSCAN(eps=self.eps, min_samples=10)
        labels = dbscan.fit_predict(data)
        print ("Number of unique labels:" + str(len(np.unique(labels))) )
        parameters.add_parameter("Unique_Labels", str(len(np.unique(labels))))
        return labels