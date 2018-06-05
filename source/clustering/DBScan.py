from sklearn.cluster import DBSCAN
import numpy as np
from clustering import clusterer


class DBScanClusterer(clusterer.Clusterer):
    miles = .75
    kilometers = miles / 0.621371
    eps = kilometers / 10

    def __init__(self):
        clusterer.Clusterer.__init__(self, "DBScan")

    def perform_clustering(self, data):
        clusterer.Clusterer.perform_clustering(self, data)
        dbscan = DBSCAN(eps=self.eps, min_samples=10)
        labels = dbscan.fit_predict(data)
        print ("Number of unique labels:" + str(len(np.unique(labels))))
        return labels