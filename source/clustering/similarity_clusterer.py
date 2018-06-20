from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from clustering import clusterer


class SimilarityClusterer(clusterer.Clusterer):

    def __init__(self):
        clusterer.Clusterer.__init__(self, "Similarity")

    def perform_clustering(self, data, parameters):
        clusterer.Clusterer.perform_clustering(self, data, parameters)
        similarities = self.compute_similarity(data)

        parameters.add_parameter("Unique_Labels", 0)

        return similarities

    def compute_similarity(self, vectors):
        similarities = cosine_similarity(vectors)
        averages = np.average(similarities, axis=0)
        return averages

