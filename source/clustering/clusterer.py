from abc import ABC, abstractmethod

class Clusterer(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def perform_clustering(self, data, parameters):
        print("Clustering performed by:" + self.name)
        pass