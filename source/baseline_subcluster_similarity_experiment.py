import json
import os
from pprint import pprint
from os import path
from data_processors.tweet_lda_dataset import TweetLDADataSet
from data_processors.cluster_selector import ClusterSelector
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from entities.tweet import Tweet
from embeddings import word2vec
from clustering import DBScan
from clustering import clusterer
from clustering.similarity_clusterer import SimilarityClusterer
import numpy as np
import csv
import time
from subcluster_experiment import SubclusterExperiment
from data_processors.noise_remover import  NoiseRemover
from parameters import Parameters

datafolder = 'data/exports/'
exports_folder = 'data/exports/sub-clusters'
fileName = 'clustering_Dataset_z_7_tweets.json_20180615-013941.csv'

#fileName = 'junk.json'
experiment_datafolder = time.strftime("%Y%m%d-%H%M%S")
filepath = os.path.join(datafolder,fileName)


dataset = TweetLDADataSet(datafolder, fileName)
dataloader = DataLoader(dataset, batch_size=1)
pre_processor = NoiseRemover()


class Baseline_Subcluster_Similarity_Experiment(SubclusterExperiment):
    parameters = Parameters()

    def __init__(self, name="Baseline subcluster experiment", eps = 0, cluster_number = 99,
                 experiment_folder = "", timestamp = ""):
        self.timestamp = timestamp
        self.experiment_folder = experiment_folder
        exports_filename = str(cluster_number) +'-sub_cluster_similarity' + fileName + '.csv'
        exports_filepath = os.path.join(self.experiment_folder,exports_filename)
        #clusterer = DBScan.DBScanClusterer(eps)
        clusterer = SimilarityClusterer()
        embedding_generator = word2vec.word2vec()
        cluster_selector = ClusterSelector(cluster_number)
        self.parameters.add_parameter("Input_File_Name", fileName)
        self.parameters.add_parameter("Clusterer", "DBScan")
        self.parameters.add_parameter("eps", eps)
        self.parameters.add_parameter("Embedding", "word2vec")
        self.parameters.add_parameter("Experiment_Type", "sub_clustering_experiment")
        self.parameters.add_parameter("Cluster_number", cluster_number)
        SubclusterExperiment.__init__(self, name, dataloader, pre_processor, embedding_generator, clusterer,
                                      exports_filepath, self.parameters,selector=cluster_selector)

    def perform_experiment(self):
        SubclusterExperiment.perform_experiment(self)
        self.parameters.write_parameters(self.experiment_folder, self.timestamp)
        print("Experiment complete")





total_clusters = 26
# for i in range(experiment_count):
#     eps = eps*10
#
#     for cluster_label in range(total_clusters):
#         print("Cluster number:",cluster_label)
#         experiment = Baseline_Subcluster_Experiment(eps = eps, cluster_number = cluster_label)
#         experiment.perform_experiment()

eps = 0
timestamp = time.strftime("%Y%m%d-%H%M%S")
experiment_folder = os.path.join(exports_folder,timestamp)
os.mkdir(experiment_folder)
for cluster_label in range(total_clusters):
        print("Cluster number:",cluster_label)
        experiment = Baseline_Subcluster_Similarity_Experiment(eps = eps, cluster_number = cluster_label, experiment_folder=experiment_folder, timestamp = timestamp)
        experiment.perform_experiment()

# for i in range(50):
#     print(tweets[i].label ,tweets[i].tweet_text)
#
# print(len(tweets))

