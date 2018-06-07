import json
import os
from pprint import pprint
from os import path
from data_processors.tweet_dataset import TweetDataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from entities.tweet import Tweet
from embeddings import word2vec
from clustering import DBScan
from clustering import clusterer
import numpy as np
import csv
import time
from experiment import Experiment
from data_processors.noise_remover import  NoiseRemover
from parameters import Parameters

datafolder = 'data/classification_data/'
exports_folder = 'data/exports/'
fileName = 'Dataset_z_42_tweets.json'

#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)


dataset = TweetDataSet(datafolder, fileName)
dataloader = DataLoader(dataset, batch_size=1)
pre_processor = NoiseRemover()


class Baseline_Experiment(Experiment):
    parameters = Parameters()

    def __init__(self, name="Baseline experiment", eps = 0):
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        exports_filename = 'clustering_' + fileName + "_"+ self.timestamp + '.csv'
        exports_filepath = os.path.join(exports_folder,exports_filename)
        clusterer = DBScan.DBScanClusterer(eps)
        embedding_generator = word2vec.word2vec()
        self.parameters.add_parameter("Input_File_Name", fileName)
        self.parameters.add_parameter("Clusterer", "DBScan")
        self.parameters.add_parameter("eps", eps)
        self.parameters.add_parameter("Embedding", "word2vec")
        Experiment.__init__(self, name, dataloader, pre_processor, embedding_generator, clusterer, exports_filepath, self.parameters)

    def perform_experiment(self):
        Experiment.perform_experiment(self)
        self.parameters.write_parameters(exports_folder, self.timestamp)
        print("Experiment complete")




experiment_count = 10
miles = .75
kilometers = miles / 0.621371
eps = kilometers / 1000
for i in range(experiment_count):
    eps = eps*5
    experiment = Baseline_Experiment(eps = eps)
    experiment.perform_experiment()


# for i in range(50):
#     print(tweets[i].label ,tweets[i].tweet_text)
#
# print(len(tweets))

