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

datafolder = 'data/classification_data/'
exports_folder = 'data/exports/'
fileName = 'Dataset_z_42_tweets.json'
exports_filename = 'clustering_' + fileName + "_"+ time.strftime("%Y%m%d-%H%M%S") + '.csv'
#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)
exports_filepath = os.path.join(exports_folder,exports_filename)

dataset = TweetDataSet(datafolder, fileName)
dataloader = DataLoader(dataset, batch_size=1)
pre_processor = NoiseRemover()

class Baseline_Experiment(Experiment):
    def __init__(self, name="Baseline experiment"):
        clusterer = DBScan.DBScanClusterer()
        embedding_generator = word2vec.word2vec()
        Experiment.__init__(self, name, dataloader, pre_processor, embedding_generator, clusterer, exports_filepath)

    def perform_experiment(self):
        Experiment.perform_experiment(self)
        print("Experiment complete")




experiment = Baseline_Experiment()
experiment.perform_experiment()


# for i in range(50):
#     print(tweets[i].label ,tweets[i].tweet_text)
#
# print(len(tweets))

