import json
import os
from pprint import pprint
from os import path

datafolder = 'data/classification_data/'
fileName = 'Dataset_z_1024_tweets.json'
#fileName = 'junk.json'
filepath = os.path.join(datafolder,fileName)

from collections import namedtuple
Metro = namedtuple('Tweet', 'text, time')

#metros = [Metro(**k) for k in data["metros"]]

with open(filepath) as f:
    content = f.read()
    content = content.replace("][",",")
    print(content[315163])

    #content.replace(('"', '\\"')
    try:
        data = json.loads("".join(content))
    except ValueError as e:
        print (e)
    data = json.loads("".join(content))

pprint(data)