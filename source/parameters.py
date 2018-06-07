import os
import json

class Parameters:
    def __init__(self):
        self.paramaters = {}

    def add_parameter(self, key, value):
        self.paramaters[str(key)] = str(value)

    def write_parameters(self, exports_folder, timestamp):
        labels_count = self.paramaters["Unique_Labels"]
        filename = "parameters_" + labels_count + "_labels_" + timestamp + '.txt'
        filepath = os.path.join(exports_folder, filename)
        exDict = {'exDict': self.paramaters}
        with open(filepath,'w') as out:
            out.write(json.dumps(exDict))