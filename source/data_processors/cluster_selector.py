

class ClusterSelector():
    def __init__(self, cluster_label):
        self.cluster_label = cluster_label

    def select(self, cluster_label):
        if type(cluster_label) is list:
            cluster_label = cluster_label[0]
        return (int(self.cluster_label) == int(cluster_label))