import os

class LabelToInt:
    def __init__(self, data_path):
        self.labels = []
        self.data_path = data_path
        self.toint = {}

        all_path = os.listdir(self.data_path)
        count = 0
        for label in all_path:
            self.toint[label] = count
            count = count + 1

    def getint(self, label):
        return self.toint[label]

