import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FRRAndFAR(object):
    def __init__(self, max_fpr=0.01):
        self.pos_score_list = []
        self.neg_score_list = []
        self.max_fpr = max_fpr

    def add(self, y_labels, y_scores):
        for y_label, y_score in zip(y_labels, y_scores):
            if y_label == 0:
                self.neg_score_list.append(y_score)
            else:
                self.pos_score_list.append(y_score)

    def reset(self):
        self.pos_score_list = []
        self.neg_score_list = []

    def calculate_eer(self, frrs, fars):
        n = len(frrs)
        eer = 1.0
        min = 1.0
        index = 0
        for i in range(n):
            if abs(frrs[i] - fars[i]) < min:
                eer = (frrs[i] + fars[i]) / 2
                min = abs(frrs[i] - fars[i])
                index = i
        return eer, index

    def calculate(self):
        frrs, fars, thresholds = [], [], []
        pos_score_list = np.array(self.pos_score_list)
        neg_score_list = np.array(self.neg_score_list)
        if len(pos_score_list) == 0:
            msg = f"The number of positive samples is 0, please add positive samples."
            logger.warning(msg)
            return frrs, fars, thresholds, None, None
        if len(neg_score_list) == 0:
            msg = f"The number of negative samples is 0, please add negative samples."
            logger.warning(msg)
            return frrs, fars, thresholds, None, None
        for i in range(0, 100):
            threshold = i / 100.
            frr = np.sum(pos_score_list < threshold) / len(pos_score_list)
            far = np.sum(neg_score_list > threshold) / len(neg_score_list)
            frrs.append(frr)
            fars.append(far)
            thresholds.append(threshold)
        eer, index = self.calculate_eer(frrs=frrs, fars=fars)
        return frrs, fars, thresholds, eer, index
