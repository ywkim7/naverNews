import pickle
import numpy as np
from sklearn.model_selection import train_test_split

FILE_PATH = '/app/dataset/data_400_4_8'

class navernewsData:
    def __init__(self):
        with open(FILE_PATH, 'rb') as f:
            input_label = pickle.load(f)

        self.X = np.array(list(input_label.keys()))
        self.Y = np.array(list(input_label.values()))
